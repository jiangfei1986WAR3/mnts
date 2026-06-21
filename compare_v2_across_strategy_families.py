from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from compare_v2_vs_classic_indicators import compute_macd, compute_rsi
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare how MNTS V2 interacts with multiple strategy families on the same out-of-sample period."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/strategy_family_compare_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    return parser.parse_args()


def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = len(df) // 2
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def persistent_breakout_signal(close: pd.Series, lookback: int) -> pd.Series:
    upper = close.shift(1).rolling(lookback).max()
    lower = close.shift(1).rolling(lookback).min()
    raw = pd.Series(np.nan, index=close.index, dtype=float)
    raw.loc[close > upper] = 1.0
    raw.loc[close < lower] = -1.0
    return raw.ffill().fillna(0.0)


def pullback_trend_signal(close: pd.Series) -> pd.Series:
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    trend = pd.Series(0.0, index=close.index, dtype=float)
    trend.loc[ema20 > ema60] = 1.0
    trend.loc[ema20 < ema60] = -1.0

    deviation = close / ema20 - 1.0
    signal = pd.Series(0.0, index=close.index, dtype=float)
    signal.loc[(trend > 0) & (deviation < -0.003)] = 1.0
    signal.loc[(trend < 0) & (deviation > 0.003)] = -1.0
    return signal


def build_strategy_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["ma_cross"] = 0.0
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    out.loc[sma20 > sma60, "ma_cross"] = 1.0
    out.loc[sma20 < sma60, "ma_cross"] = -1.0

    macd_line, macd_signal, _ = compute_macd(close)
    out["macd_regime"] = 0.0
    out.loc[macd_line > macd_signal, "macd_regime"] = 1.0
    out.loc[macd_line < macd_signal, "macd_regime"] = -1.0

    out["donchian_20"] = persistent_breakout_signal(close, lookback=20)
    out["breakout_48"] = persistent_breakout_signal(close, lookback=48)

    rsi = compute_rsi(close, 14)
    out["rsi_reversion"] = 0.0
    out.loc[rsi < 30, "rsi_reversion"] = 1.0
    out.loc[rsi > 70, "rsi_reversion"] = -1.0

    out["pullback_trend"] = pullback_trend_signal(close)
    out["next_bar_ret"] = np.log(close.shift(-1) / close)
    return out


def apply_v2_mode(signal: pd.Series, state: pd.Series, mode: str) -> pd.Series:
    signal = signal.astype(float)
    if mode == "raw":
        return signal
    if mode == "no_fracture":
        return signal.where(state != "fracture", 0.0)
    if mode == "cohesion_only":
        return signal.where(state == "cohesion", 0.0)
    if mode == "defensive_scale":
        weights = pd.Series(0.0, index=signal.index, dtype=float)
        weights[state == "cohesion"] = 1.0
        weights[state == "drift"] = 0.5
        weights[state == "fracture"] = 0.0
        return signal * weights
    raise ValueError(f"Unknown mode: {mode}")


def compute_metrics(df: pd.DataFrame, position: pd.Series, fee_rate: float) -> Dict[str, float]:
    work = df.copy()
    work["position"] = position.fillna(0.0)
    turnover = work["position"].diff().abs().fillna(work["position"].abs())
    gross_ret = work["position"] * work["next_bar_ret"].fillna(0.0)
    fee = turnover * fee_rate
    net_ret = gross_ret - fee

    gross_equity = np.exp(gross_ret.cumsum())
    net_equity = np.exp(net_ret.cumsum())
    peak = np.maximum.accumulate(net_equity)
    drawdown = net_equity / peak - 1.0
    active = work["position"] != 0

    annual_factor = np.sqrt(365 * 24 * 4)
    gross_std = float(gross_ret.std(ddof=0))
    net_std = float(net_ret.std(ddof=0))
    gross_sharpe = float(gross_ret.mean() / gross_std * annual_factor) if gross_std > 1e-12 else 0.0
    net_sharpe = float(net_ret.mean() / net_std * annual_factor) if net_std > 1e-12 else 0.0

    return {
        "bars": int(len(work)),
        "active_bars": int(active.sum()),
        "exposure": float(active.mean()),
        "avg_abs_position": float(work["position"].abs().mean()),
        "turnover_sum": float(turnover.sum()),
        "gross_total_return": float(gross_equity.iloc[-1] - 1.0),
        "net_total_return": float(net_equity.iloc[-1] - 1.0),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "active_high_vol_rate": float(work.loc[active, "high_vol_event"].mean()) if active.any() else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    raw_df = load_ohlcv_csv(args.input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    v2 = pd.read_csv(args.v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")
    merged = build_strategy_positions(merged)

    strategy_groups = {
        "ma_cross": "trend",
        "macd_regime": "trend",
        "donchian_20": "breakout",
        "breakout_48": "breakout",
        "pullback_trend": "trend",
        "rsi_reversion": "mean_reversion",
    }
    modes = ["raw", "no_fracture", "cohesion_only", "defensive_scale"]

    rows: List[Dict[str, float]] = []
    for strategy_name, family in strategy_groups.items():
        for mode in modes:
            position = apply_v2_mode(merged[strategy_name], merged["state"], mode)
            metrics = compute_metrics(merged, position, fee_rate=fee_rate)
            rows.append({"strategy": strategy_name, "family": family, "v2_mode": mode, **metrics})

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "strategy_family_compare.csv", index=False)

    improvement_rows: List[Dict[str, float]] = []
    for strategy_name, group in results.groupby("strategy"):
        raw_row = group[group["v2_mode"] == "raw"].iloc[0]
        best_row = group.sort_values("net_sharpe", ascending=False).iloc[0]
        improvement_rows.append(
            {
                "strategy": strategy_name,
                "family": str(best_row["family"]),
                "best_mode": str(best_row["v2_mode"]),
                "raw_net_sharpe": float(raw_row["net_sharpe"]),
                "best_net_sharpe": float(best_row["net_sharpe"]),
                "net_sharpe_improvement": float(best_row["net_sharpe"] - raw_row["net_sharpe"]),
                "raw_net_total_return": float(raw_row["net_total_return"]),
                "best_net_total_return": float(best_row["net_total_return"]),
                "net_return_improvement": float(best_row["net_total_return"] - raw_row["net_total_return"]),
            }
        )

    improvement = pd.DataFrame(improvement_rows).sort_values(
        ["net_sharpe_improvement", "net_return_improvement"], ascending=False
    )
    improvement.to_csv(output_dir / "strategy_family_improvement.csv", index=False)

    summary = {
        "fee_bps": float(args.fee_bps),
        "best_overall_by_net_sharpe": results.iloc[0][["strategy", "family", "v2_mode", "net_sharpe"]].to_dict(),
        "best_overall_by_net_return": results.sort_values("net_total_return", ascending=False).iloc[0][
            ["strategy", "family", "v2_mode", "net_total_return"]
        ].to_dict(),
        "most_improved_strategy": improvement.iloc[0][
            ["strategy", "family", "best_mode", "net_sharpe_improvement", "net_return_improvement"]
        ].to_dict(),
    }
    (output_dir / "strategy_family_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
