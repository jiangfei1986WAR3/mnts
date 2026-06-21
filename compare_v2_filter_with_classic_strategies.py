from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from compare_v2_vs_classic_indicators import build_indicator_frame, compute_macd, compute_rsi
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare classic strategies before and after applying MNTS 15m V2 state filter."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/filter_compare_run")
    return parser.parse_args()


def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = len(df) // 2
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ma_signal"] = 0
    out.loc[out["sma20"] > out["sma60"], "ma_signal"] = 1
    out.loc[out["sma20"] < out["sma60"], "ma_signal"] = -1

    macd_line, macd_signal, _ = compute_macd(out["close"])
    out["macd_signal_raw"] = 0
    out.loc[macd_line > macd_signal, "macd_signal_raw"] = 1
    out.loc[macd_line < macd_signal, "macd_signal_raw"] = -1

    rsi = compute_rsi(out["close"], 14)
    out["rsi_signal"] = 0
    out.loc[rsi < 30, "rsi_signal"] = 1
    out.loc[rsi > 70, "rsi_signal"] = -1

    out["next_bar_ret"] = np.log(out["close"].shift(-1) / out["close"])
    return out


def apply_filter(signal: pd.Series, state: pd.Series, mode: str) -> pd.Series:
    if mode == "raw":
        return signal
    if mode == "no_fracture":
        return signal.where(state != "fracture", 0)
    if mode == "cohesion_only":
        return signal.where(state == "cohesion", 0)
    raise ValueError(f"Unknown filter mode: {mode}")


def compute_metrics(df: pd.DataFrame, position_col: str) -> Dict[str, float]:
    ret = (df[position_col] * df["next_bar_ret"]).fillna(0.0)
    equity = np.exp(ret.cumsum())
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    active = df[position_col] != 0
    active_count = int(active.sum())
    trade_switches = int(df[position_col].diff().fillna(0).ne(0).sum())
    annual_factor = np.sqrt(365 * 24 * 4)
    ret_std = float(ret.std(ddof=0))
    sharpe = float(ret.mean() / ret_std * annual_factor) if ret_std > 1e-12 else 0.0

    active_high_vol_rate = float(df.loc[active, "high_vol_event"].mean()) if active_count > 0 else 0.0

    return {
        "bars": int(len(df)),
        "active_bars": active_count,
        "exposure": float(active.mean()),
        "trade_switches": trade_switches,
        "total_return": float(equity.iloc[-1] - 1.0),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "mean_bar_return": float(ret.mean()),
        "ret_std": ret_std,
        "active_high_vol_rate": active_high_vol_rate,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    raw_df = load_ohlcv_csv(args.input_csv)
    indicator_df = build_indicator_frame(raw_df, horizon=16)
    validation = split_validation(indicator_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    v2 = pd.read_csv(args.v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")

    merged = validation.merge(
        v2[["timestamp", "state", "high_vol_event"]],
        on="timestamp",
        how="inner",
    )
    merged = build_signals(merged)

    strategy_map = {
        "ma": "ma_signal",
        "macd": "macd_signal_raw",
        "rsi": "rsi_signal",
    }
    filter_modes = ["raw", "no_fracture", "cohesion_only"]

    rows: List[Dict[str, float]] = []
    for strategy_name, signal_col in strategy_map.items():
        for mode in filter_modes:
            pos = apply_filter(merged[signal_col], merged["state"], mode=mode)
            work = merged.copy()
            work["position"] = pos
            metrics = compute_metrics(work, "position")
            rows.append({"strategy": strategy_name, "filter_mode": mode, **metrics})

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "strategy_filter_comparison.csv", index=False)

    summary = {
        "best_sharpe": results.sort_values("annualized_sharpe", ascending=False).iloc[0][
            ["strategy", "filter_mode", "annualized_sharpe"]
        ].to_dict(),
        "best_total_return": results.sort_values("total_return", ascending=False).iloc[0][
            ["strategy", "filter_mode", "total_return"]
        ].to_dict(),
        "smallest_drawdown": results.sort_values("max_drawdown", ascending=False).iloc[0][
            ["strategy", "filter_mode", "max_drawdown"]
        ].to_dict(),
    }
    (output_dir / "strategy_filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
