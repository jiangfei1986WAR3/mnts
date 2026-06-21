from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from compare_v2_vs_classic_indicators import compute_macd
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direction B refinement: use MNTS V2 as a tactical filter / position scaler on top of MACD."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/macd_v2_tactical_run")
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=4.0,
        help="Per-turnover transaction cost in basis points. 4 bps = 0.0004.",
    )
    return parser.parse_args()


def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = len(df) // 2
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    macd_line, macd_signal, macd_hist = compute_macd(validation["close"])
    validation["macd_line"] = macd_line
    validation["macd_signal_line"] = macd_signal
    validation["macd_hist"] = macd_hist
    validation["macd_signal"] = 0.0
    validation.loc[validation["macd_line"] > validation["macd_signal_line"], "macd_signal"] = 1.0
    validation.loc[validation["macd_line"] < validation["macd_signal_line"], "macd_signal"] = -1.0
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(
        v2[["timestamp", "state", "high_vol_event"]],
        on="timestamp",
        how="inner",
    )
    return merged


def apply_mode(df: pd.DataFrame, mode: str) -> pd.Series:
    signal = df["macd_signal"]
    state = df["state"]

    if mode == "raw":
        return signal.astype(float)
    if mode == "no_fracture":
        return signal.where(state != "fracture", 0.0).astype(float)
    if mode == "cohesion_only":
        return signal.where(state == "cohesion", 0.0).astype(float)
    if mode == "defensive":
        weights = pd.Series(0.0, index=df.index)
        weights[state == "cohesion"] = 1.0
        weights[state == "drift"] = 0.5
        weights[state == "fracture"] = 0.0
        return (signal * weights).astype(float)
    if mode == "aggressive":
        weights = pd.Series(0.0, index=df.index)
        weights[state == "cohesion"] = 1.5
        weights[state == "drift"] = 1.0
        weights[state == "fracture"] = 0.0
        return (signal * weights).astype(float)
    if mode == "cohesion_boost":
        weights = pd.Series(0.0, index=df.index)
        weights[state == "cohesion"] = 1.5
        weights[state == "drift"] = 0.5
        weights[state == "fracture"] = 0.0
        return (signal * weights).astype(float)
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
    active_count = int(active.sum())
    annual_factor = np.sqrt(365 * 24 * 4)
    net_std = float(net_ret.std(ddof=0))
    gross_std = float(gross_ret.std(ddof=0))
    net_sharpe = float(net_ret.mean() / net_std * annual_factor) if net_std > 1e-12 else 0.0
    gross_sharpe = float(gross_ret.mean() / gross_std * annual_factor) if gross_std > 1e-12 else 0.0

    active_high_vol_rate = float(work.loc[active, "high_vol_event"].mean()) if active_count > 0 else 0.0

    return {
        "bars": int(len(work)),
        "active_bars": active_count,
        "exposure": float(active.mean()),
        "avg_abs_position": float(work["position"].abs().mean()),
        "turnover_sum": float(turnover.sum()),
        "gross_total_return": float(gross_equity.iloc[-1] - 1.0),
        "net_total_return": float(net_equity.iloc[-1] - 1.0),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "active_high_vol_rate": active_high_vol_rate,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df = build_base_frame(args.input_csv, args.v2_validation_csv)
    modes = [
        "raw",
        "no_fracture",
        "cohesion_only",
        "defensive",
        "aggressive",
        "cohesion_boost",
    ]

    rows: List[Dict[str, float]] = []
    for mode in modes:
        position = apply_mode(df, mode)
        metrics = compute_metrics(df, position, fee_rate=fee_rate)
        rows.append({"mode": mode, **metrics})

    result = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    result.to_csv(output_dir / "macd_v2_tactical_comparison.csv", index=False)

    summary = {
        "fee_bps": float(args.fee_bps),
        "best_net_sharpe": result.iloc[0][["mode", "net_sharpe"]].to_dict(),
        "best_net_total_return": result.sort_values("net_total_return", ascending=False).iloc[0][
            ["mode", "net_total_return"]
        ].to_dict(),
        "smallest_net_drawdown": result.sort_values("net_max_drawdown", ascending=False).iloc[0][
            ["mode", "net_max_drawdown"]
        ].to_dict(),
    }
    (output_dir / "macd_v2_tactical_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
