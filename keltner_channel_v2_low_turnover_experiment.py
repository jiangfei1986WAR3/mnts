from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import compute_metrics
from keltner_channel_v2_experiment import build_base_frame
from mnts_min_validation import ensure_output_dir


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    atr_mult: float
    signal_confirm_bars: int
    exit_confirm_bars: int
    min_hold_bars: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-turnover engineering pass for Keltner channel + V2 strategy."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/keltner_channel_v2_low_turnover_run")
    parser.add_argument("--fee-bps", type=float, default=4.0)
    return parser.parse_args()


def streaks_from_bool(series: pd.Series) -> pd.Series:
    arr = series.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(arr), dtype=int)
    streak = 0
    for i, flag in enumerate(arr):
        if flag:
            streak += 1
        else:
            streak = 0
        out[i] = streak
    return pd.Series(out, index=series.index)


def build_desired_position(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    upper = df["ema20"] + config.atr_mult * df["atr14"]
    lower = df["ema20"] - config.atr_mult * df["atr14"]

    long_break = df["close"] > upper
    short_break = df["close"] < lower
    long_streak = streaks_from_bool(long_break)
    short_streak = streaks_from_bool(short_break)
    weak_long_exit = streaks_from_bool(df["close"] < df["ema20"])
    weak_short_exit = streaks_from_bool(df["close"] > df["ema20"])
    frac_streak = streaks_from_bool(df["state"] == "fracture")
    coh_streak = streaks_from_bool(df["state"] == "cohesion")

    desired = pd.Series(0.0, index=df.index, dtype=float)
    desired.loc[long_streak >= config.signal_confirm_bars] = 1.0
    desired.loc[short_streak >= config.signal_confirm_bars] = -1.0

    long_weak = weak_long_exit >= config.exit_confirm_bars
    short_weak = weak_short_exit >= config.exit_confirm_bars
    desired.loc[(desired > 0) & long_weak] = 0.0
    desired.loc[(desired < 0) & short_weak] = 0.0

    if config.mode == "raw":
        return desired

    if config.mode == "fracture_confirm_2":
        desired.loc[frac_streak >= 2] = 0.0
        return desired

    if config.mode == "fracture_confirm_3":
        desired.loc[frac_streak >= 3] = 0.0
        return desired

    if config.mode == "cohesion_confirm_2":
        desired = desired.where(coh_streak >= 2, 0.0)
        return desired

    if config.mode == "cohesion_confirm_3":
        desired = desired.where(coh_streak >= 3, 0.0)
        return desired

    if config.mode == "defensive_scale":
        weights = pd.Series(0.5, index=df.index, dtype=float)
        weights[df["state"] == "cohesion"] = 1.0
        weights[df["state"] == "fracture"] = 0.0
        weights.loc[frac_streak >= 2] = 0.0
        desired = desired * weights
        return desired

    raise ValueError(f"Unknown mode: {config.mode}")


def apply_min_hold(desired: pd.Series, min_hold_bars: int) -> pd.Series:
    arr = desired.fillna(0.0).to_numpy(dtype=float)
    actual = np.zeros(len(arr), dtype=float)
    current = 0.0
    bars_since_change = min_hold_bars

    for i, wanted in enumerate(arr):
        if wanted == current:
            actual[i] = current
            bars_since_change += 1
            continue

        if bars_since_change >= min_hold_bars:
            current = wanted
            bars_since_change = 0
        else:
            bars_since_change += 1

        actual[i] = current

    return pd.Series(actual, index=desired.index)


def strategy_grid() -> List[StrategyConfig]:
    return [
        StrategyConfig("raw_m20_confirm2_exit2_hold16", 2.0, 2, 2, 16, "raw"),
        StrategyConfig("raw_m20_confirm2_exit2_hold24", 2.0, 2, 2, 24, "raw"),
        StrategyConfig("fracture2_m20_confirm2_exit2_hold16", 2.0, 2, 2, 16, "fracture_confirm_2"),
        StrategyConfig("fracture3_m20_confirm2_exit2_hold24", 2.0, 2, 2, 24, "fracture_confirm_3"),
        StrategyConfig("cohesion2_m20_confirm2_exit2_hold16", 2.0, 2, 2, 16, "cohesion_confirm_2"),
        StrategyConfig("cohesion3_m20_confirm2_exit2_hold24", 2.0, 2, 2, 24, "cohesion_confirm_3"),
        StrategyConfig("defensive_m20_confirm2_exit2_hold16", 2.0, 2, 2, 16, "defensive_scale"),
        StrategyConfig("defensive_m20_confirm2_exit2_hold24", 2.0, 2, 2, 24, "defensive_scale"),
    ]


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df = build_base_frame(args.input_csv, args.v2_validation_csv)
    rows: List[Dict[str, float]] = []
    for config in strategy_grid():
        desired = build_desired_position(df, config)
        actual = apply_min_hold(desired, config.min_hold_bars)
        metrics = compute_metrics(df, actual, fee_rate)
        rows.append(
            {
                "config": config.label,
                "atr_mult": config.atr_mult,
                "signal_confirm_bars": config.signal_confirm_bars,
                "exit_confirm_bars": config.exit_confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                "mode": config.mode,
                **metrics,
            }
        )

    result = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    result.to_csv(output_dir / "keltner_channel_v2_low_turnover_comparison.csv", index=False)

    summary = {
        "fee_bps": float(args.fee_bps),
        "best_net_sharpe": result.iloc[0][["config", "net_sharpe"]].to_dict(),
        "best_net_total_return": result.sort_values("net_total_return", ascending=False).iloc[0][
            ["config", "net_total_return"]
        ].to_dict(),
        "smallest_net_drawdown": result.sort_values("net_max_drawdown", ascending=False).iloc[0][
            ["config", "net_max_drawdown"]
        ].to_dict(),
    }
    (output_dir / "keltner_channel_v2_low_turnover_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
