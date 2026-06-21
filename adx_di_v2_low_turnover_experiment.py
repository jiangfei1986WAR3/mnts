from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from adx_di_v2_experiment import build_base_frame
from engineer_pullback_breakout_v2 import compute_metrics
from mnts_min_validation import ensure_output_dir


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    adx_threshold: float
    signal_confirm_bars: int
    weak_exit_bars: int
    min_hold_bars: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-turnover engineering pass for ADX+DI+V2 strategy."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/adxdi_v2_low_turnover_run")
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
    strong_adx = df["adx"] >= config.adx_threshold
    long_setup = strong_adx & (df["plus_di"] > df["minus_di"])
    short_setup = strong_adx & (df["minus_di"] > df["plus_di"])

    long_streak = streaks_from_bool(long_setup)
    short_streak = streaks_from_bool(short_setup)
    weak_streak = streaks_from_bool(~strong_adx)
    frac_streak = streaks_from_bool(df["state"] == "fracture")
    coh_streak = streaks_from_bool(df["state"] == "cohesion")

    desired = pd.Series(0.0, index=df.index, dtype=float)
    desired.loc[long_streak >= config.signal_confirm_bars] = 1.0
    desired.loc[short_streak >= config.signal_confirm_bars] = -1.0

    if config.mode == "raw":
        desired.loc[weak_streak >= config.weak_exit_bars] = 0.0
        return desired

    if config.mode == "fracture_confirm_2":
        desired.loc[frac_streak >= 2] = 0.0
        desired.loc[weak_streak >= config.weak_exit_bars] = 0.0
        return desired

    if config.mode == "fracture_confirm_3":
        desired.loc[frac_streak >= 3] = 0.0
        desired.loc[weak_streak >= config.weak_exit_bars] = 0.0
        return desired

    if config.mode == "cohesion_confirm_2":
        desired = desired.where(coh_streak >= 2, 0.0)
        desired.loc[weak_streak >= config.weak_exit_bars] = 0.0
        return desired

    if config.mode == "cohesion_confirm_3":
        desired = desired.where(coh_streak >= 3, 0.0)
        desired.loc[weak_streak >= config.weak_exit_bars] = 0.0
        return desired

    if config.mode == "defensive_scale":
        weights = pd.Series(0.5, index=df.index, dtype=float)
        weights[df["state"] == "cohesion"] = 1.0
        weights[df["state"] == "fracture"] = 0.0
        weights.loc[frac_streak >= 2] = 0.0
        desired = desired * weights
        desired.loc[weak_streak >= config.weak_exit_bars] = 0.0
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
        StrategyConfig("raw_adx20_confirm1_exit2_hold16", 20.0, 1, 2, 16, "raw"),
        StrategyConfig("raw_adx25_confirm2_exit2_hold24", 25.0, 2, 2, 24, "raw"),
        StrategyConfig("fracture2_adx20_confirm2_exit2_hold16", 20.0, 2, 2, 16, "fracture_confirm_2"),
        StrategyConfig("fracture3_adx25_confirm2_exit2_hold24", 25.0, 2, 2, 24, "fracture_confirm_3"),
        StrategyConfig("cohesion2_adx20_confirm2_exit1_hold16", 20.0, 2, 1, 16, "cohesion_confirm_2"),
        StrategyConfig("cohesion3_adx25_confirm2_exit2_hold24", 25.0, 2, 2, 24, "cohesion_confirm_3"),
        StrategyConfig("defensive_adx20_confirm2_exit2_hold16", 20.0, 2, 2, 16, "defensive_scale"),
        StrategyConfig("defensive_adx25_confirm2_exit2_hold24", 25.0, 2, 2, 24, "defensive_scale"),
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
                "adx_threshold": config.adx_threshold,
                "signal_confirm_bars": config.signal_confirm_bars,
                "weak_exit_bars": config.weak_exit_bars,
                "min_hold_bars": config.min_hold_bars,
                "mode": config.mode,
                **metrics,
            }
        )

    result = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    result.to_csv(output_dir / "adxdi_v2_low_turnover_comparison.csv", index=False)

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
    (output_dir / "adxdi_v2_low_turnover_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
