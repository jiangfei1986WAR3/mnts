from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import compute_metrics
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


@dataclass(frozen=True)
class StrategyConfig:
    gate_mode: str
    entry_confirm_bars: int
    exit_confirm_bars: int
    min_hold_bars: int

    @property
    def label(self) -> str:
        return (
            f"slowreg_{self.gate_mode}"
            f"_entry{self.entry_confirm_bars}"
            f"_exit{self.exit_confirm_bars}"
            f"_hold{self.min_hold_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a slow EMA20/EMA144 regime-hold skeleton with V2 state enhancement."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/slow_regime_hold_v2_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    return parser.parse_args()


def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = len(df) // 2
    return df.iloc[split_idx:].copy().reset_index(drop=True)


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


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    close = validation["close"]
    validation["ema20"] = close.ewm(span=20, adjust=False).mean()
    validation["ema144"] = close.ewm(span=144, adjust=False).mean()
    validation["ema20_slope_8"] = validation["ema20"].pct_change(8).fillna(0.0)
    validation["next_bar_ret"] = np.log(close.shift(-1) / close)

    validation["long_regime"] = (validation["ema20"] > validation["ema144"]) & (validation["ema20_slope_8"] > 0)
    validation["short_regime"] = (validation["ema20"] < validation["ema144"]) & (validation["ema20_slope_8"] < 0)

    validation["long_entry_setup"] = validation["long_regime"] & (validation["close"] > validation["ema20"])
    validation["short_entry_setup"] = validation["short_regime"] & (validation["close"] < validation["ema20"])
    validation["long_exit_setup"] = ~validation["long_regime"] | (validation["close"] < validation["ema20"])
    validation["short_exit_setup"] = ~validation["short_regime"] | (validation["close"] > validation["ema20"])

    validation["long_entry_streak"] = streaks_from_bool(validation["long_entry_setup"])
    validation["short_entry_streak"] = streaks_from_bool(validation["short_entry_setup"])
    validation["long_exit_streak"] = streaks_from_bool(validation["long_exit_setup"])
    validation["short_exit_streak"] = streaks_from_bool(validation["short_exit_setup"])

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")
    merged["cohesion_streak"] = streaks_from_bool(merged["state"] == "cohesion")
    merged["fracture_streak"] = streaks_from_bool(merged["state"] == "fracture")
    return merged


def gate_weight(state: str, gate_mode: str) -> float:
    if gate_mode == "raw":
        return 1.0
    if gate_mode == "no_fracture":
        return 0.0 if state == "fracture" else 1.0
    if gate_mode == "cohesion_only":
        return 1.0 if state == "cohesion" else 0.0
    if gate_mode == "defensive_scale":
        if state == "cohesion":
            return 1.0
        if state == "drift":
            return 0.5
        return 0.0
    raise ValueError(f"Unknown gate mode: {gate_mode}")


def simulate_strategy(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    positions = np.zeros(len(df), dtype=float)
    current_sign = 0.0
    bars_since_change = config.min_hold_bars

    for i, row in enumerate(df.itertuples(index=False)):
        long_entry_ready = bool(row.long_entry_setup) and int(row.long_entry_streak) >= config.entry_confirm_bars
        short_entry_ready = bool(row.short_entry_setup) and int(row.short_entry_streak) >= config.entry_confirm_bars
        long_exit_ready = int(row.long_exit_streak) >= config.exit_confirm_bars
        short_exit_ready = int(row.short_exit_streak) >= config.exit_confirm_bars
        changed = False

        if current_sign > 0 and long_exit_ready and bars_since_change >= config.min_hold_bars:
            current_sign = 0.0
            changed = True
        elif current_sign < 0 and short_exit_ready and bars_since_change >= config.min_hold_bars:
            current_sign = 0.0
            changed = True

        if current_sign == 0.0:
            if long_entry_ready:
                current_sign = 1.0
                changed = True
            elif short_entry_ready:
                current_sign = -1.0
                changed = True

        weight = gate_weight(str(row.state), config.gate_mode)
        positions[i] = current_sign * weight

        if changed:
            bars_since_change = 0
        else:
            bars_since_change += 1

    return pd.Series(positions, index=df.index)


def strategy_grid() -> List[StrategyConfig]:
    configs: List[StrategyConfig] = []
    for gate_mode in ["raw", "no_fracture", "cohesion_only", "defensive_scale"]:
        for entry_confirm_bars in [1, 2]:
            for exit_confirm_bars in [1, 2]:
                for min_hold_bars in [8, 16]:
                    configs.append(
                        StrategyConfig(
                            gate_mode=gate_mode,
                            entry_confirm_bars=entry_confirm_bars,
                            exit_confirm_bars=exit_confirm_bars,
                            min_hold_bars=min_hold_bars,
                        )
                    )
    return configs


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    base = build_base_frame(args.input_csv, args.v2_validation_csv)
    rows: List[Dict[str, float]] = []
    for config in strategy_grid():
        position = simulate_strategy(base, config)
        metrics = compute_metrics(base, position, fee_rate)
        rows.append(
            {
                "config_label": config.label,
                "gate_mode": config.gate_mode,
                "entry_confirm_bars": config.entry_confirm_bars,
                "exit_confirm_bars": config.exit_confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "slow_regime_hold_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "slow_regime_hold_best_by_gate.csv", index=False)

    raw_best = best_by_gate[best_by_gate["gate_mode"] == "raw"].iloc[0]
    best_overall = results.iloc[0]
    best_non_raw = results[results["gate_mode"] != "raw"].iloc[0]

    summary = {
        "fee_bps": float(args.fee_bps),
        "config_count": int(len(results)),
        "best_overall": {
            "config_label": str(best_overall["config_label"]),
            "gate_mode": str(best_overall["gate_mode"]),
            "net_total_return": float(best_overall["net_total_return"]),
            "net_sharpe": float(best_overall["net_sharpe"]),
            "net_max_drawdown": float(best_overall["net_max_drawdown"]),
            "turnover_sum": float(best_overall["turnover_sum"]),
        },
        "best_raw": {
            "config_label": str(raw_best["config_label"]),
            "net_total_return": float(raw_best["net_total_return"]),
            "net_sharpe": float(raw_best["net_sharpe"]),
            "net_max_drawdown": float(raw_best["net_max_drawdown"]),
        },
        "best_v2_enhanced": {
            "config_label": str(best_non_raw["config_label"]),
            "gate_mode": str(best_non_raw["gate_mode"]),
            "net_total_return": float(best_non_raw["net_total_return"]),
            "net_sharpe": float(best_non_raw["net_sharpe"]),
            "net_max_drawdown": float(best_non_raw["net_max_drawdown"]),
        },
        "improvement_vs_best_raw": {
            "net_total_return_delta": float(best_non_raw["net_total_return"] - raw_best["net_total_return"]),
            "net_sharpe_delta": float(best_non_raw["net_sharpe"] - raw_best["net_sharpe"]),
            "net_max_drawdown_delta": float(best_non_raw["net_max_drawdown"] - raw_best["net_max_drawdown"]),
        },
    }
    (output_dir / "slow_regime_hold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
