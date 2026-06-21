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
    atr_mult: float
    confirm_bars: int
    min_hold_bars: int
    exit_confirm_bars: int

    @property
    def label(self) -> str:
        mult = str(self.atr_mult).replace(".", "")
        return (
            f"kelt_{self.gate_mode}"
            f"_m{mult}"
            f"_confirm{self.confirm_bars}"
            f"_hold{self.min_hold_bars}"
            f"_exit{self.exit_confirm_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Keltner channel trend skeleton with V2 enhancement.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/keltner_channel_v2_run")
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


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    close = validation["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    atr14 = compute_atr(validation["high"], validation["low"], close, period=14)

    validation["ema20"] = ema20
    validation["atr14"] = atr14
    validation["next_bar_ret"] = np.log(close.shift(-1) / close)

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")
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
    upper = df["ema20"] + config.atr_mult * df["atr14"]
    lower = df["ema20"] - config.atr_mult * df["atr14"]

    long_break = df["close"] > upper
    short_break = df["close"] < lower
    long_streak = streaks_from_bool(long_break)
    short_streak = streaks_from_bool(short_break)
    weak_long_exit = streaks_from_bool(df["close"] < df["ema20"])
    weak_short_exit = streaks_from_bool(df["close"] > df["ema20"])
    frac_streak = streaks_from_bool(df["state"] == "fracture")

    positions = np.zeros(len(df), dtype=float)
    current_sign = 0.0
    bars_since_change = config.min_hold_bars

    for i, row in enumerate(df.itertuples(index=False)):
        long_ready = int(long_streak.iloc[i]) >= config.confirm_bars
        short_ready = int(short_streak.iloc[i]) >= config.confirm_bars
        changed = False

        if current_sign > 0:
            should_exit = (
                int(weak_long_exit.iloc[i]) >= config.exit_confirm_bars
                or int(frac_streak.iloc[i]) >= 2
                or short_ready
            )
            if should_exit and bars_since_change >= config.min_hold_bars:
                current_sign = 0.0
                changed = True
        elif current_sign < 0:
            should_exit = (
                int(weak_short_exit.iloc[i]) >= config.exit_confirm_bars
                or int(frac_streak.iloc[i]) >= 2
                or long_ready
            )
            if should_exit and bars_since_change >= config.min_hold_bars:
                current_sign = 0.0
                changed = True

        if current_sign == 0.0:
            if long_ready:
                current_sign = 1.0
                changed = True
            elif short_ready:
                current_sign = -1.0
                changed = True

        positions[i] = current_sign * gate_weight(str(row.state), config.gate_mode)
        if changed:
            bars_since_change = 0
        else:
            bars_since_change += 1

    return pd.Series(positions, index=df.index)


def strategy_grid() -> List[StrategyConfig]:
    configs: List[StrategyConfig] = []
    for gate_mode in ["raw", "no_fracture", "cohesion_only", "defensive_scale"]:
        for atr_mult in [1.5, 2.0]:
            for confirm_bars in [1, 2]:
                for min_hold_bars in [8, 16]:
                    for exit_confirm_bars in [1, 2]:
                        configs.append(
                            StrategyConfig(
                                gate_mode=gate_mode,
                                atr_mult=atr_mult,
                                confirm_bars=confirm_bars,
                                min_hold_bars=min_hold_bars,
                                exit_confirm_bars=exit_confirm_bars,
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
                "atr_mult": config.atr_mult,
                "confirm_bars": config.confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                "exit_confirm_bars": config.exit_confirm_bars,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "keltner_channel_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "keltner_channel_best_by_gate.csv", index=False)

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
    (output_dir / "keltner_channel_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
