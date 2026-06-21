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
    adx_threshold: float
    confirm_bars: int
    exit_confirm_bars: int
    min_hold_bars: int

    @property
    def label(self) -> str:
        return (
            f"adxdi_{self.gate_mode}"
            f"_adx{int(self.adx_threshold)}"
            f"_entry{self.confirm_bars}"
            f"_exit{self.exit_confirm_bars}"
            f"_hold{self.min_hold_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ADX+DI regime skeleton with V2 enhancement.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/adxdi_v2_run")
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


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
        dtype=float,
    )

    tr_components = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    alpha = 1.0 / period
    atr = true_range.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smoothed = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * plus_dm_smoothed / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_smoothed / atr.replace(0.0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame(
        {
            "atr": atr.fillna(0.0),
            "plus_di": plus_di.fillna(0.0),
            "minus_di": minus_di.fillna(0.0),
            "adx": adx.fillna(0.0),
        }
    )


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    adx_frame = compute_adx(validation["high"], validation["low"], validation["close"], period=14)
    validation = pd.concat([validation, adx_frame], axis=1)
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

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
    strong_adx = df["adx"] >= config.adx_threshold
    long_setup = strong_adx & (df["plus_di"] > df["minus_di"])
    short_setup = strong_adx & (df["minus_di"] > df["plus_di"])
    long_streak = streaks_from_bool(long_setup)
    short_streak = streaks_from_bool(short_setup)
    flat_streak = streaks_from_bool(~strong_adx)

    positions = np.zeros(len(df), dtype=float)
    current_sign = 0.0
    bars_since_change = config.min_hold_bars

    for i, row in enumerate(df.itertuples(index=False)):
        long_ready = bool(long_setup.iloc[i]) and int(long_streak.iloc[i]) >= config.confirm_bars
        short_ready = bool(short_setup.iloc[i]) and int(short_streak.iloc[i]) >= config.confirm_bars
        weak_ready = int(flat_streak.iloc[i]) >= config.exit_confirm_bars
        changed = False

        if current_sign > 0:
            should_exit = weak_ready or (bool(short_setup.iloc[i]) and int(short_streak.iloc[i]) >= config.exit_confirm_bars)
            if should_exit and bars_since_change >= config.min_hold_bars:
                current_sign = 0.0
                changed = True
        elif current_sign < 0:
            should_exit = weak_ready or (bool(long_setup.iloc[i]) and int(long_streak.iloc[i]) >= config.exit_confirm_bars)
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
        for adx_threshold in [20.0, 25.0]:
            for confirm_bars in [1, 2]:
                for exit_confirm_bars in [1, 2]:
                    for min_hold_bars in [8, 16]:
                        configs.append(
                            StrategyConfig(
                                gate_mode=gate_mode,
                                adx_threshold=adx_threshold,
                                confirm_bars=confirm_bars,
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
                "adx_threshold": config.adx_threshold,
                "confirm_bars": config.confirm_bars,
                "exit_confirm_bars": config.exit_confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "adxdi_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "adxdi_best_by_gate.csv", index=False)

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
    (output_dir / "adxdi_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
