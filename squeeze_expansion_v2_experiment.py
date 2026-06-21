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
    squeeze_quantile: float
    release_lookback: int
    confirm_bars: int
    min_hold_bars: int

    @property
    def label(self) -> str:
        q = int(self.squeeze_quantile * 100)
        return (
            f"sqz_{self.gate_mode}_q{q}"
            f"_look{self.release_lookback}"
            f"_confirm{self.confirm_bars}"
            f"_hold{self.min_hold_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate squeeze-expansion trend skeleton with V2 enhancement.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/squeeze_expansion_v2_run")
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


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    values = np.full(len(series), np.nan, dtype=float)
    arr = series.to_numpy(dtype=float)
    for end in range(window, len(arr) + 1):
        current = arr[end - window : end]
        last = current[-1]
        values[end - 1] = float(np.mean(current <= last))
    return pd.Series(values, index=series.index)


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    close = validation["close"]
    mid = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    bandwidth = (upper - lower) / mid.replace(0.0, np.nan)

    validation["bb_mid"] = mid
    validation["bb_upper"] = upper
    validation["bb_lower"] = lower
    validation["bb_bandwidth"] = bandwidth.ffill()
    validation["bb_bw_pct_rank"] = rolling_percentile_rank(validation["bb_bandwidth"], window=96)
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")
    return merged


def gate_mode_weight(state: str, gate_mode: str) -> float:
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
    squeeze = df["bb_bw_pct_rank"] <= config.squeeze_quantile
    recent_squeeze = squeeze.rolling(config.release_lookback, min_periods=1).max().fillna(0).astype(bool)

    long_break = recent_squeeze & (df["close"] > df["bb_upper"])
    short_break = recent_squeeze & (df["close"] < df["bb_lower"])
    long_streak = streaks_from_bool(long_break)
    short_streak = streaks_from_bool(short_break)
    weak_long_exit = streaks_from_bool(df["close"] < df["bb_mid"])
    weak_short_exit = streaks_from_bool(df["close"] > df["bb_mid"])
    frac_streak = streaks_from_bool(df["state"] == "fracture")

    positions = np.zeros(len(df), dtype=float)
    current_sign = 0.0
    bars_since_change = config.min_hold_bars

    for i, row in enumerate(df.itertuples(index=False)):
        long_ready = int(long_streak.iloc[i]) >= config.confirm_bars
        short_ready = int(short_streak.iloc[i]) >= config.confirm_bars
        changed = False

        if current_sign > 0:
            should_exit = int(weak_long_exit.iloc[i]) >= 2 or int(frac_streak.iloc[i]) >= 2 or short_ready
            if should_exit and bars_since_change >= config.min_hold_bars:
                current_sign = 0.0
                changed = True
        elif current_sign < 0:
            should_exit = int(weak_short_exit.iloc[i]) >= 2 or int(frac_streak.iloc[i]) >= 2 or long_ready
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

        positions[i] = current_sign * gate_mode_weight(str(row.state), config.gate_mode)
        if changed:
            bars_since_change = 0
        else:
            bars_since_change += 1

    return pd.Series(positions, index=df.index)


def strategy_grid() -> List[StrategyConfig]:
    configs: List[StrategyConfig] = []
    for gate_mode in ["raw", "no_fracture", "cohesion_only", "defensive_scale"]:
        for squeeze_quantile in [0.15, 0.25]:
            for release_lookback in [8, 16]:
                for confirm_bars in [1, 2]:
                    for min_hold_bars in [8, 16]:
                        configs.append(
                            StrategyConfig(
                                gate_mode=gate_mode,
                                squeeze_quantile=squeeze_quantile,
                                release_lookback=release_lookback,
                                confirm_bars=confirm_bars,
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
                "squeeze_quantile": config.squeeze_quantile,
                "release_lookback": config.release_lookback,
                "confirm_bars": config.confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "squeeze_expansion_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "squeeze_expansion_best_by_gate.csv", index=False)

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
    (output_dir / "squeeze_expansion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
