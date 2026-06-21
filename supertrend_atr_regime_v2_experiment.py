from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import compute_metrics
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


@dataclass(frozen=True)
class StrategyConfig:
    gate_mode: str
    atr_period: int
    atr_mult: float
    confirm_bars: int
    exit_confirm_bars: int
    min_hold_bars: int

    @property
    def label(self) -> str:
        mult = str(self.atr_mult).replace(".", "")
        return (
            f"st_{self.gate_mode}"
            f"_p{self.atr_period}"
            f"_m{mult}"
            f"_entry{self.confirm_bars}"
            f"_exit{self.exit_confirm_bars}"
            f"_hold{self.min_hold_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Supertrend / ATR regime skeleton with V2 enhancement.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/supertrend_atr_regime_v2_run")
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


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    multiplier: float,
) -> pd.DataFrame:
    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=close.index, dtype=int)
    supertrend = pd.Series(np.nan, index=close.index, dtype=float)

    if len(close) == 0:
        return pd.DataFrame({"atr": atr, "supertrend": supertrend, "trend": trend})

    supertrend.iloc[0] = basic_lower.iloc[0]

    for i in range(1, len(close)):
        prev_close = close.iloc[i - 1]

        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or prev_close > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or prev_close < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] > 0 else final_upper.iloc[i]

    return pd.DataFrame(
        {
            "atr": atr.fillna(0.0),
            "supertrend": supertrend.ffill().bfill(),
            "trend": trend,
        }
    )


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")
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


def precompute_regimes(df: pd.DataFrame) -> Dict[Tuple[int, float], Dict[str, pd.Series]]:
    cache: Dict[Tuple[int, float], Dict[str, pd.Series]] = {}
    close = df["close"]
    for atr_period in [10, 14]:
        for atr_mult in [2.0, 3.0]:
            st_frame = compute_supertrend(
                df["high"],
                df["low"],
                close,
                period=atr_period,
                multiplier=atr_mult,
            )
            st_line = st_frame["supertrend"]
            trend_up = st_frame["trend"] > 0
            trend_down = st_frame["trend"] < 0
            long_setup = trend_up & (close > st_line)
            short_setup = trend_down & (close < st_line)
            weak_long = (~trend_up) | (close < st_line)
            weak_short = (~trend_down) | (close > st_line)
            cache[(atr_period, atr_mult)] = {
                "long_setup": long_setup,
                "short_setup": short_setup,
                "long_streak": streaks_from_bool(long_setup),
                "short_streak": streaks_from_bool(short_setup),
                "weak_long_streak": streaks_from_bool(weak_long),
                "weak_short_streak": streaks_from_bool(weak_short),
            }
    return cache


def simulate_strategy(df: pd.DataFrame, config: StrategyConfig, regime_cache: Dict[Tuple[int, float], Dict[str, pd.Series]]) -> pd.Series:
    regime = regime_cache[(config.atr_period, config.atr_mult)]
    long_setup = regime["long_setup"]
    short_setup = regime["short_setup"]
    long_streak = regime["long_streak"]
    short_streak = regime["short_streak"]
    weak_long_streak = regime["weak_long_streak"]
    weak_short_streak = regime["weak_short_streak"]
    frac_streak = df["fracture_streak"]

    positions = np.zeros(len(df), dtype=float)
    current_sign = 0.0
    bars_since_change = config.min_hold_bars

    for i, row in enumerate(df.itertuples(index=False)):
        long_ready = bool(long_setup.iloc[i]) and int(long_streak.iloc[i]) >= config.confirm_bars
        short_ready = bool(short_setup.iloc[i]) and int(short_streak.iloc[i]) >= config.confirm_bars
        changed = False

        if current_sign > 0:
            should_exit = (
                int(weak_long_streak.iloc[i]) >= config.exit_confirm_bars
                or int(frac_streak.iloc[i]) >= 2
                or short_ready
            )
            if should_exit and bars_since_change >= config.min_hold_bars:
                current_sign = 0.0
                changed = True
        elif current_sign < 0:
            should_exit = (
                int(weak_short_streak.iloc[i]) >= config.exit_confirm_bars
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
        for atr_period in [10, 14]:
            for atr_mult in [2.0, 3.0]:
                for confirm_bars in [1, 2]:
                    for min_hold_bars in [8, 16]:
                        configs.append(
                            StrategyConfig(
                                gate_mode=gate_mode,
                                atr_period=atr_period,
                                atr_mult=atr_mult,
                                confirm_bars=confirm_bars,
                                exit_confirm_bars=2,
                                min_hold_bars=min_hold_bars,
                            )
                        )
    return configs


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    base = build_base_frame(args.input_csv, args.v2_validation_csv)
    regime_cache = precompute_regimes(base)
    configs = strategy_grid()
    rows: List[Dict[str, float]] = []
    for idx, config in enumerate(configs, start=1):
        position = simulate_strategy(base, config, regime_cache)
        metrics = compute_metrics(base, position, fee_rate)
        rows.append(
            {
                "config_label": config.label,
                "gate_mode": config.gate_mode,
                "atr_period": config.atr_period,
                "atr_mult": config.atr_mult,
                "confirm_bars": config.confirm_bars,
                "exit_confirm_bars": config.exit_confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                **metrics,
            }
        )
        if idx % 8 == 0 or idx == len(configs):
            pd.DataFrame(rows).to_csv(output_dir / "supertrend_atr_regime_progress.csv", index=False)
            print(f"Completed {idx}/{len(configs)} configs", flush=True)

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "supertrend_atr_regime_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "supertrend_atr_regime_best_by_gate.csv", index=False)

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
        "positive_configs": int((results["net_total_return"] > 0).sum()),
        "total_configs": int(len(results)),
    }
    (output_dir / "supertrend_atr_regime_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
