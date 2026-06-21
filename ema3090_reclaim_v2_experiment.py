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
    trigger_mode: str
    pullback_pct: float
    reclaim_lookback: int
    min_hold_bars: int
    cooldown_bars: int
    fracture_exit_bars: int

    @property
    def label(self) -> str:
        pct_bps = int(round(self.pullback_pct * 10000))
        return (
            f"ema3090_{self.gate_mode}"
            f"_{self.trigger_mode}"
            f"_pb{pct_bps}"
            f"_look{self.reclaim_lookback}"
            f"_hold{self.min_hold_bars}"
            f"_cool{self.cooldown_bars}"
            f"_fx{self.fracture_exit_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EMA30/EMA90 trend-pullback-reclaim skeleton with V2 as an enhancement layer."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/ema3090_reclaim_v2_run")
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
    validation["ema10"] = close.ewm(span=10, adjust=False).mean()
    validation["ema20"] = close.ewm(span=20, adjust=False).mean()
    validation["ema30"] = close.ewm(span=30, adjust=False).mean()
    validation["ema90"] = close.ewm(span=90, adjust=False).mean()
    validation["ema30_slope_8"] = validation["ema30"].pct_change(8).fillna(0.0)
    validation["next_bar_ret"] = np.log(close.shift(-1) / close)

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")

    merged["cohesion_streak"] = streaks_from_bool(merged["state"] == "cohesion")
    merged["fracture_streak"] = streaks_from_bool(merged["state"] == "fracture")
    return merged


def gate_passed(row: pd.Series | object, gate_mode: str) -> bool:
    state = str(row.state)
    if gate_mode == "raw":
        return True
    if gate_mode == "no_fracture":
        return state != "fracture"
    if gate_mode == "cohesion_only":
        return state == "cohesion"
    raise ValueError(f"Unknown gate mode: {gate_mode}")


def build_trigger_signals(
    df: pd.DataFrame, trigger_mode: str, long_recent_pullback: pd.Series, short_recent_pullback: pd.Series
) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    ema10 = df["ema10"]
    ema20 = df["ema20"]
    ema30 = df["ema30"]
    ema90 = df["ema90"]
    ema30_slope = df["ema30_slope_8"]

    trend_long = (ema30 > ema90) & (ema30_slope > 0)
    trend_short = (ema30 < ema90) & (ema30_slope < 0)

    if trigger_mode == "prev_bar_break":
        long_trigger = trend_long & (long_recent_pullback > 0) & (close > high.shift(1)) & (close > ema10)
        short_trigger = trend_short & (short_recent_pullback > 0) & (close < low.shift(1)) & (close < ema10)
    elif trigger_mode == "ema10_reclaim":
        long_trigger = (
            trend_long
            & (long_recent_pullback > 0)
            & (close > ema10)
            & (ema10 > ema10.shift(1))
            & (close > close.shift(1))
        )
        short_trigger = (
            trend_short
            & (short_recent_pullback > 0)
            & (close < ema10)
            & (ema10 < ema10.shift(1))
            & (close < close.shift(1))
        )
    elif trigger_mode == "ema20_reclaim":
        long_trigger = trend_long & (long_recent_pullback > 0) & (close > ema20) & (close > ema10)
        short_trigger = trend_short & (short_recent_pullback > 0) & (close < ema20) & (close < ema10)
    elif trigger_mode == "two_bar_momentum":
        long_trigger = (
            trend_long
            & (long_recent_pullback > 0)
            & (close > close.shift(1))
            & (close.shift(1) > close.shift(2))
            & (close > ema10)
        )
        short_trigger = (
            trend_short
            & (short_recent_pullback > 0)
            & (close < close.shift(1))
            & (close.shift(1) < close.shift(2))
            & (close < ema10)
        )
    else:
        raise ValueError(f"Unknown trigger mode: {trigger_mode}")

    return long_trigger.fillna(False), short_trigger.fillna(False)


def simulate_strategy(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    positions = np.zeros(len(df), dtype=float)
    current = 0.0
    bars_since_change = config.min_hold_bars
    cooldown_left = 0

    close = df["close"]
    ema20 = df["ema20"]
    ema30 = df["ema30"]
    ema90 = df["ema90"]
    ema30_slope = df["ema30_slope_8"]

    trend_long = (ema30 > ema90) & (ema30_slope > 0)
    trend_short = (ema30 < ema90) & (ema30_slope < 0)
    long_pullback = trend_long & ((close / ema20 - 1.0) <= -config.pullback_pct)
    short_pullback = trend_short & ((close / ema20 - 1.0) >= config.pullback_pct)

    long_recent_pullback = long_pullback.shift(1).rolling(config.reclaim_lookback, min_periods=1).max().fillna(0.0)
    short_recent_pullback = short_pullback.shift(1).rolling(config.reclaim_lookback, min_periods=1).max().fillna(0.0)
    long_trigger, short_trigger = build_trigger_signals(
        df, config.trigger_mode, long_recent_pullback, short_recent_pullback
    )

    for i, row in enumerate(df.itertuples(index=False)):
        if cooldown_left > 0:
            cooldown_left -= 1

        long_ready = bool(long_trigger.iloc[i]) and gate_passed(row, config.gate_mode)
        short_ready = bool(short_trigger.iloc[i]) and gate_passed(row, config.gate_mode)
        fracture_exit = config.fracture_exit_bars > 0 and int(row.fracture_streak) >= config.fracture_exit_bars
        changed = False

        if current > 0:
            should_exit = bool(close.iloc[i] < ema30.iloc[i]) or bool(not trend_long.iloc[i]) or fracture_exit
            if should_exit and bars_since_change >= config.min_hold_bars:
                current = 0.0
                cooldown_left = config.cooldown_bars
                changed = True
        elif current < 0:
            should_exit = bool(close.iloc[i] > ema30.iloc[i]) or bool(not trend_short.iloc[i]) or fracture_exit
            if should_exit and bars_since_change >= config.min_hold_bars:
                current = 0.0
                cooldown_left = config.cooldown_bars
                changed = True

        if current > 0 and short_ready and bars_since_change >= config.min_hold_bars:
            current = -1.0
            changed = True
        elif current < 0 and long_ready and bars_since_change >= config.min_hold_bars:
            current = 1.0
            changed = True
        elif current == 0.0 and cooldown_left == 0:
            if long_ready:
                current = 1.0
                changed = True
            elif short_ready:
                current = -1.0
                changed = True

        positions[i] = current
        if changed:
            bars_since_change = 0
        else:
            bars_since_change += 1

    return pd.Series(positions, index=df.index)


def strategy_grid() -> List[StrategyConfig]:
    configs: List[StrategyConfig] = []
    for gate_mode in ["raw", "no_fracture", "cohesion_only"]:
        for trigger_mode in ["prev_bar_break", "ema10_reclaim", "ema20_reclaim", "two_bar_momentum"]:
            for pullback_pct in [0.005, 0.007]:
                for reclaim_lookback in [3, 5]:
                    for min_hold_bars in [8, 12]:
                        configs.append(
                            StrategyConfig(
                                gate_mode=gate_mode,
                                trigger_mode=trigger_mode,
                                pullback_pct=pullback_pct,
                                reclaim_lookback=reclaim_lookback,
                                min_hold_bars=min_hold_bars,
                                cooldown_bars=0,
                                fracture_exit_bars=0 if gate_mode == "raw" else 2,
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
                "trigger_mode": config.trigger_mode,
                "pullback_pct": config.pullback_pct,
                "reclaim_lookback": config.reclaim_lookback,
                "min_hold_bars": config.min_hold_bars,
                "cooldown_bars": config.cooldown_bars,
                "fracture_exit_bars": config.fracture_exit_bars,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "ema3090_reclaim_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "ema3090_reclaim_best_by_gate.csv", index=False)

    best_by_trigger = (
        results.sort_values(["trigger_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("trigger_mode", as_index=False)
        .first()
    )
    best_by_trigger.to_csv(output_dir / "ema3090_reclaim_best_by_trigger.csv", index=False)

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
    (output_dir / "ema3090_reclaim_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
