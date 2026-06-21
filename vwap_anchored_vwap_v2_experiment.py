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
    anchor_bars: int
    confirm_bars: int
    exit_confirm_bars: int
    min_hold_bars: int

    @property
    def label(self) -> str:
        return (
            f"vwap_{self.gate_mode}"
            f"_anchor{self.anchor_bars}"
            f"_entry{self.confirm_bars}"
            f"_exit{self.exit_confirm_bars}"
            f"_hold{self.min_hold_bars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VWAP / anchored VWAP trend skeleton with V2 enhancement.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/vwap_anchored_vwap_v2_run")
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


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].astype(float).clip(lower=0.0)
    session_key = df["timestamp"].dt.floor("D")
    cum_pv = (typical * volume).groupby(session_key).cumsum()
    cum_vol = volume.groupby(session_key).cumsum().replace(0.0, np.nan)
    return (cum_pv / cum_vol).ffill()


def compute_anchored_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].astype(float).clip(lower=0.0)
    pv = typical * volume
    rolling_pv = pv.rolling(window, min_periods=max(20, window // 4)).sum()
    rolling_vol = volume.rolling(window, min_periods=max(20, window // 4)).sum().replace(0.0, np.nan)
    return (rolling_pv / rolling_vol).ffill()


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")
    validation["session_vwap"] = compute_session_vwap(validation)
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

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
    anchored_vwap = compute_anchored_vwap(df, config.anchor_bars)
    bullish_regime = (
        (df["close"] > df["session_vwap"])
        & (df["close"] > anchored_vwap)
        & (df["session_vwap"] > anchored_vwap)
    )
    bearish_regime = (
        (df["close"] < df["session_vwap"])
        & (df["close"] < anchored_vwap)
        & (df["session_vwap"] < anchored_vwap)
    )
    long_streak = streaks_from_bool(bullish_regime)
    short_streak = streaks_from_bool(bearish_regime)
    weak_long = streaks_from_bool((df["close"] < df["session_vwap"]) | (df["close"] < anchored_vwap))
    weak_short = streaks_from_bool((df["close"] > df["session_vwap"]) | (df["close"] > anchored_vwap))
    frac_streak = streaks_from_bool(df["state"] == "fracture")

    positions = np.zeros(len(df), dtype=float)
    current_sign = 0.0
    bars_since_change = config.min_hold_bars

    for i, row in enumerate(df.itertuples(index=False)):
        long_ready = bool(bullish_regime.iloc[i]) and int(long_streak.iloc[i]) >= config.confirm_bars
        short_ready = bool(bearish_regime.iloc[i]) and int(short_streak.iloc[i]) >= config.confirm_bars
        changed = False

        if current_sign > 0:
            should_exit = (
                int(weak_long.iloc[i]) >= config.exit_confirm_bars
                or int(frac_streak.iloc[i]) >= 2
                or short_ready
            )
            if should_exit and bars_since_change >= config.min_hold_bars:
                current_sign = 0.0
                changed = True
        elif current_sign < 0:
            should_exit = (
                int(weak_short.iloc[i]) >= config.exit_confirm_bars
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
        for anchor_bars in [192, 384]:
            for confirm_bars in [1, 2]:
                for exit_confirm_bars in [1, 2]:
                    for min_hold_bars in [8, 16]:
                        configs.append(
                            StrategyConfig(
                                gate_mode=gate_mode,
                                anchor_bars=anchor_bars,
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
                "anchor_bars": config.anchor_bars,
                "confirm_bars": config.confirm_bars,
                "exit_confirm_bars": config.exit_confirm_bars,
                "min_hold_bars": config.min_hold_bars,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "vwap_anchored_vwap_comparison.csv", index=False)

    best_by_gate = (
        results.sort_values(["gate_mode", "net_sharpe", "net_total_return"], ascending=[True, False, False])
        .groupby("gate_mode", as_index=False)
        .first()
    )
    best_by_gate.to_csv(output_dir / "vwap_anchored_vwap_best_by_gate.csv", index=False)

    raw_best = best_by_gate[best_by_gate["gate_mode"] == "raw"].iloc[0]
    best_overall = results.iloc[0]
    best_non_raw = results[results["gate_mode"] != "raw"].iloc[0]

    summary = {
        "fee_bps": float(args.fee_bps),
        "best_overall": best_overall[
            ["config_label", "gate_mode", "net_total_return", "net_sharpe", "net_max_drawdown"]
        ].to_dict(),
        "best_raw": raw_best[
            ["config_label", "gate_mode", "net_total_return", "net_sharpe", "net_max_drawdown"]
        ].to_dict(),
        "best_non_raw": best_non_raw[
            ["config_label", "gate_mode", "net_total_return", "net_sharpe", "net_max_drawdown"]
        ].to_dict(),
        "positive_configs": int((results["net_total_return"] > 0).sum()),
        "total_configs": int(len(results)),
    }
    (output_dir / "vwap_anchored_vwap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
