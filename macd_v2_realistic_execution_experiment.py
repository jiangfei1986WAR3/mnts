from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from compare_v2_vs_classic_indicators import compute_macd
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    allow_short: bool
    entry_gate: str
    entry_confirm_bars: int
    exit_fracture_bars: int
    min_hold_bars: int
    cooldown_bars: int
    cohesion_size: float = 1.0
    drift_size: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="More realistic MACD + MNTS V2 execution experiment."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/macd_v2_realistic_run")
    parser.add_argument("--fee-bps", type=float, default=4.0)
    return parser.parse_args()


def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = len(df) // 2
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def state_streaks(state: pd.Series, target_state: str) -> pd.Series:
    arr = state.to_numpy()
    out = np.zeros(len(arr), dtype=int)
    streak = 0
    for i, value in enumerate(arr):
        if value == target_state:
            streak += 1
        else:
            streak = 0
        out[i] = streak
    return pd.Series(out, index=state.index)


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    macd_line, macd_signal_line, macd_hist = compute_macd(validation["close"])
    validation["macd_line"] = macd_line
    validation["macd_signal_line"] = macd_signal_line
    validation["macd_hist"] = macd_hist
    validation["macd_regime"] = np.sign(validation["macd_line"] - validation["macd_signal_line"])
    prev_regime = validation["macd_regime"].shift(1).fillna(0.0)
    validation["bull_cross"] = (validation["macd_regime"] > 0) & (prev_regime <= 0)
    validation["bear_cross"] = (validation["macd_regime"] < 0) & (prev_regime >= 0)
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(
        v2[["timestamp", "state", "high_vol_event"]],
        on="timestamp",
        how="inner",
    )
    merged["cohesion_streak"] = state_streaks(merged["state"], "cohesion")
    merged["fracture_streak"] = state_streaks(merged["state"], "fracture")
    return merged


def entry_allowed(
    state: str,
    cohesion_streak: int,
    fracture_streak: int,
    cooldown_left: int,
    config: StrategyConfig,
) -> bool:
    if cooldown_left > 0:
        return False
    if config.entry_gate == "any":
        return True
    if config.entry_gate == "nonfracture":
        return state != "fracture"
    if config.entry_gate == "cohesion":
        return state == "cohesion" and cohesion_streak >= config.entry_confirm_bars
    if config.entry_gate == "stable":
        if state == "cohesion":
            return cohesion_streak >= config.entry_confirm_bars
        return state == "drift" and fracture_streak == 0
    raise ValueError(f"Unknown entry gate: {config.entry_gate}")


def entry_size(state: str, config: StrategyConfig) -> float:
    if state == "cohesion":
        return float(config.cohesion_size)
    if state == "drift":
        return float(config.drift_size)
    return 0.0


def simulate_strategy(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    positions = np.zeros(len(df), dtype=float)
    current = 0.0
    bars_since_change = config.min_hold_bars
    cooldown_left = 0

    for i, row in enumerate(df.itertuples(index=False)):
        if cooldown_left > 0:
            cooldown_left -= 1

        state = str(row.state)
        cohesion_streak = int(row.cohesion_streak)
        fracture_streak = int(row.fracture_streak)
        bull_cross = bool(row.bull_cross)
        bear_cross = bool(row.bear_cross)
        changed = False

        if current != 0.0 and config.exit_fracture_bars > 0 and fracture_streak >= config.exit_fracture_bars:
            current = 0.0
            cooldown_left = config.cooldown_bars
            changed = True
        elif bull_cross:
            can_enter = entry_allowed(
                state=state,
                cohesion_streak=cohesion_streak,
                fracture_streak=fracture_streak,
                cooldown_left=cooldown_left,
                config=config,
            )
            target = entry_size(state, config) if can_enter else 0.0
            if current <= 0.0 and (current == 0.0 or bars_since_change >= config.min_hold_bars):
                if current != target:
                    current = target
                    changed = True
        elif bear_cross:
            can_enter = entry_allowed(
                state=state,
                cohesion_streak=cohesion_streak,
                fracture_streak=fracture_streak,
                cooldown_left=cooldown_left,
                config=config,
            )
            target = -entry_size(state, config) if (can_enter and config.allow_short) else 0.0
            if current >= 0.0 and (current == 0.0 or bars_since_change >= config.min_hold_bars):
                if current != target:
                    current = target
                    changed = True

        positions[i] = current
        if changed:
            bars_since_change = 0
        else:
            bars_since_change += 1

    return pd.Series(positions, index=df.index)


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

    changes = work["position"].diff().fillna(work["position"])
    trade_events = int((changes != 0).sum())
    active = work["position"] != 0
    annual_factor = np.sqrt(365 * 24 * 4)
    gross_std = float(gross_ret.std(ddof=0))
    net_std = float(net_ret.std(ddof=0))

    gross_sharpe = float(gross_ret.mean() / gross_std * annual_factor) if gross_std > 1e-12 else 0.0
    net_sharpe = float(net_ret.mean() / net_std * annual_factor) if net_std > 1e-12 else 0.0

    return {
        "bars": int(len(work)),
        "trade_events": trade_events,
        "active_bars": int(active.sum()),
        "long_exposure": float((work["position"] > 0).mean()),
        "short_exposure": float((work["position"] < 0).mean()),
        "avg_abs_position": float(work["position"].abs().mean()),
        "turnover_sum": float(turnover.sum()),
        "gross_total_return": float(gross_equity.iloc[-1] - 1.0),
        "net_total_return": float(net_equity.iloc[-1] - 1.0),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "active_high_vol_rate": float(work.loc[active, "high_vol_event"].mean()) if active.any() else 0.0,
    }


def strategy_grid() -> List[StrategyConfig]:
    return [
        StrategyConfig(
            label="cross_raw_flip_hold8",
            allow_short=True,
            entry_gate="any",
            entry_confirm_bars=1,
            exit_fracture_bars=0,
            min_hold_bars=8,
            cooldown_bars=0,
        ),
        StrategyConfig(
            label="cross_raw_long_only_hold8",
            allow_short=False,
            entry_gate="any",
            entry_confirm_bars=1,
            exit_fracture_bars=0,
            min_hold_bars=8,
            cooldown_bars=0,
        ),
        StrategyConfig(
            label="cross_nonfracture_flip_exit2_hold8",
            allow_short=True,
            entry_gate="nonfracture",
            entry_confirm_bars=1,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=0,
        ),
        StrategyConfig(
            label="cross_nonfracture_flip_exit2_cool8",
            allow_short=True,
            entry_gate="nonfracture",
            entry_confirm_bars=1,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=8,
        ),
        StrategyConfig(
            label="cross_nonfracture_long_only_exit2_cool8",
            allow_short=False,
            entry_gate="nonfracture",
            entry_confirm_bars=1,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=8,
        ),
        StrategyConfig(
            label="cross_cohesion_long_only_confirm2",
            allow_short=False,
            entry_gate="cohesion",
            entry_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=8,
        ),
        StrategyConfig(
            label="cross_cohesion_long_only_confirm3",
            allow_short=False,
            entry_gate="cohesion",
            entry_confirm_bars=3,
            exit_fracture_bars=2,
            min_hold_bars=12,
            cooldown_bars=12,
        ),
        StrategyConfig(
            label="cross_state_scaled_long_only",
            allow_short=False,
            entry_gate="stable",
            entry_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=8,
            cohesion_size=1.0,
            drift_size=0.5,
        ),
    ]


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df = build_base_frame(args.input_csv, args.v2_validation_csv)
    rows: List[Dict[str, float]] = []
    for config in strategy_grid():
        position = simulate_strategy(df, config)
        metrics = compute_metrics(df, position, fee_rate)
        rows.append(
            {
                "config": config.label,
                "allow_short": config.allow_short,
                "entry_gate": config.entry_gate,
                "entry_confirm_bars": config.entry_confirm_bars,
                "exit_fracture_bars": config.exit_fracture_bars,
                "min_hold_bars": config.min_hold_bars,
                "cooldown_bars": config.cooldown_bars,
                "cohesion_size": config.cohesion_size,
                "drift_size": config.drift_size,
                **metrics,
            }
        )

    result = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    result.to_csv(output_dir / "macd_v2_realistic_comparison.csv", index=False)

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
    (output_dir / "macd_v2_realistic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
