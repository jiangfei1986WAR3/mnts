from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


@dataclass(frozen=True)
class StrategyConfig:
    label: str
    branch: str
    allow_long: bool
    allow_short: bool
    gate_mode: str
    setup_confirm_bars: int
    state_confirm_bars: int
    exit_fracture_bars: int
    min_hold_bars: int
    cooldown_bars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer pullback_trend + V2 and breakout_48 + V2 into more realistic execution rules."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/engineered_pullback_breakout_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
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


def state_streaks(state: pd.Series, target_state: str) -> pd.Series:
    return streaks_from_bool(state == target_state)


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    close = validation["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    trend = pd.Series(0.0, index=validation.index, dtype=float)
    trend.loc[ema20 > ema60] = 1.0
    trend.loc[ema20 < ema60] = -1.0
    deviation = close / ema20 - 1.0

    validation["ema20"] = ema20
    validation["ema60"] = ema60
    validation["trend"] = trend
    validation["deviation"] = deviation
    validation["pullback_long_setup"] = (trend > 0) & (deviation < -0.003)
    validation["pullback_short_setup"] = (trend < 0) & (deviation > 0.003)
    validation["pullback_long_streak"] = streaks_from_bool(validation["pullback_long_setup"])
    validation["pullback_short_streak"] = streaks_from_bool(validation["pullback_short_setup"])

    upper48 = close.shift(1).rolling(48).max()
    lower48 = close.shift(1).rolling(48).min()
    validation["breakout_upper_48"] = upper48
    validation["breakout_lower_48"] = lower48
    validation["breakout_up_event"] = close > upper48
    validation["breakout_down_event"] = close < lower48

    validation["next_bar_ret"] = np.log(close.shift(-1) / close)

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(v2[["timestamp", "state", "high_vol_event"]], on="timestamp", how="inner")
    merged["cohesion_streak"] = state_streaks(merged["state"], "cohesion")
    merged["fracture_streak"] = state_streaks(merged["state"], "fracture")
    return merged


def gate_passed(row: pd.Series | object, config: StrategyConfig) -> bool:
    state = str(row.state)
    cohesion_streak = int(row.cohesion_streak)
    fracture_streak = int(row.fracture_streak)
    if config.gate_mode == "any":
        return True
    if config.gate_mode == "nonfracture":
        return state != "fracture"
    if config.gate_mode == "cohesion":
        return state == "cohesion" and cohesion_streak >= config.state_confirm_bars
    if config.gate_mode == "stable":
        return cohesion_streak >= config.state_confirm_bars or (state == "drift" and fracture_streak == 0)
    raise ValueError(f"Unknown gate mode: {config.gate_mode}")


def simulate_pullback(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    positions = np.zeros(len(df), dtype=float)
    current = 0.0
    bars_since_change = config.min_hold_bars
    cooldown_left = 0

    for i, row in enumerate(df.itertuples(index=False)):
        if cooldown_left > 0:
            cooldown_left -= 1

        long_ready = (
            config.allow_long
            and bool(row.pullback_long_setup)
            and int(row.pullback_long_streak) >= config.setup_confirm_bars
            and gate_passed(row, config)
        )
        short_ready = (
            config.allow_short
            and bool(row.pullback_short_setup)
            and int(row.pullback_short_streak) >= config.setup_confirm_bars
            and gate_passed(row, config)
        )
        fracture_exit = config.exit_fracture_bars > 0 and int(row.fracture_streak) >= config.exit_fracture_bars
        changed = False

        if current > 0:
            should_exit = fracture_exit or float(row.trend) <= 0 or not bool(row.pullback_long_setup)
            if should_exit and bars_since_change >= config.min_hold_bars:
                current = 0.0
                cooldown_left = config.cooldown_bars
                changed = True
        elif current < 0:
            should_exit = fracture_exit or float(row.trend) >= 0 or not bool(row.pullback_short_setup)
            if should_exit and bars_since_change >= config.min_hold_bars:
                current = 0.0
                cooldown_left = config.cooldown_bars
                changed = True

        if current == 0.0 and cooldown_left == 0:
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


def simulate_breakout(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    positions = np.zeros(len(df), dtype=float)
    current = 0.0
    bars_since_change = config.min_hold_bars
    cooldown_left = 0

    for i, row in enumerate(df.itertuples(index=False)):
        if cooldown_left > 0:
            cooldown_left -= 1

        long_event = config.allow_long and bool(row.breakout_up_event) and gate_passed(row, config)
        short_event = config.allow_short and bool(row.breakout_down_event) and gate_passed(row, config)
        fracture_exit = config.exit_fracture_bars > 0 and int(row.fracture_streak) >= config.exit_fracture_bars
        changed = False

        if current > 0 and fracture_exit and bars_since_change >= config.min_hold_bars:
            current = 0.0
            cooldown_left = config.cooldown_bars
            changed = True
        elif current < 0 and fracture_exit and bars_since_change >= config.min_hold_bars:
            current = 0.0
            cooldown_left = config.cooldown_bars
            changed = True

        if current > 0 and short_event and bars_since_change >= config.min_hold_bars:
            current = -1.0 if config.allow_short else 0.0
            changed = True
        elif current < 0 and long_event and bars_since_change >= config.min_hold_bars:
            current = 1.0 if config.allow_long else 0.0
            changed = True
        elif current == 0.0 and cooldown_left == 0:
            if long_event:
                current = 1.0
                changed = True
            elif short_event:
                current = -1.0
                changed = True

        positions[i] = current
        if changed:
            bars_since_change = 0
        else:
            bars_since_change += 1

    return pd.Series(positions, index=df.index)


def compute_metrics(
    df: pd.DataFrame,
    position: pd.Series,
    fee_rate: float,
    slippage_rate: float = 0.0,
    return_column: str = "next_bar_ret",
) -> Dict[str, float]:
    work = df.copy()
    work["position"] = position.fillna(0.0)
    turnover = work["position"].diff().abs().fillna(work["position"].abs())
    gross_ret = work["position"] * work[return_column].fillna(0.0)
    fee = turnover * fee_rate
    slippage = turnover * slippage_rate
    net_ret = gross_ret - fee - slippage

    gross_equity = np.exp(gross_ret.cumsum())
    net_equity = np.exp(net_ret.cumsum())
    peak = np.maximum.accumulate(net_equity)
    drawdown = net_equity / peak - 1.0
    active = work["position"] != 0
    changes = work["position"].diff().fillna(work["position"])
    trade_events = int(changes.ne(0).sum())

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
        "fee_cost_sum": float(fee.sum()),
        "slippage_cost_sum": float(slippage.sum()),
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
            label="pullback_long_only_cohesion2_hold8",
            branch="pullback",
            allow_long=True,
            allow_short=False,
            gate_mode="cohesion",
            setup_confirm_bars=1,
            state_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=4,
        ),
        StrategyConfig(
            label="pullback_long_only_cohesion3_hold12",
            branch="pullback",
            allow_long=True,
            allow_short=False,
            gate_mode="cohesion",
            setup_confirm_bars=1,
            state_confirm_bars=3,
            exit_fracture_bars=2,
            min_hold_bars=12,
            cooldown_bars=8,
        ),
        StrategyConfig(
            label="pullback_long_short_cohesion2_hold8",
            branch="pullback",
            allow_long=True,
            allow_short=True,
            gate_mode="cohesion",
            setup_confirm_bars=1,
            state_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=4,
        ),
        StrategyConfig(
            label="pullback_short_only_cohesion2_hold8",
            branch="pullback",
            allow_long=False,
            allow_short=True,
            gate_mode="cohesion",
            setup_confirm_bars=1,
            state_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=4,
        ),
        StrategyConfig(
            label="pullback_long_only_stable2_hold8",
            branch="pullback",
            allow_long=True,
            allow_short=False,
            gate_mode="stable",
            setup_confirm_bars=1,
            state_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=4,
        ),
        StrategyConfig(
            label="breakout48_raw_flip_hold8",
            branch="breakout48",
            allow_long=True,
            allow_short=True,
            gate_mode="any",
            setup_confirm_bars=1,
            state_confirm_bars=1,
            exit_fracture_bars=0,
            min_hold_bars=8,
            cooldown_bars=0,
        ),
        StrategyConfig(
            label="breakout48_nonfracture_flip_exit2_hold8",
            branch="breakout48",
            allow_long=True,
            allow_short=True,
            gate_mode="nonfracture",
            setup_confirm_bars=1,
            state_confirm_bars=1,
            exit_fracture_bars=2,
            min_hold_bars=8,
            cooldown_bars=0,
        ),
        StrategyConfig(
            label="breakout48_nonfracture_flip_exit2_hold12_cool8",
            branch="breakout48",
            allow_long=True,
            allow_short=True,
            gate_mode="nonfracture",
            setup_confirm_bars=1,
            state_confirm_bars=1,
            exit_fracture_bars=2,
            min_hold_bars=12,
            cooldown_bars=8,
        ),
        StrategyConfig(
            label="breakout48_cohesion_flip_exit2_hold12_cool8",
            branch="breakout48",
            allow_long=True,
            allow_short=True,
            gate_mode="cohesion",
            setup_confirm_bars=1,
            state_confirm_bars=2,
            exit_fracture_bars=2,
            min_hold_bars=12,
            cooldown_bars=8,
        ),
        StrategyConfig(
            label="breakout48_long_only_nonfracture_exit2_hold12_cool8",
            branch="breakout48",
            allow_long=True,
            allow_short=False,
            gate_mode="nonfracture",
            setup_confirm_bars=1,
            state_confirm_bars=1,
            exit_fracture_bars=2,
            min_hold_bars=12,
            cooldown_bars=8,
        ),
    ]


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0
    slippage_rate = args.slippage_bps / 10000.0
    df = build_base_frame(args.input_csv, args.v2_validation_csv)

    rows: List[Dict[str, float]] = []
    for config in strategy_grid():
        if config.branch == "pullback":
            position = simulate_pullback(df, config)
        elif config.branch == "breakout48":
            position = simulate_breakout(df, config)
        else:
            raise ValueError(f"Unknown branch: {config.branch}")

        metrics = compute_metrics(df, position, fee_rate, slippage_rate)
        rows.append(
            {
                "config": config.label,
                "branch": config.branch,
                "gate_mode": config.gate_mode,
                "setup_confirm_bars": config.setup_confirm_bars,
                "state_confirm_bars": config.state_confirm_bars,
                "exit_fracture_bars": config.exit_fracture_bars,
                "min_hold_bars": config.min_hold_bars,
                "cooldown_bars": config.cooldown_bars,
                "allow_long": config.allow_long,
                "allow_short": config.allow_short,
                **metrics,
            }
        )

    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    results.to_csv(output_dir / "engineered_pullback_breakout_comparison.csv", index=False)

    branch_best_rows: List[Dict[str, float]] = []
    for branch, group in results.groupby("branch"):
        best_by_sharpe = group.sort_values("net_sharpe", ascending=False).iloc[0]
        best_by_return = group.sort_values("net_total_return", ascending=False).iloc[0]
        branch_best_rows.append(
            {
                "branch": branch,
                "best_sharpe_config": str(best_by_sharpe["config"]),
                "best_net_sharpe": float(best_by_sharpe["net_sharpe"]),
                "best_return_config": str(best_by_return["config"]),
                "best_net_total_return": float(best_by_return["net_total_return"]),
            }
        )
    branch_best = pd.DataFrame(branch_best_rows).sort_values("best_net_sharpe", ascending=False)
    branch_best.to_csv(output_dir / "engineered_pullback_breakout_branch_best.csv", index=False)

    summary = {
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "best_overall_by_net_sharpe": results.iloc[0][["config", "branch", "net_sharpe"]].to_dict(),
        "best_overall_by_net_total_return": results.sort_values("net_total_return", ascending=False).iloc[0][
            ["config", "branch", "net_total_return"]
        ].to_dict(),
        "branch_best": branch_best_rows,
    }
    (output_dir / "engineered_pullback_breakout_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
