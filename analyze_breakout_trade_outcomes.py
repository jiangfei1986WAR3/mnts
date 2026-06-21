from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from breakout_v2_parameter_stability_4y import ParamConfig, prepare_breakout_columns, simulate_breakout_param
from mnts_15m_state_layer_v2 import apply_v2_model, fit_v2_model
from rolling_breakout_v2_full_system_walkforward import (
    attach_state_streaks,
    load_cached_feature_frame,
    stitch_test_state,
    valid_training_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze closed-trade outcomes for a fixed breakout+V2 config.")
    parser.add_argument("--discovery-v2-csv", required=True)
    parser.add_argument("--validation-v2-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/breakout_trade_outcomes")
    parser.add_argument("--breakout-window", type=int, default=48)
    parser.add_argument("--gate-mode", default="cohesion", choices=["cohesion", "nonfracture"])
    parser.add_argument("--min-hold-bars", type=int, default=8)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--train-bars", type=int, default=11520)
    parser.add_argument("--test-bars", type=int, default=2880)
    parser.add_argument("--fee-bps-list", nargs="+", type=float, default=[1.0, 4.0])
    return parser.parse_args()


def extract_closed_trades(df: pd.DataFrame, position: pd.Series, fee_rate: float) -> pd.DataFrame:
    pos = position.fillna(0.0).to_numpy(dtype=float)
    ret = df["next_bar_ret"].fillna(0.0).to_numpy(dtype=float)
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    rows: List[Dict[str, object]] = []
    current_side = 0.0
    entry_idx: int | None = None
    gross_log_ret = 0.0
    net_log_ret = 0.0
    bars = 0
    prev_pos = 0.0

    for i, cur_pos in enumerate(pos):
        turnover = abs(cur_pos - prev_pos)
        gross_bar_ret = cur_pos * ret[i]
        net_bar_ret = gross_bar_ret - turnover * fee_rate

        if current_side == 0.0 and cur_pos != 0.0:
            current_side = cur_pos
            entry_idx = i
            gross_log_ret = gross_bar_ret
            net_log_ret = net_bar_ret
            bars = 1
        elif current_side != 0.0:
            if cur_pos == current_side:
                gross_log_ret += gross_bar_ret
                net_log_ret += net_bar_ret
                bars += 1
            else:
                assert entry_idx is not None
                rows.append(
                    {
                        "side": "long" if current_side > 0 else "short",
                        "entry_time": str(timestamps.iloc[entry_idx]),
                        "exit_time": str(timestamps.iloc[i]),
                        "bars_held": bars,
                        "gross_return": float(np.exp(gross_log_ret) - 1.0),
                        "net_return": float(np.exp(net_log_ret) - 1.0),
                    }
                )
                if cur_pos != 0.0:
                    current_side = cur_pos
                    entry_idx = i
                    gross_log_ret = gross_bar_ret
                    net_log_ret = net_bar_ret
                    bars = 1
                else:
                    current_side = 0.0
                    entry_idx = None
                    gross_log_ret = 0.0
                    net_log_ret = 0.0
                    bars = 0
        prev_pos = cur_pos

    if current_side != 0.0 and entry_idx is not None:
        rows.append(
            {
                "side": "long" if current_side > 0 else "short",
                "entry_time": str(timestamps.iloc[entry_idx]),
                "exit_time": str(timestamps.iloc[len(df) - 1]),
                "bars_held": bars,
                "gross_return": float(np.exp(gross_log_ret) - 1.0),
                "net_return": float(np.exp(net_log_ret) - 1.0),
            }
        )

    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> Dict[str, float]:
    net = trades["net_return"]
    long_mask = trades["side"] == "long"
    short_mask = trades["side"] == "short"
    return {
        "trade_count": int(len(trades)),
        "win_count": int((net > 0).sum()),
        "loss_count": int((net < 0).sum()),
        "flat_count": int((net == 0).sum()),
        "win_rate": float((net > 0).mean()) if len(trades) else 0.0,
        "long_trade_count": int(long_mask.sum()),
        "short_trade_count": int(short_mask.sum()),
        "long_win_count": int((long_mask & (net > 0)).sum()),
        "short_win_count": int((short_mask & (net > 0)).sum()),
        "avg_net_return_per_trade": float(net.mean()) if len(trades) else 0.0,
        "median_net_return_per_trade": float(net.median()) if len(trades) else 0.0,
        "avg_bars_held": float(trades["bars_held"].mean()) if len(trades) else 0.0,
        "median_bars_held": float(trades["bars_held"].median()) if len(trades) else 0.0,
    }


def build_trade_table(args: argparse.Namespace, fee_bps: float) -> pd.DataFrame:
    df, validation_start_idx = load_cached_feature_frame(args.discovery_v2_csv, args.validation_v2_csv)
    df = prepare_breakout_columns(df, [args.breakout_window])
    fee_rate = fee_bps / 10000.0
    param = ParamConfig(
        breakout_window=args.breakout_window,
        gate_mode=args.gate_mode,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
    )

    all_trades: List[pd.DataFrame] = []
    test_start = validation_start_idx
    while test_start + args.test_bars <= len(df):
        train_start = max(0, test_start - args.train_bars)
        train_end = test_start
        test_end = test_start + args.test_bars

        raw_train = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        raw_test = df.iloc[test_start:test_end].copy().reset_index(drop=True)
        train_for_v2 = valid_training_rows(raw_train)
        if len(train_for_v2) < max(2000, args.train_bars // 4):
            test_start += args.test_bars
            continue

        v2_model = fit_v2_model(train_for_v2)
        train_scored = attach_state_streaks(apply_v2_model(raw_train, v2_model))
        test_scored = apply_v2_model(raw_test, v2_model)
        test_scored = stitch_test_state(train_scored, test_scored)
        position = simulate_breakout_param(test_scored, param)
        trades = extract_closed_trades(test_scored, position, fee_rate)
        all_trades.append(trades)
        test_start += args.test_bars

    if not all_trades:
        return pd.DataFrame(columns=["side", "entry_time", "exit_time", "bars_held", "gross_return", "net_return"])
    return pd.concat(all_trades, ignore_index=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "config": {
            "breakout_window": args.breakout_window,
            "gate_mode": args.gate_mode,
            "min_hold_bars": args.min_hold_bars,
            "cooldown_bars": args.cooldown_bars,
        }
    }

    for fee_bps in args.fee_bps_list:
        trades = build_trade_table(args, fee_bps)
        trades.to_csv(output_dir / f"trade_outcomes_fee{int(fee_bps)}.csv", index=False)
        summary[str(float(fee_bps))] = summarize_trades(trades)

    (output_dir / "trade_outcomes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
