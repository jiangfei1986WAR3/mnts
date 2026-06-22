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
    resolve_return_column,
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
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--execution-model",
        choices=["close_to_close", "close_to_next_open"],
        default="close_to_close",
    )
    parser.add_argument("--initial-capital-usd", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    return parser.parse_args()


def extract_closed_trades(
    df: pd.DataFrame,
    position: pd.Series,
    fee_rate: float,
    slippage_rate: float = 0.0,
    return_column: str = "next_bar_ret",
    fill_price_column: str = "close",
    initial_capital_usd: float = 1000.0,
    leverage: float = 1.0,
) -> pd.DataFrame:
    pos = position.fillna(0.0).to_numpy(dtype=float)
    ret = df[return_column].fillna(0.0).to_numpy(dtype=float)
    fill_price = df[fill_price_column].to_numpy(dtype=float)
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    rows: List[Dict[str, object]] = []
    current_trade: Dict[str, object] | None = None
    prev_pos = 0.0
    equity_usd = float(initial_capital_usd)
    cost_rate = leverage * (fee_rate + slippage_rate)

    def finalize_trade(exit_idx: int, exit_price: float) -> None:
        nonlocal current_trade
        assert current_trade is not None
        entry_equity = float(current_trade["entry_equity_usd"])
        exit_equity = float(current_trade["exit_equity_usd"])
        current_trade["exit_time"] = str(timestamps.iloc[exit_idx])
        current_trade["exit_fill_price"] = float(exit_price) if np.isfinite(exit_price) else np.nan
        current_trade["gross_return"] = float(np.exp(current_trade["gross_log_ret"]) - 1.0)
        current_trade["net_return"] = float(np.exp(current_trade["net_log_ret"]) - 1.0)
        current_trade["net_pnl_usd"] = float(exit_equity - entry_equity)
        current_trade["gross_pnl_usd"] = float(entry_equity * (np.exp(current_trade["gross_log_ret"]) - 1.0))
        current_trade.pop("gross_log_ret", None)
        current_trade.pop("net_log_ret", None)
        rows.append(current_trade)
        current_trade = None

    for i, cur_pos in enumerate(pos):
        cur_pos = float(cur_pos)
        gross_bar_ret = leverage * cur_pos * ret[i]

        if prev_pos == 0.0 and cur_pos != 0.0:
            current_trade = {
                "side": "long" if cur_pos > 0 else "short",
                "entry_time": str(timestamps.iloc[i]),
                "entry_fill_price": float(fill_price[i]) if np.isfinite(fill_price[i]) else np.nan,
                "entry_equity_usd": float(equity_usd),
                "exit_equity_usd": np.nan,
                "bars_held": 1,
                "gross_log_ret": gross_bar_ret,
                "net_log_ret": gross_bar_ret - cost_rate,
            }
            equity_usd *= float(np.exp(gross_bar_ret - cost_rate))
            current_trade["exit_equity_usd"] = float(equity_usd)
        elif prev_pos != 0.0 and cur_pos == prev_pos:
            assert current_trade is not None
            current_trade["bars_held"] = int(current_trade["bars_held"]) + 1
            current_trade["gross_log_ret"] = float(current_trade["gross_log_ret"]) + gross_bar_ret
            current_trade["net_log_ret"] = float(current_trade["net_log_ret"]) + gross_bar_ret
            equity_usd *= float(np.exp(gross_bar_ret))
            current_trade["exit_equity_usd"] = float(equity_usd)
        elif prev_pos != 0.0 and cur_pos == 0.0:
            assert current_trade is not None
            current_trade["net_log_ret"] = float(current_trade["net_log_ret"]) - cost_rate
            equity_usd *= float(np.exp(-cost_rate))
            current_trade["exit_equity_usd"] = float(equity_usd)
            finalize_trade(i, fill_price[i])
        elif prev_pos != 0.0 and cur_pos != prev_pos:
            assert current_trade is not None
            current_trade["net_log_ret"] = float(current_trade["net_log_ret"]) - cost_rate
            equity_usd *= float(np.exp(-cost_rate))
            current_trade["exit_equity_usd"] = float(equity_usd)
            finalize_trade(i, fill_price[i])

            current_trade = {
                "side": "long" if cur_pos > 0 else "short",
                "entry_time": str(timestamps.iloc[i]),
                "entry_fill_price": float(fill_price[i]) if np.isfinite(fill_price[i]) else np.nan,
                "entry_equity_usd": float(equity_usd),
                "exit_equity_usd": np.nan,
                "bars_held": 1,
                "gross_log_ret": gross_bar_ret,
                "net_log_ret": gross_bar_ret - cost_rate,
            }
            equity_usd *= float(np.exp(gross_bar_ret - cost_rate))
            current_trade["exit_equity_usd"] = float(equity_usd)

        prev_pos = cur_pos

    if current_trade is not None:
        current_trade["net_log_ret"] = float(current_trade["net_log_ret"]) - cost_rate
        equity_usd *= float(np.exp(-cost_rate))
        current_trade["exit_equity_usd"] = float(equity_usd)
        finalize_trade(len(df) - 1, fill_price[len(df) - 1])

    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame, initial_capital_usd: float, leverage: float) -> Dict[str, float]:
    if trades.empty:
        return {
            "initial_capital_usd": float(initial_capital_usd),
            "leverage": float(leverage),
            "final_equity_usd": float(initial_capital_usd),
            "net_pnl_usd": 0.0,
            "compounded_total_return": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "flat_count": 0,
            "win_rate": 0.0,
            "long_trade_count": 0,
            "short_trade_count": 0,
            "long_win_count": 0,
            "short_win_count": 0,
            "avg_net_return_per_trade": 0.0,
            "median_net_return_per_trade": 0.0,
            "avg_bars_held": 0.0,
            "median_bars_held": 0.0,
        }
    net = trades["net_return"]
    long_mask = trades["side"] == "long"
    short_mask = trades["side"] == "short"
    final_equity = float(trades["exit_equity_usd"].iloc[-1]) if len(trades) else float(initial_capital_usd)
    return {
        "initial_capital_usd": float(initial_capital_usd),
        "leverage": float(leverage),
        "final_equity_usd": final_equity,
        "net_pnl_usd": float(final_equity - initial_capital_usd),
        "compounded_total_return": float(final_equity / initial_capital_usd - 1.0) if initial_capital_usd > 0 else 0.0,
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
    slippage_rate = args.slippage_bps / 10000.0
    return_column = resolve_return_column(args.execution_model)
    fill_price_column = "close" if args.execution_model == "close_to_close" else "next_open_fill_price"
    param = ParamConfig(
        breakout_window=args.breakout_window,
        gate_mode=args.gate_mode,
        min_hold_bars=args.min_hold_bars,
        cooldown_bars=args.cooldown_bars,
    )

    all_trades: List[pd.DataFrame] = []
    rolling_capital_usd = float(args.initial_capital_usd)
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
        position = simulate_breakout_param(test_scored, param, execution_model=args.execution_model)
        trades = extract_closed_trades(
            test_scored,
            position,
            fee_rate,
            slippage_rate,
            return_column=return_column,
            fill_price_column=fill_price_column,
            initial_capital_usd=rolling_capital_usd,
            leverage=float(args.leverage),
        )
        if not trades.empty:
            rolling_capital_usd = float(trades["exit_equity_usd"].iloc[-1])
        all_trades.append(trades)
        test_start += args.test_bars

    if not all_trades:
        return pd.DataFrame(
            columns=[
                "side",
                "entry_time",
                "entry_fill_price",
                "entry_equity_usd",
                "exit_time",
                "exit_fill_price",
                "exit_equity_usd",
                "bars_held",
                "gross_return",
                "net_return",
                "gross_pnl_usd",
                "net_pnl_usd",
            ]
        )
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
        },
        "execution_model": args.execution_model,
        "slippage_bps": float(args.slippage_bps),
        "initial_capital_usd": float(args.initial_capital_usd),
        "leverage": float(args.leverage),
    }

    for fee_bps in args.fee_bps_list:
        trades = build_trade_table(args, fee_bps)
        trades.to_csv(output_dir / f"trade_outcomes_fee{int(fee_bps)}.csv", index=False)
        summary[str(float(fee_bps))] = summarize_trades(
            trades,
            initial_capital_usd=float(args.initial_capital_usd),
            leverage=float(args.leverage),
        )

    (output_dir / "trade_outcomes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
