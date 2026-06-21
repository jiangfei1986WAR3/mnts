from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import compute_metrics
from keltner_channel_v2_experiment import build_base_frame
from keltner_channel_v2_low_turnover_experiment import (
    StrategyConfig,
    apply_min_hold,
    build_desired_position,
    strategy_grid,
)
from mnts_min_validation import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward rolling validation for low-turnover Keltner+V2 strategies."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/rolling_keltner_channel_v2_low_turnover_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--train-bars", type=int, default=11520, help="Rolling train window.")
    parser.add_argument("--test-bars", type=int, default=2880, help="Rolling test window.")
    return parser.parse_args()


def label_map() -> Dict[str, StrategyConfig]:
    return {config.label: config for config in strategy_grid()}


def simulate_config(df: pd.DataFrame, config: StrategyConfig, fee_rate: float) -> Dict[str, object]:
    desired = build_desired_position(df, config)
    actual = apply_min_hold(desired, config.min_hold_bars)
    metrics = compute_metrics(df, actual, fee_rate)
    return {"config": config.label, "position": actual, **metrics}


def choose_best_config(train_df: pd.DataFrame, fee_rate: float) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for config in strategy_grid():
        result = simulate_config(train_df, config, fee_rate)
        rows.append({k: v for k, v in result.items() if k != "position"})
    score_df = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    return score_df.iloc[0].to_dict()


def summarize_stitched_returns(net_ret: pd.Series) -> Dict[str, float]:
    active = net_ret != 0
    equity = np.exp(net_ret.cumsum())
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    annual_factor = np.sqrt(365 * 24 * 4)
    ret_std = float(net_ret.std(ddof=0))
    sharpe = float(net_ret.mean() / ret_std * annual_factor) if ret_std > 1e-12 else 0.0
    return {
        "bars": int(len(net_ret)),
        "active_bars": int(active.sum()),
        "net_total_return": float(equity.iloc[-1] - 1.0),
        "net_sharpe": sharpe,
        "net_max_drawdown": float(drawdown.min()),
    }


def walk_forward(df: pd.DataFrame, train_bars: int, test_bars: int, fee_rate: float) -> tuple[pd.DataFrame, pd.Series]:
    configs = label_map()
    rows: List[Dict[str, float]] = []
    stitched_net_ret = pd.Series(0.0, index=df.index, dtype=float)

    test_start = train_bars
    window_id = 0
    while test_start + test_bars <= len(df):
        train_start = test_start - train_bars
        train_end = test_start
        test_end = test_start + test_bars

        train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

        best_train = choose_best_config(train_df, fee_rate)
        selected_label = str(best_train["config"])
        selected_config = configs[selected_label]
        test_result = simulate_config(test_df, selected_config, fee_rate)
        test_position = test_result.pop("position")

        gross_ret = test_position.fillna(0.0) * test_df["next_bar_ret"].fillna(0.0)
        turnover = test_position.diff().abs().fillna(test_position.abs())
        fee = turnover * fee_rate
        net_ret = gross_ret - fee
        stitched_net_ret.iloc[test_start:test_end] = net_ret.to_numpy()

        rows.append(
            {
                "window_id": window_id,
                "train_start": str(df.iloc[train_start]["timestamp"]),
                "train_end": str(df.iloc[train_end - 1]["timestamp"]),
                "test_start": str(df.iloc[test_start]["timestamp"]),
                "test_end": str(df.iloc[test_end - 1]["timestamp"]),
                "selected_config": selected_label,
                "train_net_sharpe": float(best_train["net_sharpe"]),
                "train_net_total_return": float(best_train["net_total_return"]),
                **{f"test_{k}": v for k, v in test_result.items() if k != "config"},
            }
        )
        window_id += 1
        test_start += test_bars

    return pd.DataFrame(rows), stitched_net_ret


def evaluate_fixed_configs(df: pd.DataFrame, fee_rate: float, labels: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    configs = label_map()
    for label in labels:
        result = simulate_config(df, configs[label], fee_rate)
        rows.append({k: v for k, v in result.items() if k != "position"})
    return pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df = build_base_frame(args.input_csv, args.v2_validation_csv)
    window_results, stitched_net_ret = walk_forward(
        df=df,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_rate=fee_rate,
    )
    window_results.to_csv(output_dir / "rolling_keltner_channel_v2_low_turnover_window_results.csv", index=False)

    selected_counts = (
        window_results["selected_config"].value_counts().rename_axis("config").reset_index(name="window_count")
        if not window_results.empty
        else pd.DataFrame(columns=["config", "window_count"])
    )
    selected_counts.to_csv(output_dir / "rolling_keltner_channel_v2_low_turnover_selected_counts.csv", index=False)

    stitched_summary = summarize_stitched_returns(stitched_net_ret.iloc[args.train_bars :].reset_index(drop=True))
    fixed_results = evaluate_fixed_configs(
        df=df.iloc[args.train_bars :].copy().reset_index(drop=True),
        fee_rate=fee_rate,
        labels=[
            "defensive_m20_confirm2_exit2_hold24",
            "fracture3_m20_confirm2_exit2_hold24",
            "cohesion2_m20_confirm2_exit2_hold16",
            "raw_m20_confirm2_exit2_hold24",
        ],
    )
    fixed_results.to_csv(output_dir / "rolling_keltner_channel_v2_low_turnover_fixed_benchmarks.csv", index=False)

    summary = {
        "fee_bps": float(args.fee_bps),
        "train_bars": int(args.train_bars),
        "test_bars": int(args.test_bars),
        "window_count": int(len(window_results)),
        "stitched_walkforward_summary": stitched_summary,
        "mean_test_net_sharpe": float(window_results["test_net_sharpe"].mean()) if not window_results.empty else 0.0,
        "positive_test_windows": int((window_results["test_net_total_return"] > 0).sum()) if not window_results.empty else 0,
        "mean_train_net_sharpe": float(window_results["train_net_sharpe"].mean()) if not window_results.empty else 0.0,
    }
    (output_dir / "rolling_keltner_channel_v2_low_turnover_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
