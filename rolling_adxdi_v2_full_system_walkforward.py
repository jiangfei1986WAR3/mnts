from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from adx_di_v2_experiment import compute_adx
from adx_di_v2_low_turnover_experiment import (
    StrategyConfig,
    apply_min_hold,
    build_desired_position,
    strategy_grid,
)
from mnts_15m_state_layer_v2 import apply_v2_model, fit_v2_model
from rolling_breakout_v2_full_system_walkforward import FEATURE_COLUMNS, summarize_stitched_returns
from mnts_min_validation import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-system walk-forward validation for low-turnover ADX+DI + rolling V2 state model."
    )
    parser.add_argument("--discovery-v2-csv", required=True)
    parser.add_argument("--validation-v2-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/rolling_adxdi_full_system_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--train-bars", type=int, default=11520, help="180 days on 15m bars.")
    parser.add_argument("--test-bars", type=int, default=2880, help="45 days on 15m bars.")
    return parser.parse_args()


def config_map() -> Dict[str, StrategyConfig]:
    return {config.label: config for config in strategy_grid()}


def load_cached_feature_frame(discovery_csv: str, validation_csv: str) -> tuple[pd.DataFrame, int]:
    discovery = pd.read_csv(discovery_csv)
    validation = pd.read_csv(validation_csv)
    discovery["timestamp"] = pd.to_datetime(discovery["timestamp"], utc=True, errors="coerce")
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    discovery["source_split"] = "discovery"
    validation["source_split"] = "validation"
    full = pd.concat([discovery, validation], ignore_index=True, sort=False).sort_values("timestamp").reset_index(drop=True)
    split_idx = int((full["source_split"] == "discovery").sum())

    adx_frame = compute_adx(full["high"], full["low"], full["close"], period=14)
    full = pd.concat([full, adx_frame], axis=1)
    full["next_bar_ret"] = np.log(full["close"].shift(-1) / full["close"])
    return full, split_idx


def valid_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def simulate_over_frame(df: pd.DataFrame, config: StrategyConfig, fee_rate: float) -> Dict[str, object]:
    desired = build_desired_position(df, config)
    position = apply_min_hold(desired, config.min_hold_bars)

    gross_ret = position.fillna(0.0) * df["next_bar_ret"].fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    fee = turnover * fee_rate
    net_ret = gross_ret - fee
    active = position != 0
    equity = np.exp(net_ret.cumsum())
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    annual_factor = np.sqrt(365 * 24 * 4)
    ret_std = float(net_ret.std(ddof=0))
    sharpe = float(net_ret.mean() / ret_std * annual_factor) if ret_std > 1e-12 else 0.0

    return {
        "config": config.label,
        "position": position,
        "net_total_return": float(equity.iloc[-1] - 1.0),
        "net_sharpe": sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "turnover_sum": float(turnover.sum()),
        "active_bars": int(active.sum()),
        "trade_events": int((turnover > 0).sum()),
    }


def choose_best_config(train_scored: pd.DataFrame, fee_rate: float) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for config in strategy_grid():
        result = simulate_over_frame(train_scored, config, fee_rate)
        rows.append({k: v for k, v in result.items() if k != "position"})
    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    return results.iloc[0].to_dict()


def simulate_test_with_context(train_scored: pd.DataFrame, test_scored: pd.DataFrame, config: StrategyConfig, fee_rate: float) -> Dict[str, object]:
    combined = pd.concat([train_scored.tail(256), test_scored], ignore_index=True)
    result = simulate_over_frame(combined, config, fee_rate)
    position = result["position"].iloc[-len(test_scored):].reset_index(drop=True)

    gross_ret = position.fillna(0.0) * test_scored["next_bar_ret"].fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    fee = turnover * fee_rate
    net_ret = gross_ret - fee
    active = position != 0
    equity = np.exp(net_ret.cumsum())
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    annual_factor = np.sqrt(365 * 24 * 4)
    ret_std = float(net_ret.std(ddof=0))
    sharpe = float(net_ret.mean() / ret_std * annual_factor) if ret_std > 1e-12 else 0.0

    return {
        "config": config.label,
        "position": position,
        "net_total_return": float(equity.iloc[-1] - 1.0),
        "net_sharpe": sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "turnover_sum": float(turnover.sum()),
        "active_bars": int(active.sum()),
        "trade_events": int((turnover > 0).sum()),
        "net_ret": net_ret,
    }


def walk_forward_full_system(
    df: pd.DataFrame,
    validation_start_idx: int,
    train_bars: int,
    test_bars: int,
    fee_rate: float,
) -> tuple[pd.DataFrame, pd.Series]:
    label_lookup = config_map()
    rows: List[Dict[str, float]] = []
    stitched_net_ret = pd.Series(0.0, index=df.index, dtype=float)

    test_start = validation_start_idx
    window_id = 0
    while test_start + test_bars <= len(df):
        train_start = max(0, test_start - train_bars)
        train_end = test_start
        test_end = test_start + test_bars

        raw_train = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        raw_test = df.iloc[test_start:test_end].copy().reset_index(drop=True)
        train_for_v2 = valid_training_rows(raw_train)
        if len(train_for_v2) < max(2000, train_bars // 4):
            test_start += test_bars
            continue

        v2_model = fit_v2_model(train_for_v2)
        train_scored = apply_v2_model(raw_train, v2_model)
        best_train = choose_best_config(train_scored, fee_rate)
        selected_label = str(best_train["config"])
        selected_config = label_lookup[selected_label]

        test_scored = apply_v2_model(raw_test, v2_model)
        test_result = simulate_test_with_context(train_scored, test_scored, selected_config, fee_rate)
        test_position = test_result["position"]
        net_ret = test_result["net_ret"]
        stitched_net_ret.iloc[test_start:test_end] = net_ret.to_numpy()

        rows.append(
            {
                "window_id": window_id,
                "train_start": str(df.iloc[train_start]["timestamp"]),
                "train_end": str(df.iloc[train_end - 1]["timestamp"]),
                "test_start": str(df.iloc[test_start]["timestamp"]),
                "test_end": str(df.iloc[test_end - 1]["timestamp"]),
                "selected_config": selected_label,
                "train_rows_for_v2": int(len(train_for_v2)),
                "train_v2_score_q30": float(v2_model["score_q30"]),
                "train_v2_score_q70": float(v2_model["score_q70"]),
                "train_v2_high_vol_threshold": float(v2_model["high_vol_threshold"]),
                "train_net_sharpe": float(best_train["net_sharpe"]),
                "train_net_total_return": float(best_train["net_total_return"]),
                "test_net_total_return": float(test_result["net_total_return"]),
                "test_net_sharpe": float(test_result["net_sharpe"]),
                "test_net_max_drawdown": float(test_result["net_max_drawdown"]),
                "test_turnover_sum": float(test_result["turnover_sum"]),
                "test_active_bars": int(test_result["active_bars"]),
                "test_trade_events": int(test_result["trade_events"]),
                "test_end_position": float(test_position.iloc[-1]) if len(test_position) else 0.0,
            }
        )
        window_id += 1
        test_start += test_bars

    return pd.DataFrame(rows), stitched_net_ret


def evaluate_fixed_full_system(df: pd.DataFrame, validation_start_idx: int, fee_rate: float) -> pd.DataFrame:
    initial_discovery = df.iloc[:validation_start_idx].copy().reset_index(drop=True)
    validation = df.iloc[validation_start_idx:].copy().reset_index(drop=True)
    v2_model = fit_v2_model(valid_training_rows(initial_discovery))
    discovery_scored = apply_v2_model(initial_discovery, v2_model)
    validation_scored = apply_v2_model(validation, v2_model)

    rows: List[Dict[str, float]] = []
    for label in [
        "cohesion2_adx20_confirm2_exit1_hold16",
        "defensive_adx20_confirm2_exit2_hold16",
        "cohesion3_adx25_confirm2_exit2_hold24",
        "raw_adx20_confirm1_exit2_hold16",
    ]:
        result = simulate_test_with_context(discovery_scored, validation_scored, config_map()[label], fee_rate)
        rows.append({k: v for k, v in result.items() if k not in {"position", "net_ret"}})
    return pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df, validation_start_idx = load_cached_feature_frame(args.discovery_v2_csv, args.validation_v2_csv)
    window_results, stitched_net_ret = walk_forward_full_system(
        df=df,
        validation_start_idx=validation_start_idx,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_rate=fee_rate,
    )
    window_results.to_csv(output_dir / "rolling_adxdi_full_system_window_results.csv", index=False)

    selected_counts = (
        window_results["selected_config"].value_counts().rename_axis("config").reset_index(name="window_count")
        if not window_results.empty
        else pd.DataFrame(columns=["config", "window_count"])
    )
    selected_counts.to_csv(output_dir / "rolling_adxdi_full_system_selected_counts.csv", index=False)

    fixed_benchmarks = evaluate_fixed_full_system(df, validation_start_idx, fee_rate)
    fixed_benchmarks.to_csv(output_dir / "rolling_adxdi_full_system_fixed_benchmarks.csv", index=False)

    stitched_summary = summarize_stitched_returns(stitched_net_ret.iloc[validation_start_idx:].reset_index(drop=True))
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
    (output_dir / "rolling_adxdi_full_system_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
