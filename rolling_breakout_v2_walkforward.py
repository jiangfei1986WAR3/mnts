from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import (
    StrategyConfig,
    compute_metrics,
    simulate_breakout,
    state_streaks,
    strategy_grid,
)
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward rolling validation for engineered breakout_48 + V2 strategies."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--discovery-v2-csv", required=True)
    parser.add_argument("--validation-v2-csv", required=True)
    parser.add_argument("--v2-summary-json", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/rolling_breakout_walkforward_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--train-bars", type=int, default=11520, help="180 days on 15m bars.")
    parser.add_argument("--test-bars", type=int, default=2880, help="45 days on 15m bars.")
    parser.add_argument("--start-mode", default="validation_only", choices=["validation_only", "full_series"])
    return parser.parse_args()


def zscore_series(series: pd.Series, mean: float, std: float) -> pd.Series:
    std = 1.0 if abs(std) < 1e-8 else std
    return (series - mean) / std


def compute_instability_score(
    df: pd.DataFrame,
    feature_stats: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for feature, params in feature_stats.items():
        z = zscore_series(df[feature], params["mean"], params["std"])
        score += weights[feature] * params["direction"] * z
    return score


def apply_v2_model(df: pd.DataFrame, model: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    feature_stats = model["feature_stats"]
    weights = model["weights"]
    out["instability_score"] = compute_instability_score(out, feature_stats, weights)
    out["state"] = "drift"
    out.loc[out["instability_score"] <= float(model["score_q30"]), "state"] = "cohesion"
    out.loc[out["instability_score"] >= float(model["score_q70"]), "state"] = "fracture"
    out["high_vol_event"] = (out["fwd_rv"] >= float(model["high_vol_threshold"])).astype(int)
    return out


def load_full_v2_states(
    discovery_csv: str,
    validation_csv: str,
    summary_json: str,
) -> pd.DataFrame:
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    model = summary["model"]

    discovery = pd.read_csv(discovery_csv)
    discovery["timestamp"] = pd.to_datetime(discovery["timestamp"], utc=True, errors="coerce")
    discovery_scored = apply_v2_model(discovery, model)

    validation = pd.read_csv(validation_csv)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    full = pd.concat(
        [
            discovery_scored[["timestamp", "state", "high_vol_event"]],
            validation[["timestamp", "state", "high_vol_event"]],
        ],
        ignore_index=True,
    ).sort_values("timestamp")
    return full.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def build_breakout_frame(
    input_csv: str,
    discovery_v2_csv: str,
    validation_v2_csv: str,
    v2_summary_json: str,
) -> tuple[pd.DataFrame, int]:
    raw_df = load_ohlcv_csv(input_csv)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
    split_idx = len(raw_df) // 2

    close = raw_df["close"]
    upper48 = close.shift(1).rolling(48).max()
    lower48 = close.shift(1).rolling(48).min()
    raw_df["breakout_upper_48"] = upper48
    raw_df["breakout_lower_48"] = lower48
    raw_df["breakout_up_event"] = close > upper48
    raw_df["breakout_down_event"] = close < lower48
    raw_df["next_bar_ret"] = np.log(close.shift(-1) / close)

    states = load_full_v2_states(discovery_v2_csv, validation_v2_csv, v2_summary_json)
    merged = raw_df.merge(states, on="timestamp", how="inner")
    merged["cohesion_streak"] = state_streaks(merged["state"], "cohesion")
    merged["fracture_streak"] = state_streaks(merged["state"], "fracture")
    return merged.reset_index(drop=True), split_idx


def breakout_configs() -> List[StrategyConfig]:
    return [config for config in strategy_grid() if config.branch == "breakout48"]


def choose_best_config(train_df: pd.DataFrame, configs: List[StrategyConfig], fee_rate: float) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for config in configs:
        position = simulate_breakout(train_df, config)
        metrics = compute_metrics(train_df, position, fee_rate)
        rows.append({"config": config.label, **metrics})
    score_df = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    return score_df.iloc[0].to_dict()


def config_map() -> Dict[str, StrategyConfig]:
    return {config.label: config for config in breakout_configs()}


def walk_forward(
    df: pd.DataFrame,
    validation_start_idx: int,
    train_bars: int,
    test_bars: int,
    fee_rate: float,
    start_mode: str,
) -> tuple[pd.DataFrame, pd.Series]:
    configs = breakout_configs()
    label_to_config = config_map()
    rows: List[Dict[str, float]] = []
    stitched_net_ret = pd.Series(0.0, index=df.index, dtype=float)

    if start_mode == "validation_only":
        test_start = validation_start_idx
    else:
        test_start = train_bars

    window_id = 0
    while test_start + test_bars <= len(df):
        train_start = max(0, test_start - train_bars)
        train_end = test_start
        test_end = test_start + test_bars
        train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

        if len(train_df) < max(500, train_bars // 3):
            test_start += test_bars
            continue

        best_train = choose_best_config(train_df, configs, fee_rate)
        selected_label = str(best_train["config"])
        selected_config = label_to_config[selected_label]
        test_position = simulate_breakout(test_df, selected_config)
        test_metrics = compute_metrics(test_df, test_position, fee_rate)

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
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        window_id += 1
        test_start += test_bars

    return pd.DataFrame(rows), stitched_net_ret


def evaluate_fixed_configs(
    df: pd.DataFrame,
    start_idx: int,
    fee_rate: float,
    labels: List[str],
) -> pd.DataFrame:
    work_df = df.iloc[start_idx:].copy().reset_index(drop=True)
    label_to_config = config_map()
    rows: List[Dict[str, float]] = []
    for label in labels:
        config = label_to_config[label]
        position = simulate_breakout(work_df, config)
        metrics = compute_metrics(work_df, position, fee_rate)
        rows.append({"config": label, **metrics})
    return pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)


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


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df, validation_start_idx = build_breakout_frame(
        input_csv=args.input_csv,
        discovery_v2_csv=args.discovery_v2_csv,
        validation_v2_csv=args.validation_v2_csv,
        v2_summary_json=args.v2_summary_json,
    )

    window_results, stitched_net_ret = walk_forward(
        df=df,
        validation_start_idx=validation_start_idx,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_rate=fee_rate,
        start_mode=args.start_mode,
    )
    window_results.to_csv(output_dir / "rolling_breakout_window_results.csv", index=False)

    stitched_summary = summarize_stitched_returns(stitched_net_ret.iloc[validation_start_idx:].reset_index(drop=True))
    fixed_results = evaluate_fixed_configs(
        df=df,
        start_idx=validation_start_idx,
        fee_rate=fee_rate,
        labels=[
            "breakout48_cohesion_flip_exit2_hold12_cool8",
            "breakout48_nonfracture_flip_exit2_hold12_cool8",
            "breakout48_raw_flip_hold8",
        ],
    )
    fixed_results.to_csv(output_dir / "rolling_breakout_fixed_benchmarks.csv", index=False)

    selected_counts = (
        window_results["selected_config"].value_counts().rename_axis("config").reset_index(name="window_count")
        if not window_results.empty
        else pd.DataFrame(columns=["config", "window_count"])
    )
    selected_counts.to_csv(output_dir / "rolling_breakout_selected_counts.csv", index=False)

    summary = {
        "fee_bps": float(args.fee_bps),
        "train_bars": int(args.train_bars),
        "test_bars": int(args.test_bars),
        "window_count": int(len(window_results)),
        "stitched_walkforward_summary": stitched_summary,
        "mean_test_net_sharpe": float(window_results["test_net_sharpe"].mean()) if not window_results.empty else 0.0,
        "positive_test_windows": int((window_results["test_net_total_return"] > 0).sum()) if not window_results.empty else 0,
        "best_fixed_benchmark": fixed_results.iloc[0][["config", "net_sharpe", "net_total_return"]].to_dict()
        if not fixed_results.empty
        else {},
    }
    (output_dir / "rolling_breakout_walkforward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
