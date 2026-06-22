from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import compute_metrics, gate_passed, simulate_breakout, state_streaks, strategy_grid
from mnts_min_validation import ensure_output_dir
from mnts_15m_state_layer_v2 import apply_v2_model, compute_instability_score, fit_v2_model


FEATURE_COLUMNS = [
    "distribution_shift_l1",
    "token_entropy",
    "entropy_delta",
    "switch_rate",
    "embedding_anomaly",
    "dominant_token_share",
    "fwd_rv",
]

HOURLY_RISK_FEATURES = [
    "hourly_fracture_ratio",
    "hourly_cohesion_ratio",
    "hourly_instability_mean",
    "hourly_instability_max",
    "hourly_shift_mean",
    "hourly_shift_max",
    "hourly_anomaly_mean",
    "hourly_anomaly_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-system walk-forward validation for breakout_48 + rolling V2 state model."
    )
    parser.add_argument("--discovery-v2-csv", required=True)
    parser.add_argument("--validation-v2-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/rolling_breakout_full_system_run")
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument(
        "--execution-model",
        choices=["close_to_close", "close_to_next_open"],
        default="close_to_close",
        help="Execution timing. close_to_close is the legacy model; close_to_next_open uses close signals and next-open fills.",
    )
    parser.add_argument("--hourly-risk-layer", choices=["off", "v1"], default="off")
    parser.add_argument("--hourly-bars", type=int, default=4, help="1h aggregation over 15m bars.")
    parser.add_argument("--hourly-orange-quantile", type=float, default=0.60)
    parser.add_argument("--hourly-red-quantile", type=float, default=0.85)
    parser.add_argument("--train-bars", type=int, default=11520, help="180 days on 15m bars.")
    parser.add_argument("--test-bars", type=int, default=2880, help="45 days on 15m bars.")
    return parser.parse_args()


def breakout_configs() -> Dict[str, object]:
    return {config.label: config for config in strategy_grid() if config.branch == "breakout48"}


def load_cached_feature_frame(discovery_csv: str, validation_csv: str) -> tuple[pd.DataFrame, int]:
    discovery = pd.read_csv(discovery_csv)
    validation = pd.read_csv(validation_csv)
    discovery["timestamp"] = pd.to_datetime(discovery["timestamp"], utc=True, errors="coerce")
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    discovery["source_split"] = "discovery"
    validation["source_split"] = "validation"
    full = pd.concat([discovery, validation], ignore_index=True, sort=False).sort_values("timestamp").reset_index(drop=True)
    split_idx = int((full["source_split"] == "discovery").sum())

    close = full["close"]
    upper48 = close.shift(1).rolling(48).max()
    lower48 = close.shift(1).rolling(48).min()
    full["breakout_upper_48"] = upper48
    full["breakout_lower_48"] = lower48
    full["breakout_up_event"] = close > upper48
    full["breakout_down_event"] = close < lower48
    full["next_bar_ret"] = np.log(close.shift(-1) / close)
    full["next_open_fill_price"] = full["open"].shift(-1)
    full["next_open_ret"] = np.log(full["open"].shift(-2) / full["open"].shift(-1))
    return full, split_idx


def resolve_return_column(execution_model: str) -> str:
    if execution_model == "close_to_close":
        return "next_bar_ret"
    if execution_model == "close_to_next_open":
        return "next_open_ret"
    raise ValueError(f"Unknown execution model: {execution_model}")


def valid_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def attach_state_streaks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cohesion_streak"] = state_streaks(out["state"], "cohesion")
    out["fracture_streak"] = state_streaks(out["state"], "fracture")
    return out


def build_hourly_risk_features(df: pd.DataFrame, hourly_bars: int) -> pd.DataFrame:
    out = df.copy()
    fracture_flag = (out["state"] == "fracture").astype(float)
    cohesion_flag = (out["state"] == "cohesion").astype(float)
    out["hourly_fracture_ratio"] = fracture_flag.rolling(hourly_bars, min_periods=1).mean()
    out["hourly_cohesion_ratio"] = cohesion_flag.rolling(hourly_bars, min_periods=1).mean()
    out["hourly_instability_mean"] = out["instability_score"].rolling(hourly_bars, min_periods=1).mean()
    out["hourly_instability_max"] = out["instability_score"].rolling(hourly_bars, min_periods=1).max()
    out["hourly_shift_mean"] = out["distribution_shift_l1"].rolling(hourly_bars, min_periods=1).mean()
    out["hourly_shift_max"] = out["distribution_shift_l1"].rolling(hourly_bars, min_periods=1).max()
    out["hourly_anomaly_mean"] = out["embedding_anomaly"].rolling(hourly_bars, min_periods=1).mean()
    out["hourly_anomaly_max"] = out["embedding_anomaly"].rolling(hourly_bars, min_periods=1).max()
    return out


def fit_hourly_risk_model(
    train_scored: pd.DataFrame,
    hourly_bars: int,
    orange_quantile: float,
    red_quantile: float,
) -> Dict[str, object]:
    work = build_hourly_risk_features(train_scored, hourly_bars)
    work = work.dropna(subset=HOURLY_RISK_FEATURES + ["high_vol_event"]).reset_index(drop=True)
    target = work["high_vol_event"].astype(int)

    feature_directions = {
        "hourly_fracture_ratio": 1.0,
        "hourly_cohesion_ratio": -1.0,
        "hourly_instability_mean": 1.0,
        "hourly_instability_max": 1.0,
        "hourly_shift_mean": 1.0,
        "hourly_shift_max": 1.0,
        "hourly_anomaly_mean": 1.0,
        "hourly_anomaly_max": 1.0,
    }

    feature_stats: Dict[str, Dict[str, float]] = {}
    weights: Dict[str, float] = {}
    pos = work[target == 1]
    neg = work[target == 0]

    for feature, direction in feature_directions.items():
        mean = float(work[feature].mean())
        std = float(work[feature].std(ddof=0))
        if std == 0 or np.isnan(std):
            std = 1.0
        feature_stats[feature] = {"mean": mean, "std": std, "direction": direction}
        pos_mean = float(pos[feature].mean()) if not pos.empty else mean
        neg_mean = float(neg[feature].mean()) if not neg.empty else mean
        raw_weight = direction * (pos_mean - neg_mean) / std
        weights[feature] = abs(raw_weight)

    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {feature: 1.0 / len(feature_directions) for feature in feature_directions}
    else:
        weights = {feature: value / weight_sum for feature, value in weights.items()}

    work["hourly_risk_score"] = compute_instability_score(work, feature_stats, weights)
    orange_threshold = float(work["hourly_risk_score"].quantile(orange_quantile))
    red_threshold = float(work["hourly_risk_score"].quantile(red_quantile))
    return {
        "hourly_bars": int(hourly_bars),
        "orange_quantile": float(orange_quantile),
        "red_quantile": float(red_quantile),
        "orange_threshold": orange_threshold,
        "red_threshold": red_threshold,
        "feature_stats": feature_stats,
        "weights": weights,
    }


def apply_hourly_risk_model(df: pd.DataFrame, model: Dict[str, object]) -> pd.DataFrame:
    out = build_hourly_risk_features(df, int(model["hourly_bars"]))
    out["hourly_risk_score"] = compute_instability_score(out, model["feature_stats"], model["weights"])
    out["hourly_state"] = "green"
    out.loc[out["hourly_risk_score"] >= float(model["orange_threshold"]), "hourly_state"] = "orange"
    out.loc[out["hourly_risk_score"] >= float(model["red_threshold"]), "hourly_state"] = "red"
    out["hourly_allow_entry"] = (out["hourly_state"] == "green").astype(int)
    out["hourly_force_exit"] = (out["hourly_state"] == "red").astype(int)
    return out


def simulate_breakout_with_hourly_risk(df: pd.DataFrame, config: object) -> pd.Series:
    positions = np.zeros(len(df), dtype=float)
    current = 0.0
    bars_since_change = config.min_hold_bars
    cooldown_left = 0
    branch = str(getattr(config, "branch", ""))
    breakout_window = branch.replace("breakout", "") if branch.startswith("breakout") else ""
    up_col = f"breakout_up_{breakout_window}" if breakout_window and f"breakout_up_{breakout_window}" in df.columns else "breakout_up_event"
    down_col = (
        f"breakout_down_{breakout_window}"
        if breakout_window and f"breakout_down_{breakout_window}" in df.columns
        else "breakout_down_event"
    )

    for i, row in enumerate(df.itertuples(index=False)):
        if cooldown_left > 0:
            cooldown_left -= 1

        hourly_state = str(getattr(row, "hourly_state", "green"))
        hourly_allow_entry = hourly_state == "green"
        hourly_force_exit = hourly_state == "red"
        long_event = config.allow_long and bool(getattr(row, up_col)) and hourly_allow_entry and gate_passed(row, config)
        short_event = config.allow_short and bool(getattr(row, down_col)) and hourly_allow_entry and gate_passed(row, config)

        fracture_exit = config.exit_fracture_bars > 0 and int(row.fracture_streak) >= config.exit_fracture_bars
        changed = False

        if current != 0.0 and hourly_force_exit:
            current = 0.0
            cooldown_left = config.cooldown_bars
            changed = True
        elif current > 0 and fracture_exit and bars_since_change >= config.min_hold_bars:
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


def choose_best_config(
    train_scored: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
    hourly_risk_enabled: bool,
    execution_model: str,
) -> Dict[str, float]:
    label_map = breakout_configs()
    return_column = resolve_return_column(execution_model)
    rows: List[Dict[str, float]] = []
    for label, config in label_map.items():
        position = (
            simulate_breakout_with_hourly_risk(train_scored, config)
            if hourly_risk_enabled
            else simulate_breakout(train_scored, config)
        )
        metrics = compute_metrics(
            train_scored,
            position,
            fee_rate,
            slippage_rate,
            return_column=return_column,
        )
        rows.append({"config": label, **metrics})
    results = pd.DataFrame(rows).sort_values(["net_sharpe", "net_total_return"], ascending=False)
    return results.iloc[0].to_dict()


def stitch_test_state(train_scored: pd.DataFrame, test_scored: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([train_scored.tail(256), test_scored], ignore_index=True)
    combined = attach_state_streaks(combined)
    return combined.iloc[len(combined) - len(test_scored) :].reset_index(drop=True)


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


def walk_forward_full_system(
    df: pd.DataFrame,
    validation_start_idx: int,
    train_bars: int,
    test_bars: int,
    fee_rate: float,
    slippage_rate: float,
    execution_model: str,
    hourly_risk_layer: str,
    hourly_bars: int,
    hourly_orange_quantile: float,
    hourly_red_quantile: float,
) -> tuple[pd.DataFrame, pd.Series]:
    label_map = breakout_configs()
    return_column = resolve_return_column(execution_model)
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
        test_for_v2 = raw_test.copy()

        if len(train_for_v2) < max(2000, train_bars // 4):
            test_start += test_bars
            continue

        v2_model = fit_v2_model(train_for_v2)
        train_scored = attach_state_streaks(apply_v2_model(raw_train, v2_model))
        hourly_model: Dict[str, object] | None = None
        if hourly_risk_layer == "v1":
            hourly_model = fit_hourly_risk_model(
                train_scored=train_scored,
                hourly_bars=hourly_bars,
                orange_quantile=hourly_orange_quantile,
                red_quantile=hourly_red_quantile,
            )
            train_scored = apply_hourly_risk_model(train_scored, hourly_model)

        best_train = choose_best_config(
            train_scored,
            fee_rate,
            slippage_rate,
            hourly_risk_enabled=hourly_model is not None,
            execution_model=execution_model,
        )
        selected_label = str(best_train["config"])
        selected_config = label_map[selected_label]

        test_scored = apply_v2_model(test_for_v2, v2_model)
        test_scored = stitch_test_state(train_scored, test_scored)
        if hourly_model is not None:
            combined_for_hourly = pd.concat(
                [train_scored.tail(max(8, hourly_bars * 2)), test_scored],
                ignore_index=True,
            )
            combined_for_hourly = apply_hourly_risk_model(combined_for_hourly, hourly_model)
            test_scored = combined_for_hourly.iloc[len(combined_for_hourly) - len(test_scored) :].reset_index(drop=True)
            test_position = simulate_breakout_with_hourly_risk(test_scored, selected_config)
        else:
            test_position = simulate_breakout(test_scored, selected_config)
        test_metrics = compute_metrics(
            test_scored,
            test_position,
            fee_rate,
            slippage_rate,
            return_column=return_column,
        )

        gross_ret = test_position.fillna(0.0) * test_scored[return_column].fillna(0.0)
        turnover = test_position.diff().abs().fillna(test_position.abs())
        fee = turnover * fee_rate
        slippage = turnover * slippage_rate
        net_ret = gross_ret - fee - slippage
        stitched_net_ret.iloc[test_start:test_end] = net_ret.to_numpy()

        rows.append(
            {
                "window_id": window_id,
                "train_start": str(df.iloc[train_start]["timestamp"]),
                "train_end": str(df.iloc[train_end - 1]["timestamp"]),
                "test_start": str(df.iloc[test_start]["timestamp"]),
                "test_end": str(df.iloc[test_end - 1]["timestamp"]),
                "selected_config": selected_label,
                "execution_model": execution_model,
                "train_rows_for_v2": int(len(train_for_v2)),
                "train_v2_score_q30": float(v2_model["score_q30"]),
                "train_v2_score_q70": float(v2_model["score_q70"]),
                "train_v2_high_vol_threshold": float(v2_model["high_vol_threshold"]),
                "hourly_risk_layer": hourly_risk_layer,
                "train_hourly_orange_threshold": float(hourly_model["orange_threshold"]) if hourly_model else np.nan,
                "train_hourly_red_threshold": float(hourly_model["red_threshold"]) if hourly_model else np.nan,
                "test_hourly_green_share": float((test_scored["hourly_state"] == "green").mean())
                if "hourly_state" in test_scored
                else np.nan,
                "test_hourly_orange_share": float((test_scored["hourly_state"] == "orange").mean())
                if "hourly_state" in test_scored
                else np.nan,
                "test_hourly_red_share": float((test_scored["hourly_state"] == "red").mean())
                if "hourly_state" in test_scored
                else np.nan,
                "train_net_sharpe": float(best_train["net_sharpe"]),
                "train_net_total_return": float(best_train["net_total_return"]),
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        window_id += 1
        test_start += test_bars

    return pd.DataFrame(rows), stitched_net_ret


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0
    slippage_rate = args.slippage_bps / 10000.0

    df, validation_start_idx = load_cached_feature_frame(args.discovery_v2_csv, args.validation_v2_csv)
    window_results, stitched_net_ret = walk_forward_full_system(
        df=df,
        validation_start_idx=validation_start_idx,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_model=args.execution_model,
        hourly_risk_layer=args.hourly_risk_layer,
        hourly_bars=args.hourly_bars,
        hourly_orange_quantile=args.hourly_orange_quantile,
        hourly_red_quantile=args.hourly_red_quantile,
    )
    window_results.to_csv(output_dir / "rolling_breakout_full_system_window_results.csv", index=False)

    selected_counts = (
        window_results["selected_config"].value_counts().rename_axis("config").reset_index(name="window_count")
        if not window_results.empty
        else pd.DataFrame(columns=["config", "window_count"])
    )
    selected_counts.to_csv(output_dir / "rolling_breakout_full_system_selected_counts.csv", index=False)

    stitched_summary = summarize_stitched_returns(stitched_net_ret.iloc[validation_start_idx:].reset_index(drop=True))
    summary = {
        "fee_bps": float(args.fee_bps),
        "slippage_bps": float(args.slippage_bps),
        "execution_model": args.execution_model,
        "hourly_risk_layer": args.hourly_risk_layer,
        "hourly_bars": int(args.hourly_bars),
        "hourly_orange_quantile": float(args.hourly_orange_quantile),
        "hourly_red_quantile": float(args.hourly_red_quantile),
        "train_bars": int(args.train_bars),
        "test_bars": int(args.test_bars),
        "window_count": int(len(window_results)),
        "stitched_walkforward_summary": stitched_summary,
        "mean_test_net_sharpe": float(window_results["test_net_sharpe"].mean()) if not window_results.empty else 0.0,
        "positive_test_windows": int((window_results["test_net_total_return"] > 0).sum()) if not window_results.empty else 0,
        "mean_train_net_sharpe": float(window_results["train_net_sharpe"].mean()) if not window_results.empty else 0.0,
    }
    (output_dir / "rolling_breakout_full_system_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
