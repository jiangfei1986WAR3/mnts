from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

from engineer_pullback_breakout_v2 import StrategyConfig, gate_passed
from mnts_min_validation import ensure_output_dir
from mnts_15m_state_layer_v2 import apply_v2_model, fit_v2_model
from rolling_breakout_v2_full_system_walkforward import (
    attach_state_streaks,
    load_cached_feature_frame,
    stitch_test_state,
    summarize_stitched_returns,
    valid_training_rows,
)


@dataclass(frozen=True)
class ParamConfig:
    breakout_window: int
    gate_mode: str
    min_hold_bars: int
    cooldown_bars: int

    @property
    def state_confirm_bars(self) -> int:
        return 2 if self.gate_mode == "cohesion" else 1

    @property
    def label(self) -> str:
        return (
            f"bw{self.breakout_window}_{self.gate_mode}"
            f"_hold{self.min_hold_bars}_cool{self.cooldown_bars}"
        )

    def to_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            label=self.label,
            branch=f"breakout{self.breakout_window}",
            allow_long=True,
            allow_short=True,
            gate_mode=self.gate_mode,
            setup_confirm_bars=1,
            state_confirm_bars=self.state_confirm_bars,
            exit_fracture_bars=2,
            min_hold_bars=self.min_hold_bars,
            cooldown_bars=self.cooldown_bars,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter stability map for breakout + rolling V2 using 4-year cached features."
    )
    parser.add_argument("--discovery-v2-csv", required=True)
    parser.add_argument("--validation-v2-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/breakout_v2_param_stability_4y_run")
    parser.add_argument("--train-bars", type=int, default=11520, help="180 days on 15m bars.")
    parser.add_argument("--test-bars", type=int, default=2880, help="45 days on 15m bars.")
    parser.add_argument("--breakout-windows", nargs="+", type=int, default=[40, 48, 56, 64])
    parser.add_argument("--min-holds", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--cooldowns", nargs="+", type=int, default=[0, 4, 8, 12])
    parser.add_argument("--gate-modes", nargs="+", default=["nonfracture", "cohesion"])
    parser.add_argument("--fee-bps-list", nargs="+", type=float, default=[1.0, 4.0])
    return parser.parse_args()


def build_param_grid(args: argparse.Namespace) -> List[ParamConfig]:
    configs = [
        ParamConfig(
            breakout_window=breakout_window,
            gate_mode=gate_mode,
            min_hold_bars=min_hold_bars,
            cooldown_bars=cooldown_bars,
        )
        for breakout_window, min_hold_bars, cooldown_bars, gate_mode in product(
            args.breakout_windows,
            args.min_holds,
            args.cooldowns,
            args.gate_modes,
        )
    ]
    return configs


def prepare_breakout_columns(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for window in windows:
        upper = close.shift(1).rolling(window).max()
        lower = close.shift(1).rolling(window).min()
        out[f"breakout_upper_{window}"] = upper
        out[f"breakout_lower_{window}"] = lower
        out[f"breakout_up_{window}"] = close > upper
        out[f"breakout_down_{window}"] = close < lower
    return out


def simulate_breakout_param(df: pd.DataFrame, param: ParamConfig) -> pd.Series:
    config = param.to_strategy_config()
    up_col = f"breakout_up_{param.breakout_window}"
    down_col = f"breakout_down_{param.breakout_window}"

    positions = np.zeros(len(df), dtype=float)
    current = 0.0
    bars_since_change = config.min_hold_bars
    cooldown_left = 0

    for i, row in enumerate(df.itertuples(index=False)):
        if cooldown_left > 0:
            cooldown_left -= 1

        long_event = config.allow_long and bool(getattr(row, up_col)) and gate_passed(row, config)
        short_event = config.allow_short and bool(getattr(row, down_col)) and gate_passed(row, config)
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


def compute_fee_metrics(df: pd.DataFrame, position: pd.Series, fee_rate: float) -> tuple[Dict[str, float], np.ndarray]:
    next_bar_ret = df["next_bar_ret"].fillna(0.0)
    work_pos = position.fillna(0.0)
    turnover = work_pos.diff().abs().fillna(work_pos.abs())
    gross_ret = work_pos * next_bar_ret
    fee = turnover * fee_rate
    net_ret = gross_ret - fee

    gross_equity = np.exp(gross_ret.cumsum())
    net_equity = np.exp(net_ret.cumsum())
    peak = np.maximum.accumulate(net_equity)
    drawdown = net_equity / peak - 1.0
    active = work_pos != 0
    changes = work_pos.diff().fillna(work_pos)
    trade_events = int(changes.ne(0).sum())

    annual_factor = np.sqrt(365 * 24 * 4)
    gross_std = float(gross_ret.std(ddof=0))
    net_std = float(net_ret.std(ddof=0))
    gross_sharpe = float(gross_ret.mean() / gross_std * annual_factor) if gross_std > 1e-12 else 0.0
    net_sharpe = float(net_ret.mean() / net_std * annual_factor) if net_std > 1e-12 else 0.0

    metrics = {
        "bars": int(len(df)),
        "trade_events": trade_events,
        "active_bars": int(active.sum()),
        "long_exposure": float((work_pos > 0).mean()),
        "short_exposure": float((work_pos < 0).mean()),
        "avg_abs_position": float(work_pos.abs().mean()),
        "turnover_sum": float(turnover.sum()),
        "gross_total_return": float(gross_equity.iloc[-1] - 1.0),
        "net_total_return": float(net_equity.iloc[-1] - 1.0),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "active_high_vol_rate": float(df.loc[active, "high_vol_event"].mean()) if active.any() else 0.0,
    }
    return metrics, net_ret.to_numpy(dtype=float)


def evaluate_grid(
    df: pd.DataFrame,
    validation_start_idx: int,
    configs: List[ParamConfig],
    train_bars: int,
    test_bars: int,
    fee_bps_list: List[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fee_rates = {fee_bps: fee_bps / 10000.0 for fee_bps in fee_bps_list}
    net_return_chunks: Dict[float, Dict[str, List[np.ndarray]]] = {
        fee_bps: {config.label: [] for config in configs} for fee_bps in fee_bps_list
    }
    window_rows: List[Dict[str, float]] = []

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
        train_scored = attach_state_streaks(apply_v2_model(raw_train, v2_model))
        test_scored = apply_v2_model(raw_test, v2_model)
        test_scored = stitch_test_state(train_scored, test_scored)

        for config in configs:
            test_position = simulate_breakout_param(test_scored, config)
            for fee_bps, fee_rate in fee_rates.items():
                metrics, net_ret = compute_fee_metrics(test_scored, test_position, fee_rate)
                net_return_chunks[fee_bps][config.label].append(net_ret)
                window_rows.append(
                    {
                        "fee_bps": float(fee_bps),
                        "config_label": config.label,
                        "breakout_window": int(config.breakout_window),
                        "gate_mode": config.gate_mode,
                        "state_confirm_bars": int(config.state_confirm_bars),
                        "min_hold_bars": int(config.min_hold_bars),
                        "cooldown_bars": int(config.cooldown_bars),
                        "window_id": window_id,
                        "train_start": str(df.iloc[train_start]["timestamp"]),
                        "train_end": str(df.iloc[train_end - 1]["timestamp"]),
                        "test_start": str(df.iloc[test_start]["timestamp"]),
                        "test_end": str(df.iloc[test_end - 1]["timestamp"]),
                        "train_rows_for_v2": int(len(train_for_v2)),
                        "train_v2_score_q30": float(v2_model["score_q30"]),
                        "train_v2_score_q70": float(v2_model["score_q70"]),
                        "train_v2_high_vol_threshold": float(v2_model["high_vol_threshold"]),
                        **metrics,
                    }
                )

        window_id += 1
        test_start += test_bars

    window_df = pd.DataFrame(window_rows)

    summary_rows: List[Dict[str, float]] = []
    for fee_bps in fee_bps_list:
        fee_window_df = window_df[window_df["fee_bps"] == float(fee_bps)].copy()
        for config in configs:
            chunks = net_return_chunks[fee_bps][config.label]
            if not chunks:
                continue
            stitched = pd.Series(np.concatenate(chunks, axis=0), dtype=float)
            stitched_summary = summarize_stitched_returns(stitched)
            config_windows = fee_window_df[fee_window_df["config_label"] == config.label]
            summary_rows.append(
                {
                    "fee_bps": float(fee_bps),
                    "config_label": config.label,
                    "breakout_window": int(config.breakout_window),
                    "gate_mode": config.gate_mode,
                    "state_confirm_bars": int(config.state_confirm_bars),
                    "min_hold_bars": int(config.min_hold_bars),
                    "cooldown_bars": int(config.cooldown_bars),
                    "window_count": int(len(config_windows)),
                    "positive_windows": int((config_windows["net_total_return"] > 0).sum()),
                    "positive_window_ratio": float((config_windows["net_total_return"] > 0).mean()),
                    "mean_window_net_total_return": float(config_windows["net_total_return"].mean()),
                    "mean_window_net_sharpe": float(config_windows["net_sharpe"].mean()),
                    "mean_window_turnover_sum": float(config_windows["turnover_sum"].mean()),
                    **{f"stitched_{k}": v for k, v in stitched_summary.items()},
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, window_df


def build_json_summary(summary_df: pd.DataFrame) -> Dict[str, object]:
    by_fee: Dict[str, object] = {}
    for fee_bps, fee_group in summary_df.groupby("fee_bps"):
        fee_group = fee_group.sort_values(["stitched_net_sharpe", "stitched_net_total_return"], ascending=False)
        by_fee[str(float(fee_bps))] = {
            "config_count": int(len(fee_group)),
            "positive_config_count": int((fee_group["stitched_net_total_return"] > 0).sum()),
            "configs_with_sharpe_above_1": int((fee_group["stitched_net_sharpe"] > 1.0).sum()),
            "configs_with_sharpe_above_1_5": int((fee_group["stitched_net_sharpe"] > 1.5).sum()),
            "best_config": fee_group.iloc[0][
                [
                    "config_label",
                    "breakout_window",
                    "gate_mode",
                    "min_hold_bars",
                    "cooldown_bars",
                    "stitched_net_total_return",
                    "stitched_net_sharpe",
                    "stitched_net_max_drawdown",
                    "positive_window_ratio",
                ]
            ].to_dict(),
        }
    return {"by_fee_bps": by_fee}


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    configs = build_param_grid(args)
    df, validation_start_idx = load_cached_feature_frame(args.discovery_v2_csv, args.validation_v2_csv)
    df = prepare_breakout_columns(df, args.breakout_windows)

    summary_df, window_df = evaluate_grid(
        df=df,
        validation_start_idx=validation_start_idx,
        configs=configs,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_bps_list=[float(x) for x in args.fee_bps_list],
    )
    summary_df = summary_df.sort_values(
        ["fee_bps", "stitched_net_sharpe", "stitched_net_total_return"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    window_df = window_df.sort_values(["fee_bps", "config_label", "window_id"]).reset_index(drop=True)

    summary_df.to_csv(output_dir / "parameter_stability_summary.csv", index=False)
    window_df.to_csv(output_dir / "parameter_stability_window_results.csv", index=False)

    top_df = summary_df.groupby("fee_bps", group_keys=False).head(20).reset_index(drop=True)
    top_df.to_csv(output_dir / "parameter_stability_top20.csv", index=False)

    anchor_region = summary_df[
        summary_df["breakout_window"].isin([40, 48, 56])
        & summary_df["min_hold_bars"].isin([8, 12, 16])
        & summary_df["cooldown_bars"].isin([4, 8, 12])
    ].copy()
    anchor_region.to_csv(output_dir / "parameter_stability_anchor_region.csv", index=False)

    json_summary = build_json_summary(summary_df)
    (output_dir / "parameter_stability_summary.json").write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
