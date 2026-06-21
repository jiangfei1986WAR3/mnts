from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from compare_v2_vs_classic_indicators import compute_macd
from mnts_min_validation import ensure_output_dir, load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-turnover MACD + MNTS V2 tactical filter experiment."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/macd_v2_low_turnover_run")
    parser.add_argument("--fee-bps", type=float, default=4.0)
    return parser.parse_args()


def split_validation(df: pd.DataFrame) -> pd.DataFrame:
    split_idx = len(df) // 2
    return df.iloc[split_idx:].copy().reset_index(drop=True)


def build_base_frame(input_csv: str, v2_validation_csv: str) -> pd.DataFrame:
    raw_df = load_ohlcv_csv(input_csv)
    validation = split_validation(raw_df)
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")

    macd_line, macd_signal, _ = compute_macd(validation["close"])
    validation["macd_line"] = macd_line
    validation["macd_signal_line"] = macd_signal
    validation["macd_signal"] = 0.0
    validation.loc[validation["macd_line"] > validation["macd_signal_line"], "macd_signal"] = 1.0
    validation.loc[validation["macd_line"] < validation["macd_signal_line"], "macd_signal"] = -1.0
    validation["next_bar_ret"] = np.log(validation["close"].shift(-1) / validation["close"])

    v2 = pd.read_csv(v2_validation_csv)
    v2["timestamp"] = pd.to_datetime(v2["timestamp"], utc=True, errors="coerce")
    merged = validation.merge(
        v2[["timestamp", "state", "high_vol_event"]],
        on="timestamp",
        how="inner",
    )
    return merged


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


def build_desired_position(df: pd.DataFrame, mode: str) -> pd.Series:
    signal = df["macd_signal"].astype(float)
    state = df["state"]
    frac_streak = state_streaks(state, "fracture")
    coh_streak = state_streaks(state, "cohesion")

    if mode == "raw":
        return signal
    if mode == "fracture_confirm_2":
        return signal.where(frac_streak < 2, 0.0)
    if mode == "fracture_confirm_3":
        return signal.where(frac_streak < 3, 0.0)
    if mode == "cohesion_confirm_2":
        return signal.where(coh_streak >= 2, 0.0)
    if mode == "cohesion_confirm_3":
        return signal.where(coh_streak >= 3, 0.0)
    if mode == "coarse_weight_confirm":
        weight = pd.Series(0.5, index=df.index, dtype=float)
        weight[state == "fracture"] = 0.0
        weight[state == "cohesion"] = 1.0
        return signal * weight
    raise ValueError(f"Unknown mode: {mode}")


def apply_min_hold(desired: pd.Series, min_hold_bars: int) -> pd.Series:
    arr = desired.fillna(0.0).to_numpy(dtype=float)
    actual = np.zeros(len(arr), dtype=float)
    current = 0.0
    bars_since_change = min_hold_bars

    for i, wanted in enumerate(arr):
        if wanted == current:
            actual[i] = current
            bars_since_change += 1
            continue

        if bars_since_change >= min_hold_bars:
            current = wanted
            bars_since_change = 0
        else:
            bars_since_change += 1

        actual[i] = current

    return pd.Series(actual, index=desired.index)


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

    active = work["position"] != 0
    annual_factor = np.sqrt(365 * 24 * 4)
    gross_std = float(gross_ret.std(ddof=0))
    net_std = float(net_ret.std(ddof=0))

    gross_sharpe = float(gross_ret.mean() / gross_std * annual_factor) if gross_std > 1e-12 else 0.0
    net_sharpe = float(net_ret.mean() / net_std * annual_factor) if net_std > 1e-12 else 0.0

    return {
        "bars": int(len(work)),
        "active_bars": int(active.sum()),
        "exposure": float(active.mean()),
        "avg_abs_position": float(work["position"].abs().mean()),
        "turnover_sum": float(turnover.sum()),
        "gross_total_return": float(gross_equity.iloc[-1] - 1.0),
        "net_total_return": float(net_equity.iloc[-1] - 1.0),
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "net_max_drawdown": float(drawdown.min()),
        "active_high_vol_rate": float(work.loc[active, "high_vol_event"].mean()) if active.any() else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    fee_rate = args.fee_bps / 10000.0

    df = build_base_frame(args.input_csv, args.v2_validation_csv)
    configs: List[Tuple[str, str, int]] = [
        ("raw", "raw", 1),
        ("fracture_confirm2_hold8", "fracture_confirm_2", 8),
        ("fracture_confirm3_hold12", "fracture_confirm_3", 12),
        ("cohesion_confirm2_hold8", "cohesion_confirm_2", 8),
        ("cohesion_confirm3_hold12", "cohesion_confirm_3", 12),
        ("coarse_weight_confirm_hold8", "coarse_weight_confirm", 8),
    ]

    rows: List[Dict[str, float]] = []
    for label, mode, hold in configs:
        desired = build_desired_position(df, mode)
        actual = apply_min_hold(desired, min_hold_bars=hold)
        metrics = compute_metrics(df, actual, fee_rate)
        rows.append({"config": label, "mode": mode, "min_hold_bars": hold, **metrics})

    result = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    result.to_csv(output_dir / "macd_v2_low_turnover_comparison.csv", index=False)

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
    (output_dir / "macd_v2_low_turnover_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
