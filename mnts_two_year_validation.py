from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnts_min_validation import (
    KronosTokenizerAdapter,
    ensure_output_dir,
    future_realized_vol,
    l1_distribution_shift,
    load_ohlcv_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-year discovery/validation test for MNTS risk warning layer."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/two_year_run")
    parser.add_argument("--kronos-repo-dir", required=True)
    parser.add_argument("--kronos-tokenizer-name", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--rolling-window", type=int, default=96)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--top-risk-tokens", type=int, default=8)
    return parser.parse_args()


def compute_future_min_drawdown(df: pd.DataFrame, horizon: int) -> pd.Series:
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    values = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df) - horizon):
        future_min = np.min(low[i + 1 : i + horizon + 1])
        values[i] = future_min / close[i] - 1.0
    return pd.Series(values, index=df.index)


def tokenize_full_dataset(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    adapter = KronosTokenizerAdapter(
        repo_dir=args.kronos_repo_dir,
        tokenizer_name=args.kronos_tokenizer_name,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    artifacts = adapter.fit_transform(df)
    out = artifacts.frame.copy()
    out["distribution_shift_l1"] = l1_distribution_shift(
        artifacts.token_ids, window=args.rolling_window
    )
    out["fwd_rv"] = future_realized_vol(out["close"], args.horizon)
    out["future_drawdown"] = compute_future_min_drawdown(out, args.horizon)
    return out


def split_discovery_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = len(df) // 2
    discovery = df.iloc[:split_idx].copy().reset_index(drop=True)
    validation = df.iloc[split_idx:].copy().reset_index(drop=True)
    return discovery, validation


def summarize_vol_by_token(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for token_id, group in df.groupby("token_id"):
        vals = group["fwd_rv"].dropna()
        if len(vals) == 0:
            continue
        rows.append(
            {
                "token_id": int(token_id),
                "count": int(len(vals)),
                "mean_fwd_rv": float(vals.mean()),
                "median_fwd_rv": float(vals.median()),
                "q10_fwd_rv": float(vals.quantile(0.10)),
                "q90_fwd_rv": float(vals.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def fit_discovery_rules(
    discovery: pd.DataFrame,
    horizon: int,
    min_count: int,
    top_risk_tokens: int,
    rolling_window: int,
) -> Dict[str, object]:
    token_vol = summarize_vol_by_token(discovery)
    eligible = token_vol[token_vol["count"] >= min_count].copy()
    eligible = eligible.sort_values(["mean_fwd_rv", "count"], ascending=[False, False])
    risk_tokens = eligible.head(top_risk_tokens)["token_id"].astype(int).tolist()

    discovery = discovery.copy()
    discovery["high_risk_token_flag"] = discovery["token_id"].isin(risk_tokens).astype(int)
    discovery["rolling_high_risk_ratio"] = (
        discovery["high_risk_token_flag"].rolling(rolling_window, min_periods=1).mean()
    )

    shift_valid = discovery["distribution_shift_l1"].dropna()
    shift_q70 = float(shift_valid.quantile(0.70))
    shift_q90 = float(shift_valid.quantile(0.90))
    ratio_q70 = float(discovery["rolling_high_risk_ratio"].quantile(0.70))
    ratio_q90 = float(discovery["rolling_high_risk_ratio"].quantile(0.90))
    high_vol_threshold = float(discovery["fwd_rv"].dropna().quantile(0.90))

    past_rv = np.log(discovery["close"]).diff().rolling(horizon).std()
    baseline_threshold = float(past_rv.dropna().quantile(0.90))

    return {
        "risk_tokens": risk_tokens,
        "shift_q70": shift_q70,
        "shift_q90": shift_q90,
        "ratio_q70": ratio_q70,
        "ratio_q90": ratio_q90,
        "high_vol_threshold": high_vol_threshold,
        "baseline_threshold": baseline_threshold,
        "horizon": int(horizon),
        "min_count": int(min_count),
        "rolling_window": int(rolling_window),
        "eligible_risk_token_count": int(len(eligible)),
    }


def apply_warning_rules(df: pd.DataFrame, rules: Dict[str, object], horizon: int) -> pd.DataFrame:
    out = df.copy()
    risk_tokens = set(int(x) for x in rules["risk_tokens"])
    out["high_risk_token_flag"] = out["token_id"].isin(risk_tokens).astype(int)
    out["rolling_high_risk_ratio"] = (
        out["high_risk_token_flag"].rolling(int(rules["rolling_window"]), min_periods=1).mean()
    )
    out["shift_flag_mid"] = (out["distribution_shift_l1"] >= float(rules["shift_q70"])).astype(int)
    out["shift_flag_high"] = (out["distribution_shift_l1"] >= float(rules["shift_q90"])).astype(int)
    out["risk_flag_mid"] = (out["rolling_high_risk_ratio"] >= float(rules["ratio_q70"])).astype(int)
    out["risk_flag_high"] = (out["rolling_high_risk_ratio"] >= float(rules["ratio_q90"])).astype(int)

    out["warning_level"] = "green"
    out.loc[(out["shift_flag_mid"] + out["risk_flag_mid"]) >= 1, "warning_level"] = "orange"
    out.loc[(out["shift_flag_high"] + out["risk_flag_high"]) >= 2, "warning_level"] = "red"

    out["warning_binary"] = (out["warning_level"] != "green").astype(int)
    out["high_vol_event"] = (out["fwd_rv"] >= float(rules["high_vol_threshold"])).astype(int)
    out["crash_event"] = (out["future_drawdown"] <= -0.02).astype(int)

    out["past_rv"] = np.log(out["close"]).diff().rolling(horizon).std()
    out["baseline_warning"] = (out["past_rv"] >= float(rules["baseline_threshold"])).astype(int)
    return out


def summarize_warning_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for level, group in df.groupby("warning_level"):
        rows.append(
            {
                "warning_level": level,
                "count": int(len(group)),
                "mean_future_rv": float(group["fwd_rv"].dropna().mean()),
                "high_vol_event_rate": float(group["high_vol_event"].mean()),
                "crash_event_rate": float(group["crash_event"].mean()),
                "mean_future_drawdown": float(group["future_drawdown"].dropna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("warning_level")


def summarize_binary_compare(df: pd.DataFrame, signal_col: str, name: str) -> Dict[str, float]:
    flagged = df[df[signal_col] == 1]
    quiet = df[df[signal_col] == 0]
    return {
        "signal_name": name,
        "flagged_count": int(len(flagged)),
        "quiet_count": int(len(quiet)),
        "flagged_mean_future_rv": float(flagged["fwd_rv"].dropna().mean()),
        "quiet_mean_future_rv": float(quiet["fwd_rv"].dropna().mean()),
        "flagged_high_vol_rate": float(flagged["high_vol_event"].mean()),
        "quiet_high_vol_rate": float(quiet["high_vol_event"].mean()),
        "flagged_crash_rate": float(flagged["crash_event"].mean()),
        "quiet_crash_rate": float(quiet["crash_event"].mean()),
    }


def compute_event_coverage(df: pd.DataFrame, signal_col: str, lookback: int = 16) -> float:
    event_idx = np.where(df["high_vol_event"].to_numpy(dtype=int) == 1)[0]
    signal = df[signal_col].to_numpy(dtype=int)
    hits = 0
    total = 0
    for idx in event_idx:
        if idx <= 0:
            continue
        total += 1
        start = max(0, idx - lookback)
        if signal[start:idx].max() > 0:
            hits += 1
    return float(hits / total) if total > 0 else 0.0


def plot_validation_warning(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df["timestamp"], df["rolling_high_risk_ratio"], color="#dc2626", linewidth=1.2)
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("High-risk ratio", color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["distribution_shift_l1"], color="#2563eb", linewidth=1.0, alpha=0.8)
    ax2.set_ylabel("Distribution shift", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")

    red_mask = df["warning_level"] == "red"
    orange_mask = df["warning_level"] == "orange"
    ax1.scatter(df.loc[orange_mask, "timestamp"], df.loc[orange_mask, "rolling_high_risk_ratio"], s=8, color="#f59e0b", alpha=0.6)
    ax1.scatter(df.loc[red_mask, "timestamp"], df.loc[red_mask, "rolling_high_risk_ratio"], s=10, color="#991b1b", alpha=0.8)

    ax1.set_title("Validation-Year Risk Warning Layer")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_year_warning_layer.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    raw_df = load_ohlcv_csv(args.input_csv)
    tokenized = tokenize_full_dataset(args, raw_df)
    discovery, validation = split_discovery_validation(tokenized)

    rules = fit_discovery_rules(
        discovery=discovery,
        horizon=args.horizon,
        min_count=args.min_count,
        top_risk_tokens=args.top_risk_tokens,
        rolling_window=args.rolling_window,
    )
    validation_scored = apply_warning_rules(validation, rules, args.horizon)

    warning_groups = summarize_warning_groups(validation_scored)
    mnts_binary = summarize_binary_compare(validation_scored, "warning_binary", "mnts_warning")
    baseline_binary = summarize_binary_compare(validation_scored, "baseline_warning", "vol_baseline")
    mnts_coverage = compute_event_coverage(validation_scored, "warning_binary", lookback=args.horizon)
    baseline_coverage = compute_event_coverage(validation_scored, "baseline_warning", lookback=args.horizon)

    discovery.to_csv(output_dir / "discovery_year_tokenized.csv", index=False)
    validation_scored.to_csv(output_dir / "validation_year_scored.csv", index=False)
    warning_groups.to_csv(output_dir / "validation_warning_group_summary.csv", index=False)

    summary = {
        "discovery_rows": int(len(discovery)),
        "validation_rows": int(len(validation_scored)),
        "discovery_start": str(discovery["timestamp"].iloc[0]),
        "discovery_end": str(discovery["timestamp"].iloc[-1]),
        "validation_start": str(validation_scored["timestamp"].iloc[0]),
        "validation_end": str(validation_scored["timestamp"].iloc[-1]),
        "rules": rules,
        "validation_warning_counts": {
            level: int(count)
            for level, count in validation_scored["warning_level"].value_counts().to_dict().items()
        },
        "mnts_warning_binary": mnts_binary,
        "vol_baseline_binary": baseline_binary,
        "mnts_high_vol_event_coverage": mnts_coverage,
        "vol_baseline_high_vol_event_coverage": baseline_coverage,
    }
    (output_dir / "two_year_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_validation_warning(validation_scored, output_dir)
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

