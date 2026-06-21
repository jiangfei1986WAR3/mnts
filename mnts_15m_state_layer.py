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
    forward_returns,
    future_realized_vol,
    l1_distribution_shift,
    load_ohlcv_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal 15m MNTS tactical state layer based on token distribution dynamics."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/15m_state_run")
    parser.add_argument("--kronos-repo-dir", required=True)
    parser.add_argument("--kronos-tokenizer-name", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    return parser.parse_args()


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
        artifacts.token_ids, window=args.state_window
    )
    out["fwd_ret"] = forward_returns(out["close"], args.horizon)
    out["fwd_rv"] = future_realized_vol(out["close"], args.horizon)
    return out


def split_discovery_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = len(df) // 2
    discovery = df.iloc[:split_idx].copy().reset_index(drop=True)
    validation = df.iloc[split_idx:].copy().reset_index(drop=True)
    return discovery, validation


def rolling_entropy_and_dominance(token_ids: np.ndarray, window: int) -> Tuple[pd.Series, pd.Series]:
    entropies = np.full(len(token_ids), np.nan, dtype=float)
    dominance = np.full(len(token_ids), np.nan, dtype=float)

    for end in range(window, len(token_ids) + 1):
        current = token_ids[end - window : end]
        probs = pd.Series(current).value_counts(normalize=True)
        p = probs.to_numpy(dtype=float)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        entropies[end - 1] = entropy
        dominance[end - 1] = float(p.max())

    return pd.Series(entropies), pd.Series(dominance)


def fit_state_rules(discovery: pd.DataFrame) -> Dict[str, float]:
    shift = discovery["distribution_shift_l1"].dropna()
    entropy = discovery["token_entropy"].dropna()
    dominance = discovery["dominant_token_share"].dropna()
    future_rv = discovery["fwd_rv"].dropna()

    rules = {
        "shift_q50": float(shift.quantile(0.50)),
        "shift_q65": float(shift.quantile(0.65)),
        "shift_q85": float(shift.quantile(0.85)),
        "entropy_q30": float(entropy.quantile(0.30)),
        "entropy_q50": float(entropy.quantile(0.50)),
        "entropy_q70": float(entropy.quantile(0.70)),
        "dominance_q50": float(dominance.quantile(0.50)),
        "dominance_q70": float(dominance.quantile(0.70)),
        "high_vol_threshold": float(future_rv.quantile(0.90)),
    }
    return rules


def apply_state_rules(df: pd.DataFrame, rules: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["state"] = "drift"

    cohesion_mask = (
        (out["token_entropy"] <= rules["entropy_q30"])
        & (out["dominant_token_share"] >= rules["dominance_q70"])
        & (out["distribution_shift_l1"] <= rules["shift_q50"])
    )
    fracture_mask = (
        (out["distribution_shift_l1"] >= rules["shift_q85"])
        & (
            (out["token_entropy"] >= rules["entropy_q50"])
            | (out["dominant_token_share"] <= rules["dominance_q50"])
        )
    )

    out.loc[cohesion_mask, "state"] = "cohesion"
    out.loc[fracture_mask, "state"] = "fracture"
    out["high_vol_event"] = (out["fwd_rv"] >= rules["high_vol_threshold"]).astype(int)
    out["abs_fwd_ret"] = out["fwd_ret"].abs()
    return out


def summarize_states(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for state, group in df.groupby("state"):
        rows.append(
            {
                "state": state,
                "count": int(len(group)),
                "share": float(len(group) / len(df)),
                "mean_shift": float(group["distribution_shift_l1"].dropna().mean()),
                "mean_entropy": float(group["token_entropy"].dropna().mean()),
                "mean_dominance": float(group["dominant_token_share"].dropna().mean()),
                "mean_fwd_ret": float(group["fwd_ret"].dropna().mean()),
                "median_fwd_ret": float(group["fwd_ret"].dropna().median()),
                "mean_abs_fwd_ret": float(group["abs_fwd_ret"].dropna().mean()),
                "mean_fwd_rv": float(group["fwd_rv"].dropna().mean()),
                "high_vol_event_rate": float(group["high_vol_event"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("state")


def plot_state_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    level_map = {"cohesion": 0, "drift": 1, "fracture": 2}
    numeric_state = df["state"].map(level_map)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(df["timestamp"], df["close"], linewidth=1.0, color="#111827")
    axes[0].set_title("BTCUSDT 15m Close")
    axes[0].set_ylabel("Close")

    axes[1].plot(df["timestamp"], df["distribution_shift_l1"], linewidth=1.0, color="#2563eb", label="Shift")
    axes[1].plot(df["timestamp"], df["token_entropy"], linewidth=1.0, color="#16a34a", alpha=0.8, label="Entropy")
    axes[1].plot(df["timestamp"], df["dominant_token_share"], linewidth=1.0, color="#f59e0b", alpha=0.8, label="Dominance")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("Metrics")

    axes[2].scatter(df["timestamp"], numeric_state, s=6, c=numeric_state, cmap="viridis", alpha=0.7)
    axes[2].set_yticks([0, 1, 2], ["cohesion", "drift", "fracture"])
    axes[2].set_ylabel("State")
    axes[2].set_xlabel("Timestamp")

    fig.tight_layout()
    fig.savefig(output_dir / "validation_state_timeline.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    raw_df = load_ohlcv_csv(args.input_csv)
    full_df = tokenize_full_dataset(args, raw_df)
    full_df["token_entropy"], full_df["dominant_token_share"] = rolling_entropy_and_dominance(
        full_df["token_id"].to_numpy(dtype=int),
        window=args.state_window,
    )

    discovery, validation = split_discovery_validation(full_df)
    rules = fit_state_rules(discovery)
    validation_scored = apply_state_rules(validation, rules)
    state_summary = summarize_states(validation_scored)

    discovery.to_csv(output_dir / "discovery_year_15m_tokenized.csv", index=False)
    validation_scored.to_csv(output_dir / "validation_year_15m_states.csv", index=False)
    state_summary.to_csv(output_dir / "validation_15m_state_summary.csv", index=False)
    plot_state_timeline(validation_scored, output_dir)

    summary = {
        "discovery_rows": int(len(discovery)),
        "validation_rows": int(len(validation_scored)),
        "state_window": int(args.state_window),
        "horizon": int(args.horizon),
        "rules": rules,
        "validation_state_counts": {
            state: int(count) for state, count in validation_scored["state"].value_counts().to_dict().items()
        },
        "state_judgement": {
            "has_state_separation_on_future_rv": bool(
                state_summary["mean_fwd_rv"].max() - state_summary["mean_fwd_rv"].min() > 0.00015
            ),
            "has_state_separation_on_high_vol_event": bool(
                state_summary["high_vol_event_rate"].max() - state_summary["high_vol_event_rate"].min() > 0.015
            ),
        },
    }
    (output_dir / "15m_state_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
