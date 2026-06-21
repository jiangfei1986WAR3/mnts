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
        description="MNTS 15m state layer V2: lightweight scoring model over token distribution dynamics."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/15m_state_v2_run")
    parser.add_argument("--kronos-repo-dir", required=True)
    parser.add_argument("--kronos-tokenizer-name", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state-window", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    return parser.parse_args()


def tokenize_full_dataset(
    input_csv: str,
    repo_dir: str,
    tokenizer_name: str,
    chunk_size: int,
    device: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw_df = load_ohlcv_csv(input_csv)
    adapter = KronosTokenizerAdapter(
        repo_dir=repo_dir,
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        device=device,
    )
    artifacts = adapter.fit_transform(raw_df)
    out = artifacts.frame.copy()
    return out, artifacts.token_ids, artifacts.embeddings


def split_discovery_validation(
    df: pd.DataFrame, embeddings: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    split_idx = len(df) // 2
    discovery = df.iloc[:split_idx].copy().reset_index(drop=True)
    validation = df.iloc[split_idx:].copy().reset_index(drop=True)
    return discovery, embeddings[:split_idx], validation, embeddings[split_idx:]


def rolling_entropy_and_dominance(token_ids: np.ndarray, window: int) -> Tuple[pd.Series, pd.Series]:
    entropies = np.full(len(token_ids), np.nan, dtype=float)
    dominance = np.full(len(token_ids), np.nan, dtype=float)

    for end in range(window, len(token_ids) + 1):
        current = token_ids[end - window : end]
        probs = pd.Series(current).value_counts(normalize=True)
        p = probs.to_numpy(dtype=float)
        entropies[end - 1] = float(-(p * np.log(p + 1e-12)).sum())
        dominance[end - 1] = float(p.max())

    return pd.Series(entropies), pd.Series(dominance)


def rolling_switch_rate(token_ids: np.ndarray, window: int) -> pd.Series:
    values = np.full(len(token_ids), np.nan, dtype=float)
    if window < 2:
        return pd.Series(values)
    for end in range(window, len(token_ids) + 1):
        current = token_ids[end - window : end]
        switches = np.not_equal(current[1:], current[:-1]).mean()
        values[end - 1] = float(switches)
    return pd.Series(values)


def embedding_anomaly(embeddings: np.ndarray, center: np.ndarray, scale: float) -> pd.Series:
    distances = np.linalg.norm(embeddings - center[None, :], axis=1)
    return pd.Series(distances / max(scale, 1e-8))


def prepare_features(
    df: pd.DataFrame,
    token_ids: np.ndarray,
    embeddings: np.ndarray,
    state_window: int,
    horizon: int,
    center: np.ndarray | None = None,
    scale: float | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    out = df.copy()
    out["distribution_shift_l1"] = l1_distribution_shift(token_ids, window=state_window)
    out["token_entropy"], out["dominant_token_share"] = rolling_entropy_and_dominance(
        token_ids, window=state_window
    )
    out["switch_rate"] = rolling_switch_rate(token_ids, window=state_window)
    out["entropy_delta"] = out["token_entropy"].diff().fillna(0.0)
    out["fwd_ret"] = forward_returns(out["close"], horizon)
    out["fwd_rv"] = future_realized_vol(out["close"], horizon)

    if center is None:
        center = embeddings.mean(axis=0)
    if scale is None:
        raw_dist = np.linalg.norm(embeddings - center[None, :], axis=1)
        scale = float(np.std(raw_dist))
        if scale == 0:
            scale = 1.0
    out["embedding_anomaly"] = embedding_anomaly(embeddings, center, scale)
    return out, center, scale


def zscore_series(series: pd.Series, mean: float, std: float) -> pd.Series:
    std = 1.0 if abs(std) < 1e-8 else std
    return (series - mean) / std


def fit_v2_model(discovery: pd.DataFrame) -> Dict[str, object]:
    high_vol_threshold = float(discovery["fwd_rv"].dropna().quantile(0.90))
    work = discovery.copy()
    work["high_vol_event"] = (work["fwd_rv"] >= high_vol_threshold).astype(int)

    feature_directions = {
        "distribution_shift_l1": 1.0,
        "token_entropy": 1.0,
        "entropy_delta": 1.0,
        "switch_rate": 1.0,
        "embedding_anomaly": 1.0,
        "dominant_token_share": -1.0,
    }

    feature_stats: Dict[str, Dict[str, float]] = {}
    weights: Dict[str, float] = {}
    pos = work[work["high_vol_event"] == 1]
    neg = work[work["high_vol_event"] == 0]

    for feature, direction in feature_directions.items():
        mean = float(work[feature].dropna().mean())
        std = float(work[feature].dropna().std(ddof=0))
        if std == 0 or np.isnan(std):
            std = 1.0
        feature_stats[feature] = {"mean": mean, "std": std, "direction": direction}

        pos_mean = float(pos[feature].dropna().mean()) if not pos.empty else mean
        neg_mean = float(neg[feature].dropna().mean()) if not neg.empty else mean
        raw_weight = direction * (pos_mean - neg_mean) / std
        weights[feature] = abs(raw_weight)

    weight_sum = sum(weights.values())
    if weight_sum == 0:
        weights = {feature: 1.0 / len(weights) for feature in weights}
    else:
        weights = {feature: value / weight_sum for feature, value in weights.items()}

    work["instability_score"] = compute_instability_score(work, feature_stats, weights)
    score_q30 = float(work["instability_score"].quantile(0.30))
    score_q70 = float(work["instability_score"].quantile(0.70))

    return {
        "high_vol_threshold": high_vol_threshold,
        "feature_stats": feature_stats,
        "weights": weights,
        "score_q30": score_q30,
        "score_q70": score_q70,
    }


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
                "mean_score": float(group["instability_score"].mean()),
                "mean_shift": float(group["distribution_shift_l1"].dropna().mean()),
                "mean_entropy": float(group["token_entropy"].dropna().mean()),
                "mean_entropy_delta": float(group["entropy_delta"].dropna().mean()),
                "mean_switch_rate": float(group["switch_rate"].dropna().mean()),
                "mean_embedding_anomaly": float(group["embedding_anomaly"].dropna().mean()),
                "mean_dominance": float(group["dominant_token_share"].dropna().mean()),
                "mean_fwd_ret": float(group["fwd_ret"].dropna().mean()),
                "mean_abs_fwd_ret": float(group["abs_fwd_ret"].dropna().mean()),
                "mean_fwd_rv": float(group["fwd_rv"].dropna().mean()),
                "high_vol_event_rate": float(group["high_vol_event"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("state")


def plot_v2_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    level_map = {"cohesion": 0, "drift": 1, "fracture": 2}
    numeric_state = df["state"].map(level_map)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df["timestamp"], df["close"], linewidth=1.0, color="#111827")
    axes[0].set_title("BTCUSDT 15m Close")
    axes[0].set_ylabel("Close")

    axes[1].plot(df["timestamp"], df["instability_score"], linewidth=1.0, color="#7c3aed")
    axes[1].set_ylabel("Score")
    axes[1].set_title("V2 Instability Score")

    axes[2].plot(df["timestamp"], df["distribution_shift_l1"], linewidth=1.0, color="#2563eb", label="Shift")
    axes[2].plot(df["timestamp"], df["token_entropy"], linewidth=1.0, color="#16a34a", alpha=0.8, label="Entropy")
    axes[2].plot(df["timestamp"], df["switch_rate"], linewidth=1.0, color="#dc2626", alpha=0.8, label="Switch rate")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_ylabel("Metrics")

    axes[3].scatter(df["timestamp"], numeric_state, s=6, c=numeric_state, cmap="viridis", alpha=0.7)
    axes[3].set_yticks([0, 1, 2], ["cohesion", "drift", "fracture"])
    axes[3].set_ylabel("State")
    axes[3].set_xlabel("Timestamp")

    fig.tight_layout()
    fig.savefig(output_dir / "validation_state_v2_timeline.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    tokenized_df, token_ids, embeddings = tokenize_full_dataset(
        input_csv=args.input_csv,
        repo_dir=args.kronos_repo_dir,
        tokenizer_name=args.kronos_tokenizer_name,
        chunk_size=args.chunk_size,
        device=args.device,
    )

    discovery_df_raw, discovery_emb, validation_df_raw, validation_emb = split_discovery_validation(
        tokenized_df, embeddings
    )
    discovery_tokens = token_ids[: len(discovery_df_raw)]
    validation_tokens = token_ids[len(discovery_df_raw) :]

    discovery, center, scale = prepare_features(
        discovery_df_raw,
        discovery_tokens,
        discovery_emb,
        state_window=args.state_window,
        horizon=args.horizon,
    )
    validation, _, _ = prepare_features(
        validation_df_raw,
        validation_tokens,
        validation_emb,
        state_window=args.state_window,
        horizon=args.horizon,
        center=center,
        scale=scale,
    )

    model = fit_v2_model(discovery)
    validation_scored = apply_v2_model(validation, model)
    state_summary = summarize_states(validation_scored)

    discovery.to_csv(output_dir / "discovery_year_15m_v2_tokenized.csv", index=False)
    validation_scored.to_csv(output_dir / "validation_year_15m_v2_states.csv", index=False)
    state_summary.to_csv(output_dir / "validation_15m_v2_state_summary.csv", index=False)
    plot_v2_timeline(validation_scored, output_dir)

    summary = {
        "discovery_rows": int(len(discovery)),
        "validation_rows": int(len(validation_scored)),
        "state_window": int(args.state_window),
        "horizon": int(args.horizon),
        "model": model,
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
    (output_dir / "15m_state_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
