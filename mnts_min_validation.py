from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]
OPTIONAL_COLUMNS = ["volume", "amount", "quote_asset_volume"]


@dataclass
class ValidationArtifacts:
    frame: pd.DataFrame
    token_ids: np.ndarray
    embeddings: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal MNTS validation pipeline for BTC 15m tokenizer research."
    )
    parser.add_argument("--input-csv", required=True, help="Path to OHLCV CSV file.")
    parser.add_argument(
        "--output-dir",
        default="validation_outputs",
        help="Directory to save tables and figures.",
    )
    parser.add_argument(
        "--mode",
        choices=["pseudo", "kronos"],
        default="pseudo",
        help="Tokenizer backend. Start with pseudo if Kronos is not installed yet.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=24,
        help="Number of pseudo tokens for KMeans mode.",
    )
    parser.add_argument(
        "--lookaheads",
        default="4,8,16",
        help="Comma-separated forward horizons in bars.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=96,
        help="Rolling window size used for token distribution shifts.",
    )
    parser.add_argument(
        "--kronos-repo-dir",
        default="",
        help="Optional local Kronos repository path when mode=kronos.",
    )
    parser.add_argument(
        "--kronos-tokenizer-name",
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="Tokenizer model name used by KronosTokenizer.from_pretrained().",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for Kronos tokenization.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for Kronos mode. Use cpu on lightweight machines.",
    )
    return parser.parse_args()


def ensure_output_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_ohlcv_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    df = pd.read_csv(path)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    use_columns = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[use_columns].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    numeric_columns = [c for c in df.columns if c != "timestamp"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        if "quote_asset_volume" in df.columns:
            df["amount"] = pd.to_numeric(df["quote_asset_volume"], errors="coerce").fillna(0.0)
        else:
            df["amount"] = 0.0

    return df


def build_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    eps = 1e-8

    out["ret_1"] = np.log(df["close"]).diff().fillna(0.0)
    out["range"] = (df["high"] - df["low"]) / (df["close"].replace(0, np.nan) + eps)
    out["body"] = (df["close"] - df["open"]) / (df["open"].replace(0, np.nan) + eps)
    out["upper_wick"] = (
        df["high"] - np.maximum(df["open"], df["close"])
    ) / (df["close"].replace(0, np.nan) + eps)
    out["lower_wick"] = (
        np.minimum(df["open"], df["close"]) - df["low"]
    ) / (df["close"].replace(0, np.nan) + eps)
    out["hl_spread"] = (df["high"] - df["low"]).rolling(8).mean().fillna(method="bfill")
    out["vol_z"] = (
        (df["volume"] - df["volume"].rolling(48).mean())
        / (df["volume"].rolling(48).std().replace(0, np.nan) + eps)
    ).fillna(0.0)
    out["rv_8"] = out["ret_1"].rolling(8).std().fillna(0.0)
    out["rv_32"] = out["ret_1"].rolling(32).std().fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


class BaseTokenizerAdapter:
    def fit_transform(self, df: pd.DataFrame) -> ValidationArtifacts:
        raise NotImplementedError


class PseudoTokenizerAdapter(BaseTokenizerAdapter):
    def __init__(self, num_tokens: int = 24, random_state: int = 42):
        self.num_tokens = num_tokens
        self.random_state = random_state

    def fit_transform(self, df: pd.DataFrame) -> ValidationArtifacts:
        features = build_handcrafted_features(df)
        scaled = standardize(features.values)
        token_ids, _ = simple_kmeans(
            scaled,
            n_clusters=self.num_tokens,
            random_state=self.random_state,
            n_init=8,
            max_iter=60,
        )
        embeddings = scaled.astype(np.float32)

        out = df.copy()
        out["token_id"] = token_ids
        return ValidationArtifacts(frame=out, token_ids=token_ids, embeddings=embeddings)


class KronosTokenizerAdapter(BaseTokenizerAdapter):
    def __init__(
        self,
        repo_dir: str,
        tokenizer_name: str,
        chunk_size: int = 512,
        device: str = "cpu",
        clip: float = 5.0,
    ):
        self.repo_dir = repo_dir
        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.device = device
        self.clip = clip

    def fit_transform(self, df: pd.DataFrame) -> ValidationArtifacts:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for Kronos mode") from exc

        repo_dir = Path(self.repo_dir).expanduser().resolve()
        if not repo_dir.exists():
            raise RuntimeError("Kronos repository path does not exist")

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        try:
            from model.kronos import KronosTokenizer  # type: ignore
        except ImportError:
            from model import KronosTokenizer  # type: ignore

        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.eval()
        tokenizer.to(self.device)

        data = df[["open", "high", "low", "close", "volume", "amount"]].astype(np.float32).values
        token_chunks: List[np.ndarray] = []
        s1_chunks: List[np.ndarray] = []
        s2_chunks: List[np.ndarray] = []
        embed_chunks: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(data), self.chunk_size):
                chunk = data[start : start + self.chunk_size]
                chunk_mean = np.mean(chunk, axis=0, keepdims=True)
                chunk_std = np.std(chunk, axis=0, keepdims=True)
                chunk_std = np.where(chunk_std == 0, 1.0, chunk_std)
                chunk_norm = (chunk - chunk_mean) / (chunk_std + 1e-5)
                chunk_norm = np.clip(chunk_norm, -self.clip, self.clip)

                x = torch.tensor(chunk_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

                z = tokenizer.embed(x)
                for layer in tokenizer.encoder:
                    z = layer(z)
                pre_quant = tokenizer.quant_embed(z)
                s1_ids, s2_ids = tokenizer.encode(x, half=True)
                combined_ids = s1_ids * (2 ** tokenizer.s2_bits) + s2_ids

                token_chunks.append(combined_ids.detach().cpu().numpy().reshape(-1))
                s1_chunks.append(s1_ids.detach().cpu().numpy().reshape(-1))
                s2_chunks.append(s2_ids.detach().cpu().numpy().reshape(-1))
                embed_chunks.append(pre_quant.detach().cpu().numpy().reshape(len(chunk), -1))

        token_ids = np.concatenate(token_chunks, axis=0)
        s1_ids = np.concatenate(s1_chunks, axis=0)
        s2_ids = np.concatenate(s2_chunks, axis=0)
        embeddings = np.concatenate(embed_chunks, axis=0).astype(np.float32)

        out = df.copy()
        out["token_s1"] = s1_ids
        out["token_s2"] = s2_ids
        out["token_id"] = token_ids
        return ValidationArtifacts(frame=out, token_ids=token_ids, embeddings=embeddings)


def get_adapter(args: argparse.Namespace) -> BaseTokenizerAdapter:
    if args.mode == "pseudo":
        return PseudoTokenizerAdapter(num_tokens=args.num_tokens)
    return KronosTokenizerAdapter(
        repo_dir=args.kronos_repo_dir,
        tokenizer_name=args.kronos_tokenizer_name,
        chunk_size=args.chunk_size,
        device=args.device,
    )


def forward_returns(series: pd.Series, horizon: int) -> pd.Series:
    return np.log(series.shift(-horizon) / series)


def future_realized_vol(close: pd.Series, horizon: int) -> pd.Series:
    rets = np.log(close).diff()
    return rets.rolling(horizon).std().shift(-horizon + 1)


def summarize_by_token(
    df: pd.DataFrame, lookaheads: Sequence[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records_ret: List[Dict[str, float]] = []
    records_vol: List[Dict[str, float]] = []

    for horizon in lookaheads:
        df[f"fwd_ret_{horizon}"] = forward_returns(df["close"], horizon)
        df[f"fwd_rv_{horizon}"] = future_realized_vol(df["close"], horizon)

    grouped = df.groupby("token_id")
    for token_id, group in grouped:
        for horizon in lookaheads:
            ret_col = f"fwd_ret_{horizon}"
            vol_col = f"fwd_rv_{horizon}"
            ret_vals = group[ret_col].dropna()
            vol_vals = group[vol_col].dropna()
            if len(ret_vals) > 0:
                records_ret.append(
                    {
                        "token_id": int(token_id),
                        "horizon": int(horizon),
                        "count": int(len(ret_vals)),
                        "mean": float(ret_vals.mean()),
                        "median": float(ret_vals.median()),
                        "std": float(ret_vals.std(ddof=0)),
                        "q10": float(ret_vals.quantile(0.10)),
                        "q90": float(ret_vals.quantile(0.90)),
                    }
                )
            if len(vol_vals) > 0:
                records_vol.append(
                    {
                        "token_id": int(token_id),
                        "horizon": int(horizon),
                        "count": int(len(vol_vals)),
                        "mean": float(vol_vals.mean()),
                        "median": float(vol_vals.median()),
                        "std": float(vol_vals.std(ddof=0)),
                        "q10": float(vol_vals.quantile(0.10)),
                        "q90": float(vol_vals.quantile(0.90)),
                    }
                )

    return pd.DataFrame(records_ret), pd.DataFrame(records_vol)


def build_transition_matrix(token_ids: np.ndarray) -> pd.DataFrame:
    current_ids = token_ids[:-1]
    next_ids = token_ids[1:]
    token_space = np.unique(token_ids)
    matrix = pd.DataFrame(0, index=token_space, columns=token_space, dtype=float)

    for cur, nxt in zip(current_ids, next_ids):
        matrix.loc[cur, nxt] += 1.0

    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    matrix = matrix.div(row_sums, axis=0).fillna(0.0)
    matrix.index.name = "current_token"
    return matrix


def l1_distribution_shift(token_ids: np.ndarray, window: int) -> pd.Series:
    values: List[float] = [math.nan] * len(token_ids)
    token_space = np.unique(token_ids)

    for end in range(window * 2, len(token_ids) + 1):
        left = token_ids[end - 2 * window : end - window]
        right = token_ids[end - window : end]
        left_counts = pd.Series(left).value_counts(normalize=True).reindex(token_space, fill_value=0.0)
        right_counts = pd.Series(right).value_counts(normalize=True).reindex(token_space, fill_value=0.0)
        values[end - 1] = float(np.abs(left_counts.values - right_counts.values).sum())

    return pd.Series(values)


def reduce_embeddings(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = embeddings.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:n_components].T
    return x @ components


def standardize(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std


def simple_kmeans(
    x: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 8,
    max_iter: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    best_labels: Optional[np.ndarray] = None
    best_centers: Optional[np.ndarray] = None
    best_inertia = float("inf")

    n_samples = x.shape[0]
    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot exceed number of samples")

    for _ in range(n_init):
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = x[init_idx].copy()

        for _ in range(max_iter):
            distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    new_centers[i] = x[mask].mean(axis=0)
                else:
                    new_centers[i] = x[rng.integers(0, n_samples)]

            if np.allclose(new_centers, centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers

        final_distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(final_distances, axis=1)
        inertia = float(final_distances[np.arange(n_samples), labels].sum())

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    assert best_labels is not None
    assert best_centers is not None
    return best_labels, best_centers


def plot_token_usage(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["token_id"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Token Usage Distribution")
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "token_usage_distribution.png", dpi=160)
    plt.close(fig)


def plot_shift_curve(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["distribution_shift_l1"], linewidth=1.0)
    ax.set_title("Rolling Token Distribution Shift (L1)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("L1 shift")
    fig.tight_layout()
    fig.savefig(output_dir / "rolling_distribution_shift.png", dpi=160)
    plt.close(fig)


def plot_embedding_projection(df: pd.DataFrame, reduced: np.ndarray, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=df["token_id"].astype(int),
        s=6,
        cmap="tab20",
        alpha=0.65,
    )
    ax.set_title("Embedding Projection (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Token ID")
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_projection_pca.png", dpi=160)
    plt.close(fig)


def plot_risk_warning_layer(warning_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(
        warning_df["timestamp"],
        warning_df["rolling_high_risk_ratio"],
        color="#dc2626",
        linewidth=1.2,
        label="Rolling High-Risk Token Ratio",
    )
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("High-risk ratio", color="#dc2626")
    ax1.tick_params(axis="y", labelcolor="#dc2626")

    ax2 = ax1.twinx()
    ax2.plot(
        warning_df["timestamp"],
        warning_df["distribution_shift_l1"],
        color="#2563eb",
        linewidth=1.0,
        alpha=0.75,
        label="Distribution Shift L1",
    )
    ax2.set_ylabel("Distribution shift", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")

    ax1.set_title("Risk Warning Layer")
    fig.tight_layout()
    fig.savefig(output_dir / "risk_warning_layer.png", dpi=160)
    plt.close(fig)


def derive_interpretation_tables(
    df: pd.DataFrame,
    return_summary: pd.DataFrame,
    vol_summary: pd.DataFrame,
    target_horizon: int,
    min_count: int = 30,
    top_n: int = 8,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    token_counts = (
        df["token_id"].value_counts().rename_axis("token_id").reset_index(name="occurrences")
    )
    eligible_returns = return_summary[
        (return_summary["horizon"] == target_horizon) & (return_summary["count"] >= min_count)
    ].copy()
    eligible_vol = vol_summary[
        (vol_summary["horizon"] == target_horizon) & (vol_summary["count"] >= min_count)
    ].copy()

    if not eligible_returns.empty:
        overall_return_mean = float(eligible_returns["mean"].mean())
        strong_tokens = eligible_returns.sort_values(["mean", "count"], ascending=[False, False]).head(top_n)
        weak_tokens = eligible_returns.sort_values(["mean", "count"], ascending=[True, False]).head(top_n)
    else:
        overall_return_mean = 0.0
        strong_tokens = eligible_returns
        weak_tokens = eligible_returns

    if not eligible_vol.empty:
        overall_vol_mean = float(eligible_vol["mean"].mean())
        risk_tokens = eligible_vol.sort_values(["mean", "count"], ascending=[False, False]).head(top_n)
        calm_tokens = eligible_vol.sort_values(["mean", "count"], ascending=[True, False]).head(top_n)
    else:
        overall_vol_mean = 0.0
        risk_tokens = eligible_vol
        calm_tokens = eligible_vol

    top_counts = token_counts.head(top_n).copy()
    if not top_counts.empty:
        top_counts["share"] = top_counts["occurrences"] / len(df)

    labeled_frames: List[pd.DataFrame] = []
    for label, source in [
        ("strong_tokens", strong_tokens),
        ("weak_tokens", weak_tokens),
        ("risk_tokens", risk_tokens),
        ("calm_tokens", calm_tokens),
    ]:
        if source.empty:
            continue
        tmp = source.copy()
        tmp.insert(0, "category", label)
        labeled_frames.append(tmp)

    if not top_counts.empty:
        tmp = top_counts.copy()
        tmp.insert(0, "category", "frequent_tokens")
        labeled_frames.append(tmp)

    combined = pd.concat(labeled_frames, ignore_index=True) if labeled_frames else pd.DataFrame()
    interpretation = {
        "target_horizon": int(target_horizon),
        "min_count": int(min_count),
        "eligible_return_tokens": int(len(eligible_returns)),
        "eligible_risk_tokens": int(len(eligible_vol)),
        "overall_return_mean": overall_return_mean,
        "overall_vol_mean": overall_vol_mean,
        "judgement": {
            "token_distribution_ok": bool(df["token_id"].nunique() > 20),
            "has_return_separation": bool(
                not eligible_returns.empty
                and (eligible_returns["mean"].max() - eligible_returns["mean"].min()) > 0.004
            ),
            "has_risk_separation": bool(
                not eligible_vol.empty
                and (eligible_vol["mean"].max() - eligible_vol["mean"].min()) > 0.0008
            ),
        },
    }
    return interpretation, combined


def derive_risk_warning_layer(
    df: pd.DataFrame,
    vol_summary: pd.DataFrame,
    target_horizon: int,
    min_count: int = 30,
    top_n: int = 8,
    rolling_window: int = 96,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    eligible_vol = vol_summary[
        (vol_summary["horizon"] == target_horizon) & (vol_summary["count"] >= min_count)
    ].copy()
    risk_tokens = (
        eligible_vol.sort_values(["mean", "count"], ascending=[False, False])
        .head(top_n)["token_id"]
        .astype(int)
        .tolist()
    )

    warning_df = df[["timestamp", "token_id", "distribution_shift_l1"]].copy()
    warning_df["high_risk_token_flag"] = warning_df["token_id"].isin(risk_tokens).astype(int)
    warning_df["rolling_high_risk_ratio"] = (
        warning_df["high_risk_token_flag"].rolling(rolling_window, min_periods=1).mean()
    )

    shift = warning_df["distribution_shift_l1"].copy()
    shift_valid = shift.dropna()
    if shift_valid.empty:
        shift_q70 = 0.0
        shift_q90 = 0.0
    else:
        shift_q70 = float(shift_valid.quantile(0.70))
        shift_q90 = float(shift_valid.quantile(0.90))

    ratio = warning_df["rolling_high_risk_ratio"]
    ratio_q70 = float(ratio.quantile(0.70))
    ratio_q90 = float(ratio.quantile(0.90))

    warning_df["shift_flag_mid"] = (warning_df["distribution_shift_l1"] >= shift_q70).astype(int)
    warning_df["shift_flag_high"] = (warning_df["distribution_shift_l1"] >= shift_q90).astype(int)
    warning_df["risk_flag_mid"] = (warning_df["rolling_high_risk_ratio"] >= ratio_q70).astype(int)
    warning_df["risk_flag_high"] = (warning_df["rolling_high_risk_ratio"] >= ratio_q90).astype(int)

    warning_df["warning_level"] = "green"
    orange_mask = (warning_df["shift_flag_mid"] + warning_df["risk_flag_mid"]) >= 1
    red_mask = (warning_df["shift_flag_high"] + warning_df["risk_flag_high"]) >= 2
    warning_df.loc[orange_mask, "warning_level"] = "orange"
    warning_df.loc[red_mask, "warning_level"] = "red"

    latest = warning_df.iloc[-1].to_dict() if len(warning_df) else {}
    summary = {
        "target_horizon": int(target_horizon),
        "rolling_window": int(rolling_window),
        "high_risk_tokens": risk_tokens,
        "shift_q70": shift_q70,
        "shift_q90": shift_q90,
        "ratio_q70": ratio_q70,
        "ratio_q90": ratio_q90,
        "latest_warning_level": latest.get("warning_level", "green"),
        "latest_distribution_shift_l1": float(latest.get("distribution_shift_l1", 0.0) or 0.0),
        "latest_high_risk_ratio": float(latest.get("rolling_high_risk_ratio", 0.0) or 0.0),
    }
    return warning_df, summary


def save_summary(
    df: pd.DataFrame,
    return_summary: pd.DataFrame,
    vol_summary: pd.DataFrame,
    transition: pd.DataFrame,
    reduced: np.ndarray,
    output_dir: Path,
) -> None:
    df.to_csv(output_dir / "tokenized_dataset.csv", index=False)
    return_summary.to_csv(output_dir / "token_future_return_summary.csv", index=False)
    vol_summary.to_csv(output_dir / "token_future_vol_summary.csv", index=False)
    transition.to_csv(output_dir / "token_transition_matrix.csv")

    reduced_df = pd.DataFrame(reduced, columns=["pc1", "pc2"])
    reduced_df.insert(0, "token_id", df["token_id"].values)
    reduced_df.insert(0, "timestamp", df["timestamp"].astype(str).values)
    reduced_df.to_csv(output_dir / "embedding_projection_pca.csv", index=False)

    summary = {
        "num_rows": int(len(df)),
        "num_unique_tokens": int(df["token_id"].nunique()),
        "token_top5_share": (
            df["token_id"].value_counts(normalize=True).head(5).sum().item()
            if len(df) > 0
            else 0.0
        ),
        "mean_distribution_shift_l1": float(df["distribution_shift_l1"].dropna().mean()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    target_horizon = int(return_summary["horizon"].max()) if not return_summary.empty else 0
    interpretation, interpretation_table = derive_interpretation_tables(
        df=df,
        return_summary=return_summary,
        vol_summary=vol_summary,
        target_horizon=target_horizon,
    )
    (output_dir / "interpretation.json").write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    interpretation_table.to_csv(output_dir / "interpretation_table.csv", index=False)

    warning_df, warning_summary = derive_risk_warning_layer(
        df=df,
        vol_summary=vol_summary,
        target_horizon=target_horizon,
        rolling_window=96,
    )
    warning_df.to_csv(output_dir / "risk_warning_layer.csv", index=False)
    (output_dir / "risk_warning_summary.json").write_text(
        json.dumps(warning_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_risk_warning_layer(warning_df, output_dir)


def main() -> None:
    args = parse_args()
    lookaheads = [int(x.strip()) for x in args.lookaheads.split(",") if x.strip()]
    output_dir = ensure_output_dir(args.output_dir)

    df = load_ohlcv_csv(args.input_csv)
    adapter = get_adapter(args)
    artifacts = adapter.fit_transform(df)

    result_df = artifacts.frame.copy()
    result_df["distribution_shift_l1"] = l1_distribution_shift(
        artifacts.token_ids, window=args.rolling_window
    )

    return_summary, vol_summary = summarize_by_token(result_df, lookaheads)
    transition = build_transition_matrix(artifacts.token_ids)
    reduced = reduce_embeddings(artifacts.embeddings, n_components=2)

    save_summary(
        result_df,
        return_summary,
        vol_summary,
        transition,
        reduced,
        output_dir,
    )
    plot_token_usage(result_df, output_dir)
    plot_shift_curve(result_df, output_dir)
    plot_embedding_projection(result_df, reduced, output_dir)

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
