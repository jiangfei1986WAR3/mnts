from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from mnts_min_validation import ensure_output_dir, future_realized_vol, load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MNTS 15m V2 state layer against MA/RSI/MACD on the same out-of-sample task."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--v2-validation-csv", required=True)
    parser.add_argument("--output-dir", default="validation_outputs/classic_compare_run")
    parser.add_argument("--horizon", type=int, default=16)
    return parser.parse_args()


def split_discovery_validation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = len(df) // 2
    return (
        df.iloc[:split_idx].copy().reset_index(drop=True),
        df.iloc[split_idx:].copy().reset_index(drop=True),
    )


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def build_indicator_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["fwd_rv"] = future_realized_vol(close, horizon)
    out["sma20"] = close.rolling(20).mean()
    out["sma60"] = close.rolling(60).mean()
    out["ma_price_gap_20"] = (close / out["sma20"] - 1.0).abs()
    out["ma_gap_20_60"] = (out["sma20"] / out["sma60"] - 1.0).abs()
    out["ma_slope_20"] = out["sma20"].pct_change(4).abs()

    out["rsi14"] = compute_rsi(close, 14)
    out["rsi_extreme"] = ((out["rsi14"] - 50.0).abs() / 50.0).fillna(0.0)
    out["rsi_delta"] = out["rsi14"].diff().abs().fillna(0.0)

    macd_line, signal, hist = compute_macd(close)
    out["macd_line"] = macd_line
    out["macd_signal"] = signal
    out["macd_hist"] = hist
    out["macd_abs"] = out["macd_line"].abs()
    out["macd_hist_abs"] = out["macd_hist"].abs()
    out["macd_hist_delta"] = out["macd_hist"].diff().abs().fillna(0.0)

    return out


def zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    std = 1.0 if abs(std) < 1e-8 or np.isnan(std) else std
    return (series - mean) / std


def fit_score_model(
    discovery: pd.DataFrame,
    feature_names: Iterable[str],
    high_vol_threshold: float,
) -> Dict[str, object]:
    work = discovery.copy()
    work["high_vol_event"] = (work["fwd_rv"] >= high_vol_threshold).astype(int)
    pos = work[work["high_vol_event"] == 1]
    neg = work[work["high_vol_event"] == 0]

    feature_stats: Dict[str, Dict[str, float]] = {}
    weights: Dict[str, float] = {}
    for feature in feature_names:
        mean = float(work[feature].dropna().mean())
        std = float(work[feature].dropna().std(ddof=0))
        if std == 0 or np.isnan(std):
            std = 1.0
        feature_stats[feature] = {"mean": mean, "std": std}
        pos_mean = float(pos[feature].dropna().mean()) if not pos.empty else mean
        neg_mean = float(neg[feature].dropna().mean()) if not neg.empty else mean
        weights[feature] = abs((pos_mean - neg_mean) / std)

    weight_sum = sum(weights.values())
    if weight_sum == 0:
        weights = {feature: 1.0 / len(weights) for feature in weights}
    else:
        weights = {feature: value / weight_sum for feature, value in weights.items()}

    score = compute_score(work, feature_stats, weights)
    return {
        "feature_stats": feature_stats,
        "weights": weights,
        "score_q30": float(score.quantile(0.30)),
        "score_q70": float(score.quantile(0.70)),
    }


def compute_score(
    df: pd.DataFrame,
    feature_stats: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for feature in weights:
        params = feature_stats[feature]
        score += weights[feature] * zscore(df[feature], params["mean"], params["std"])
    return score


def apply_score_model(
    df: pd.DataFrame,
    model: Dict[str, object],
    high_vol_threshold: float,
    prefix: str,
) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_score"] = compute_score(out, model["feature_stats"], model["weights"])
    out[f"{prefix}_state"] = "drift"
    out.loc[out[f"{prefix}_score"] <= float(model["score_q30"]), f"{prefix}_state"] = "cohesion"
    out.loc[out[f"{prefix}_score"] >= float(model["score_q70"]), f"{prefix}_state"] = "fracture"
    out["high_vol_event"] = (out["fwd_rv"] >= high_vol_threshold).astype(int)
    return out


def summarize_states(df: pd.DataFrame, state_col: str, score_col: str, method_name: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for state, group in df.groupby(state_col):
        rows.append(
            {
                "method": method_name,
                "state": state,
                "count": int(len(group)),
                "share": float(len(group) / len(df)),
                "mean_score": float(group[score_col].mean()),
                "mean_fwd_rv": float(group["fwd_rv"].dropna().mean()),
                "high_vol_event_rate": float(group["high_vol_event"].mean()),
            }
        )
    return rows


def summarize_binary(df: pd.DataFrame, state_col: str, method_name: str) -> Dict[str, float]:
    flagged = df[df[state_col] == "fracture"]
    quiet = df[df[state_col] != "fracture"]
    return {
        "method": method_name,
        "flagged_count": int(len(flagged)),
        "quiet_count": int(len(quiet)),
        "flagged_mean_fwd_rv": float(flagged["fwd_rv"].dropna().mean()),
        "quiet_mean_fwd_rv": float(quiet["fwd_rv"].dropna().mean()),
        "flagged_high_vol_rate": float(flagged["high_vol_event"].mean()),
        "quiet_high_vol_rate": float(quiet["high_vol_event"].mean()),
        "rv_gap": float(flagged["fwd_rv"].dropna().mean() - quiet["fwd_rv"].dropna().mean()),
        "event_gap": float(flagged["high_vol_event"].mean() - quiet["high_vol_event"].mean()),
    }


def load_v2_validation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    raw_df = load_ohlcv_csv(args.input_csv)
    indicator_df = build_indicator_frame(raw_df, args.horizon)
    discovery_raw, validation_raw = split_discovery_validation(indicator_df)

    high_vol_threshold = float(discovery_raw["fwd_rv"].dropna().quantile(0.90))

    v2_validation = load_v2_validation(Path(args.v2_validation_csv))
    validation = validation_raw.copy()
    validation["timestamp"] = pd.to_datetime(validation["timestamp"], utc=True, errors="coerce")
    v2_small = v2_validation[["timestamp", "state", "instability_score"]].rename(
        columns={"state": "v2_state", "instability_score": "v2_score"}
    )
    validation = validation.merge(v2_small, on="timestamp", how="left")
    validation["high_vol_event"] = (validation["fwd_rv"] >= high_vol_threshold).astype(int)

    methods = {
        "ma": ["ma_price_gap_20", "ma_gap_20_60", "ma_slope_20"],
        "rsi": ["rsi_extreme", "rsi_delta"],
        "macd": ["macd_abs", "macd_hist_abs", "macd_hist_delta"],
    }

    all_state_rows = summarize_states(validation, "v2_state", "v2_score", "v2")
    all_binary_rows = [summarize_binary(validation, "v2_state", "v2")]
    model_info: Dict[str, object] = {}

    for method_name, features in methods.items():
        model = fit_score_model(discovery_raw, features, high_vol_threshold)
        scored = apply_score_model(validation, model, high_vol_threshold, prefix=method_name)
        all_state_rows.extend(
            summarize_states(scored, f"{method_name}_state", f"{method_name}_score", method_name)
        )
        all_binary_rows.append(summarize_binary(scored, f"{method_name}_state", method_name))
        model_info[method_name] = model

    state_summary = pd.DataFrame(all_state_rows)
    binary_summary = pd.DataFrame(all_binary_rows).sort_values("event_gap", ascending=False)

    judgement = {
        "best_event_gap_method": str(binary_summary.iloc[0]["method"]) if not binary_summary.empty else "",
        "best_rv_gap_method": str(binary_summary.sort_values("rv_gap", ascending=False).iloc[0]["method"])
        if not binary_summary.empty
        else "",
    }

    state_summary.to_csv(output_dir / "state_summary_by_method.csv", index=False)
    binary_summary.to_csv(output_dir / "fracture_binary_compare.csv", index=False)
    summary = {
        "horizon": int(args.horizon),
        "high_vol_threshold": high_vol_threshold,
        "judgement": judgement,
        "model_info": model_info,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
