#%% Regime analysis helpers using ruptures on mill data
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import ruptures as rpt
except Exception as e:  # pragma: no cover
    rpt = None
    _RUPTURES_IMPORT_ERROR = e
else:
    _RUPTURES_IMPORT_ERROR = None

# Matplotlib defaults
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Columns typically excluded from change-point detection (targets/lab)
EXCLUDE_DEFAULT = [
    "PSI200", "Class_15", "Class_12", "Grano", "Daiki", "Shisti"
]


@dataclass
class DetectionResult:
    regime_labels: np.ndarray  # length N labels 0..K-1
    change_points: List[int]   # indices without the last endpoint
    change_times: List[pd.Timestamp]  # timestamps for change_points
    features: List[str]


def get_features(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Return numeric feature columns, excluding known target/lab columns.
    """
    if exclude is None:
        exclude = EXCLUDE_DEFAULT
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    return features


def _check_ruptures():
    if rpt is None:
        raise ImportError(
            "ruptures is not installed. Please install with 'pip install ruptures'.\n"
            f"Original import error: {_RUPTURES_IMPORT_ERROR}"
        )


def detect_regimes(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "pelt",
    model: str = "rbf",
    min_size: int = 60,
    jump: int = 5,
    pen: Optional[float] = 10.0,
    n_bkps: Optional[int] = None,
) -> DetectionResult:
    """Detect regimes using ruptures and return labels and change-points.

    Notes:
    - If n_bkps is provided, it takes precedence over 'pen'.
    - change_points don't include the final endpoint (len(X)).
    """
    _check_ruptures()

    if features is None:
        features = get_features(df)

    if len(features) == 0:
        raise ValueError("No numeric features found for change point detection.")

    X = df[features].copy()
    # Fill missing values
    X = X.fillna(method="ffill").fillna(method="bfill")
    X_values = X.values

    if method == "pelt":
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    elif method == "binseg":
        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump)
    elif method == "window":
        # width ~ 3*min_size is a reasonable starting point
        algo = rpt.Window(width=max(min_size * 3, min_size + 1), model=model, min_size=min_size, jump=jump)
    else:
        raise ValueError("method must be one of: 'pelt', 'binseg', 'window'")

    def _fit_predict(_algo):
        _algo.fit(X_values)
        if n_bkps is not None:
            return _algo.predict(n_bkps=n_bkps)
        else:
            _pen = 10.0 if pen is None else pen
            return _algo.predict(pen=_pen)

    try:
        cpts = _fit_predict(algo)
    except Exception as e:
        # RBF cost builds an NxN Gram matrix (O(N^2) memory). Fallback to l2 on memory issues.
        mem_err_types = (MemoryError,)
        try:
            from numpy.core._exceptions import _ArrayMemoryError as ArrayMemoryError  # type: ignore
            mem_err_types = (MemoryError, ArrayMemoryError)
        except Exception:
            pass

        if model == "rbf" and isinstance(e, mem_err_types) or (model == "rbf" and "ArrayMemoryError" in str(type(e))):
            warnings.warn("ruptures RBF model ran out of memory; falling back to model='l2'.")
            # Rebuild algo with l2 model and retry
            if method == "pelt":
                algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump)
            elif method == "binseg":
                algo = rpt.Binseg(model="l2", min_size=min_size, jump=jump)
            else:
                algo = rpt.Window(width=max(min_size * 3, min_size + 1), model="l2", min_size=min_size, jump=jump)
            cpts = _fit_predict(algo)
        elif model == "rbf":
            warnings.warn(f"ruptures RBF model failed with error: {e}. Falling back to model='l2'.")
            if method == "pelt":
                algo = rpt.Pelt(model="l2", min_size=min_size, jump=jump)
            elif method == "binseg":
                algo = rpt.Binseg(model="l2", min_size=min_size, jump=jump)
            else:
                algo = rpt.Window(width=max(min_size * 3, min_size + 1), model="l2", min_size=min_size, jump=jump)
            cpts = _fit_predict(algo)
        else:
            raise

    # Ruptures returns cpts including the final index len(X)
    cpts_no_end = list(cpts[:-1])

    labels = np.zeros(len(X_values), dtype=int)
    start = 0
    for i, cp in enumerate(cpts_no_end):
        labels[start:cp] = i
        start = cp
    # Assign the tail (last segment) to a new label index
    if start < len(X_values):
        labels[start:] = len(cpts_no_end)

    change_times = [df.index[i - 1] if i > 0 else df.index[0] for i in cpts_no_end]

    return DetectionResult(
        regime_labels=labels,
        change_points=cpts_no_end,
        change_times=change_times,
        features=features,
    )


def summarize_segments(
    df: pd.DataFrame,
    labels: np.ndarray,
    target: str = "PSI200",
    extra_stats: bool = True,
) -> pd.DataFrame:
    """Build a per-segment summary with timings and stats (incl. target if present)."""
    if len(df) != len(labels):
        raise ValueError("labels length must match df length")

    seg_ids = pd.Series(labels, index=df.index, name="regime")
    groups = df.join(seg_ids).groupby("regime", sort=True, observed=True)

    rows = []
    for seg_id, g in groups:
        start_time = g.index.min()
        end_time = g.index.max()
        n = len(g)
        duration_min = float((end_time - start_time).total_seconds() / 60.0) if n > 1 else 0.0
        row: Dict[str, float] = {
            "regime": int(seg_id),
            "start_time": start_time,
            "end_time": end_time,
            "n_samples": int(n),
            "duration_min": duration_min,
        }
        # Add target stats if present
        if target in g.columns:
            row.update({
                f"{target}_mean": float(g[target].mean()),
                f"{target}_std": float(g[target].std(ddof=0)),
            })
        if extra_stats:
            # Selected core features if present
            core = [c for c in ["Ore", "WaterMill", "WaterZumpf", "MotorAmp", "DensityHC", "PressureHC"] if c in g.columns]
            for c in core:
                row[f"{c}_mean"] = float(g[c].mean())
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)

    # Compute deltas on target between consecutive segments
    if target in df.columns and f"{target}_mean" in summary.columns:
        summary["psi200_delta_vs_prev"] = summary[f"{target}_mean"].diff()
    return summary


def visualize_regimes(
    df: pd.DataFrame,
    labels: np.ndarray,
    change_points: List[int],
    features: Optional[List[str]] = None,
    max_plots: int = 4,
    title: str = "Detected regimes",
    save_path: Optional[Path] = None,
):
    """Time-indexed visualization of selected features and the target."""
    if features is None:
        # Prioritize PSI200 + a few core features if present
        preferred = ["PSI200", "Ore", "WaterMill", "MotorAmp"]
        features = [c for c in preferred if c in df.columns]
        if not features:
            features = df.select_dtypes(include=[np.number]).columns.tolist()[:max_plots]
    features = features[:max_plots]

    time_index = df.index
    fig, axes = plt.subplots(len(features), 1, figsize=(14, 3 * len(features)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    regime_colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))

    for ax, col in zip(axes, features):
        # Plot full series
        ax.plot(time_index, df[col].values, color="#1f77b4", linewidth=0.8)
        # Overlay regimes as background colors
        start = 0
        for r, cp in enumerate(change_points + [len(df)]):
            end = cp
            ax.axvspan(time_index[start], time_index[end - 1], color=regime_colors[r % len(regime_colors)], alpha=0.07)
            start = cp
        # Vertical lines for boundaries
        for cp in change_points:
            ax.axvline(time_index[cp - 1], color="red", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def visualize_regimes_ru(
    df: pd.DataFrame,
    labels: np.ndarray,
    change_points: List[int],
    features: Optional[List[str]] = None,
    max_plots: int = 3,
    title: str = "Ruptures-style visualization",
    save_path: Optional[Path] = None,
    alt_colors: Tuple[str, str] = ("#cfe8ff", "#ffdbe1"),  # light blue / light pink
    show_means: bool = True,
):
    """Ruptures GitHub-style plots with alternating background colors and dashed lines.

    - Alternating shaded regions for each regime (two colors)
    - Dashed black vertical lines at change-points
    - Optional piecewise-constant overlay of per-segment means for each feature
    """
    if features is None:
        preferred = ["PSI200", "Ore", "WaterMill"]
        features = [c for c in preferred if c in df.columns]
        if not features:
            features = df.select_dtypes(include=[np.number]).columns.tolist()[:max_plots]
    features = features[:max_plots]

    time_index = df.index
    fig, axes = plt.subplots(len(features), 1, figsize=(16, 3.4 * len(features)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Build segment boundaries including end
    boundaries = change_points + [len(df)]
    starts = [0] + change_points

    for ax, col in zip(axes, features):
        y = df[col].values
        # main signal
        ax.plot(time_index, y, color="#1f77b4", linewidth=0.9)

        # alternating backgrounds
        for i, (s, e) in enumerate(zip(starts, boundaries)):
            if e <= s:
                continue
            color = alt_colors[i % 2]
            ax.axvspan(time_index[s], time_index[e - 1], color=color, alpha=0.35)

        # dashed vertical boundaries
        for cp in change_points:
            ax.axvline(time_index[cp - 1], color="k", linestyle=(0, (5, 5)), linewidth=1.0, alpha=0.9)

        # piecewise means
        if show_means:
            for s, e in zip(starts, boundaries):
                if e <= s:
                    continue
                mu = float(np.nanmean(y[s:e]))
                ax.plot([time_index[s], time_index[e - 1]], [mu, mu], color="#ff7f0e", linewidth=1.6)

        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def _segment_r2_for_target(y: np.ndarray, labels: np.ndarray) -> float:
    """Return R^2 of predicting y using segment means (ANOVA-style)."""
    if len(y) == 0:
        return np.nan
    y = np.asarray(y)
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    if sst == 0:
        return 0.0
    # SSE within segments
    sse = 0.0
    for seg in np.unique(labels):
        mask = labels == seg
        if not np.any(mask):
            continue
        y_seg = y[mask]
        mu = float(np.mean(y_seg))
        sse += float(np.sum((y_seg - mu) ** 2))
    r2 = max(0.0, 1.0 - sse / sst)
    return r2


def tune_penalty(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    penalties: Optional[List[float]] = None,
    method: str = "pelt",
    model: str = "rbf",
    min_size: int = 60,
    jump: int = 5,
    target: str = "PSI200",
) -> pd.DataFrame:
    """Sweep penalty values and report segmentation characteristics.

    Returns a DataFrame with columns: penalty, n_regimes, avg_seg_len, r2_target (if target exists)
    """
    _check_ruptures()

    if penalties is None:
        penalties = [1, 3, 5, 10, 20, 30, 50, 80, 120, 200]
    if features is None:
        features = get_features(df)

    X = df[features].fillna(method="ffill").fillna(method="bfill")
    y = df[target].values if target in df.columns else None

    rows = []
    for pen in penalties:
        try:
            res = detect_regimes(
                df=df,
                features=features,
                method=method,
                model=model,
                min_size=min_size,
                jump=jump,
                pen=pen,
                n_bkps=None,
            )
            n_reg = int(np.max(res.regime_labels) + 1)
            avg_len = len(df) / max(n_reg, 1)
            r2 = np.nan
            if y is not None and len(df) > 0:
                r2 = _segment_r2_for_target(y, res.regime_labels)

            rows.append({
                "penalty": float(pen),
                "n_regimes": n_reg,
                "avg_seg_len": float(avg_len),
                "r2_target": float(r2) if not np.isnan(r2) else np.nan,
            })
        except Exception as e:
            warnings.warn(f"Penalty {pen} failed during tuning: {e}")
            rows.append({
                "penalty": float(pen),
                "n_regimes": np.nan,
                "avg_seg_len": np.nan,
                "r2_target": np.nan,
            })

    out = pd.DataFrame(rows)
    return out


def pick_penalty(results: pd.DataFrame, min_regimes: int = 4, max_regimes: int = 20, min_avg_len: int = 90) -> float:
    """Pick a reasonable penalty from tuning results.

    Strategy:
      - Keep candidates with n_regimes in [min_regimes, max_regimes] and avg_seg_len >= min_avg_len
      - Among them, prefer the one with highest r2_target; if NaN, choose median n_regimes.
    """
    if results.empty:
        raise ValueError("Empty tuning results")

    cand = results.copy()
    # drop failed rows
    cand = cand.dropna(subset=["n_regimes", "avg_seg_len"])
    cand = cand[(cand["n_regimes"] >= min_regimes) & (cand["n_regimes"] <= max_regimes) & (cand["avg_seg_len"] >= min_avg_len)]
    if cand.empty:
        # Fall back to mid penalty
        return float(np.median(results["penalty"]))

    if "r2_target" in cand and cand["r2_target"].notna().any():
        best = cand.sort_values(["r2_target", "n_regimes"], ascending=[False, True]).iloc[0]
    else:
        # choose near-median n_regimes
        target_n = int(np.median(cand["n_regimes"]))
        best = cand.iloc[(cand["n_regimes"] - target_n).abs().argsort().iloc[0]]
    return float(best["penalty"]) 


def per_regime_correlations(
    df: pd.DataFrame,
    labels: np.ndarray,
    target: str = "PSI200",
    features: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlations between the target and features inside each regime.

    Returns a DataFrame with one row per regime, columns per feature correlation,
    plus n_samples. If target or features are missing, returns empty DataFrame.
    """
    if len(df) != len(labels):
        raise ValueError("labels length must match df length")

    if target not in df.columns:
        return pd.DataFrame()

    if features is None:
        # Use numeric features excluding lab/targets
        features = get_features(df)
    else:
        features = [c for c in features if c in df.columns]

    if not features:
        return pd.DataFrame()

    seg_ids = pd.Series(labels, index=df.index, name="regime")
    groups = df.join(seg_ids).groupby("regime", sort=True, observed=True)

    rows: List[Dict[str, float]] = []
    for seg_id, g in groups:
        row: Dict[str, float] = {"regime": int(seg_id), "n_samples": int(len(g))}
        # Compute correlation for each feature with the target inside this segment
        for c in features:
            try:
                if method == "spearman":
                    corr = float(g[[target, c]].corr(method="spearman").iloc[0, 1])
                else:
                    corr = float(g[[target, c]].corr(method="pearson").iloc[0, 1])
            except Exception:
                corr = np.nan
            row[f"corr_{c}"] = corr
        rows.append(row)

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
