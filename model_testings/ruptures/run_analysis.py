#%% Run ruptures-based regime analysis on mill data
import sys
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ruptures_df import load_data
from regime_analysis import (
    tune_penalty,
    pick_penalty,
    detect_regimes,
    summarize_segments,
    visualize_regimes,
    visualize_regimes_ru,
    per_regime_correlations,
    get_features,
)

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ruptures-based regime analysis")
    # Resampling and features
    p.add_argument("--resample", default="5T", help="Resample rule (e.g., 5T) or empty for none")
    p.add_argument(
        "--features",
        default="Ore,WaterMill,WaterZumpf,MotorAmp,DensityHC",
        help="Comma-separated feature subset to use; if some are missing, they are skipped; leave empty to auto-select",
    )
    # Ruptures config
    p.add_argument("--method", default="pelt", choices=["pelt", "binseg", "window"], help="Detection method")
    p.add_argument("--model", default="l2", choices=["l2", "rbf"], help="Cost model")
    p.add_argument("--min-size", type=int, default=120, help="Minimum segment length (samples)")
    p.add_argument("--min-size-min", type=int, default=None, help="Minimum segment length in minutes (overrides --min-size if set)")
    p.add_argument("--jump", type=int, default=10, help="Sub-sampling step for speed")
    # Penalty sweep
    p.add_argument(
        "--penalties",
        default="30,50,80,120,200,300,500,800",
        help="Comma-separated penalty values to sweep",
    )
    # Picker constraints
    p.add_argument("--pick-min-regimes", type=int, default=5)
    p.add_argument("--pick-max-regimes", type=int, default=120)
    p.add_argument("--pick-min-avg-len", type=int, default=180, help="Min average segment length in minutes")
    # Directly specify number of change points (overrides penalty during detection)
    p.add_argument("--n-bkps", type=int, default=None, help="If set, detect exactly this many change points")
    # Convenience: desired number of regimes (segments). Sets n_bkps=K-1.
    p.add_argument("--target-regimes", type=int, default=None, help="If set, detect approximately this many regimes (uses n_bkps=K-1)")
    # Correlations
    p.add_argument("--corr-method", default="pearson", choices=["pearson", "spearman"], help="Correlation type")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    print("=== Ruptures Regime Analysis ===")
    # 1) Load cleaned, filtered time series from ruptures_df.py
    resample_rule = None if (args.resample or "").strip() == "" else args.resample
    df = load_data(resample=resample_rule)
    print(f"Loaded data: shape={df.shape}, time span: {df.index.min()} -> {df.index.max()}")

    # Minutes per sample from resample rule (used for conversions)
    try:
        minutes_per_sample = 1.0 if resample_rule is None else (pd.to_timedelta(resample_rule).total_seconds() / 60.0)
        if minutes_per_sample <= 0:
            minutes_per_sample = 1.0
    except Exception:
        minutes_per_sample = 1.0

    # 2) Choose features: prefer manual subset if available else auto
    feat_list = [s.strip() for s in (args.features or "").split(",") if s.strip()]
    manual = [c for c in feat_list if c in df.columns]
    features = manual if len(manual) >= 2 else get_features(df)
    print(f"Using {len(features)} features for change-point detection: {features[:8]}{('...' if len(features) > 8 else '')}")

    # 3) Penalty sweep to understand segmentation behavior
    method = args.method
    model = args.model
    min_size = args.min_size
    # If user provided minutes-based minimum, convert to samples and enforce
    if args.min_size_min is not None:
        min_size_conv = max(1, int(np.ceil(float(args.min_size_min) / minutes_per_sample)))
        if min_size_conv > min_size:
            print(f"Enforcing min_size from minutes: >= {min_size_conv} samples (~{min_size_conv*minutes_per_sample:.0f} min)")
        min_size = max(min_size, min_size_conv)
    jump = args.jump

    print("\nTuning penalties...")
    pen_values = [float(x) for x in (args.penalties or "").split(",") if x.strip()]
    tuning = tune_penalty(
        df=df,
        features=features,
        penalties=pen_values,
        method=method,
        model=model,
        min_size=min_size,
        jump=jump,
        target="PSI200",
    )
    tuning.to_csv(OUT_DIR / "penalty_tuning.csv", index=False)
    print(tuning)

    # 4) Choose a reasonable penalty (convert min_avg_len minutes -> samples)
    try:
        minutes_per_sample = 1.0 if resample_rule is None else (pd.to_timedelta(resample_rule).total_seconds() / 60.0)
        if minutes_per_sample <= 0:
            minutes_per_sample = 1.0
    except Exception:
        minutes_per_sample = 1.0
    min_avg_len_samples = max(1, int(np.ceil(float(args.pick_min_avg_len) / minutes_per_sample)))

    chosen_pen = pick_penalty(
        tuning,
        min_regimes=args.pick_min_regimes,
        max_regimes=args.pick_max_regimes,
        min_avg_len=min_avg_len_samples,
    )
    print(f"\nChosen penalty: {chosen_pen} (picker min_avg_len >= {min_avg_len_samples} samples ~ {min_avg_len_samples*minutes_per_sample:.0f} min)")

    # 5) Detect regimes with chosen penalty
    # Determine n_bkps based on explicit flags
    nbkps = args.n_bkps
    if args.target_regimes is not None:
        nbkps = max(0, int(args.target_regimes) - 1)
        print(f"Using target_regimes={args.target_regimes} -> n_bkps={nbkps} (overrides penalty during detection)")

    det = detect_regimes(
        df=df,
        features=features,
        method=method,
        model=model,
        min_size=min_size,
        jump=jump,
        pen=chosen_pen,
        n_bkps=nbkps,
    )

    n_reg = int(np.max(det.regime_labels) + 1)
    avg_len = len(df) / max(n_reg, 1)
    print(f"Detected {n_reg} regimes (avg length ~ {avg_len:.1f} samples ~ {avg_len*minutes_per_sample:.0f} min); change-points (indices): {det.change_points[:20]}{('...' if len(det.change_points) > 20 else '')}")

    # 6) Summarize segments (including PSI200 stats if present)
    summary = summarize_segments(df, det.regime_labels, target="PSI200", extra_stats=True)
    summary.to_csv(OUT_DIR / "regime_summary.csv", index=False)

    # Also save labels aligned with timestamps
    labels_df = pd.DataFrame({
        "TimeStamp": df.index,
        "regime": det.regime_labels,
    })
    labels_df.set_index("TimeStamp").to_csv(OUT_DIR / "regime_labels.csv")

    # 7) Visualize selected features with regime overlays
    fig = visualize_regimes(
        df=df,
        labels=det.regime_labels,
        change_points=det.change_points,
        features=None,  # auto-pick PSI200 + key features if present
        max_plots=4,
        title=f"Regimes ({method}/{model}, pen={chosen_pen}, min_size={min_size}, jump={jump})",
        save_path=OUT_DIR / "regimes_plot.png",
    )

    # 7b) Ruptures GitHub-style visualization with alternating shaded regimes and dashed lines
    fig2 = visualize_regimes_ru(
        df=df,
        labels=det.regime_labels,
        change_points=det.change_points,
        features=None,  # auto-pick
        max_plots=3,
        title=f"Ruptures-style ({method}/{model}, pen={chosen_pen})",
        save_path=OUT_DIR / "regimes_plot_ruptures_style.png",
    )

    # 8) Print quick insights
    print("\nTop 10 segments by PSI200 mean (if PSI200 present):")
    if "PSI200_mean" in summary.columns:
        print(summary.sort_values("PSI200_mean").head(5))
        print(summary.sort_values("PSI200_mean", ascending=False).head(5))

    print("\nLargest PSI200 jumps between consecutive segments:")
    if "psi200_delta_vs_prev" in summary.columns:
        jumps = summary[["regime", "start_time", "psi200_delta_vs_prev"]].copy()
        print(jumps.reindex(jumps["psi200_delta_vs_prev"].abs().sort_values(ascending=False).index).head(10))

    # 9) Per-regime feature correlations vs target
    corr_df = per_regime_correlations(df, det.regime_labels, target="PSI200", features=features, method=args.corr_method)
    if not corr_df.empty:
        corr_df.to_csv(OUT_DIR / "regime_correlations.csv", index=False)
        print("\nSaved per-regime correlations to regime_correlations.csv")

    print(f"\nSaved outputs to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
