import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from raw_idea import smart_binning, analyze_binning_quality

sns.set(style="whitegrid")

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def _make_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    data = {
        "temperature": np.random.normal(350, 50, n_samples) + np.random.normal(0, 10, n_samples),
        "pressure": np.random.normal(15, 3, n_samples) + np.random.normal(0, 2, n_samples),
        "flow_rate": np.random.exponential(2, n_samples) + np.random.normal(0, 0.5, n_samples),
        "target": np.random.normal(100, 20, n_samples),
    }
    return pd.DataFrame(data)


def _plot_hist_with_bins(series: pd.Series, bins: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(8, 4))
    sns.histplot(series, kde=True, bins=30, color="#4C78A8", edgecolor="white")
    for edge in bins:
        plt.axvline(edge, color="#E45756", linestyle="--", alpha=0.8)
    plt.title(title)
    plt.xlabel(series.name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _serialize_bin_info(bin_info: dict) -> dict:
    """Convert bin_info to JSON-serializable format."""
    out = {}
    for col, info in bin_info.items():
        out[col] = {
            "method": info.get("method"),
            "bins": list(map(float, info.get("bins", []))),
            "bin_counts": list(map(int, info.get("bin_counts", []))),
        }
    return out


def _plot_bin_counts(counts: pd.Series, title: str, out_path: str):
    plt.figure(figsize=(6, 3.5))
    counts = counts.sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#72B7B2", edgecolor="black")
    plt.title(title)
    plt.xlabel("Bin index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_metric_bar(metric_df: pd.DataFrame, metric_col: str, feature: str, out_path: str):
    plt.figure(figsize=(6, 3.5))
    sns.barplot(data=metric_df, x="method", y=metric_col, hue="method", dodge=False)
    plt.title(f"{feature}: {metric_col} by method")
    plt.xlabel("Method")
    plt.ylabel(metric_col)
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = _make_data()
    features = ["temperature", "pressure", "flow_rate"]
    methods = ["quantile", "kmeans", "entropy", "equal_width"]
    n_bins = 5

    # Generate per-feature, per-method figures
    for feat in features:
        for method in methods:
            # Limit to the feature + target to avoid clutter
            sub_df = df[[feat, "target"]].copy()
            binned_df, bin_info = smart_binning(sub_df, n_bins=n_bins, method=method, target_col="target")
            bins = bin_info[feat]["bins"]

            # Distribution with bin edges
            hist_path = os.path.join(FIG_DIR, f"{feat}_{method}_bins.png")
            _plot_hist_with_bins(sub_df[feat], np.array(bins), f"{feat} - {method} bins", hist_path)

            # Bin counts
            counts = binned_df[feat].value_counts().sort_index()
            counts_path = os.path.join(FIG_DIR, f"{feat}_{method}_counts.png")
            _plot_bin_counts(counts, f"{feat} - {method} bin counts", counts_path)

    # Quality summaries per feature across methods
    for feat in features:
        rows = []
        for method in methods:
            sub_df = df[[feat, "target"]].copy()
            binned_df, bin_info = smart_binning(sub_df, n_bins=n_bins, method=method, target_col="target")
            analysis = analyze_binning_quality(sub_df, binned_df, bin_info)
            metrics = analysis.get(feat, {})
            rows.append({
                "method": method,
                "balance_score": metrics.get("balance_score", np.nan),
                "information_retention": metrics.get("information_retention", np.nan),
            })
        metr_df = pd.DataFrame(rows)

        # Info retention bar
        ir_path = os.path.join(FIG_DIR, f"{feat}_info_retention_by_method.png")
        _plot_metric_bar(metr_df, "information_retention", feat, ir_path)

        # Balance score bar
        bal_path = os.path.join(FIG_DIR, f"{feat}_balance_score_by_method.png")
        _plot_metric_bar(metr_df, "balance_score", feat, bal_path)

    # Create and save binned DataFrames for ALL features per method
    print("\nGenerating full binned DataFrames for all features...")
    for method in methods:
        all_binned_df, all_bin_info = smart_binning(df, n_bins=n_bins, method=method, target_col="target")
        csv_path = os.path.join(OUT_DIR, f"binned_all_features_{method}.csv")
        json_path = os.path.join(OUT_DIR, f"bin_info_{method}.json")

        # Save binned dataframe
        all_binned_df.to_csv(csv_path, index=True)

        # Save bin_info metadata
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_bin_info(all_bin_info), f, ensure_ascii=False, indent=2)

        print(f"- Saved: {csv_path}")
        print(f"- Saved: {json_path}")

    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Binned DataFrames and metadata saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
