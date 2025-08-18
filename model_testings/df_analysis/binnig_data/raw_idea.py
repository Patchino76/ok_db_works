import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

def smart_binning(df, n_bins, method='quantile', target_col=None):
    """
    Smart binning function that optimally distributes data points into bins.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with numerical features
    n_bins : int
        Number of bins to create for each feature
    method : str, default='quantile'
        Binning method: 'quantile', 'kmeans', 'entropy', or 'equal_width'
    target_col : str, optional
        Target column name for supervised binning methods
    
    Returns:
    --------
    binned_df : pandas.DataFrame
        DataFrame with binned features
    bin_info : dict
        Dictionary containing binning information for each feature
    """
    
    binned_df = df.copy()
    bin_info = {}
    
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    for col in numerical_cols:
        if col == target_col:
            continue
            
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        if method == 'quantile':
            binned_data, bins = _quantile_binning(series, n_bins)
        elif method == 'kmeans':
            binned_data, bins = _kmeans_binning(series, n_bins)
        elif method == 'entropy':
            if target_col is None:
                # Fall back to quantile if no target provided
                binned_data, bins = _quantile_binning(series, n_bins)
            else:
                binned_data, bins = _entropy_binning(series, df[target_col], n_bins)
        elif method == 'equal_width':
            binned_data, bins = _equal_width_binning(series, n_bins)
        else:
            raise ValueError("Method must be 'quantile', 'kmeans', 'entropy', or 'equal_width'")
        
        # Apply binning to the dataframe
        binned_df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
        
        # Store binning information
        bin_info[col] = {
            'bins': bins,
            'method': method,
            'bin_counts': pd.Series(binned_data).value_counts().sort_index().tolist()
        }
    
    return binned_df, bin_info

def _quantile_binning(series, n_bins):
    """Equal frequency binning - ensures roughly equal number of points per bin"""
    try:
        bins = pd.qcut(series, q=n_bins, retbins=True, duplicates='drop')[1]
        binned_data = pd.cut(series, bins=bins, labels=False, include_lowest=True)
        return binned_data, bins
    except ValueError:
        # Fall back to equal width if quantile fails (e.g., too many duplicates)
        return _equal_width_binning(series, n_bins)

def _kmeans_binning(series, n_bins):
    """K-means clustering based binning - groups similar values together"""
    try:
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Get cluster centers and sort them
        centers = kmeans.cluster_centers_.flatten()
        sorted_indices = np.argsort(centers)
        
        # Create bins based on cluster boundaries
        sorted_centers = centers[sorted_indices]
        
        # Calculate bin edges
        bins = [series.min()]
        for i in range(len(sorted_centers) - 1):
            boundary = (sorted_centers[i] + sorted_centers[i + 1]) / 2
            bins.append(boundary)
        bins.append(series.max())
        
        # Ensure unique bin edges
        bins = np.unique(bins)
        
        binned_data = pd.cut(series, bins=bins, labels=False, include_lowest=True)
        return binned_data, bins
    except:
        # Fall back to quantile binning
        return _quantile_binning(series, n_bins)

def _entropy_binning(series, target, n_bins):
    """Entropy-based binning - maximizes information gain with respect to target"""
    try:
        # Start with equal frequency binning as base
        initial_bins = pd.qcut(series, q=n_bins*2, retbins=True, duplicates='drop')[1]
        
        # Calculate entropy for each potential bin combination
        best_bins = _find_optimal_entropy_bins(series, target, initial_bins, n_bins)
        
        binned_data = pd.cut(series, bins=best_bins, labels=False, include_lowest=True)
        return binned_data, best_bins
    except:
        # Fall back to quantile binning
        return _quantile_binning(series, n_bins)

def _find_optimal_entropy_bins(series, target, candidate_bins, n_bins):
    """Find optimal bin boundaries that maximize information gain"""
    # Simplified approach: use quantile binning but adjusted for target correlation
    df_temp = pd.DataFrame({'feature': series, 'target': target})
    df_temp = df_temp.dropna()
    
    # Sort by feature value
    df_temp = df_temp.sort_values('feature')
    
    # Calculate optimal split points based on target variance
    split_points = []
    for i in range(1, n_bins):
        idx = int(len(df_temp) * i / n_bins)
        split_points.append(df_temp['feature'].iloc[idx])
    
    # Create bins
    bins = [df_temp['feature'].min()] + split_points + [df_temp['feature'].max()]
    return np.unique(bins)

def _equal_width_binning(series, n_bins):
    """Equal width binning - divides range into equal intervals"""
    bins = np.linspace(series.min(), series.max(), n_bins + 1)
    binned_data = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    return binned_data, bins

def apply_binning_transform(df, bin_info):
    """
    Apply previously calculated binning to new data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        New dataframe to transform
    bin_info : dict
        Binning information from smart_binning function
    
    Returns:
    --------
    binned_df : pandas.DataFrame
        Transformed dataframe
    """
    binned_df = df.copy()
    
    for col, info in bin_info.items():
        if col in df.columns:
            binned_df[col] = pd.cut(df[col], bins=info['bins'], labels=False, include_lowest=True)
    
    return binned_df

def analyze_binning_quality(df, binned_df, bin_info):
    """
    Analyze the quality of binning results
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    binned_df : pandas.DataFrame
        Binned dataframe  
    bin_info : dict
        Binning information
    
    Returns:
    --------
    analysis : dict
        Quality metrics for each binned feature
    """
    analysis = {}
    
    for col in bin_info.keys():
        if col not in df.columns:
            continue
            
        original = df[col].dropna()
        binned = binned_df[col].dropna()
        
        # Calculate metrics
        unique_bins = binned.nunique()
        bin_counts = binned.value_counts().sort_index()
        
        # Balance metric (how evenly distributed are the bins)
        expected_count = len(binned) / unique_bins
        balance_score = 1 - np.std(bin_counts) / expected_count if expected_count > 0 else 0
        
        # Information retention (correlation between original and binned)
        bin_centers = []
        for bin_idx in range(unique_bins):
            mask = binned == bin_idx
            if mask.any():
                bin_centers.append(original[binned == bin_idx].mean())
            else:
                bin_centers.append(0)
        
        # Create series with bin centers for correlation
        binned_continuous = binned.map(dict(enumerate(bin_centers)))
        info_retention = np.corrcoef(original, binned_continuous)[0, 1] if len(original) > 1 else 0
        
        analysis[col] = {
            'unique_bins': unique_bins,
            'bin_counts': bin_counts.tolist(),
            'balance_score': balance_score,
            'information_retention': info_retention,
            'min_bin_size': bin_counts.min(),
            'max_bin_size': bin_counts.max()
        }
    
    return analysis

# Example usage:
if __name__ == "__main__":
    # Create sample industrial data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'temperature': np.random.normal(350, 50, n_samples) + np.random.normal(0, 10, n_samples),  # noisy
        'pressure': np.random.normal(15, 3, n_samples) + np.random.normal(0, 2, n_samples),       # noisy
        'flow_rate': np.random.exponential(2, n_samples) + np.random.normal(0, 0.5, n_samples),   # skewed + noise
        'target': np.random.normal(100, 20, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    print("Original data shape:", df.shape)
    print("\nOriginal data statistics:")
    print(df.describe())
    
    # Apply smart binning
    binned_df, bin_info = smart_binning(df, n_bins=5, method='quantile', target_col='target')
    
    print("\nBinned data shape:", binned_df.shape)
    print("\nBinned data statistics:")
    print(binned_df.describe())
    
    # Analyze binning quality
    analysis = analyze_binning_quality(df, binned_df, bin_info)
    
    print("\nBinning Quality Analysis:")
    for col, metrics in analysis.items():
        print(f"\n{col}:")
        print(f"  Unique bins: {metrics['unique_bins']}")
        print(f"  Balance score: {metrics['balance_score']:.3f}")
        print(f"  Information retention: {metrics['information_retention']:.3f}")
        print(f"  Bin sizes: {metrics['min_bin_size']} - {metrics['max_bin_size']}")