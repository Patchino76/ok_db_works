#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('combined_data_mill8.csv')

# Convert TimeStamp to datetime and set as index
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df.set_index('TimeStamp', inplace=True)

# Filter rows where PSI200 < 35
df = df[df['PSI200'] < 35]
df = df[df['Ore'] > 150]

# Display the first few rows to verify
# print("Filtered data (PSI200 < 35):")
# print(df.head())
# print(f"\nNumber of rows after filtering: {len(df)}")

# %%
def resample_dataframe(df, freq='8H', agg_func='mean'):
    """
    Resample the DataFrame to a given frequency.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with a datetime index
    freq : str, default '8H'
        Frequency string for resampling (e.g., '1H', 'D', 'W')
    agg_func : str or dict, default 'mean'
        Aggregation function(s) to use when resampling
        
    Returns:
    --------
    pandas.DataFrame
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
        
    return df.resample(freq).agg(agg_func)

# %%
resampled_df = resample_dataframe(df, freq='8H', agg_func='mean')
print(resampled_df.head(2))


# %%

def plot_trends_stack(df, columns=None, figsize=(14, 10), hspace=0.4, plot_mean=True):
    """
    Plot time series trends in vertically stacked subplots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with a datetime index
    columns : list, optional
        List of column names to plot. If None, uses all numeric columns
    figsize : tuple, default (14, 10)
        Figure size (width, height) in inches
    hspace : float, default 0.4
        Vertical space between subplots
    plot_mean : bool, default True
        If True, plots a horizontal dashed red line at the mean value for each subplot
    """
    if columns is None:
        # If no columns specified, use all numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Use the data as is without resampling
    plot_df = df[columns]
    
    # Calculate number of subplots
    n_plots = len(columns)
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    # If there's only one subplot, axes won't be an array
    if n_plots == 1:
        axes = [axes]
    
    # Plot each column in its own subplot
    for i, col in enumerate(columns):
        ax = axes[i]
        # Plot the data
        sns.lineplot(data=plot_df, x=plot_df.index, y=col, ax=ax, linewidth=1.5)
        
        # Set title and labels
        ax.set_title(f'{col} Trend', fontsize=12, pad=10)
        ax.set_ylabel(col, fontsize=10)
        
        # Set y-axis limits with some padding
        y_min = plot_df[col].min()
        y_max = plot_df[col].max()
        padding = (y_max - y_min) * 0.1  # 10% padding
        ax.set_ylim(y_min - padding, y_max + padding)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add mean line if requested
        if plot_mean:
            mean_val = plot_df[col].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        # Rotate x-tick labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Set x-label for the bottom subplot
    axes[-1].set_xlabel('Date', fontsize=10)
    
    # Adjust layout with specified spacing
    plt.subplots_adjust(hspace=hspace)
    plt.tight_layout()
    plt.show()

# Example usage - uncomment to run
def plot_features_vs_target(df, target='PSI200', features=None, n_cols=2, 
                          figsize=None, alpha=0.5):
    """
    Create scatter plots with regression lines for multiple features against a target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    target : str, default 'PSI200'
        Name of the target variable (y-axis)
    features : list, optional
        List of feature names to plot against the target.
        If None, uses all numeric columns except the target.
    n_cols : int, default 2
        Number of subplot columns
    figsize : tuple, optional
        Figure size (width, height) in inches.
        If None, it will be automatically determined.
    alpha : float, default 0.5
        Transparency of the scatter points
        
    Example:
    --------
    # Plot all numeric features against PSI200
    plot_features_vs_target(df)
    
    # Plot specific features against a custom target
    plot_features_vs_target(df, target='MotorAmp', 
                           features=['PSI200', 'Ore', 'PressureHC'])
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import to_rgba
    
    # Get features if not provided
    if features is None:
        features = [col for col in df.select_dtypes(include=['number']).columns 
                   if col != target]
    
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Set default figure size if not provided
    if figsize is None:
        fig_width = min(20, 8 * n_cols)
        fig_height = 5 * n_rows
        figsize = (fig_width, fig_height)
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easier iteration
    if n_features > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Define a color palette (avoiding red)
    colors = sns.color_palette("husl", n_colors=n_features)
    
    # Plot each feature against the target
    for i, (feature, ax) in enumerate(zip(features, axes)):
        # Filter out NaN values
        clean_df = df[[feature, target]].dropna()
        
        if clean_df.empty:
            ax.text(0.5, 0.5, 'No data', 
                   ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f"{target} vs {feature}", fontsize=12)
            continue
            
        # Create scatter plot with regression line
        sns.regplot(x=feature, y=target, data=clean_df, 
                   scatter_kws={'alpha': alpha, 'color': colors[i]},
                   line_kws={'color': colors[i], 'linestyle': '--'},
                   ax=ax)
        
        # Calculate correlation and R²
        corr = clean_df[feature].corr(clean_df[target])
        r_squared = corr ** 2
        
        # Add title with statistics
        ax.set_title(f"{target} vs {feature}\n$r$ = {corr:.2f}, $R^2$ = {r_squared:.2f}", 
                    fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_scatter_with_regression(df, 'PSI200', 'MotorAmp')
# plot_scatter_with_regression(df, 'PSI200', 'MotorAmp', 'Ore',  figsize=(8, 8), alpha=0.3, color='green')
# %%

plot_trends_stack(resampled_df, columns=['PSI200', 'MotorAmp', 'Ore', 'PressureHC', 'DensityHC', 'WaterMill', 'WaterZumpf'])



# %%
def calculate_windowed_correlations(df, features=None, window='8H', min_periods=10):
    """
    Calculate correlations between all pairs of features in non-overlapping time windows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with a datetime index
    features : list, optional
        List of feature names to calculate correlations between.
        If None, uses all numeric columns in the DataFrame.
    window : str, default '8H'
        Size of the time window for correlation calculation (e.g., '4H' for 4 hours).
    min_periods : int, default 10
        Minimum number of observations required in a window to calculate correlation.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with one row per time window, containing correlation values.
        Columns are named as 'corr_[feature1]_[feature2]'.
    """
    if features is None:
        features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create a copy to avoid modifying the original DataFrame
    df_working = df[features].copy()
    
    # Generate all unique pairs of features
    from itertools import combinations
    feature_pairs = list(combinations(features, 2))
    
    # Define a function to calculate correlations for each window
    def calculate_correlations(group):
        if len(group) < min_periods:
            return pd.Series()
        result = {}
        for f1, f2 in feature_pairs:
            # Calculate correlation for this window
            corr = group[f1].corr(group[f2])
            result[f'corr_{f1}_{f2}'] = corr
        return pd.Series(result)
    
    # Group by the specified time window and calculate correlations
    result_series = (
        df_working
        .groupby(pd.Grouper(freq=window))
        .apply(calculate_correlations)
    )
    
    # Unstack the MultiIndex to get correlation pairs as columns
    if not result_series.empty:
        result_df = result_series.unstack()
        # Drop any rows with all NaN values
        result_df = result_df.dropna(how='all')
        return result_df
    else:
        return pd.DataFrame()
# %%
# Example usage:
df_corrs = calculate_windowed_correlations(
    df, 
    features=['PSI200', 'MotorAmp', 'Ore', 'PressureHC', 'DensityHC', 'WaterMill', 'WaterZumpf'],
    window='8H',
    min_periods=10
)
# %%
plot_trends_stack(df_corrs, columns=['corr_PSI200_MotorAmp', 'corr_PSI200_Ore', 'corr_PSI200_PressureHC', 'corr_PSI200_DensityHC', 'corr_PSI200_WaterMill', 'corr_PSI200_WaterZumpf'], plot_mean=True)
# %%

# plot_scatter_matrix(
#     df,
#     features=['PSI200', 'MotorAmp', 'Ore'],
#     figsize=(15, 12),  # Custom figure size
#     alpha=0.3,         # Point transparency
#     color='green',     # Point color
#     line_color='red'   # Regression line color
# )
# %%
plot_features_vs_target(
    df,
    features=['MotorAmp', 'Ore', 'PressureHC', 'DensityHC', 'WaterMill', 'WaterZumpf']
)
# %%
