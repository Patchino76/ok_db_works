#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from parameters import get_formatted_name, get_parameter_info, parameter_colors

# %%
df = pd.read_csv('combined_data_mill8.csv')

# Convert TimeStamp to datetime and set as index
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df.set_index('TimeStamp', inplace=True)

# Filter rows where PSI200 < 35
df = df[df['PSI200'] < 35]
df = df[df['PSI200'] > 15]
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
        color = parameter_colors.get(col, 'blue')
        sns.lineplot(data=plot_df, x=plot_df.index, y=col, ax=ax, linewidth=1.5, color=color)
        
        # Set title and labels with Bulgarian names and units
        formatted_name = get_formatted_name(col)
        ax.set_title(f'{formatted_name} Тренд', fontsize=12, pad=10)
        ax.set_ylabel(formatted_name, fontsize=10)
        
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
            param_info = get_parameter_info(col)
            unit = param_info['unit'] if param_info else ''
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                      label=f'Средно: {mean_val:.2f} {unit}')
            ax.legend(fontsize=9)
        
        # Rotate x-tick labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Set x-label for the bottom subplot
    axes[-1].set_xlabel('Дата', fontsize=10)
    
    # Adjust layout with specified spacing
    plt.subplots_adjust(hspace=hspace)
    plt.tight_layout()
    plt.show()

# Example usage - uncomment to run
def plot_features_vs_target(df, target='PSI200', features=None, n_cols=2, 
                          figsize=None, alpha=0.5, wspace=0.5, hspace=0.8):
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
    
    # Set default figure size if not provided - much larger to prevent overlapping
    if figsize is None:
        fig_width = max(12, 6 * n_cols)
        fig_height = max(5, 5 * n_rows)
        figsize = (fig_width, fig_height)
    
    # Create figure and axes with specified spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)  # Add more space between subplots
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Get target parameter info
    target_formatted = get_formatted_name(target)
    
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
            
        # Get feature parameter info and color
        feature_formatted = get_formatted_name(feature)
        color = parameter_colors.get(feature, 'blue')
        
        # Create scatter plot with regression line (red line, colored points)
        sns.regplot(x=feature, y=target, data=clean_df, 
                   scatter_kws={'alpha': alpha, 'color': color, 's': 20},  # s controls point size
                   line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2},
                   ax=ax)
        
        # Set axis labels with Bulgarian names and units - smaller font and better rotation
        ax.set_xlabel(feature_formatted, fontsize=9)
        ax.set_ylabel(target_formatted, fontsize=9)
        
        # Rotate x-tick labels more and adjust their position
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Adjust tick label padding
        ax.tick_params(axis='x', which='major', pad=8)
        ax.tick_params(axis='y', which='major', pad=8)
        
        # Adjust axis limits to provide some padding
        x_min, x_max = clean_df[feature].min(), clean_df[feature].max()
        y_min, y_max = clean_df[target].min(), clean_df[target].max()
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Calculate correlation and R²
        corr = clean_df[feature].corr(clean_df[target])
        r_squared = corr ** 2
        
        # Create a more compact title with statistics
        title = f"{target_formatted} vs {feature_formatted}"
        stats = f"$r$ = {corr:.2f}, $R^2$ = {r_squared:.2f}"
        
        # Add title with smaller font and more padding
        ax.set_title(title, fontsize=9, pad=12)
        
        # Add stats text below the title
        ax.text(0.5, 0.95, stats, transform=ax.transAxes, 
                fontsize=8, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # First adjust subplot spacing with more padding
    plt.subplots_adjust(wspace=wspace, hspace=hspace,
                       left=0.1, right=0.95,
                       bottom=0.1, top=0.95)
    
    # Then apply tight layout with very generous padding
    plt.tight_layout(pad=4.0, h_pad=3.0, w_pad=3.0)
    
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
# Add a function to format correlation column names with Bulgarian names
def format_corr_column_name(col_name):
    """Format correlation column names with Bulgarian parameter names"""
    if not col_name.startswith('corr_'):
        return get_formatted_name(col_name)
    
    # Extract the two parameter IDs from the correlation column name
    parts = col_name.split('_')
    if len(parts) >= 3:
        param1 = parts[1]
        param2 = '_'.join(parts[2:])  # Handle parameters with underscore in name
        
        # Get Bulgarian names
        name1 = get_parameter_info(param1)['name'] if get_parameter_info(param1) else param1
        name2 = get_parameter_info(param2)['name'] if get_parameter_info(param2) else param2
        
        return f'Корелация {name1} - {name2}'
    return col_name

# Override get_formatted_name for correlation columns
original_get_formatted_name = get_formatted_name
def enhanced_get_formatted_name(parameter_id):
    """Enhanced version that handles correlation column names"""
    if parameter_id.startswith('corr_'):
        return format_corr_column_name(parameter_id)
    return original_get_formatted_name(parameter_id)

# Replace the imported function with our enhanced version
get_formatted_name = enhanced_get_formatted_name

# Now plot with Bulgarian names
plot_trends_stack(df_corrs, columns=['corr_PSI200_MotorAmp', 'corr_PSI200_Ore', 'corr_PSI200_PressureHC', 'corr_PSI200_DensityHC', 'corr_PSI200_WaterMill', 'corr_PSI200_WaterZumpf'], plot_mean=True)
# %%


# %%
plot_features_vs_target(
    df,
    features=['MotorAmp', 'Ore', 'PressureHC', 'DensityHC', 'WaterMill', 'WaterZumpf']
)
# %%
