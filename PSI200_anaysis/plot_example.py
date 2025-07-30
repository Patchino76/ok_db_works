"""
Example showing how to use the parameters file for plotting with Bulgarian labels and units.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from parameters import get_formatted_name, parameter_colors, parameter_icons

# Sample data (replace with your actual data loading code)
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
data = pd.DataFrame({
    'Timestamp': dates,
    'Ore': np.random.normal(180, 10, 100),
    'WaterMill': np.random.normal(15, 3, 100),
    'PSI80': np.random.normal(50, 5, 100)
})

# Example 1: Simple line plot with Bulgarian labels
def plot_parameter_timeseries(df, parameter_id):
    """Plot a single parameter with proper Bulgarian labels and units"""
    plt.figure(figsize=(12, 6))
    
    # Get the formatted name with units
    formatted_name = get_formatted_name(parameter_id)
    
    # Plot with parameter color
    color = parameter_colors.get(parameter_id, 'blue')
    plt.plot(df['Timestamp'], df[parameter_id], color=color, linewidth=2)
    
    # Add icon to title if available
    icon = parameter_icons.get(parameter_id, '')
    plt.title(f"{icon} {formatted_name}", fontsize=14)
    
    plt.xlabel("Време", fontsize=12)
    plt.ylabel(formatted_name, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

# Example 2: Multiple parameters on same plot
def plot_multiple_parameters(df, parameter_ids):
    """Plot multiple parameters on the same figure with proper labels"""
    plt.figure(figsize=(14, 8))
    
    for param_id in parameter_ids:
        color = parameter_colors.get(param_id, 'blue')
        formatted_name = get_formatted_name(param_id)
        icon = parameter_icons.get(param_id, '')
        
        plt.plot(df['Timestamp'], df[param_id], 
                 label=f"{icon} {formatted_name}", 
                 color=color, linewidth=2)
    
    plt.title("Сравнение на параметри", fontsize=14)
    plt.xlabel("Време", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Example 1: Single parameter plot
    fig1 = plot_parameter_timeseries(data, 'Ore')
    fig1.savefig('ore_timeseries.png')
    
    # Example 2: Multiple parameters
    fig2 = plot_multiple_parameters(data, ['Ore', 'WaterMill', 'PSI80'])
    fig2.savefig('multiple_parameters.png')
    
    plt.show()
