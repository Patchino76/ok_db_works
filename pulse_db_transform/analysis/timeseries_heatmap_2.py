import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample realistic data for solar photovoltaic production
date_range = pd.date_range(start='2022-09-01 00:00:00', end='2023-08-31 23:00:00', freq='H')
# Simulate data with some daily and seasonal variation
values = (np.sin(2 * np.pi * date_range.dayofyear / 365) + 
          np.random.normal(0, 0.1, len(date_range)) + 
          np.sin(2 * np.pi * date_range.hour / 24)) * 50 + 100
data = {
    'time': date_range,
    'value': values
}
df = pd.DataFrame(data)

# Extract month, day, hour from the timestamp
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Create a pivot table with hours on the y-axis and months on the x-axis
pivot_table = df.pivot_table(index='hour', columns='month', values='value', aggfunc='mean')
print(pivot_table)

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, cmap='Oranges', annot=False)
plt.title('Solar Photovoltaic Production Heatmap')
plt.xlabel('Month')
plt.ylabel('Hour of the Day')
plt.show()
