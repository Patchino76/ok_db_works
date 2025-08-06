#%%
import pandas as pd
import numpy as np

# Create example DataFrame with 2-hour frequency
rng = pd.date_range('2025-07-10 00:00', periods=5, freq='2H')
df = pd.DataFrame({'value': np.arange(5)}, index=rng)

print("Original DataFrame:")
print(df)
#%%
# Resample to 1-minute frequency, forward fill
df_resampled = df.resample('1min').ffill()

print("\nResampled DataFrame (2H → 1min, step-style):")
print(df_resampled)
# %%
