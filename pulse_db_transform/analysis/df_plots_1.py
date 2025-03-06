#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_db_to_df import MillDataLoader
# %%

loader = MillDataLoader()
df = loader.get_mill_dataframe("MILL_01")

# For DatetimeIndex, access properties directly without using .dt
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour
print(df.head(3))
print(df.info())

# %%
