#%%
import pandas as pd
from sqlite_loader import SQLiteDBLoader

start_date = "2024-07-01 06:00"
end_date = "2025-02-26 15:00"

# %%

db_loader = SQLiteDBLoader()

db_loader.connect()

# %%

db_loader.get_table_names()

# %%
db_loader.get_table_schema("Dispatchers")
# %%
df_dispatchers = db_loader.table_to_dataframe("Dispatchers")
df_dispatchers = df_dispatchers[(df_dispatchers['timestamp'] >= start_date) & (df_dispatchers['timestamp'] <= end_date)]

# Convert timestamp to datetime and set as index
df_dispatchers['timestamp'] = pd.to_datetime(df_dispatchers['timestamp'])
df_dispatchers = df_dispatchers.set_index('timestamp')

df_dispatchers = df_dispatchers.drop(['shift', 'dispatcher', 'original_sheet'], axis=1)
df_dispatchers = df_dispatchers.resample('1h').mean()
df_dispatchers = df_dispatchers.interpolate(method='time')
df_dispatchers = df_dispatchers.fillna(method='ffill')
df_dispatchers = df_dispatchers.fillna(method='bfill')

df_dispatchers
# %%
df_mill_01 = db_loader.table_to_dataframe("MILL_01")
df_mill_01 = df_mill_01[(df_mill_01['TimeStamp'] >= start_date) & (df_mill_01['TimeStamp'] <= end_date)]

# Convert timestamp to datetime and set as index for mill data too
df_mill_01['TimeStamp'] = pd.to_datetime(df_mill_01['TimeStamp'])
df_mill_01 = df_mill_01.set_index('TimeStamp')

df_mill_01 = df_mill_01.resample('1h').mean()
df_mill_01 = df_mill_01.interpolate(method='time')
df_mill_01 = df_mill_01.fillna(method='ffill')
df_mill_01 = df_mill_01.fillna(method='bfill')

df_mill_01
# %%
