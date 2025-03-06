#%%
import pandas as pd
from columns_dictionary import translation_dict

#%%
# Load the data
df = pd.read_csv('combined_dispatcher_data.csv', encoding='utf-8')
df.rename(columns=translation_dict, inplace=True)
df = df.dropna(subset=['dispatcher'])

#dealing with the date time shits...
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.drop(columns=['date'])
cols = df.columns.tolist()
cols.remove('timestamp')
cols = ['timestamp'] + cols
df = df[cols]
df.tail()

#%%
# Replace NaN values with column means
df = df.fillna(df.mean(numeric_only=True))

#%%
# Save the translated data
df.to_csv('combined_dispatcher_data_en.csv', index=False)

# %%
