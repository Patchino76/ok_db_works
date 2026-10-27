import pandas as pd
import numpy as np
from sqlalchemy import create_engine

e = create_engine(
    'mssql+pyodbc:///?odbc_connect='
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.20.2.10;'
    'DATABASE=pulse;UID=Pulse_RO;PWD=PD@T@r3@der'
)

frames = []
for tbl in ('LoggerValues', 'LoggerValues_Archive_Sep2026'):
    q = f"""
    SELECT IndexTime, Value FROM {tbl}
    WHERE LoggerTagID = 5328 AND IndexTime >= '2026-09-01'
    """
    frames.append(pd.read_sql_query(q, e))

d = pd.concat(frames, ignore_index=True)
d['IndexTime'] = pd.to_datetime(d['IndexTime'])
s = d.drop_duplicates('IndexTime').set_index('IndexTime')['Value'].sort_index()

# --- replicate build() PSI200 logic ---
psi = s.dropna()
out = pd.DataFrame(index=psi.index[1:])
out['PSI200'] = psi.values[1:]
out['dt_sec'] = np.diff(psi.index.values) / np.timedelta64(1, 's')
out = out[out['dt_sec'] <= 60 * 60]          # max_gap=60min

# --- replicate mask + split ---
mask = out['PSI200'].between(15, 30)
out = out[mask]
split_idx = int(len(out) * 0.8)
test = out.iloc[split_idx:]

dead = test.loc['2026-10-04':'2026-10-20']   # exclude Oct 21 (data resumed)
print(f'total rows after build+mask : {len(out)}')
print(f'test rows                   : {len(test)}   {test.index.min()} .. {test.index.max()}')
print(f'test rows in DEAD window    : {len(dead)}  ({100*len(dead)/len(test):.1f}% of test)')
print(f'  dead-window PSI200 values : min={dead.PSI200.min()} max={dead.PSI200.max()} '
      f'nunique={dead.PSI200.nunique()}')
print(f'  value counts              : {dead.PSI200.value_counts().to_dict()}')
print()
alive = test.drop(dead.index)
print('test rows OUTSIDE dead window:')
print(f'  n={len(alive)}  PSI200 mean={alive.PSI200.mean():.2f} std={alive.PSI200.std():.2f}')
print(f'  dead  PSI200 mean={dead.PSI200.mean():.2f} std={dead.PSI200.std():.2f}')
print()
print(f'frozen (==18.0) rows in whole test: {int((test.PSI200 == 18.0).sum())}')
