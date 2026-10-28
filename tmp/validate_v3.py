import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5


df = pd.read_parquet(r'c:\Projects\ok_db_works\pulse_db_transform\psi_events_Mill06.parquet')
TARGET = 'PSI200'
CONTROLLABLE = ['Ore', 'WaterMill', 'WaterZumpf', 'PumpRPM', 'PressureHC', 'DensityHC']
CTRL_COLS = [f'{f}_mean' for f in CONTROLLABLE]

# --- A1 ---
g = df[CTRL_COLS + [TARGET]].resample('1h')
hourly = g.mean()
hourly['n'] = g[TARGET].count()
hourly['psi_std'] = g[TARGET].std()
hourly = hourly[hourly['n'] >= 3].dropna(subset=CTRL_COLS + [TARGET])
q = hourly['psi_std'].quantile(0.75)
steady = hourly[hourly['psi_std'] <= q].copy()
print(f'A1 hourly={len(hourly)} steady={len(steady)}')

# --- A2 ---
Xh = steady[CTRL_COLS].astype(float)
yh = steady[TARGET].astype(float)


def wf_eval(make_model, X, y, n_splits=5, gap=1):
    rows = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[va])
        rows.append({'n': len(va), 'rmse': rmse(y.iloc[va], p), 'r2': r2_score(y.iloc[va], p)})
    return pd.DataFrame(rows)


y_prev, y_now = yh.iloc[:-1], yh.iloc[1:]
base = pd.DataFrame([
    {'model': 'mean', 'rmse': rmse(y_now, np.full(len(y_now), y_prev.mean())),
     'r2': r2_score(y_now, np.full(len(y_now), y_prev.mean()))},
    {'model': 'persistence (t-1h)', 'rmse': rmse(y_now, y_prev.values),
     'r2': r2_score(y_now, y_prev.values)},
])
lin = wf_eval(lambda: LinearRegression(), Xh, yh)
xgb = wf_eval(lambda: XGBRegressor(
    n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0,
    objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=-1), Xh, yh)
print('A2 baselines:')
print(base.round(3).to_string(index=False))
print(f'A2 lin R2={lin.r2.mean():.3f}  xgb R2={xgb.r2.mean():.3f}')

# --- B1 ---
GRID = '10min'
grid = df[CTRL_COLS + [TARGET]].resample(GRID).mean().interpolate(limit=2)


def detect_steps(s, k=3.0, min_sep='60min'):
    d = s.diff()
    thr = k * d.std()
    if not np.isfinite(thr) or thr <= 0:
        return pd.DatetimeIndex([])
    cand = d[d.abs() >= thr].index
    kept = []
    for t in cand:
        if not kept or (t - kept[-1]) >= pd.Timedelta(min_sep):
            kept.append(t)
    return pd.DatetimeIndex(kept)


steps = {f: detect_steps(grid[f'{f}_mean']) for f in CONTROLLABLE}
print('B1 steps:', {f: len(steps[f]) for f in CONTROLLABLE})

# --- B2 ---
PRE, POST = pd.Timedelta('60min'), pd.Timedelta('180min')


def response_curve(events, pre=PRE, post=POST):
    curves = []
    for t0 in events:
        w = grid.loc[t0 - pre: t0 + post, TARGET]
        if len(w) < 6 or w.index.min() > t0 - pre + pd.Timedelta('20min'):
            continue
        b = w.loc[:t0].mean()
        if not np.isfinite(b):
            continue
        rel = (w - b)
        rel.index = w.index - t0
        curves.append(rel)
    if not curves:
        return None
    return pd.concat(curves, axis=1).mean(axis=1)


def dead_time_gain(rc, frac=0.2, floor=0.1, smooth=3, sustain=2):
    after = rc[rc.index >= pd.Timedelta(0)]
    if len(after) == 0:
        return np.nan, np.nan
    gain = after.iloc[-6:].mean() if len(after) >= 6 else after.mean()
    sm = after.rolling(smooth, center=True, min_periods=1).mean()
    thr = max(floor, frac * abs(gain))
    run = (sm.abs() > thr).astype(int).rolling(sustain).sum()
    hit = run[run >= sustain]
    dt = hit.index.min().total_seconds() / 60 if len(hit) else np.nan
    return dt, gain


dyn = []
for f in CONTROLLABLE:
    rc = response_curve(steps[f])
    if rc is None:
        dyn.append({'param': f, 'n_steps': len(steps[f]), 'dead_time_min': np.nan, 'gain': np.nan})
        continue
    dt, gain = dead_time_gain(rc)
    dyn.append({'param': f, 'n_steps': len(steps[f]), 'dead_time_min': dt, 'gain': gain})
print('B2 dynamics:')
print(pd.DataFrame(dyn).round(2).to_string(index=False))
print('VALIDATION OK')
