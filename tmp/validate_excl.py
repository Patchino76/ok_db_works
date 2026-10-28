import pandas as pd
import numpy as np

df = pd.read_parquet(r'c:\Projects\ok_db_works\pulse_db_transform\psi_events_Mill06.parquet')
TARGET = 'PSI200'

# FEATURES без Power, ZumpfLevel, PumpRPM (и FE, който и без това пада по покритие)
FEATURES = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 'DensityHC', 'PulpHC', 'MotorAmp']
drop = [c for c in df.columns if c.split('_')[0] in ('Power', 'ZumpfLevel', 'PumpRPM')]
df = df.drop(columns=drop, errors='ignore')


def add_features(d):
    out = d.copy()
    psi = out[TARGET].where(out[TARGET].between(15, 30))
    eps = 1e-6
    for k in (1, 2, 3, 6, 12):
        out[f'psi_lag{k}'] = psi.shift(k)
    for w in (3, 6, 12):
        r = psi.shift(1).rolling(w, min_periods=1)
        out[f'psi_roll{w}_mean'] = r.mean()
        out[f'psi_roll{w}_std'] = r.std()
        out[f'psi_roll{w}_min'] = r.min()
        out[f'psi_roll{w}_max'] = r.max()
    base = psi.shift(1)
    out['psi_diff1'] = base.diff(1)
    out['psi_diff3'] = base.diff(3)
    out['psi_slope6'] = (psi.shift(1) - psi.shift(6)) / 5.0
    for f in FEATURES:
        c = f'{f}_mean'
        if c in out.columns:
            out[f'{f}_lag1'] = out[c].shift(1)
            out[f'{f}_diff1'] = out[c].diff(1)
    if {'WaterMill_mean', 'Ore_mean'} <= set(out.columns):
        out['dilution_mill'] = out['WaterMill_mean'] / (out['Ore_mean'] + eps)
    if {'WaterZumpf_mean', 'Ore_mean'} <= set(out.columns):
        out['dilution_zumpf'] = out['WaterZumpf_mean'] / (out['Ore_mean'] + eps)
    if {'Ore_mean', 'PulpHC_mean'} <= set(out.columns):
        out['solids_load'] = out['Ore_mean'] * out['PulpHC_mean']
    if {'Power_mean', 'Ore_mean'} <= set(out.columns):
        out['spec_energy'] = out['Power_mean'] / (out['Ore_mean'] + eps)
    if {'MotorAmp_mean', 'PumpRPM_mean'} <= set(out.columns):
        out['amp_per_rpm'] = out['MotorAmp_mean'] / (out['PumpRPM_mean'] + eps)
    if {'DensityHC_mean', 'PulpHC_mean'} <= set(out.columns):
        out['density_minus_pulp'] = out['DensityHC_mean'] - out['PulpHC_mean']
    return out


df2 = add_features(df)
print('add_features OK')
print('spec_energy present (should be False):', 'spec_energy' in df2.columns)
print('amp_per_rpm present (should be False):', 'amp_per_rpm' in df2.columns)
print('dilution_mill present (should be True):', 'dilution_mill' in df2.columns)

CONTROLLABLE = [f for f in ['Ore', 'WaterMill', 'WaterZumpf', 'PumpRPM', 'PressureHC', 'DensityHC']
                if f in FEATURES]
CTRL_COLS = [f'{f}_mean' for f in CONTROLLABLE]
g = df[CTRL_COLS + [TARGET]].resample('1h')
hourly = g.mean()
hourly['n'] = g[TARGET].count()
hourly['psi_std'] = g[TARGET].std()
hourly = hourly[hourly['n'] >= 3].dropna(subset=CTRL_COLS + [TARGET])
print('A1 OK, CONTROLLABLE:', CONTROLLABLE)
print('FIX OK')
