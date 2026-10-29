"""Sanity-test на функциите от data_quality_assessment.ipynb върху синтетични данни.

Изпълнява само клетките с дефиниции (без SQL) и инжектира:
  - байпас на вода (WaterZumpf пада наполовина) в прозорец
  - замръзнал сензор (Ore константа) в прозорец
  - PSI200 глитч (6553) и flatline
После проверява, че флаговете хващат точно тези прозорци.
"""
import json
import os

import numpy as np
import pandas as pd

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'pulse_db_transform', 'data_quality_assessment.ipynb')

with open(NB, encoding='utf-8') as fh:
    nb = json.load(fh)

# изпълняваме само клетките с дефиниции (config, read_raw/build, flags, score)
DEF_CELLS = [1, 2, 4, 6, 8, 10, 12, 21]
ns = {}
for i in DEF_CELLS:
    src = nb['cells'][i]['source']
    if i == 21:  # train_eval клетката завършва с print(train_eval(df)) - махаме го
        src = src.split('print(train_eval(df)')[0]
    exec(src, ns)

print('Дефиниции заредени OK')

# ---------------------------------------------------------------------------
# синтетични сурови данни
rng = np.random.default_rng(0)
MILL = 'Mill06'
t0 = pd.Timestamp('2026-01-01')
t1 = pd.Timestamp('2026-01-10')

psi_idx = pd.date_range(t0, t1, freq='8min')
min_idx = pd.date_range(t0, t1, freq='1min')

base = {
    'Ore': 170.0, 'WaterMill': 10.0, 'WaterZumpf': 170.0, 'Power': 1850.0,
    'ZumpfLevel': 1575.0, 'PressureHC': 0.43, 'DensityHC': 1660.0,
    'FE': 50.0, 'PulpHC': 482.0, 'PumpRPM': 300.0, 'MotorAmp': 196.0,
}
noise = {
    'Ore': 5.0, 'WaterMill': 0.5, 'WaterZumpf': 8.0, 'Power': 40.0,
    'ZumpfLevel': 20.0, 'PressureHC': 0.01, 'DensityHC': 10.0,
    'FE': 2.0, 'PulpHC': 15.0, 'PumpRPM': 10.0, 'MotorAmp': 4.0,
}

raw = {}
for feat, b in base.items():
    s = pd.Series(b + rng.normal(0, noise[feat], len(min_idx)), index=min_idx)
    raw.setdefault(feat, {})[MILL] = s

# инжектираме байпас: WaterZumpf пада наполовина на 2026-01-05
bypass = (min_idx >= '2026-01-05') & (min_idx < '2026-01-06')
raw['WaterZumpf'][MILL].loc[bypass] *= 0.45

# инжектираме замръзнал Ore на 2026-01-07
frozen = (min_idx >= '2026-01-07') & (min_idx < '2026-01-08')
raw['Ore'][MILL].loc[frozen] = 170.0

# PSI200: нормално ~20, глитч 6553, flatline 18.0
psi = pd.Series(20.0 + rng.normal(0, 1.5, len(psi_idx)), index=psi_idx)
psi.iloc[300] = 6553.0
psi.iloc[500:520] = 18.0
raw.setdefault('PSI200', {})[MILL] = psi

# ---------------------------------------------------------------------------
df = ns['build'](raw, MILL)
df, sensor_flags, skipped = ns['add_sensor_flags'](df)
df, phys_flags = ns['add_physical_checks'](df)
df = ns['add_multivariate_flags'](df)
score, all_flags = ns['composite_score'](df, sensor_flags, phys_flags)
df['quality'] = score

print(f'редове={len(df)}  пропуснати={skipped}')
print(f'средно качество={score.mean():.3f}')

# --- проверки ---
bypass_rows = df['bypass_water']
in_bypass = (df.index >= '2026-01-05') & (df.index < '2026-01-06')
print(f'\nbypass флаг: общо={int(bypass_rows.sum())}  '
      f'в инжектирания прозорец={int(bypass_rows[in_bypass].sum())}/{int(in_bypass.sum())}  '
      f'извън={int(bypass_rows[~in_bypass].sum())}')
assert bypass_rows[in_bypass].mean() > 0.5, 'байпасът не е хванат'
assert bypass_rows[~in_bypass].mean() < 0.1, 'твърде много фалшиви байпас флагове'

frozen_rows = df['Ore_frozen_flag']
in_frozen = (df.index >= '2026-01-07') & (df.index < '2026-01-08')
print(f'Ore_frozen флаг: в инжектирания прозорец={int(frozen_rows[in_frozen].sum())}/{int(in_frozen.sum())}  '
      f'извън={int(frozen_rows[~in_frozen].sum())}')
assert frozen_rows[in_frozen].mean() > 0.5, 'замръзналият сензор не е хванат'

print(f'PSI200_glitch: {int(df["PSI200_glitch"].sum())}  (очакваме >=1)')
assert df['PSI200_glitch'].sum() >= 1, 'глитчът не е хванат'

print(f'PSI200_flat: {int(df["PSI200_flat"].sum())}  (очакваме >=1)')
assert df['PSI200_flat'].sum() >= 1, 'flatline не е хванат'

# качеството в лошите прозорци трябва да е по-ниско
q_bad = df.loc[in_bypass | in_frozen, 'quality'].mean()
q_good = df.loc[~(in_bypass | in_frozen), 'quality'].mean()
print(f'\nкачество: лоши прозорци={q_bad:.3f}  добри={q_good:.3f}')
assert q_bad < q_good, 'quality score не различава лошите прозорци'

# --- period_shift / adversarial ---
ref = df.index < '2026-01-05'
cmp = df.index >= '2026-01-05'
sh = ns['period_shift'](df, ref, cmp)
auc, imp = ns['adversarial_validation'](df, ref, cmp)
print(f'\nperiod_shift top:\n{sh.head(3).round(3).to_string(index=False)}')
print(f'adversarial AUC={auc:.3f}')

# --- train_eval ---
res = ns['train_eval'](df)
print(f'\ntrain_eval:\n{res.round(3).to_string(index=False)}')

print('\nALL CHECKS PASSED')
