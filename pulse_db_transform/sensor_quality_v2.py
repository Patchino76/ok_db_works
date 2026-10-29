# %% [markdown]
# # Качествени стабилни периоди за обучение на PSI200 (v2)
#
# Цел: интервали, в които мелницата работи в **стабилен режим** или в **плавен преход**
# между режими, с валиден таргет и живи сензори. Резките промени (спиране на руда,
# пускове, скокове) и времето за успокояване след тях се изключват.
#
# Стъпки:
# 0. Профил на записване на таговете (кога "липсва" е реално).
# 1. 1-минутна мрежа: последна известна стойност + staleness (без сляп ffill).
# 2. Състояние на мелницата: работи / спряна / пуск (буфер).
# 3. Режим: steady / ramp (плавен) / abrupt (рязък) + settling буфер след рязко събитие.
# 4. Валиден таргет PSI200.
# 5. Сегменти (интервали) + почасов датасет за модела.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
plt.rcParams['figure.figsize'] = (14, 4)

ENGINE = create_engine(
    'mssql+pyodbc:///?odbc_connect='
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.20.2.10;'
    'DATABASE=pulse;UID=Pulse_RO;PWD=PD@T@r3@der'
)

SQL_TAGS = {
    'Ore':        {'Mill06': '500',  'Mill07': '455',  'Mill08': '467'},
    'WaterMill':  {'Mill06': '1219', 'Mill07': '461',  'Mill08': '474'},
    'WaterZumpf': {'Mill06': '1220', 'Mill07': '462',  'Mill08': '475'},
    'Power':      {'Mill06': '502',  'Mill07': '460',  'Mill08': '471'},
    'ZumpfLevel': {'Mill06': '501',  'Mill07': '458',  'Mill08': '470'},
    'PressureHC': {'Mill06': '1217', 'Mill07': '459',  'Mill08': '472'},
    'DensityHC':  {'Mill06': '1216', 'Mill07': '457',  'Mill08': '469'},
    'PulpHC':     {'Mill06': '2777', 'Mill07': '2837', 'Mill08': '1833'},
    'PumpRPM':    {'Mill06': '2796', 'Mill07': '2856', 'Mill08': '1800'},
    'MotorAmp':   {'Mill06': '2770', 'Mill07': '2830', 'Mill08': '1805'},
    'PSI200':     {'Mill06': '5328', 'Mill07': '5330', 'Mill08': '5333'},
}

TABLE_NAMES = ['LoggerValues'] + [
    f'LoggerValues_Archive_{m}{y}'
    for y in (2026, 2025)
    for m in ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    if not (y == 2026 and m in ('Oct', 'Nov', 'Dec'))
]

TARGET = 'PSI200'
CORE_FEATURES = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 'DensityHC', 'MotorAmp']
OPTIONAL_FEATURES = ['PulpHC']   # винаги в датасета; влияе на филтъра/модела само ако USE_PULPHC
USE_PULPHC = True
MODEL_FEATURES = CORE_FEATURES + (OPTIONAL_FEATURES if USE_PULPHC else [])
DATASET_FEATURES = CORE_FEATURES + OPTIONAL_FEATURES

START, END = '2026-01-01', '2026-10-15'
MILL = os.environ.get('MILL', 'Mill06')

# лаг (мин) на входовете спрямо PSI200: стойността от t-lag обяснява PSI200(t)
LAGS_MIN = {'Ore': 15, 'WaterMill': 15, 'WaterZumpf': 15}

# състав на рудата (mills.ore_quality, 1 проба на смяна; TimeStamp = начало на смяната 06/14/22)
PG_ENGINE = create_engine(
    'postgresql://s.lyubenov:tP9uB7sH7mK6zA7t@em-m-db4.ellatzite-med.com:5432/em_pulse_data'
)
ORE_FEATURES = ['Class_15', 'Class_12', 'Grano', 'Daiki', 'Shisti']
ORE_TIME_OFFSET_H = 1          # предполагаме проба в 07/15/23 -> валидна от тогава
ORE_MAX_AGE_H = 9              # не държим стойността повече от една смяна (+буфер)
CACHE_DIR = Path(__file__).parent / 'cache' if '__file__' in globals() else Path('cache')

# физически граници - извън тях стойността е невалидна (NaN)
PHYSICAL_LIMITS = {
    'Ore': (0, 400), 'WaterMill': (0, 100), 'WaterZumpf': (0, 600), 'Power': (0, 4000),
    'ZumpfLevel': (0, 3000), 'PressureHC': (0, 2), 'DensityHC': (1000, 2200),
    'PulpHC': (0, 1200), 'PumpRPM': (0, 1000), 'MotorAmp': (0, 500), 'PSI200': (10, 40),
}

# --- състояние на мелницата ---
RUNNING_RULES = {'Ore': 100, 'Power': 1000, 'MotorAmp': 100}   # всички > прага
STARTUP_BUFFER_MIN = 60        # минути след (пре)пускане, които изключваме

# --- режим ---
SMOOTH_MIN = 15                # изглаждане преди смятане на скоростта
RATE_WIN_MIN = 10              # скорост = промяна за RATE_WIN_MIN минути
# рязка промяна: |Δ за RATE_WIN_MIN| > лимита (в единиците на сензора). Калибрирай с клетката за квантили.
RATE_LIMITS = {
    'Ore': 20, 'WaterMill': 4, 'WaterZumpf': 40,
    'MotorAmp': 6, 'DensityHC': 60, 'PressureHC': 0.03,
}
RATE_LIMITS_BY_MILL = {'Mill08': {'MotorAmp': 10}}   # Mill08: MotorAmp е по-шумен (p99.9 ≈ 10 A)
RATE_LIMITS = {**RATE_LIMITS, **RATE_LIMITS_BY_MILL.get(MILL, {})}
STEADY_FRAC = 0.3              # |Δ| < STEADY_FRAC * лимит за всички -> steady, иначе ramp
SETTLE_MIN = 30                # минути след рязко събитие, докато процесът се успокои

# --- таргет ---
PSI_MAX_HOLD_MIN = 25          # PSI200 се чете ~на 11 мин; по-старо -> няма таргет
PSI_FROZEN_MIN = 45            # PSI200 без промяна по-дълго -> анализаторът е замръзнал

# --- сегменти ---
MIN_SEGMENT_MIN = 120          # по-кратки добри интервали не ни интересуват
HOUR_MIN_GOOD_FRAC = 0.8       # час влиза в датасета, ако >= 80% от минутите са добри

# %% [markdown]
# ## Четене (с кеш в parquet, за да не удряме SQL при всяко пускане)

# %%
def read_raw(start, end, use_cache=True):
    """Връща long DataFrame [IndexTime, LoggerTagID, Value] за всички тагове."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f'raw_{start}_{end}.parquet'
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    tags = ','.join(sorted({t for m in SQL_TAGS.values() for t in m.values()}))
    frames = []
    for tbl in TABLE_NAMES:
        q = (f'SELECT IndexTime, LoggerTagID, Value FROM {tbl} '
             f"WHERE LoggerTagID IN ({tags}) AND IndexTime >= '{start}' AND IndexTime < '{end}'")
        d = pd.read_sql_query(q, ENGINE)
        if not d.empty:
            print(f'{tbl:<32} {len(d):>9}')
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw['LoggerTagID'] = raw['LoggerTagID'].astype(str)
    raw['IndexTime'] = pd.to_datetime(raw['IndexTime'])
    raw = raw.drop_duplicates(['IndexTime', 'LoggerTagID'], keep='last').sort_values('IndexTime')
    raw.to_parquet(cache)
    return raw


def tag_series(raw, feat, mill):
    tag = SQL_TAGS[feat][mill]
    s = raw.loc[raw['LoggerTagID'] == tag].set_index('IndexTime')['Value'].astype(float).sort_index()
    return s[~s.index.duplicated(keep='last')]


raw = read_raw(START, END)
print(f'{len(raw):,} сурови реда')

# %% [markdown]
# ## Стъпка 0: Профил на записване
#
# - `dt_med`, `dt_p95`, `dt_max` – интервали между записите.
# - `eq_frac` – дял последователни еднакви стойности. ~0 => записва се **само при промяна**
#   (тогава "няма запис" = "без промяна", не "липсва").
# - `max_hold_min` – колко дълго държим последната стойност на 1-мин мрежата.

# %%
def sampling_profile(raw, mill):
    rows = []
    for feat in SQL_TAGS:
        s = tag_series(raw, feat, mill)
        if len(s) < 3:
            rows.append({'feature': feat, 'n': len(s)})
            continue
        dt = s.index.to_series().diff().dt.total_seconds().dropna() / 60
        rows.append({
            'feature': feat, 'n': len(s),
            'first': s.index.min(), 'last': s.index.max(),
            'dt_med': dt.median(), 'dt_p95': dt.quantile(.95), 'dt_p99': dt.quantile(.99),
            'dt_max': dt.max(), 'eq_frac': (s.diff() == 0).mean(), 'n_unique': s.nunique(),
        })
    p = pd.DataFrame(rows).set_index('feature')
    # държим стойността до 3 x p99 на интервала, но поне 5 и най-много 60 мин
    p['max_hold_min'] = (3 * p['dt_p99']).clip(5, 60).round(1)
    p.loc[TARGET, 'max_hold_min'] = PSI_MAX_HOLD_MIN
    return p


profile = sampling_profile(raw, MILL)
print(profile.round(3).to_string())

# %% [markdown]
# ## Стъпка 1: 1-минутна мрежа (sample-and-hold + staleness)

# %%
def to_grid(s, grid, max_hold_min, limits=None):
    """Последна известна стойност към всяка минута; NaN, ако е по-стара от max_hold_min."""
    if limits is not None:
        s = s[(s >= limits[0]) & (s <= limits[1])]
    if s.empty:
        return pd.Series(np.nan, index=grid), pd.Series(np.inf, index=grid)
    t = s.index.values
    pos = np.searchsorted(t, grid.values, side='right') - 1
    has = pos >= 0
    pc = pos.clip(0)
    val = np.where(has, s.values[pc], np.nan)
    age = np.where(has, (grid.values - t[pc]) / np.timedelta64(1, 'm'), np.inf)
    val[age > max_hold_min] = np.nan
    return pd.Series(val, index=grid), pd.Series(age, index=grid)


def build_grid(raw, mill, profile):
    grid = pd.date_range(pd.Timestamp(START), pd.Timestamp(END), freq='1min', inclusive='left')
    g = pd.DataFrame(index=grid)
    g.index.name = 'TimeStamp'
    for feat in SQL_TAGS:
        s = tag_series(raw, feat, mill)
        hold = profile.loc[feat, 'max_hold_min'] if feat in profile.index else 10
        hold = 10 if pd.isna(hold) else hold
        g[feat], g[f'{feat}_age'] = to_grid(s, grid, hold, PHYSICAL_LIMITS.get(feat))
    # кога PSI200 е бил реално прочетен (за flat-проверка по време)
    psi = tag_series(raw, TARGET, mill)
    psi = psi[psi.between(*PHYSICAL_LIMITS[TARGET])]
    chg = psi[psi.diff() != 0]
    _, g['PSI200_unchanged_min'] = to_grid(chg, grid, np.inf)
    return g


g = build_grid(raw, MILL, profile)
print(g[list(SQL_TAGS)].notna().mean().mul(100).round(1).rename('% покритие').to_string())

# %% [markdown]
# ## Стъпки 2-4: състояние, режим, таргет
#
# Всяка минута получава **един** клас (по приоритет):
# `no_data` > `stopped` > `startup` > `abrupt` > `settling` > `bad_target` > `ramp` / `steady`.
# За обучение ползваме само `steady` и `ramp`.

# %%
def smooth(x, minutes):
    return (x.rolling(f'{max(3, minutes // 3)}min', center=True).median()
             .rolling(f'{minutes}min', center=True, min_periods=minutes // 2).mean())


def classify(g):
    c = pd.DataFrame(index=g.index)

    # --- данни ---
    c['no_data'] = g[MODEL_FEATURES].isna().any(axis=1)

    # --- мелницата работи ли (NaN != спряна) ---
    run_cols = [k for k in RUNNING_RULES if k in g]
    known = g[run_cols].notna().all(axis=1)
    running = pd.concat([g[k] > v for k, v in RUNNING_RULES.items() if k in g], axis=1).all(axis=1)
    c['stopped'] = known & ~running
    # startup: STARTUP_BUFFER_MIN минути след последното спиране
    last_stop = pd.Series(np.where(c['stopped'], g.index.values, np.datetime64('NaT')),
                          index=g.index).ffill()
    since_stop = (g.index.to_series() - last_stop).dt.total_seconds() / 60
    c['startup'] = ~c['stopped'] & (since_stop <= STARTUP_BUFFER_MIN)

    # --- скорост на промяна ---
    rates = pd.DataFrame(index=g.index)
    for f, lim in RATE_LIMITS.items():
        sm = smooth(g[f], SMOOTH_MIN)
        rates[f] = (sm - sm.shift(RATE_WIN_MIN)).abs() / lim   # в единици "лимит"
    max_rate = rates.max(axis=1)
    c['abrupt'] = max_rate > 1
    last_abr = pd.Series(np.where(c['abrupt'], g.index.values, np.datetime64('NaT')),
                         index=g.index).ffill()
    since_abr = (g.index.to_series() - last_abr).dt.total_seconds() / 60
    c['settling'] = ~c['abrupt'] & (since_abr <= SETTLE_MIN)
    c['ramp'] = max_rate.between(STEADY_FRAC, 1, inclusive='left')

    # --- таргет ---
    c['bad_target'] = g[TARGET].isna() | (g['PSI200_unchanged_min'] > PSI_FROZEN_MIN)

    order = ['no_data', 'stopped', 'startup', 'abrupt', 'settling', 'bad_target', 'ramp']
    label = pd.Series('steady', index=g.index)
    for k in reversed(order):
        label[c[k]] = k
    return label, rates, c


g['regime'], rates, cls_flags = classify(g)
g['good'] = g['regime'].isin(['steady', 'ramp'])
print((g['regime'].value_counts(normalize=True) * 100).round(1).to_string())

# %% [markdown]
# ### Калибриране на `RATE_LIMITS`
# Квантили на |Δ за RATE_WIN_MIN| (в единиците на сензора) **само докато мелницата работи**.
# Лимитът трябва да е над "нормалното" движение (p95) и под истинските скокове (p99.9).

# %%
run_mask = ~g['regime'].isin(['no_data', 'stopped', 'startup'])
raw_rates = rates.mul(pd.Series(RATE_LIMITS))[run_mask]
q = raw_rates.quantile([.5, .9, .95, .99, .999]).T
q['limit'] = pd.Series(RATE_LIMITS)
q['% над лимита'] = (raw_rates > pd.Series(RATE_LIMITS)).mean() * 100
print(q.round(3).to_string())

# %% [markdown]
# ## Стъпка 5: Сегменти

# %%
def segments(g):
    grp = (g['regime'] != g['regime'].shift()).cumsum()
    seg = g.groupby(grp).agg(start=('regime', lambda s: s.index.min()),
                             end=('regime', lambda s: s.index.max()),
                             regime=('regime', 'first'))
    seg['minutes'] = (seg['end'] - seg['start']).dt.total_seconds() / 60 + 1
    return seg.reset_index(drop=True)


def good_segments(g, min_minutes=MIN_SEGMENT_MIN):
    grp = (g['good'] != g['good'].shift()).cumsum()
    rows = []
    for _, s in g.groupby(grp):
        if not s['good'].iloc[0] or len(s) < min_minutes:
            continue
        rows.append({
            'start': s.index.min(), 'end': s.index.max(), 'hours': len(s) / 60,
            'steady_%': (s['regime'] == 'steady').mean() * 100,
            'PSI200_mean': s[TARGET].mean(), 'PSI200_std': s[TARGET].std(),
            **{f'{f}_mean': s[f].mean() for f in DATASET_FEATURES},
        })
    return pd.DataFrame(rows)


seg_all = segments(g)
seg_good = good_segments(g)
print(f'Добри сегменти (>= {MIN_SEGMENT_MIN} мин): {len(seg_good)},  '
      f'общо {seg_good["hours"].sum():.0f} ч от {len(g) / 60:.0f} ч '
      f'({seg_good["hours"].sum() / (len(g) / 60) * 100:.1f}%)')
print(seg_good.round(2).to_string())

print('\nЗагубено време по причина (часове):')
print((seg_all.groupby('regime')['minutes'].sum() / 60).sort_values(ascending=False).round(1).to_string())

# %% [markdown]
# ## Графики: времева линия на режимите

# %%
COLORS = {'steady': 'tab:green', 'ramp': 'yellowgreen', 'settling': 'gold', 'abrupt': 'tab:red',
          'startup': 'tab:orange', 'stopped': 'black', 'bad_target': 'tab:purple', 'no_data': 'lightgray'}


def plot_timeline(g, start=None, end=None, feats=('Ore', 'WaterZumpf', 'MotorAmp', 'DensityHC')):
    d = g.loc[start:end]
    fig, axes = plt.subplots(len(feats) + 2, 1, figsize=(15, 2 * (len(feats) + 2)), sharex=True)
    axes[0].plot(d.index, d[TARGET], lw=.7)
    axes[0].set_ylabel(TARGET)
    for ax, f in zip(axes[1:], feats):
        ax.plot(d.index, d[f], lw=.6)
        ax.set_ylabel(f)
    codes = pd.Categorical(d['regime'], categories=list(COLORS))
    for i, (k, col) in enumerate(COLORS.items()):
        m = codes == k
        axes[-1].scatter(d.index[m], np.full(m.sum(), i), s=2, color=col, label=k)
    axes[-1].set_yticks(range(len(COLORS)))
    axes[-1].set_yticklabels(list(COLORS), fontsize=7)
    for ax in axes[:-1]:
        for _, r in seg_good.iterrows():
            ax.axvspan(r['start'], r['end'], color='tab:green', alpha=.08)
    for ax in axes:
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


plot_timeline(g)

# %%
# Зуум върху една седмица
plot_timeline(g, START, pd.Timestamp(START) + pd.Timedelta(days=7))

# %% [markdown]
# ## Почасов датасет за модела
# Само часове с >= `HOUR_MIN_GOOD_FRAC` добри минути; средните са **само по добрите минути**.

# %%
def read_ore_quality(start, end):
    q = (f'SELECT "TimeStamp", {", ".join(chr(34) + c + chr(34) for c in ORE_FEATURES)} '
         f'FROM mills.ore_quality WHERE "TimeStamp" >= \'{start}\'::timestamp - interval \'1 day\' '
         f'AND "TimeStamp" < \'{end}\'')
    oq = pd.read_sql_query(q, PG_ENGINE, parse_dates=['TimeStamp'])
    oq['TimeStamp'] += pd.Timedelta(hours=ORE_TIME_OFFSET_H)
    oq[ORE_FEATURES] = oq[ORE_FEATURES].astype(float)
    return oq.sort_values('TimeStamp').drop_duplicates('TimeStamp', keep='last')


def hourly_dataset(g, ore=None):
    x = g[DATASET_FEATURES + [TARGET, 'Power', 'ZumpfLevel']].copy()
    for f, lag in LAGS_MIN.items():
        x[f] = x[f].shift(lag)          # 1-мин мрежа -> shift(lag) = lag минути
    h = x[g['good']].resample('1h').mean()
    h['good_frac'] = g['good'].resample('1h').mean()
    h['steady_frac'] = (g['regime'] == 'steady').resample('1h').mean()
    h = h[h['good_frac'] >= HOUR_MIN_GOOD_FRAC].dropna(subset=MODEL_FEATURES + [TARGET])
    if ore is not None and not ore.empty:
        h = pd.merge_asof(h.reset_index(), ore, on='TimeStamp', direction='backward',
                          tolerance=pd.Timedelta(hours=ORE_MAX_AGE_H)).set_index('TimeStamp')
    return h


ore = read_ore_quality(START, END)
print(f'Проби за състав на рудата: {len(ore)}')
hourly = hourly_dataset(g, ore)
print(f'Почасови редове: {len(hourly)}   с състав на рудата: '
      f'{hourly[ORE_FEATURES].notna().all(axis=1).sum() if set(ORE_FEATURES) <= set(hourly) else 0}')
out = CACHE_DIR / f'hourly_{MILL}_{START}_{END}.parquet'
hourly.to_parquet(out)
seg_good.to_csv(CACHE_DIR / f'segments_{MILL}_{START}_{END}.csv', index=False)
print(f'Записано: {out}')

# %% [markdown]
# ## Стъпка 4: Дрифт и стъпки на сензорите
#
# Бавният дрифт не се хваща от скоростта на промяна. Затова гледаме **дневни медиани**
# само по добрите минути и търсим **точки на промяна** (change points).
#
# - **Директно** – нивото на самия сензор (може да е и реална смяна на режима).
# - **По двойки** – остатък от връзка, която физически трябва да е стабилна
#   (напр. `Power ~ MotorAmp`). Скок в остатъка = сензорът се е изместил,
#   а не режимът, защото режимът мести и двата сензора заедно.
#
# Отклонението е в единици `noise` (робустно дневно σ); `|shift| > 3` е сериозно.

# %%
DRIFT_PAIRS = [            # (y, x): y трябва да се обяснява линейно от x
    ('MotorAmp', 'Power'),
    ('PulpHC', 'DensityHC'),
    ('WaterZumpf', 'PumpRPM'),
    ('ZumpfLevel', 'WaterZumpf'),
]
CP_MIN_DAYS = 5            # минимална дължина на участък между две точки на промяна
CP_MIN_SHIFT = 3.0         # минимален скок в единици noise
REF_DAYS = 14              # референтен период за връзките по двойки (първите N дни)


def daily_median(x, good):
    d = x[good].resample('1D').median()
    n = good.resample('1D').sum()
    return d[n >= 6 * 60]            # поне 6 ч добри данни в деня


def robust_noise(d):
    """Робустно σ на дневния шум (по разликите ден-за-ден, не се влияе от стъпки)."""
    return 1.4826 * d.diff().abs().median() / np.sqrt(2)


def change_points(d, min_size=CP_MIN_DAYS, min_shift=CP_MIN_SHIFT):
    """Бинарна сегментация по медиана: рекурсивно намира най-силния скок в нивото."""
    d = d.dropna()
    noise = robust_noise(d)
    if not np.isfinite(noise) or noise == 0:
        return [], noise
    found = []

    def split(lo, hi):
        seg = d.iloc[lo:hi]
        if len(seg) < 2 * min_size:
            return
        best, best_k = 0.0, None
        for k in range(min_size, len(seg) - min_size + 1):
            shift = abs(seg.iloc[k:].median() - seg.iloc[:k].median()) / noise
            if shift > best:
                best, best_k = shift, k
        if best_k is None or best < min_shift:
            return
        found.append(lo + best_k)
        split(lo, lo + best_k)
        split(lo + best_k, hi)

    split(0, len(d))
    return sorted(found), noise


def describe_cps(d, cps, noise, name, kind):
    rows = []
    bounds = [0] + cps + [len(d)]
    for i, k in enumerate(cps):
        before = d.iloc[bounds[i]:k].median()
        after = d.iloc[k:bounds[i + 2]].median()
        rows.append({'sensor': name, 'kind': kind, 'date': d.index[k].date(),
                     'before': before, 'after': after, 'delta': after - before,
                     'shift_noise': (after - before) / noise})
    return rows


good = g['good']
dm = pd.DataFrame({f: daily_median(g[f], good) for f in SQL_TAGS if f != TARGET}).dropna(how='all')

ref_end = g.index.min() + pd.Timedelta(days=REF_DAYS)
pair_res = pd.DataFrame(index=dm.index)
pair_fit = {}
for y, x in DRIFT_PAIRS:
    ref = g.loc[good & (g.index < ref_end), [x, y]].dropna()
    if len(ref) < 1000:
        continue
    k, b = np.polyfit(ref[x], ref[y], 1)
    pair_fit[f'{y}~{x}'] = (k, b)
    pair_res[f'{y}~{x}'] = daily_median(g[y] - (k * g[x] + b), good)

drift_rows = []
for col in dm.columns:
    d = dm[col].dropna()
    cps, noise = change_points(d)
    drift_rows += describe_cps(d, cps, noise, col, 'level')
for col in pair_res.columns:
    d = pair_res[col].dropna()
    cps, noise = change_points(d)
    drift_rows += describe_cps(d, cps, noise, col, 'pair')

drift = pd.DataFrame(drift_rows)
print('Връзки по двойки (референтни първите', REF_DAYS, 'дни):')
for name, (k, b) in pair_fit.items():
    print(f'  {name}:  y = {k:.4f} * x + {b:.2f}')
print()
if drift.empty:
    print('Няма открити точки на промяна.')
else:
    drift = drift.sort_values(['kind', 'sensor', 'date'])
    print(drift.round(3).to_string(index=False))

# %% [markdown]
# ## Валидация: нива и промени (delta)
#
# Rolling-origin (разширяващ се train, следващият блок е test). Два въпроса:
#
# - **Ниво:** `PSI200(t) ~ X(t)`. Базова линия: persistence `PSI200(t-1)`.
# - **Промяна:** `ΔPSI200 ~ ΔX` в рамките на непрекъснати добри часове.
#   Базова линия: "няма промяна" (Δ=0). Това маха бавните скрити фактори (руда, износване)
#   и отговаря директно на въпроса на оптимизацията: "ако променя X, какво става с PSI200".
#   `R2 > 0` = входовете обясняват част от промените.
#
# Модели: Ridge (линеен, устойчив при малко данни) и XGB (плитък).

# %%
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

N_FOLDS = 5
EXCITATION_WIN_H = 4           # прозорец за оценка колко се движат входовете

FEATURE_SETS = {
    'core': CORE_FEATURES,
    'core+PulpHC': CORE_FEATURES + OPTIONAL_FEATURES,
    'core+PulpHC+ore': CORE_FEATURES + OPTIONAL_FEATURES + ORE_FEATURES,
}


def make_models():
    return {
        'Ridge': make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        'XGB': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                            reg_lambda=5.0, random_state=42, n_jobs=-1),
    }


def delta_frame(h, cols):
    """Разлики час-за-час само между последователни добри часове."""
    contiguous = h.index.to_series().diff() == pd.Timedelta('1h')
    d = h[cols + [TARGET]].diff()[contiguous]
    return d.dropna()


def rolling_eval(h, mode, n_folds=N_FOLDS):
    rows = []
    for fs_name, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in h.columns]
        base = h.dropna(subset=cols + [TARGET]).sort_index()
        if mode == 'delta':
            d = delta_frame(base, cols)
        else:
            d = base[cols + [TARGET]].copy()
            prev = base[TARGET].shift(1)
            ok = base.index.to_series().diff() == pd.Timedelta('1h')
        if len(d) < 25 * (n_folds + 1):
            print(f'  [{mode}] {fs_name}: само {len(d)} реда - пропускам')
            continue
        edges = np.linspace(0, len(d), n_folds + 2).astype(int)
        for i in range(1, n_folds + 1):
            tr, te = d.iloc[:edges[i]], d.iloc[edges[i]:edges[i + 1]]
            y = te[TARGET]
            if mode == 'delta':
                bp = pd.Series(0.0, index=te.index)
                rows.append({'mode': mode, 'features': fs_name, 'model': 'no-change', 'fold': i,
                             'n_test': len(te), 'r2': r2_score(y, bp), 'mae': mean_absolute_error(y, bp)})
            else:
                pm = ok.loc[te.index] & prev.loc[te.index].notna()
                rows.append({'mode': mode, 'features': fs_name, 'model': 'persistence', 'fold': i,
                             'n_test': int(pm.sum()), 'r2': r2_score(y[pm], prev.loc[te.index][pm]),
                             'mae': mean_absolute_error(y[pm], prev.loc[te.index][pm])})
            for m_name, m in make_models().items():
                m.fit(tr[cols], tr[TARGET])
                p = m.predict(te[cols])
                rows.append({'mode': mode, 'features': fs_name, 'model': m_name, 'fold': i,
                             'n_test': len(te), 'r2': r2_score(y, p), 'mae': mean_absolute_error(y, p)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res, res
    summ = res.groupby(['features', 'model'])[['r2', 'mae']].agg(['mean', 'std'])
    return res, summ


for mode in ('level', 'delta'):
    res, summ = rolling_eval(hourly, mode)
    print(f'\n=== {mode.upper()} ===')
    print(summ.round(3).to_string())

# %% [markdown]
# ### Кои входове движат PSI200 (delta-модел, Ridge върху целия период)
# Коефициентът е в единици PSI200 за едно стандартно отклонение на промяната на входа
# (`coef_std`) и в физически единици (`coef_per_unit`). Знакът трябва да е физично логичен.

# %%
cols = [c for c in CORE_FEATURES + OPTIONAL_FEATURES if c in hourly]
d_all = delta_frame(hourly.dropna(subset=cols + [TARGET]), cols)
ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(d_all[cols], d_all[TARGET])
sc, rg = ridge.named_steps['standardscaler'], ridge.named_steps['ridgecv']
coef = pd.DataFrame({'coef_std': rg.coef_, 'coef_per_unit': rg.coef_ / sc.scale_,
                     'corr_dX_dPSI': [d_all[c].corr(d_all[TARGET]) for c in cols]}, index=cols)
print(f'Редове (часови разлики): {len(d_all)},  alpha={rg.alpha_:.3g},  in-sample R2={ridge.score(d_all[cols], d_all[TARGET]):.3f}')
print(coef.sort_values('coef_std', key=abs, ascending=False).round(4).to_string())

# %% [markdown]
# ## Информативност: движат ли се входовете достатъчно?
#
# За всеки час: плъзгащо стандартно отклонение на всеки вход (EXCITATION_WIN_H часа),
# делено на общото стандартно отклонение. `excitation` = средното по входове.
# Сравняваме delta-модела върху "движещи" срещу "спокойни" часове.

# %%
exc_cols = [c for c in CORE_FEATURES + OPTIONAL_FEATURES if c in hourly]
hh = hourly.asfreq('1h')
rel = hh[exc_cols].rolling(EXCITATION_WIN_H, min_periods=EXCITATION_WIN_H // 2).std() / hh[exc_cols].std()
hourly['excitation'] = rel.mean(axis=1).reindex(hourly.index)
print('Разпределение на excitation:')
print(hourly['excitation'].describe().round(3).to_string())

seg_info = []
for _, r in seg_good.iterrows():
    s = hourly.loc[r['start']:r['end']]
    if len(s) < 2:
        continue
    seg_info.append({'start': r['start'], 'end': r['end'], 'hours': r['hours'],
                     'excitation': s['excitation'].mean(),
                     'PSI200_range': s[TARGET].max() - s[TARGET].min(),
                     **{f'{c}_range': s[c].max() - s[c].min() for c in exc_cols}})
seg_info = pd.DataFrame(seg_info).sort_values('excitation', ascending=False)
print('\nНай-информативни сегменти (входовете се движат):')
print(seg_info.head(15).round(2).to_string(index=False))

# delta-моделът (същият rolling CV) оценен отделно върху движещи/спокойни test часове
d = delta_frame(hourly.dropna(subset=cols + [TARGET]), cols)
d['excitation'] = hourly['excitation'].reindex(d.index)
thr = d['excitation'].median()
edges = np.linspace(0, len(d), N_FOLDS + 2).astype(int)
rows = []
for i in range(1, N_FOLDS + 1):
    tr, te = d.iloc[:edges[i]], d.iloc[edges[i]:edges[i + 1]]
    m = make_models()['Ridge'].fit(tr[cols], tr[TARGET])
    te = te.assign(p=m.predict(te[cols]))
    for name, sub in (('движещи', te[te['excitation'] >= thr]), ('спокойни', te[te['excitation'] < thr])):
        if len(sub) > 10:
            rows.append({'subset': name, 'fold': i, 'n': len(sub), 'r2': r2_score(sub[TARGET], sub['p'])})
print('\nDelta Ridge R2 по движещи / спокойни часове:')
if rows:
    print(pd.DataFrame(rows).groupby('subset')[['n', 'r2']].agg(['mean', 'std']).round(3).to_string())
else:
    print('  недостатъчно данни')

# %%
def plot_drift(series_df, title):
    cols = list(series_df.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 1.8 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        d = series_df[c].dropna()
        ax.plot(d.index, d.values, marker='o', ms=2, lw=.8)
        if not drift.empty:
            for dt in drift.loc[drift['sensor'] == c, 'date']:
                ax.axvline(pd.Timestamp(dt), color='r', ls='--', lw=1)
        ax.set_ylabel(c, fontsize=8)
        ax.grid(alpha=.3)
    axes[0].set_title(title)
    plt.tight_layout()
    plt.show()


plot_drift(dm, 'Дневни медиани (добри минути) + точки на промяна')
plot_drift(pair_res, 'Остатъци по двойки - скок = изместен сензор')

# %% [markdown]
# ## Динамичен модел на промените (15 мин)
#
# `ΔPSI200(t) = Σ_i Σ_k b_ik · ΔX_i(t-k)`, k = 0..DYN_LAGS-1 блока по 15 мин.
# Сумата на b_ik по k = **пълен ефект** на единична стъпка в X_i след DYN_LAGS·15 мин;
# кумулативната сума по k = крива на реакцията (step response). Използват се суровите входове
# (без `LAGS_MIN`) - лагът се учи от данните. CI: bootstrap по дни.

# %%
DYN_BLOCK = '15min'
DYN_LAGS = 4                   # 0, 15, 30, 45 мин
DYN_MIN_GOOD = 0.8
DYN_INPUTS = [c for c in CORE_FEATURES + OPTIONAL_FEATURES if c in g]
N_BOOT = 200


def dyn_blocks(g):
    b = g.loc[g['good'], DYN_INPUTS + [TARGET]].resample(DYN_BLOCK).mean()
    gf = g['good'].resample(DYN_BLOCK).mean()
    return b.where(gf >= DYN_MIN_GOOD).asfreq(DYN_BLOCK)      # редовна мрежа, дупките са NaN


def dyn_design(b):
    dX = b[DYN_INPUTS].diff()
    X = pd.concat({f'{c}_l{k}': dX[c].shift(k) for c in DYN_INPUTS for k in range(DYN_LAGS)}, axis=1)
    X[TARGET] = b[TARGET].diff()
    return X                                                   # NaN там, където има дупка


def dyn_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))


def total_effects(model, cols):
    sc, rg = model.named_steps['standardscaler'], model.named_steps['ridgecv']
    per_unit = pd.Series(rg.coef_ / sc.scale_, index=cols)
    return pd.DataFrame({c: [per_unit[f'{c}_l{k}'] for k in range(DYN_LAGS)] for c in DYN_INPUTS},
                        index=[f'{k * 15}мин' for k in range(DYN_LAGS)])


blocks = dyn_blocks(g)
X_full = dyn_design(blocks)
D = X_full.dropna()
feat_cols = [c for c in D.columns if c != TARGET]
print(f'15-мин редове за динамичния модел: {len(D)}')

# rolling CV
edges = np.linspace(0, len(D), N_FOLDS + 2).astype(int)
rows = []
for i in range(1, N_FOLDS + 1):
    tr, te = D.iloc[:edges[i]], D.iloc[edges[i]:edges[i + 1]]
    m = dyn_ridge().fit(tr[feat_cols], tr[TARGET])
    p = m.predict(te[feat_cols])
    rows.append({'fold': i, 'n': len(te), 'r2_model': r2_score(te[TARGET], p),
                 'r2_no_change': r2_score(te[TARGET], np.zeros(len(te))),
                 'mae_model': mean_absolute_error(te[TARGET], p),
                 'mae_no_change': te[TARGET].abs().mean()})
dyn_cv = pd.DataFrame(rows)
print(dyn_cv.round(3).to_string(index=False))
print(f'Средно R2: {dyn_cv["r2_model"].mean():.3f}  (no-change {dyn_cv["r2_no_change"].mean():.3f})')

# пълен ефект + bootstrap CI по дни
dyn_model = dyn_ridge().fit(D[feat_cols], D[TARGET])
resp = total_effects(dyn_model, feat_cols).cumsum()
days = D.index.normalize()
uniq = days.unique()
rng = np.random.default_rng(42)
boot = []
for _ in range(N_BOOT):
    pick = rng.choice(uniq, len(uniq), replace=True)
    Db = pd.concat([D[days == d] for d in pick])
    boot.append(total_effects(dyn_ridge().fit(Db[feat_cols], Db[TARGET]), feat_cols).sum())
boot = pd.DataFrame(boot)

unit_step = {'Ore': 10, 'WaterMill': 2, 'WaterZumpf': 20, 'PressureHC': 0.02,
             'DensityHC': 20, 'MotorAmp': 5, 'PulpHC': 20}
eff = pd.DataFrame({
    'step': pd.Series(unit_step).reindex(DYN_INPUTS),
    'effect_per_unit': resp.iloc[-1],
    'ci_lo': boot.quantile(0.05), 'ci_hi': boot.quantile(0.95),
    'sign_stable_%': (np.sign(boot) == np.sign(resp.iloc[-1])).mean() * 100,
})
eff['effect_per_step'] = eff['effect_per_unit'] * eff['step']
eff['ci_step_lo'] = eff['ci_lo'] * eff['step']
eff['ci_step_hi'] = eff['ci_hi'] * eff['step']
print(f'\nПълен ефект след {DYN_LAGS * 15} мин (90% CI, bootstrap по дни):')
print(eff[['step', 'effect_per_step', 'ci_step_lo', 'ci_step_hi', 'sign_stable_%']].round(3).to_string())
print('\nКрива на реакцията (кумулативен ефект на стъпката step):')
print((resp * eff['step']).round(3).to_string())

# %% [markdown]
# ## Проверка върху реални стъпки (out-of-sample)
#
# Събитие за вход i: за STEP_H блока |ΔX_i| е в горните 10%, а всички останали
# входове са под медианата си (спокойни). Предсказаната промяна = сума на прогнозите
# на модел, обучен **без седмицата на събитието** (GroupKFold по седмици).

# %%
from sklearn.model_selection import GroupKFold

STEP_H = 4                     # 60 мин след началото на стъпката

weeks = D.index.to_period('W').astype(str)
oos = pd.Series(np.nan, index=D.index)
for tr_i, te_i in GroupKFold(n_splits=min(5, weeks.nunique())).split(D, groups=weeks):
    m = dyn_ridge().fit(D.iloc[tr_i][feat_cols], D.iloc[tr_i][TARGET])
    oos.iloc[te_i] = m.predict(D.iloc[te_i][feat_cols])
oos = oos.reindex(blocks.index)

dXH = blocks[DYN_INPUTS].shift(-STEP_H) - blocks[DYN_INPUTS]
dYH = blocks[TARGET].shift(-STEP_H) - blocks[TARGET]
predH = oos[::-1].rolling(STEP_H, min_periods=STEP_H).sum()[::-1].shift(-1)   # Σ oos(t+1..t+H)
q_hi = dXH.abs().quantile(0.9)
q_med = dXH.abs().median()

ev_rows = []
for c in DYN_INPUTS:
    others = [o for o in DYN_INPUTS if o != c]
    mask = (dXH[c].abs() >= q_hi[c]) & (dXH[others].abs() < q_med[others]).all(axis=1) \
        & dYH.notna() & predH.notna()
    idx, last = [], None
    for t in dXH.index[mask]:
        if last is None or t - last >= pd.Timedelta(minutes=15 * STEP_H):
            idx.append(t)
            last = t
    if len(idx) < 5:
        ev_rows.append({'input': c, 'n_events': len(idx)})
        continue
    a, p = dYH.loc[idx], predH.loc[idx]
    slope = np.polyfit(p, a, 1)[0] if p.std() > 0 else np.nan
    ev_rows.append({'input': c, 'n_events': len(idx),
                    'mean_dX': dXH.loc[idx, c].abs().mean(),
                    'sign_agree_%': (np.sign(a) == np.sign(p)).mean() * 100,
                    'corr': a.corr(p), 'slope_actual_vs_pred': slope,
                    'mae_model': (a - p).abs().mean(), 'mae_no_change': a.abs().mean()})
events = pd.DataFrame(ev_rows)
print(f'Стъпкови събития (хоризонт {STEP_H * 15} мин, out-of-sample по седмици):')
print(events.round(3).to_string(index=False))
