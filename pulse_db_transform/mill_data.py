"""Основен слой за данни на мелниците: SQL -> pandas DataFrame с 1-минутен DatetimeIndex.

    df = load_mill('Mill06', '2026-01-01', '2026-10-15')
    h  = resample_good(df, '1h')

Колони на df:
    измерени (PV):  Ore, WaterMill, WaterZumpf, Power, ZumpfLevel, PressureHC, DensityHC,
                    PulpHC, PumpRPM, MotorAmp, PSI200
    сетпойнти (SP): Ore_SP, WaterMill_SP, WaterZumpf_SP
    руда:           Class_15, Class_12, Grano, Daiki, Shisti
    ремонти:        days_since_cylinder, days_since_feed_cover, days_since_discharge_cover,
                    nozzle_mm, days_since_nozzle, campaign_id
    качество:       regime, good
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ----------------------------------------------------------------------------- конфигурация
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / 'cache'
LINERS_CSV = ROOT.parent / 'tmp' / 'obliczovki_melnitsi.csv'
NOZZLES_CSV = ROOT.parent / 'tmp' / 'hidrociklony_dyuzi_iztok.csv'

MSSQL = create_engine(
    'mssql+pyodbc:///?odbc_connect='
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.20.2.10;'
    'DATABASE=pulse;UID=Pulse_RO;PWD=PD@T@r3@der'
)
PG = create_engine(
    'postgresql://s.lyubenov:tP9uB7sH7mK6zA7t@em-m-db4.ellatzite-med.com:5432/em_pulse_data'
)

MILLS = ['Mill06', 'Mill08']   # Mill07 - временно изключена (технологични промени)

PV_TAGS = {
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
SP_TAGS = {
    'Ore_SP':        {'Mill06': '2775', 'Mill07': '2835', 'Mill08': '1823'},
    'WaterMill_SP':  {'Mill06': '2787', 'Mill07': '2847', 'Mill08': '1817'},
    'WaterZumpf_SP': {'Mill06': '2790', 'Mill07': '2850', 'Mill08': '1820'},
}
TAGS = {**PV_TAGS, **SP_TAGS}

TABLES = ['LoggerValues'] + [
    f'LoggerValues_Archive_{m}{y}' for y in (2026, 2025)
    for m in ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    if not (y == 2026 and m in ('Oct', 'Nov', 'Dec'))
]

TARGET = 'PSI200'
MODEL_FEATURES = ['Ore', 'WaterMill', 'WaterZumpf', 'PressureHC', 'DensityHC', 'MotorAmp', 'PulpHC']
ORE_FEATURES = ['Class_15', 'Class_12', 'Grano', 'Daiki', 'Shisti']

# невалидни показания -> NaN
PHYSICAL_LIMITS = {
    'Ore': (0, 400), 'WaterMill': (5, 100), 'WaterZumpf': (0, 600), 'Power': (0, 4000),
    'ZumpfLevel': (0, 3000), 'PressureHC': (0, 2), 'DensityHC': (1000, 1900),
    'PulpHC': (0, 1200), 'PumpRPM': (0, 1000), 'MotorAmp': (0, 500), 'PSI200': (16, 30),
}
# по мелници: под тези стойности сензорът е задръстен / байпас
PHYSICAL_LIMITS_BY_MILL = {
    'Mill06': {'WaterZumpf': (140, 600)},
    'Mill08': {'WaterZumpf': (140, 600)},
}
# технологични граници (mills-parameters.ts, Ore max 200, FE 2-4, PulpHC min 300) - за оптимизатора
TECH_LIMITS = {
    'Ore': (140, 200), 'WaterMill': (5, 25), 'WaterZumpf': (140, 250), 'PulpHC': (300, 600),
    'DensityHC': (1500, 1900), 'PressureHC': (0.2, 0.5), 'PumpRPM': (0, 800),
    'MotorAmp': (150, 250), 'FE': (2, 4), 'PSI200': (16, 30),
}

PSI_MAX_HOLD_MIN = 25          # PSI200 се чете на ~11 мин
PSI_FROZEN_MIN = 45            # без промяна по-дълго -> анализаторът е замръзнал
ORE_OFFSET_H = 1               # пробата на смяната е валидна от 07/15/23
ORE_MAX_AGE_H = 9

RUNNING_RULES = {'Ore': 100, 'Power': 1000, 'MotorAmp': 100}
STARTUP_MIN = 60
SMOOTH_MIN = 15
RATE_WIN_MIN = 10
RATE_LIMITS = {'Ore': 20, 'WaterMill': 4, 'WaterZumpf': 40,
               'MotorAmp': 6, 'DensityHC': 60, 'PressureHC': 0.03}
RATE_LIMITS_BY_MILL = {'Mill08': {'MotorAmp': 10}}
STEADY_FRAC = 0.3
SETTLE_MIN = 30

LINER_TYPES = {'Цилиндър': 'cylinder', 'Капак захранващ': 'feed_cover',
               'Капак разтоварващ': 'discharge_cover'}

REGIME_ORDER = ['repair', 'no_data', 'stopped', 'startup', 'abrupt', 'settling', 'bad_target', 'ramp']


# ----------------------------------------------------------------------------- SQL
def read_raw(start, end, use_cache=True):
    """Long DataFrame [IndexTime, LoggerTagID, Value] за всички PV и SP тагове."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f'raw_pvsp_{start}_{end}.parquet'
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    tags = ','.join(sorted({t for m in TAGS.values() for t in m.values()}))
    frames = []
    for tbl in TABLES:
        q = (f'SELECT IndexTime, LoggerTagID, Value FROM {tbl} '
             f"WHERE LoggerTagID IN ({tags}) AND IndexTime >= '{start}' AND IndexTime < '{end}'")
        d = pd.read_sql_query(q, MSSQL)
        if not d.empty:
            frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw['LoggerTagID'] = raw['LoggerTagID'].astype(str)
    raw['IndexTime'] = pd.to_datetime(raw['IndexTime'])
    raw = raw.drop_duplicates(['IndexTime', 'LoggerTagID'], keep='last').sort_values('IndexTime')
    raw.to_parquet(cache)
    return raw


def read_ore_quality(start, end):
    cols = ', '.join(f'"{c}"' for c in ORE_FEATURES)
    q = (f'SELECT "TimeStamp", {cols} FROM mills.ore_quality '
         f"WHERE \"TimeStamp\" >= '{start}'::timestamp - interval '1 day' AND \"TimeStamp\" < '{end}'")
    oq = pd.read_sql_query(q, PG, parse_dates=['TimeStamp'])
    oq['TimeStamp'] += pd.Timedelta(hours=ORE_OFFSET_H)
    oq[ORE_FEATURES] = oq[ORE_FEATURES].astype(float)
    return oq.sort_values('TimeStamp').drop_duplicates('TimeStamp', keep='last')


# ----------------------------------------------------------------------------- 1-мин мрежа
def tag_series(raw, tag):
    s = raw.loc[raw['LoggerTagID'] == tag].set_index('IndexTime')['Value'].astype(float).sort_index()
    return s[~s.index.duplicated(keep='last')]


def hold_minutes(s):
    """Колко дълго държим последната стойност: 3 x p99 на интервала, в [5, 60] мин."""
    if len(s) < 3:
        return 10
    dt = s.index.to_series().diff().dt.total_seconds().dropna() / 60
    return float(np.clip(3 * dt.quantile(.99), 5, 60))


def to_grid(s, grid, max_hold_min):
    """Последна известна стойност към всяка минута; NaN, ако е по-стара от max_hold_min."""
    if s.empty:
        return pd.Series(np.nan, index=grid)
    t = s.index.values
    pos = np.searchsorted(t, grid.values, side='right') - 1
    pc = pos.clip(0)
    age = np.where(pos >= 0, (grid.values - t[pc]) / np.timedelta64(1, 'm'), np.inf)
    val = np.where(age <= max_hold_min, s.values[pc], np.nan)
    return pd.Series(val, index=grid)


def build_grid(raw, mill, start, end):
    grid = pd.date_range(start, end, freq='1min', inclusive='left', name='TimeStamp')
    df = pd.DataFrame(index=grid)
    limits = {**PHYSICAL_LIMITS, **PHYSICAL_LIMITS_BY_MILL.get(mill, {})}
    for col, tags in PV_TAGS.items():
        s = tag_series(raw, tags[mill])
        lo, hi = limits[col]
        s = s[s.between(lo, hi)]
        hold = PSI_MAX_HOLD_MIN if col == TARGET else hold_minutes(s)
        df[col] = to_grid(s, grid, hold)
    for col, tags in SP_TAGS.items():
        df[col] = to_grid(tag_series(raw, tags[mill]), grid, np.inf)   # SP се записва при промяна
    psi = tag_series(raw, PV_TAGS[TARGET][mill])
    psi = psi[psi.between(*PHYSICAL_LIMITS[TARGET])]
    changes = psi[psi.diff() != 0]
    last_change = pd.Series(changes.index, index=changes.index).reindex(grid, method='ffill')
    df['psi_unchanged_min'] = (grid - last_change).dt.total_seconds().values / 60
    return df


# ----------------------------------------------------------------------------- ремонти
def read_liners(mill):
    d = pd.read_csv(LINERS_CSV, sep=';', dtype=str)
    d.columns = ['mill', 'kind', 'date', 'next']
    d = d[d['mill'] == str(int(mill[-2:]))]
    d['kind'] = d['kind'].str.split(' - ').str[0].map(LINER_TYPES)
    d['date'] = pd.to_datetime(d['date'], format='%d.%m.%Y')
    return d[['kind', 'date']]


def read_nozzles(mill):
    d = pd.read_csv(NOZZLES_CSV, sep=';', dtype=str)
    d.columns = ['mill', 'date', 'nozzle_mm']
    d = d[d['mill'] == str(int(mill[-2:]))]
    d['date'] = pd.to_datetime(d['date'], format='%d.%m.%Y')
    d['nozzle_mm'] = pd.to_numeric(d['nozzle_mm'], errors='coerce')
    return d[['date', 'nozzle_mm']].sort_values('date')


def days_since(index, dates):
    """Дни от последното събитие до всяка минута (NaN преди първото)."""
    dates = pd.DatetimeIndex(sorted(dates))
    last = pd.Series(dates, index=dates).reindex(index, method='ffill')
    return (index - last).dt.total_seconds().values / 86400


def add_repairs(df, mill):
    liners, nozzles = read_liners(mill), read_nozzles(mill)
    idx = df.index
    for kind in LINER_TYPES.values():
        df[f'days_since_{kind}'] = days_since(idx, liners.loc[liners['kind'] == kind, 'date'])
    df['nozzle_mm'] = nozzles.set_index('date')['nozzle_mm'].reindex(idx, method='ffill').values
    df['days_since_nozzle'] = days_since(idx, nozzles['date'])
    events = pd.DatetimeIndex(sorted(set(liners['date']) | set(nozzles['date'])))
    df['campaign_id'] = np.searchsorted(events.values, idx.values, side='right')
    df['repair_day'] = idx.normalize().isin(events)                 # цялото денонощие = престой
    return df


# ----------------------------------------------------------------------------- режим
def smooth(x, minutes):
    return (x.rolling(f'{max(3, minutes // 3)}min', center=True).median()
             .rolling(f'{minutes}min', center=True, min_periods=minutes // 2).mean())


def minutes_since(flag):
    last = pd.Series(np.where(flag, flag.index.values, np.datetime64('NaT')), index=flag.index).ffill()
    return (flag.index.to_series() - last).dt.total_seconds() / 60


def classify(df, mill):
    """Един клас за минута по приоритет REGIME_ORDER; за обучение - само steady и ramp."""
    f = pd.DataFrame(index=df.index)
    f['repair'] = df['repair_day']
    f['no_data'] = df[MODEL_FEATURES].isna().any(axis=1)

    known = df[list(RUNNING_RULES)].notna().all(axis=1)
    running = pd.concat([df[k] > v for k, v in RUNNING_RULES.items()], axis=1).all(axis=1)
    f['stopped'] = known & ~running
    f['startup'] = ~f['stopped'] & (minutes_since(f['stopped']) <= STARTUP_MIN)

    limits = {**RATE_LIMITS, **RATE_LIMITS_BY_MILL.get(mill, {})}
    rates = pd.concat({c: (smooth(df[c], SMOOTH_MIN).diff(RATE_WIN_MIN).abs() / lim)
                       for c, lim in limits.items()}, axis=1).max(axis=1)
    f['abrupt'] = rates > 1
    f['settling'] = ~f['abrupt'] & (minutes_since(f['abrupt']) <= SETTLE_MIN)
    f['ramp'] = rates.between(STEADY_FRAC, 1, inclusive='left')
    f['bad_target'] = df[TARGET].isna() | (df['psi_unchanged_min'] > PSI_FROZEN_MIN)

    regime = pd.Series('steady', index=df.index)
    for k in reversed(REGIME_ORDER):
        regime[f[k]] = k
    return regime


# ----------------------------------------------------------------------------- публично API
def load_mill(mill, start, end, raw=None, ore=None):
    """1-минутен DataFrame за една мелница: PV, SP, руда, ремонти, regime/good."""
    raw = read_raw(start, end) if raw is None else raw
    ore = read_ore_quality(start, end) if ore is None else ore
    df = build_grid(raw, mill, start, end)
    df = pd.merge_asof(df.reset_index(), ore, on='TimeStamp', direction='backward',
                       tolerance=pd.Timedelta(hours=ORE_MAX_AGE_H)).set_index('TimeStamp')
    df = add_repairs(df, mill)
    df['regime'] = classify(df, mill)
    df['good'] = df['regime'].isin(['steady', 'ramp'])
    return df.drop(columns=['psi_unchanged_min', 'repair_day'])


def load_mills(mills, start, end):
    """Общ DataFrame за няколко мелници, с колона 'mill'."""
    raw, ore = read_raw(start, end), read_ore_quality(start, end)
    return pd.concat([load_mill(m, start, end, raw, ore).assign(mill=m) for m in mills])


def resample_good(df, freq='1h', min_good_frac=0.8):
    """Средни само по добрите минути; периоди с < min_good_frac добри минути отпадат."""
    num = df.select_dtypes('number').columns.drop('campaign_id', errors='ignore')
    out = df.loc[df['good'], num].resample(freq).mean()
    out['good_frac'] = df['good'].resample(freq).mean()
    out['campaign_id'] = df['campaign_id'].resample(freq).last()
    return out[out['good_frac'] >= min_good_frac]


def limits_report(df, freq='MS'):
    """% от минутите на работа (без stopped/startup/repair) под и над TECH_LIMITS."""
    run = df[~df['regime'].isin(['stopped', 'startup', 'repair'])]
    cols = [c for c in TECH_LIMITS if c in run]
    low = pd.DataFrame({c: run[c] < TECH_LIMITS[c][0] for c in cols})
    high = pd.DataFrame({c: run[c] > TECH_LIMITS[c][1] for c in cols})
    valid = run[cols].notna()
    total = pd.DataFrame({'min': [TECH_LIMITS[c][0] for c in cols],
                          'max': [TECH_LIMITS[c][1] for c in cols],
                          'p01': run[cols].quantile(.01).values,
                          'p50': run[cols].median().values,
                          'p99': run[cols].quantile(.99).values,
                          '% под': (low.sum() / valid.sum() * 100).values,
                          '% над': (high.sum() / valid.sum() * 100).values}, index=cols)
    monthly = ((low | high).resample(freq).sum() / valid.resample(freq).sum() * 100)
    return total, monthly


# ----------------------------------------------------------------------------- проверка
if __name__ == '__main__':
    START, END = '2026-01-01', '2026-10-15'
    raw, ore = read_raw(START, END), read_ore_quality(START, END)
    for mill in MILLS:
        df = load_mill(mill, START, END, raw, ore)
        df.to_parquet(CACHE_DIR / f'mill_{mill}_{START}_{END}.parquet')
        print(f'\n===== {mill}: {len(df):,} мин, добри {df["good"].mean() * 100:.1f}% =====')
        print('Покритие %:', df[list(TAGS)].notna().mean().mul(100).round(1).to_dict())
        print('Режими %:', df['regime'].value_counts(normalize=True).mul(100).round(1).to_dict())
        sp_err = {c: (df.loc[df['good'], c] - df.loc[df['good'], f'{c}_SP']).abs().median()
                  for c in ['Ore', 'WaterMill', 'WaterZumpf']}
        print('Медиана |PV - SP| (добри):', {k: round(v, 2) for k, v in sp_err.items()})
        camp = df.groupby('campaign_id').agg(start=('good', lambda s: s.index.min()),
                                             nozzle_mm=('nozzle_mm', 'last'),
                                             good_h=('good', lambda s: s.sum() / 60))
        print(camp.round(1).to_string())
