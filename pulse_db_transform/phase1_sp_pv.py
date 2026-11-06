"""Фаза 1: колко точно и колко бързо измерените стойности (PV) следват сетпойнтите (SP)."""
import numpy as np
import pandas as pd

from mill_data import CACHE_DIR, MILLS

START, END = '2026-01-01', '2026-10-15'
LOOPS = ['Ore', 'WaterMill', 'WaterZumpf']
MIN_STEP = {'Ore': 5, 'WaterMill': 1, 'WaterZumpf': 10}   # минимална стъпка на SP
QUIET_MIN = 30                                             # без друга промяна на SP толкова минути
TOL_FRAC = 0.05                                            # |PV - SP| > 5% от SP = не следва


def tracking(df, loop):
    run = df[~df['regime'].isin(['stopped', 'startup', 'repair'])]
    err = (run[loop] - run[f'{loop}_SP']).dropna()
    sp = run.loc[err.index, f'{loop}_SP']
    return {'median_abs_err': err.abs().median(), 'p95_abs_err': err.abs().quantile(.95),
            'bias': err.median(), '% извън 5%': (err.abs() > TOL_FRAC * sp.abs()).mean() * 100}


def step_responses(df, loop):
    """Изолирани стъпки на SP: време (мин) до 63% и 90% от стъпката в PV."""
    sp, pv = df[f'{loop}_SP'], df[loop]
    change = sp.diff().fillna(0)
    times = change.index[change.abs() >= MIN_STEP[loop]]
    rows = []
    for t in times:
        win = slice(t, t + pd.Timedelta(minutes=QUIET_MIN))
        if (change[win].iloc[1:] != 0).any() or not df.loc[win, 'good'].all():
            continue
        p = pv[win]
        if p.isna().any() or len(p) < QUIET_MIN:
            continue
        step, start = change[t], pv.shift(1)[t]
        frac = (p - start) / step
        t63 = (frac >= 0.63).idxmax() if (frac >= 0.63).any() else pd.NaT
        t90 = (frac >= 0.90).idxmax() if (frac >= 0.90).any() else pd.NaT
        rows.append({'step': step, 't63': (t63 - t).total_seconds() / 60 if t63 is not pd.NaT else np.nan,
                     't90': (t90 - t).total_seconds() / 60 if t90 is not pd.NaT else np.nan,
                     'reached_%': frac.iloc[-5:].mean() * 100})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    for mill in MILLS:
        df = pd.read_parquet(CACHE_DIR / f'mill_{mill}_{START}_{END}.parquet')
        rows = {}
        for loop in LOOPS:
            r = step_responses(df, loop)
            rows[loop] = {**tracking(df, loop), 'n_steps': len(r),
                          't63_med': r['t63'].median() if len(r) else np.nan,
                          't90_med': r['t90'].median() if len(r) else np.nan,
                          'reached_%_med': r['reached_%'].median() if len(r) else np.nan}
        print(f'\n===== {mill} =====')
        print(pd.DataFrame(rows).T.round(2).to_string())
