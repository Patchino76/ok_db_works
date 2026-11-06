"""Фаза 2: управляващи (Ore, WaterMill, WaterZumpf) -> контролирани (DensityHC, PulpHC, PressureHC, PumpRPM).

Модел на промените на 15 мин: dCV(t) = сума по лагове k=0..3 на b_k * dMV(t-k).
Проверка: групово по кампании + симулация 60 мин напред спрямо "без промяна".
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mill_data import CACHE_DIR, MILLS, resample_good

START, END = '2026-01-01', '2026-10-15'
MVS = ['Ore', 'WaterMill', 'WaterZumpf']
CVS = ['DensityHC', 'PulpHC', 'PressureHC', 'PumpRPM']
LAGS = range(4)                    # 0, 15, 30, 45 мин
HORIZON = 4                        # 4 x 15 мин = 60 мин
STEPS = {'Ore': 10, 'WaterMill': 2, 'WaterZumpf': 20}


def make_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def build(df, use_sp=False):
    """15-мин мрежа с промени на MV (с лагове) и CV. use_sp: входове от *_SP."""
    h = resample_good(df, '15min').asfreq('15min')
    d = h[MVS + CVS + [f'{mv}_SP' for mv in MVS]].diff()
    src = {mv: f'{mv}_SP' if use_sp else mv for mv in MVS}
    X = pd.concat({f'd{mv}_l{k}': d[src[mv]].shift(k) for mv in MVS for k in LAGS}, axis=1)
    return h, d, X


def fit_cv(h, d, X, cv):
    """Предсказания извън извадката (по кампании), симулационен skill и усилвания."""
    y = d[cv]
    ok = X.notna().all(axis=1) & y.notna()
    Xo, yo, groups = X[ok], y[ok], h.loc[ok, 'campaign_id']

    pred = pd.Series(np.nan, index=X.index)
    for tr, te in GroupKFold(n_splits=groups.nunique()).split(Xo, yo, groups):
        m = make_model().fit(Xo.iloc[tr], yo.iloc[tr])
        pred.loc[Xo.index[te]] = m.predict(Xo.iloc[te])

    r2 = 1 - ((yo - pred[ok]) ** 2).sum() / ((yo - yo.mean()) ** 2).sum()

    # симулация: y(t+H) ~ y(t) + сума на предсказаните промени (t+1..t+H)
    sim_sum = sum(pred.shift(-k) for k in range(1, HORIZON + 1))
    true_chg = h[cv].shift(-HORIZON) - h[cv]
    v = sim_sum.notna() & true_chg.notna()
    skill = 1 - ((true_chg[v] - sim_sum[v]) ** 2).mean() / (true_chg[v] ** 2).mean()

    m = make_model().fit(Xo, yo)
    coef = m[-1].coef_ / m[0].scale_
    gains = {mv: coef[[i for i, c in enumerate(Xo.columns) if c.startswith(f'd{mv}_')]].sum() * STEPS[mv]
             for mv in MVS}
    return {'n': int(ok.sum()), 'R2_15min': r2, 'skill_60min': skill, 'n_sim': int(v.sum()),
            **{f'{mv} +{STEPS[mv]}': g for mv, g in gains.items()}}


def sp_step_events(df, mv='WaterZumpf', min_step=10, quiet_min=60,
                   cols=('DensityHC', 'ZumpfLevel', 'PumpRPM', 'PulpHC', 'Ore', 'WaterMill', 'WaterZumpf')):
    """Средна реакция (на +20 m3/h) след изолирани стъпки на SP: само тази стъпка за quiet_min."""
    sp = df[f'{mv}_SP']
    chg = sp.diff().fillna(0)
    others = (df[[f'{m}_SP' for m in MVS if m != mv]].diff().abs() > 0).any(axis=1)
    rows = []
    for t in chg.index[chg.abs() >= min_step]:
        win = df.loc[t - pd.Timedelta(minutes=5): t + pd.Timedelta(minutes=quiet_min)]
        if len(win) < quiet_min or not win['good'].all():
            continue
        if (chg[t:t + pd.Timedelta(minutes=quiet_min)].iloc[1:] != 0).any() or others[win.index].any():
            continue
        base = win[list(cols)].iloc[:5].mean()
        resp = {m: (df.loc[t + pd.Timedelta(minutes=m), list(cols)] - base) * 20 / chg[t]
                for m in (5, 15, 30, 45, 60)}
        rows.append(pd.concat(resp, names=['min', 'col']))
    if not rows:
        return None, 0
    return pd.concat(rows, axis=1).median(axis=1).unstack('col')[list(cols)], len(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    for mill in MILLS:
        df = pd.read_parquet(CACHE_DIR / f'mill_{mill}_{START}_{END}.parquet')
        h, d, X = build(df)
        res = pd.DataFrame({cv: fit_cv(h, d, X, cv) for cv in CVS}).T
        print(f'\n===== {mill}: {h["good_frac"].notna().sum()} добри 15-мин интервала =====')
        print(res.round(3).to_string())
        _, _, Xsp = build(df, use_sp=True)
        print('--- входове = SP ---')
        print(pd.DataFrame({cv: fit_cv(h, d, Xsp, cv) for cv in CVS}).T.round(3).to_string())
        print('DensityHC - ефекти по кампании:')
        for c, idx in h.groupby('campaign_id').groups.items():
            sel = h.index.isin(idx)
            try:
                r = fit_cv(h[sel].assign(campaign_id=h.loc[sel].index.month), d[sel], X[sel], 'DensityHC')
                print(f'  кампания {c}: n={r["n"]}, ' + ', '.join(f'{k}: {r[k]:.1f}' for k in r if '+' in k))
            except ValueError:
                pass
        ev, n = sp_step_events(df)
        print(f'Изолирани стъпки WaterZumpf_SP: {n}; медиана на реакцията на +20 m3/h:')
        print(ev.round(2).to_string() if ev is not None else '  няма')
