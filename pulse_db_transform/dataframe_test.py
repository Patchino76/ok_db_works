"""
dataframe_test.py - ЕКСПЕРИМЕНТАЛЕН standalone файл за оглед на данните.

ИДЕЯТА
------
PSI200 се записва на неравни интервали (~10 мин), а останалите сензори се
архивират по-често. Вместо да разтягаме всичко до 1 час (както прави
pulse_to_postgresql.py), тук правим "event-based" dataset:

    * един ред = едно четене на PSI200  -> това е TARGET-ът
    * всяка feature = средна стойност на сензора в интервала между
      ПРЕДИШНОТО и ТЕКУЩОТО четене на PSI200

Така target-ът не се изглажда (няма ffill + resample) и features съвпадат
с физическия интервал, който PSI200 измерва.

PSI80 е АЛТЕРНАТИВЕН target - записва се като колона, но НЕ влиза в features.

ЗА ВСЯКА FEATURE ПАЗИМ 4 КОЛОНИ
-------------------------------
    <feature>_mean    - средна стойност в прозореца
    <feature>_std     - стандартно отклонение (динамиката в прозореца)
    <feature>_n       - брой сурови проби в прозореца
    <feature>_frozen  - True ако има >1 проба, но всички са еднакви
                        (замръзнал/мъртъв сензор)

УПОТРЕБА
--------
    python dataframe_test.py --mill Mill06 --start 2026-01-01
    python dataframe_test.py --all-mills --start 2026-01-01
    python dataframe_test.py --mill Mill07 --start 2026-06-01 --lag 10 --max-gap 30
"""

import argparse

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# КОНФИГУРАЦИЯ - копирана от pulse_to_postgresql.py (PulseDBTransformer.__init__)
# ---------------------------------------------------------------------------

SERVER = '10.20.2.10'
DATABASE = 'pulse'
USERNAME = 'Pulse_RO'
PASSWORD = 'PD@T@r3@der'
CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};"
    f"DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"
)

# feature -> { MillName: TagID }
SQL_TAGS = {
    'Ore': {"Mill01": "485", "Mill02": "488", "Mill03": "491", "Mill04": "494", "Mill05": "497", "Mill06": "500",
            "Mill07": "455", "Mill08": "467", "Mill09": "476", "Mill10": "479", "Mill11": "482", "Mill12": "3786"},

    'WaterMill': {"Mill01": "560", "Mill02": "580", "Mill03": "530", "Mill04": "1227", "Mill05": "1237", "Mill06": "1219",
                  "Mill07": "461", "Mill08": "474", "Mill09": "509", "Mill10": "517", "Mill11": "522", "Mill12": "3790"},

    'WaterZumpf': {"Mill01": "561", "Mill02": "581", "Mill03": "531", "Mill04": "1228", "Mill05": "1238", "Mill06": "1220",
                   "Mill07": "462", "Mill08": "475", "Mill09": "510", "Mill10": "518", "Mill11": "523", "Mill12": "3792"},

    'Power': {"Mill01": "487", "Mill02": "490", "Mill03": "493", "Mill04": "496", "Mill05": "499", "Mill06": "502",
              "Mill07": "460", "Mill08": "471", "Mill09": "478", "Mill10": "481", "Mill11": "483", "Mill12": "3773"},

    'ZumpfLevel': {"Mill01": "486", "Mill02": "489", "Mill03": "492", "Mill04": "495", "Mill05": "498", "Mill06": "501",
                   "Mill07": "458", "Mill08": "470", "Mill09": "477", "Mill10": "480", "Mill11": "484", "Mill12": "3747"},

    'PressureHC': {"Mill01": "558", "Mill02": "578", "Mill03": "528", "Mill04": "1225", "Mill05": "1235", "Mill06": "1217",
                   "Mill07": "459", "Mill08": "472", "Mill09": "507", "Mill10": "515", "Mill11": "2687", "Mill12": "3774"},

    'DensityHC': {"Mill01": "557", "Mill02": "577", "Mill03": "527", "Mill04": "1224", "Mill05": "1234", "Mill06": "1216",
                  "Mill07": "457", "Mill08": "469", "Mill09": "506", "Mill10": "514", "Mill11": "2658", "Mill12": "3742"},

    # FE е един глобален таг, споделен от всички мелници
    'FE': {"Mill01": "1762", "Mill02": "1762", "Mill03": "1762", "Mill04": "1762", "Mill05": "1762", "Mill06": "1762",
           "Mill07": "1762", "Mill08": "1762", "Mill09": "1762", "Mill10": "1762", "Mill11": "1762", "Mill12": "1762"},

    'PulpHC': {"Mill01": "2386", "Mill02": "2160", "Mill03": "2609", "Mill04": "2726", "Mill05": "2217", "Mill06": "2777",
               "Mill07": "2837", "Mill08": "1833", "Mill09": "2485", "Mill10": "2434", "Mill11": "2670", "Mill12": "3755"},

    'PumpRPM': {"Mill01": "2405", "Mill02": "2198", "Mill03": "2629", "Mill04": "2745", "Mill05": "1652", "Mill06": "2796",
                "Mill07": "2856", "Mill08": "1800", "Mill09": "2471", "Mill10": "2452", "Mill11": "2690", "Mill12": "3780"},

    'MotorAmp': {"Mill01": "2379", "Mill02": "2153", "Mill03": "2602", "Mill04": "2719", "Mill05": "2210", "Mill06": "2770",
                 "Mill07": "2830", "Mill08": "1805", "Mill09": "2478", "Mill10": "2427", "Mill11": "2663", "Mill12": "3748"},

    'PSI80': {"Mill01": "2379", "Mill02": "2153", "Mill03": "2602", "Mill04": "2719", "Mill05": "2210", "Mill06": "5329",
              "Mill07": "5331", "Mill08": "5332", "Mill09": "2478", "Mill10": "2427", "Mill11": "2663", "Mill12": "3748"},

    'PSI200': {"Mill01": "2379", "Mill02": "2153", "Mill03": "2602", "Mill04": "2719", "Mill05": "2210", "Mill06": "5328",
               "Mill07": "5330", "Mill08": "5333", "Mill09": "2478", "Mill10": "2427", "Mill11": "2663", "Mill12": "3748"},
}

TABLE_NAMES = [
    'LoggerValues',
    'LoggerValues_Archive_Sep2026', 'LoggerValues_Archive_Aug2026',
    'LoggerValues_Archive_Jul2026', 'LoggerValues_Archive_Jun2026',
    'LoggerValues_Archive_May2026', 'LoggerValues_Archive_Apr2026',
    'LoggerValues_Archive_Mar2026', 'LoggerValues_Archive_Feb2026',
    'LoggerValues_Archive_Jan2026',
    'LoggerValues_Archive_Dec2025', 'LoggerValues_Archive_Nov2025',
    'LoggerValues_Archive_Oct2025', 'LoggerValues_Archive_Sep2025',
    'LoggerValues_Archive_Aug2025', 'LoggerValues_Archive_Jul2025',
    'LoggerValues_Archive_Jun2025',
    'LoggerValues_Archive_May2025', 'LoggerValues_Archive_Apr2025',
    'LoggerValues_Archive_Mar2025', 'LoggerValues_Archive_Feb2025',
    'LoggerValues_Archive_Jan2025',
]

TARGET = 'PSI200'
ALT_TARGET = 'PSI80'

# PSI200/PSI80 имат смисъл само за тези мелници - за останалите таговете са
# копие на MotorAmp и стойностите не са PSI.
VALID_MILLS = ['Mill06', 'Mill07', 'Mill08']


class PsiEventDataset:
    """Чете суровите данни от SQL Server и строи event-based dataframe."""

    def __init__(self):
        self.engine = create_engine("mssql+pyodbc:///?odbc_connect=" + CONNECTION_STRING)
        # всички features = всичко без target-ите
        self.features = [f for f in SQL_TAGS if f not in (TARGET, ALT_TARGET)]
        # тук пазим суровите серии: {feature: {mill: Series}}
        self.raw = {}

    # ------------------------------------------------------------------ четене
    def read_all_raw(self, start=None):
        """Чете СУРОВИТЕ данни - без resample, без ffill, без bfill.

        Една заявка на таблица за всички тагове наведнъж (по-бързо от
        оригиналния подход, който прави отделна заявка за всяка feature).
        """
        # таг -> [(feature, mill), ...]  (един таг може да обслужва няколко мелници)
        tag_map = {}
        for feature, mills in SQL_TAGS.items():
            for mill, tag in mills.items():
                tag_map.setdefault(str(tag), []).append((feature, mill))
        tag_ids = sorted(tag_map.keys())
        tag_list = ','.join(tag_ids)

        frames = []
        for tbl in TABLE_NAMES:
            q = (f"SELECT IndexTime, LoggerTagID, Value FROM {tbl} "
                 f"WHERE LoggerTagID IN ({tag_list})")
            if start:
                q += f" AND IndexTime >= '{start}'"
            q += " ORDER BY IndexTime ASC"

            print(f"  четa {tbl:<32}", end=' ', flush=True)
            df = pd.read_sql_query(q, self.engine)
            print(f"{len(df):>8} реда")
            if not df.empty:
                frames.append(df)

        if not frames:
            print("  ⚠️  Няма върнати данни.")
            return {}

        raw = pd.concat(frames, ignore_index=True)
        raw['LoggerTagID'] = raw['LoggerTagID'].astype(str)
        raw['IndexTime'] = pd.to_datetime(raw['IndexTime'])

        # ВАЖНО: dedup по (време, таг). Ако е само по време (както е в
        # pulse_to_postgresql.py), при 12 мелници с еднакъв IndexTime
        # остава само един ред и губим мелници.
        raw = raw.drop_duplicates(subset=['IndexTime', 'LoggerTagID'], keep='last')
        raw = raw.sort_values('IndexTime')

        # раздаваме сериите по feature/mill
        out = {}
        for tag, targets in tag_map.items():
            s = raw.loc[raw['LoggerTagID'] == tag, ['IndexTime', 'Value']]
            s = s.set_index('IndexTime')['Value'].sort_index()
            s = s[~s.index.duplicated(keep='last')]
            for feature, mill in targets:
                out.setdefault(feature, {})[mill] = s

        self.raw = out
        return out

    # ------------------------------------------------------- агрегация по прозорци
    @staticmethod
    def _window_stats(series, edges, n_windows):
        """Средна/std/брой на series в прозорците между edges.

        edges[i] .. edges[i+1] -> прозорец i (дясно затворен: (edges[i], edges[i+1]]).
        Връща DataFrame с индекс 0..n_windows-1 или None.
        """
        if series is None or len(series) == 0:
            return None
        s = series.dropna()
        if s.empty:
            return None

        # в кой прозорец попада всяка сурова стойност
        pos = np.searchsorted(edges, s.index.values, side='left') - 1
        ok = (pos >= 0) & (pos < n_windows)
        if not ok.any():
            return None

        df = pd.DataFrame({'pos': pos[ok], 'v': s.values[ok]})
        g = df.groupby('pos')['v']
        stats = pd.DataFrame({'mean': g.mean(), 'std': g.std(), 'n': g.count()})
        # прозорци без нито една проба остават NaN
        return stats.reindex(range(n_windows))

    # ------------------------------------------------------------------ строене
    def build(self, mill, lag_minutes=0, max_gap_minutes=60, psi80_tolerance='15min'):
        """Един ред на всяко четене на PSI200 + агрегирани features."""
        psi = self.raw.get(TARGET, {}).get(mill)
        if psi is None or psi.dropna().empty:
            print(f"  ⚠️  Няма PSI200 за {mill}")
            return pd.DataFrame()
        psi = psi.dropna()

        # Граници на прозорците. lag_minutes=0 -> прозорецът завършва точно
        # в момента на четенето. lag_minutes=N -> смятаме, че пробата е взета
        # N минути ПРЕДИ записа, т.е. местим границите назад.
        edges = (psi.index - pd.Timedelta(minutes=lag_minutes)).values
        n_windows = len(psi) - 1  # N четения -> N-1 интервала
        if n_windows <= 0:
            return pd.DataFrame()

        # индекс на резултата = времената на PSI200 (първото четене няма
        # лява граница, затова го пропускаме)
        out = pd.DataFrame(index=psi.index[1:])
        out.index.name = 'TimeStamp'
        out[TARGET] = psi.values[1:]

        # дължина на прозореца в секунди - за да виждаме неравномерността
        out['dt_sec'] = np.diff(psi.index.values) / np.timedelta64(1, 's')

        # алтернативен target - най-близкото PSI80 четене (НЕ е feature).
        # tolerance: ако най-близкото PSI80 е по-далеч от това -> NaN
        psi80 = self.raw.get(ALT_TARGET, {}).get(mill)
        if psi80 is not None and not psi80.dropna().empty:
            out[ALT_TARGET] = psi80.dropna().reindex(
                out.index, method='nearest', tolerance=psi80_tolerance)
        else:
            out[ALT_TARGET] = np.nan

        # features
        for feature in self.features:
            stats = self._window_stats(
                self.raw.get(feature, {}).get(mill), edges, n_windows)

            if stats is None:
                out[f'{feature}_mean'] = np.nan
                out[f'{feature}_std'] = np.nan
                out[f'{feature}_n'] = 0
                out[f'{feature}_frozen'] = True
                continue

            out[f'{feature}_mean'] = stats['mean'].values
            out[f'{feature}_std'] = stats['std'].values
            out[f'{feature}_n'] = stats['n'].fillna(0).astype(int).values
            # замръзнал сензор: имаме >1 проба, но всички са еднакви
            # ВАЖНО: .values - иначе pandas подравнява по индекс (0..n-1 срещу
            # datetime) и цялата колона става NaN
            out[f'{feature}_frozen'] = (
                (stats['n'] > 1) & (stats['std'].fillna(0) == 0)).values

        # колко features имат поне една проба в този ред (за филтриране на
        # редове, в които почти всичко липсва)
        mean_cols = [f'{f}_mean' for f in self.features]
        out['n_features_ok'] = out[mean_cols].notna().sum(axis=1)

        # изхвърляме прекалено дългите прозорци (иначе средна за 3 часа не е
        # сравнима със средна за 10 минути)
        if max_gap_minutes:
            before = len(out)
            out = out[out['dt_sec'] <= max_gap_minutes * 60]
            print(f"  ✂️  Пуснати {before - len(out)} реда с прозорец > {max_gap_minutes} мин")

        return out

    # ------------------------------------------------------------------ отчет
    def report(self, df, mill):
        print(f"\n{'=' * 78}")
        print(f"  ОТЧЕТ: {mill}")
        print('=' * 78)
        if df.empty:
            print("  празен dataframe")
            return

        print(f"  Редове: {len(df)}")
        print(f"  Период: {df.index.min()}  ->  {df.index.max()}")

        print("\n  Дължина на прозореца (сек):")
        print(df['dt_sec'].describe(percentiles=[.05, .25, .5, .75, .95]).to_string())

        print("\n  Target:")
        print(f"    {TARGET}: mean={df[TARGET].mean():.2f}  std={df[TARGET].std():.2f}  "
              f"NaN={df[TARGET].isna().sum()}")
        print(f"    {ALT_TARGET}: mean={df[ALT_TARGET].mean():.2f}  std={df[ALT_TARGET].std():.2f}  "
              f"NaN={df[ALT_TARGET].isna().sum()}")
        # автокорелация - колко силно е "просто предишното" (persistence baseline)
        print(f"    {TARGET} автокорелация lag-1: {df[TARGET].autocorr(1):.3f}")

        # таблица по features
        rows = []
        for f in self.features:
            m = df[f'{f}_mean']
            s = df[f'{f}_std']
            rows.append({
                'feature': f,
                'NaN_%': round(m.isna().mean() * 100, 1),
                'frozen_%': round(df[f'{f}_frozen'].mean() * 100, 1),
                'n_med': int(df[f'{f}_n'].median()),
                'corr_mean': round(m.corr(df[TARGET], method='spearman'), 3),
                'corr_std': round(s.corr(df[TARGET], method='spearman'), 3),
            })
        print("\n  Features (corr = Spearman спрямо PSI200):")
        print(pd.DataFrame(rows).to_string(index=False))

        # колко реда имат поне един замръзнал сензор
        frozen_cols = [f'{f}_frozen' for f in self.features]
        any_frozen = df[frozen_cols].any(axis=1).mean() * 100
        print(f"\n  Редове с поне един замръзнал сензор: {any_frozen:.1f}%")

        # --- защо има NaN: прозорци, в които сензорът няма НИТО една проба ---
        print("\n  Липсващи данни (NaN = прозорец без нито една сурова проба):")
        miss = []
        for f in self.features:
            n = df[f'{f}_n']
            miss.append({
                'feature': f,
                'празни_прозорци_%': round((n == 0).mean() * 100, 1),
                'n_med': int(n.median()),
                'n_max': int(n.max()),
            })
        print(pd.DataFrame(miss).to_string(index=False))

        print("\n  n_features_ok (колко features имат данни в реда):")
        print(df['n_features_ok'].describe().to_string())

        # --- първите 10 реда, за да се види структурата ---
        print("\n  df.head(10):")
        with pd.option_context('display.max_columns', None, 'display.width', 250):
            print(df.head(10).to_string())


def main():
    ap = argparse.ArgumentParser(description="Event-based PSI200 dataset (експеримент)")
    ap.add_argument('--mill', default='Mill06', help='коя мелница (default Mill06)')
    ap.add_argument('--all-mills', action='store_true', help='обработи Mill06/07/08')
    ap.add_argument('--start', default=None, help='YYYY-MM-DD - филтър по IndexTime')
    ap.add_argument('--lag', type=int, default=0,
                    help='минути; 0 = прозорецът завършва в момента на четенето')
    ap.add_argument('--max-gap', type=int, default=60,
                    help='макс дължина на прозореца в минути (0 = без филтър)')
    ap.add_argument('--psi80-tol', default='15min',
                    help="допустимо разстояние до най-близкото PSI80 (default 15min)")
    ap.add_argument('--out', default='psi_events', help='префикс на изходните файлове')
    args = ap.parse_args()

    mills = VALID_MILLS if args.all_mills else [args.mill]

    print("🚀 Четене на сурови данни от SQL Server...")
    ds = PsiEventDataset()
    ds.read_all_raw(start=args.start)

    for mill in mills:
        print(f"\n🔧 Строене на dataframe за {mill} (lag={args.lag} мин)...")
        df = ds.build(mill, lag_minutes=args.lag, max_gap_minutes=args.max_gap,
                      psi80_tolerance=args.psi80_tol)
        ds.report(df, mill)

        if not df.empty:
            path = f"{args.out}_{mill}.parquet"
            df.to_parquet(path)
            print(f"\n  💾 Записано: {path}")


if __name__ == '__main__':
    main()
