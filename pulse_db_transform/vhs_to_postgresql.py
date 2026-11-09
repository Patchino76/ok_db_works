"""Синхронизация на VHS тагове: Pulse SQL Server -> PostgreSQL (schema vhs, table VHS_TAGS).

Логиката е същата като при мелниците (pulse_to_postgresql.py):
  - таблицата се създава, ако не съществува (първоначален backfill от START_DATE);
  - при следващи пускания се добавят само нови редове (append-only).

Таговете (LoggerTagId -> tag_name, type) се четат от vhs_tags.csv, който се
генерира от tmp/vhs_tags_with_ids.xlsx (sheet "Mapped").

Подготовка на данните (както при мелниците):
  - аналогови тагове: ffill().bfill() - запълване без линейна интерполация;
  - цифрови (boolean) тагове: само 0/1, при липса на данни -> 0;
  - редове без никакви данни се изключват.
"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from pulse_config import (
    PULSE_SERVER, PULSE_DATABASE, PULSE_USER, PULSE_PASSWORD,
    PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD,
    create_pulse_engine,
)

ROOT = Path(__file__).resolve().parent
TAG_CSV = ROOT / 'vhs_tags.csv'

# Начало на първоначалния backfill
START_DATE = '2026-10-01'

# Таговете се пазят в същите таблици като при мелниците
TABLE_NAMES = ['LoggerValues'] + [
    f'LoggerValues_Archive_{m}{y}' for y in (2026,)
    for m in ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    if not (y == 2026 and m in ('Oct', 'Nov', 'Dec'))
]

SCHEMA = 'vhs'
TABLE_NAME = 'VHS_TAGS'


class VHSTransformer:
    def __init__(self, pg_host=PG_HOST, pg_port=PG_PORT, pg_dbname=PG_DBNAME,
                 pg_user=PG_USER, pg_password=PG_PASSWORD, engine=None, pg_engine=None):
        # SQL Server (Pulse) connection parameters
        self.server = PULSE_SERVER
        self.database = PULSE_DATABASE
        self.username = PULSE_USER
        self.password = PULSE_PASSWORD
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};"
            f"DATABASE={self.database};UID={self.username};PWD={self.password}"
        )
        self.engine = engine if engine is not None else create_pulse_engine()

        # PostgreSQL (target) connection parameters
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_dbname = pg_dbname
        self.pg_user = pg_user
        self.pg_password = pg_password
        self.pg_engine = pg_engine if pg_engine is not None else create_engine(
            f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_dbname}"
        )

        # LoggerTagId -> tag_name + множество на цифровите (boolean) тагове
        tags_df = self._load_tags()
        self.tags = dict(zip(tags_df['LoggerTagId'], tags_df['tag_name']))
        self.bool_tags = set(tags_df.loc[tags_df['type'] == 'цифров', 'tag_name'])
        self.tag_ids = list(self.tags.keys())
        self.tag_names = list(self.tags.values())

        # Append-mode timestamp filter (None -> backfill from START_DATE)
        self.filter_timestamp = None
        self.creating_new_table = False

    @staticmethod
    def _load_tags():
        """Зарежда LoggerTagId, tag_name, type от vhs_tags.csv."""
        if not TAG_CSV.exists():
            raise FileNotFoundError(
                f"Липсва {TAG_CSV}. Генерирай го от tmp/vhs_tags_with_ids.xlsx (sheet 'Mapped')."
            )
        d = pd.read_csv(TAG_CSV, dtype={'LoggerTagId': str, 'tag_name': str, 'type': str})
        d = d.dropna(subset=['LoggerTagId', 'tag_name'])
        d['LoggerTagId'] = d['LoggerTagId'].str.strip()
        d['tag_name'] = d['tag_name'].str.strip()
        d['type'] = d['type'].fillna('').str.strip()
        return d

    def read_sql_table(self, table_name):
        """Чете всички VHS тагове от една SQL таблица и връща wide 1-мин DataFrame."""
        tag_conditions = " OR ".join([f"LoggerTagID = {tid}" for tid in self.tag_ids])
        where_clauses = [f"({tag_conditions})"]

        if self.creating_new_table:
            where_clauses.append(f"IndexTime >= '{START_DATE} 00:00:00'")
            print(f"  - Creating new table: fetching data from {START_DATE}")
        elif self.filter_timestamp is not None:
            since_str = self.filter_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            where_clauses.append(f"IndexTime > '{since_str}'")
            print(f"  - Filtering data after: {since_str}")
        else:
            where_clauses.append(f"IndexTime >= '{START_DATE} 00:00:00'")
            print(f"  - No filter set, defaulting to {START_DATE}")

        where_clause = " AND ".join(where_clauses)
        query_str = f"""
        SELECT IndexTime, LoggerTagID, Value
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY IndexTime ASC
        """

        print(f"  - Executing query on {table_name}")
        query = pd.read_sql_query(query_str, self.engine)

        if query.empty:
            print(f"  - No data found in {table_name}")
            return pd.DataFrame()

        print(f"  - Found {len(query)} rows in {table_name}")

        query['LoggerTagID'] = query['LoggerTagID'].astype(str)
        query = query.drop_duplicates(subset=['IndexTime', 'LoggerTagID'], keep='last')
        df = query.pivot(index='IndexTime', columns='LoggerTagID', values='Value')

        # Rename tag ids -> tag names
        df.columns = [self.tags.get(str(c), str(c)) for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = 'TimeStamp'
        df = df.sort_index()

        # 1-минутна мрежа (запълването/типовете се прилагат в _finalize)
        df = df.resample('1min').mean()

        # Гарантираме стабилна схема: всички тагове като колони
        df = df.reindex(columns=self.tag_names)
        return df

    def compose_data(self, seed=None):
        """Обединява данните от всички SQL таблици."""
        frames = []
        for tbl in TABLE_NAMES:
            print(f"Processing {tbl}")
            frames.append(self.read_sql_table(tbl))

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        # Същото +2h отместване като при мелниците (SQL времената са в централна зона)
        df = df.shift(2, freq='h')
        return self._finalize(df, seed=seed)

    def _finalize(self, df, seed=None):
        """Изключва празните редове; аналогови -> ffill, цифрови -> 0/1 (0 при липса).

        seed: последният ред от таблицата (Series по tag_name). Използва се при append,
        за да не се губят бавно обновявани тагове, чиито последни стойности са преди
        началото на изтегления прозорец.
        """
        if df.empty:
            return df

        bool_cols = [c for c in df.columns if c in self.bool_tags]
        analog_cols = [c for c in df.columns if c not in self.bool_tags]

        # Seed: за тагове без нито едно четене в прозореца използваме последната
        # известна стойност от таблицата (иначе ffill/bfill не могат да ги запълнят).
        if seed is not None:
            seed_row = seed.reindex(df.columns)
            empty_cols = df.columns[df.isna().all()]
            if len(empty_cols) > 0:
                df = df.fillna(seed_row[empty_cols].to_dict())

        # Аналогови: ffill + bfill (както при мелниците) - без линейна интерполация
        if analog_cols:
            df[analog_cols] = df[analog_cols].ffill().bfill()

        # Изключваме редове без никакви данни
        df = df.dropna(how='all')

        # Цифрови: само 0/1; при липса на данни -> 0
        if bool_cols:
            df[bool_cols] = (df[bool_cols].fillna(0) >= 0.5).astype('int8')

        return df

    def _ensure_schema(self):
        try:
            with self.pg_engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
                conn.commit()
            print(f"✅ Schema '{SCHEMA}' created or already exists.")
            return True
        except SQLAlchemyError as e:
            print(f"❌ Error creating schema: {e}")
            return False

    def _table_exists(self):
        with self.pg_engine.connect() as conn:
            return conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = :schema AND tablename = :table
                );
            """), {'schema': SCHEMA, 'table': TABLE_NAME}).scalar()

    def append_to_postgresql(self):
        """Създава таблицата при нужда и добавя само нови редове."""
        print("\n🚀 Starting VHS -> PostgreSQL synchronization...")
        start_time = datetime.now()

        if not self._ensure_schema():
            return

        try:
            if self._table_exists():
                with self.pg_engine.connect() as conn:
                    latest_ts = conn.execute(
                        text(f'SELECT MAX("TimeStamp") FROM {SCHEMA}."{TABLE_NAME}"')
                    ).scalar()

                if latest_ts:
                    latest_dt = pd.to_datetime(latest_ts)
                    # 6-часов буфер срещу закъснения/часови зони
                    self.filter_timestamp = latest_dt - pd.Timedelta('6 hours')
                    self.creating_new_table = False
                    print(f"  ⏱️  Last record in DB: {latest_dt}")
                    print(f"  🔍 Fetching data after: {self.filter_timestamp} (6-hour buffer)")

                    # Последният ред служи за seed на ffill (бавни тагове)
                    last_row = pd.read_sql_query(
                        f'SELECT * FROM {SCHEMA}."{TABLE_NAME}" '
                        f'ORDER BY "TimeStamp" DESC LIMIT 1',
                        self.pg_engine
                    )
                    seed = (last_row.drop(columns=['TimeStamp']).iloc[0]
                            if not last_row.empty else None)

                    df = self.compose_data(seed=seed)
                    if not df.empty:
                        new_data = df[df.index > latest_dt]
                        if not new_data.empty:
                            new_data.to_sql(
                                name=TABLE_NAME, schema=SCHEMA, con=self.pg_engine,
                                if_exists='append', index=True, index_label='TimeStamp',
                                method='multi', chunksize=200
                            )
                            duration = (datetime.now() - start_time).total_seconds()
                            print(f"  ✅ Appended {len(new_data)} new rows in {duration:.1f} seconds")
                            print(f"  📊 New data range: {new_data.index.min()} to {new_data.index.max()}")
                        else:
                            print("  ℹ️  No new data to append")
                    else:
                        print("  ℹ️  No data returned from source")
                else:
                    print("  ⚠️  Table exists but is empty - will repopulate")
                    self._repopulate_table()
            else:
                print("  🆕 Table doesn't exist - creating with initial data")
                self._repopulate_table()
        except Exception as e:
            print(f"  ❌ Error during synchronization: {e}")
            import traceback
            traceback.print_exc()

        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Synchronization completed in {total_duration:.1f} seconds")

    def _repopulate_table(self):
        """Първоначален backfill от START_DATE; създава/презаписва таблицата."""
        print(f"  🔄 Repopulating {SCHEMA}.{TABLE_NAME} from {START_DATE}...")
        self.filter_timestamp = None
        self.creating_new_table = True
        df = self.compose_data()

        if df.empty:
            print("  ⚠️  No data available to populate table")
            return

        df.to_sql(
            name=TABLE_NAME, schema=SCHEMA, con=self.pg_engine,
            if_exists='replace', index=True, index_label='TimeStamp',
            method='multi', chunksize=200
        )
        print(f"  ✅ Created {SCHEMA}.{TABLE_NAME} with {len(df)} rows")
        print(f"  📅 Data range: {df.index.min()} to {df.index.max()}")


def main():
    pg_host = 'em-m-db4.ellatzite-med.com'
    pg_port = 5432
    pg_dbname = 'em_pulse_data'
    pg_user = 's.lyubenov'
    pg_password = 'tP9uB7sH7mK6zA7t'

    print("🚀 Starting VHS to PostgreSQL data synchronization...")
    print(f"Connecting to {pg_host}:{pg_port}/{pg_dbname} as {pg_user}")

    try:
        transformer = VHSTransformer(
            pg_host=pg_host, pg_port=pg_port, pg_dbname=pg_dbname,
            pg_user=pg_user, pg_password=pg_password
        )
        transformer.append_to_postgresql()
    except Exception as e:
        print(f"❌ Error during synchronization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("✅ Data synchronization completed successfully!")
    return 0


if __name__ == "__main__":
    main()
