"""Общи настройки за връзка с Pulse SQL Server и PostgreSQL.

Използва се от трансформаторите (pulse_to_postgresql.py, vhs_to_postgresql.py)
и от обединения scheduler (scheduled_pulse_postgresql.py), за да има една
централна точка за креденшъли и създаване на engine-и.
"""
from sqlalchemy import create_engine

# Pulse SQL Server
PULSE_SERVER = '10.20.2.10'
PULSE_DATABASE = 'pulse'
PULSE_USER = 'Pulse_RO'
PULSE_PASSWORD = 'PD@T@r3@der'

# PostgreSQL (target)
PG_HOST = 'em-m-db4.ellatzite-med.com'
PG_PORT = 5432
PG_DBNAME = 'em_pulse_data'
PG_USER = 's.lyubenov'
PG_PASSWORD = 'tP9uB7sH7mK6zA7t'


def create_pulse_engine():
    """Създава engine към Pulse SQL Server."""
    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={PULSE_SERVER};"
        f"DATABASE={PULSE_DATABASE};UID={PULSE_USER};PWD={PULSE_PASSWORD}"
    )
    return create_engine("mssql+pyodbc:///?odbc_connect=" + connection_string)


def create_pg_engine():
    """Създава engine към PostgreSQL."""
    return create_engine(
        f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
    )
