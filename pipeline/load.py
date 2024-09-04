# load.py

from sqlalchemy import create_engine
import pandas as pd

def get_postgres_engine(db_config):
    db_url = f"postgresql://{db_config['user']}:{db_config['pw']}@{db_config['host']}:{db_config['port']}/{db_config['db']}"
    return create_engine(db_url)

def load_data_to_postgres(df, table_name, db_config):
    engine = get_postgres_engine(db_config)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data has been successfully inserted into the table '{table_name}'.")

def load_filtered_data(filtered_df):
    db_config = {
        'host': "127.0.0.1",
        'port': "5432",
        'user': "postgres",
        'pw': "qwe123",
        'db': "project",
        'location': "localhost_target",
        'engine': "postgre",
        'table_name': "your_table_name"
    }
    load_data_to_postgres(filtered_df, db_config['table_name'], db_config)
