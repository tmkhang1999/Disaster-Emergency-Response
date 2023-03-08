import joblib
import pandas as pd

from sqlalchemy import create_engine, text


def load_data(sql_path, table_name, model_path):
    # load clean data from sqlite
    engine = create_engine('sqlite:///' + sql_path)
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(sql=text(query), con=engine.connect())

    # load model
    model = joblib.load(model_path)

    return df, model
