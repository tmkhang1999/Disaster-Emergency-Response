import joblib
import pandas as pd
from sqlalchemy import create_engine, text

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])


def load_data(sql_path, table_name, model_path):
    # load clean data from sqlite
    engine = create_engine('sqlite:///' + sql_path)
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(sql=text(query), con=engine.connect())

    # load model
    model = joblib.load(model_path)

    return df, model


def tokenize(txt):
    """
    This 'tokenize' function is used for deployment

    INPUT:
    txt - Raw text

    OUTPUT:
    clean_tokens - a list of tokens that are lemmatized, lowered, and stripped from text
    """
    # tokenize text
    tokens = nltk.word_tokenize(txt)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, lower, then strip token
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
