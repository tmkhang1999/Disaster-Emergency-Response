import argparse
import pandas as pd
from sqlalchemy import create_engine, text

import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(database_path, table_name):
    """
    INPUT:
    database_path - The path to the database
    table_name - Name of the table containing the data to be extracted in the database

    OUTPUT:
    X, y - Input and Output
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_path)
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(sql=text(query), con=engine.connect())

    # Split input and output
    X = df['message']
    y = df.iloc[:, 4:]

    return X, y


def tokenize(txt):
    """
    INPUT:
    txt - Raw text

    OUTPUT:
    clean_tokens - a list of tokens that are lemmatized, lowered, and stripped from text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # replace each url with placeholder
    for url in re.findall(url_regex, txt):
        txt = txt.replace(url, 'url_placeholder')

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


def build_model():
    """
    Create a Machine Learning pipeline with CountVectorizer, TfidfTransformer,
    and Random Forest Classifier with multiple labels
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=True)))
    ])

    return model


def get_score(y_true, y_pred, target_names):
    """
    INPUT:
    y_true - True values
    y_pred - Predicted values
    target_names - output labels

    OUTPUT:
    df - the score table
    """
    df = pd.DataFrame()

    for i, target in enumerate(target_names):
        # Calculate accuracy, f1-score, precision, recall
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i], average='weighted')
        precision = precision_score(y_true[:, i], y_pred[:, i], average='weighted')
        recall = recall_score(y_true[:, i], y_pred[:, i], average='weighted')

        # Append to a new dataframe
        df = df.append({'index': target, 'Accuracy': accuracy, 'F1 Score': f1,
                        'Precision': precision, 'Recall': recall}, ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, default=r'data/DisasterResponse',
                        help='The path to the database')
    parser.add_argument('--table_name', type=str, default='response',
                        help='Name of the table containing the data to be extracted in the database')
    parser.add_argument('--model_path', type=str, default='model.joblib',
                        help='The path you want to save the model')
    opt = parser.parse_args()

    # Load the data
    X, y = load_data(opt.database_path, opt.table_name)

    # Build model, and split the dataset
    model = build_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)

    # Get the evaluation score
    y_pred = model.predict(X_test)
    res = get_score(y_test.values, y_pred, y.columns)
    print(res)

    # Save the model
    joblib.dump(model, opt.save_path)


if __name__ == '__main__':
    main()
