import argparse
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_path, categories_path):
    """
    INPUT:
    messages_path - The path to the disaster_messages data
    categories_path - The path to the disaster_categories data

    OUTPUT:
    new_df - a new dataframe created from merging messages and categories data based on 'id'
    """
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    INPUT:
    df - a dataframe with the messy 'categories' column and some duplicated rows

    OUTPUT:
    new_df - a new dataframe which is cleaner
    """
    # Split `categories` into separate category columns
    categories = df['categories'].str.split(";", expand=True)

    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to number 0 & 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    # Replace the categories column with new category columns
    df.drop(['categories'], inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)

    return df


def save_data(df, db_name, table_name):
    """
    Save the processed data to sqlite database
    """
    engine = create_engine('sqlite:///' + db_name, pool_pre_ping=True)
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--messages_path', type=str, default='disaster_messages.csv',
                        help='The path to disaster-messages data')
    parser.add_argument('--categories_path', type=str, default='disaster_categories.csv',
                        help='The path to disaster-categories data')
    parser.add_argument('--database_name', type=str, default='DisasterResponse')
    parser.add_argument('--table_name', type=str, default='response')

    opt = parser.parse_args()
    df = load_data(opt.messages_path, opt.categories_path)
    df = clean_data(df)
    save_data(df, opt.database_name, opt.table_name)


if __name__ == '__main__':
    main()
