import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """_summary_

    Args:
        messages_filepath (_type_): _description_
        categories_filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset 
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages,categories, how = 'left', on = 'id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # categories values to numbers 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop("categories", axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)

    return df 


def clean_data(df):
    
    # drop duplicates
    df = df.drop_duplicates().reset_index(drop = True)

    # remove values in the related column so that it is a binary variable 
    df = df[(df['related'] == 0)|(df['related'] == 1)].reset_index(drop = True)

    return df 


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages_category', engine, index=False)  
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()