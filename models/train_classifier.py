import sys

# The fundamentals 
import pandas as pd
import numpy as np 

# Load data 
from sqlalchemy import create_engine

# Text 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt','stopwords', 'wordnet', 'averaged_perceptron_tagger'])
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

# ML models 
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb 

# Multilabel train test split (external script)
from multilabel_split import multilabel_train_test_split


# Save model 
import joblib

def load_data(database_filepath):
    """
    Load data from SQLite database

    Args:
        database_filepath (string): Path to database 

    Returns:
        X (pd.DataFrame): X variables for ML model 
        Y (pd.DataFrame): Y variable for ML model (multilabel)
        category_names (list): Names of the categories 

    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM Messages_category", engine)

    X = df['message']
    Y = df.drop(columns = {'id', 'message', 'original', 'genre'})
    category_names = Y.columns

    return X,Y,category_names


def tokenize(text):
    """
    Tokenize and lemmatize text removing stop words 

    Args:
        text (string): Text to tokenize

    Returns:
        tokens (list): Tokenized text 
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    Create ML model 

    Returns:
        pipeline (Sklearn pipeline): ML model pipeline
    """

    # the parameters have been obtained using GridSearch in the Jupyter Notebook 
    pipeline =Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(xgb.XGBClassifier(max_depth = 9, 
                learning_rate = 0.1, n_estimators = 180))) ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate ML model using precission, recall, F1 score and Hamming score. Prints the results.  

    Args:
        model (Sklearn pipeline): ML model pipeline
        X_test (pd.DataFrame): X_test variable for ML model 
        Y_test (pd.DataFrame): Y_test variable for ML model (Multilabel)
        category_names (list): Names of the categories
    """
    Y_pred = model.predict(X_test)
    
    # classification report on test data
    print('\n', classification_report(Y_test.values, Y_pred, target_names=category_names ))

    print("The Hamming Loss Score is: {0:.2f}".format(hamming_loss(Y_pred, Y_test)))


def save_model(model, model_filepath):
    """
    Save ML model 

    Args:
        model (Sklearn pipeline): ML model pipeline
        model_filepath (string): Path to save location 
    """
    joblib.dump(model, model_filepath)

    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = multilabel_train_test_split(X, Y, stratify = Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()