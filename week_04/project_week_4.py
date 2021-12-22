#### Get libraries

import requests
import re
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import numpy as np
from sklearn.naive_bayes import MultinomialNB 
from PIL import Image
from sklearn.linear_model import LogisticRegression
import argparse
from matplotlib import pyplot as plt
import wordcloud

parser = argparse.ArgumentParser(description='Predicts to what band input lyrics snippets belong to. The code works for Dream Theater, Angra and King Crimson')
parser.add_argument("-v", "--verbosity", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")



def load_data():
    """The function loads the data and specifies the model that will be estimated.#
    """
    
    df = pd.read_csv('data/output.csv')
    
    df = df.dropna()
    
    corpus = df['Lyrics'].to_list()
    for words in corpus:
        words.split()
    
    labels = df['Artist'].to_list()
    
    X = corpus
    y = labels
    
    return X, y 

def feature_engineering(X):
    """The function performs feature engineering
    """
   
    tf_vec = TfidfVectorizer(stop_words= ['is'])
    tv_vec = tf_vec.fit(X)
    X_trans = tf_vec.transform(X).todense()
    
    return X_trans, tf_vec

def train_model(X, y):
    """
    Trains a scikit-learn classification model on text.
    
    Parameters
    ----------
    text : list
    labels : list
    
    Returns
    -------
    model : Trained scikit-learn model.
    
    """
    tf_vec = TfidfVectorizer()
    nb = MultinomialNB(alpha = 1)
    model = make_pipeline(tf_vec, nb)
    model.fit(X, y)
    
    return model

def build_model_RF():
    """
    The function builds a machine the machine learning model. First, the pipeline is created, then the parameters dictionary is 
    created, and lastly the grid search object is built.
    
    Input:
    None

    Output: 
    Grid search object
    """    

    m = RandomForestClassifier(class_weight = "balanced", random_state = 42)
    
    parameters = {'max_depth':[10, 50, 100],'n_estimators':[50, 100, 200]}
    
    cv = GridSearchCV(m, param_grid = parameters)
    return cv



##### Make prediction

def main():
    X, y = load_data()
    print('Loading the data...')    
    
    print('---------------------------------------------------------------------------------------------------')
    
    print('Let us first attempt the Naive Bayes estimator!')
     
    print('Building and fitting the model..')
    model = train_model(X, y)
    snippet = [input("Please type a snippet from a lyric:")]
    prediction = model.predict(snippet)
    probabilities = model.predict_proba(snippet)
    probabilities = np.round(probabilities, 2)
    print("The most likely artist for this snippet is:")
    print(prediction)
    print ("These are the two model classes and their respective probabilities:")
    print(model.classes_)
    print(probabilities)
      
    print('---------------------------------------------------------------------------------------------------')

    print('Now let us try the Random Forest...')
    
    X_trans, tf_vec = feature_engineering(X)
    print('Performing feature engineering for the Random Forest...')
    
    m = build_model_RF()
    print('Building the model...')
    m = m.fit(np.asarray(X_trans), y)   
    print('Fitting the model...')
    snippet = [input("Please type a snippet from a lyric:")]
    a = m.predict(tf_vec.transform(snippet))      
    print("The most likely artist for this snippet is:")
    print(a)
    b = m.predict_proba(tf_vec.transform(snippet))    
    b = np.round(b, 2)
    print ("These are the two model classes and their respective probabilities:")
    print (m.classes_)
    print(b)
    print('---------------------------------------------------------------------------------------------------')
    print("In your opinion, which estimator performed better?")
         
if __name__ == '__main__':
	main() 