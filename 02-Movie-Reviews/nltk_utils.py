import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re

def remove_punctuation(string:str):
    return re.sub("[^a-zA-Z0-9\s]", " ", string)

def remove_numbers(string:str):
    return re.sub("[^\D]", " ", string)

def remove_stopwords(series: pd.Series, stop_words: list):
    pat = r'\b(?:{})\b'.format('|'.join(stop_words))
    return series.str.replace(pat, " ")

def lemmatize_text(string: str):
    lemmatizer = WordNetLemmatizer()
    string = string.split()
    for i in range(len(string)):
        string[i] = lemmatizer.lemmatize(string[i], pos= "v")
        string[i] = lemmatizer.lemmatize(string[i], pos= "n")
        string[i] = lemmatizer.lemmatize(string[i], pos= "a")
    return " ".join(string)

def vectorize_text(series: pd.Series, ngrams = 1):
    matrix = CountVectorizer(ngram_range=(ngrams, ngrams))
    matrix.fit(series)
    return matrix.transform(series)

def multinomial_naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
    multinomial = MultinomialNB()
    multinomial.fit(X_train, y_train)
    y_pred = multinomial.predict(X_test)
    return accuracy_score(y_test, y_pred)

def global_processing(series: pd.Series):
    stop_words = stopwords.words('english')
    return remove_stopwords(series.str.replace(r"[\W_]", " ").str.lower(), stop_words)
