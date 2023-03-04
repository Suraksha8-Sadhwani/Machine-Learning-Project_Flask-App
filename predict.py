
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM, Embedding, Dense
import os
import pickle
import re
from nltk.corpus import stopwords
import nltk
import tensorflow as tf
import pandas as pd


def init_model():
    model = Sequential()
    model.add(
        Embedding(2000, 100, input_length=100))
    model.add(layers.Bidirectional(LSTM(32)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights("./static/aug_model.hdf5")
    model.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def load_text_vectorizer():
    filepath = "./static/vectorize_layer_model"
    loaded_vectorize_layer_model = tf.keras.models.load_model(filepath)
    loaded_vectorize_layer = loaded_vectorize_layer_model.layers[0]
    return loaded_vectorize_layer


def preprocess_text(sentence):
    nltk.download('stopwords')
    nltk.download('wordnet')
    stemmer = WordNetLemmatizer()
    swords = stopwords.words('english')
    tweet = re.sub(r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', sentence)
    tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
    tweet = tweet.lower()
    tweet = tweet.split(' ')
    tweet = [word for word in tweet if word not in swords]
    tweet = [stemmer.lemmatize(word) for word in tweet]
    tweet = ' '.join(tweet)
    return tweet


def predict_news_status(model, txt):
    txt = preprocess_text(txt)
    txt = load_text_vectorizer()(pd.DataFrame([[txt]]))
    prob = model.predict(txt)[0][0]
    print(prob)
    return 'Real' if prob >= 0.5 else 'Fake'
