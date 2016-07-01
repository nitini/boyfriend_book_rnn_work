from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np
import sys
import pandas as pd
import file_paths as fp
import time
import json
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import lstm_initial_work as liw


def get_training_data():


def create_model(maxlen=7, chars={}, num_layers=2):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, len(chars)))
    for i in range(num_layers - 2):
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')




def main():
    tweet_data = pd.read_csv('./old_tweet_data/old_tweet_data_unweighted.csv')
    text, chars, char_indices, indices_char = liw.convert_csv_to_text(tweet_data)

    maxlen = 7
    step = 1

    train_data = liw.convert_text_to_train(text, maxlen, step)

    percent_to_train = 0.3

    train_subset = train_data.sample(frac=percent_to_train).copy()

    param_grid = dict()
    



if __name__ == '__main__':
    main()
