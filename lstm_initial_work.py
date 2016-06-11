# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 07:20:56 2016

@author: nitini
"""

#%% Importing necessary libraries
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import codecs
import pandas as pd
import pydot
from keras.utils.visualize_util import plot

#%% Load in training data, setting up char indexes

nietzsche_data = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = codecs.open(nietzsche_data, encoding='utf-8').read().lower()

boyfriend_data = pd.read_csv('./boyfriend_lines_csv.csv')
boyfriend_data.drop(['empty1','empty2','empty3','empty4'],axis=1,inplace=True)

text = ''
for i in range(boyfriend_data.shape[0]):
    text = text + ' ' + boyfriend_data.line.iloc[i]


print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

#%% Chop text into semi-redundant chunks

maxlen = 7
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

sentence_data = pd.DataFrame({'sentence':sentences, 'next_char':next_chars})

#%% Vectorization

X = np.zeros((sentence_data.shape[0], maxlen, len(chars)), dtype=np.bool)
y = np.zeros((sentence_data.shape[0], len(chars)), dtype=np.bool)


for i, sentence in enumerate(sentence_data.sentence):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[sentence_data.next_char.iloc[i]]] = 1

#%% LSTM Model

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#%% Random text generation

def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # Modified to go faster, not training on all data
    model.fit(X, y, batch_size=256, nb_epoch=1)


   #start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [1.0]:
        print()
        print()
        sentence_starts_dict = {'I want ':'I want ', 'I like ':'I like ','I need ':'I need '}
        for s in sentence_starts_dict.keys():
            orig_sentence = s
            sentence = s
            print()
            for i in range(70):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                sentence_starts_dict[orig_sentence] += next_char
                sentence = sentence[1:] + next_char

        for key, value in sentence_starts_dict.iteritems():
            print("----- Sentence seed: ")
            print("----- " + key)
            print()
            print("----- Generated Sentence: ")
            print("----- " + value)
            print()
        print()

print("Completed running script")

