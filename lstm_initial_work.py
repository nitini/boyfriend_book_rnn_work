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
import numpy as np
import sys
import pandas as pd
import file_paths as fp
import time
import json

def convert_csv_to_text(data):
    """
    Data should be a pandas dataframe with a single column.
    Each column is a line of text (a rule in the boyfriend book case)
    otherwise it can be some arbitrary chunk of text. This method takes
    data and then returns three pieces listed below

    Inputs:
        data: pandas Dataframe of text
    Outputs:
        text: single string of all training data
        char_indices: mapping of characters to indices
        indices_char: reverse mapping of the above

    """
    text = ''

    for i in range(data.shape[0]):
        text = text + ' ' + data.line.iloc[i]

    chars = set(text)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return text, chars, char_indices, indices_char

def convert_text_to_train(text, maxlen, step):
    """
    This method takes the raw text and chunkifies it into training data.
    Training data here is a sequence of characters, and then the next
    character following the sequence.

    Inputs:
        text: The large string of text data
        maxlen: The length for each training example
        step: How many characters to move the sliding window
              to get the next training example
    
    Outputs:
        training_data: A pandas dataframe of each sequence and
                       it's corresponding next character

    """
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    return pd.DataFrame({'sentence':sentences, 'next_char':next_chars})

def vectorize_training_data(training_data, maxlen, step, chars, char_indices):
    """
    Take the training data in text form and convert it to a 3D tensor
    that represents the text in a form that can be inputted to the model

    Inputs:
        training_data: The pandas dataframe of examples
        maxlen: The length of each sequence in the training data
        step: sliding window size
        chars: The set of all characters in training data
        char_indices: mapping of characters to indices

    Outputs:
        X: A 3D tensor of sequence examples encoded into numerical form
        y: A 3D tensor of the next char for each sequence example

    """

    X = np.zeros((training_data.shape[0], maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((training_data.shape[0], len(chars)), dtype=np.bool)

    for i, sentence in enumerate(training_data.sentence):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[training_data.next_char.iloc[i]]] = 1

    return X, y

def get_model_v1(maxlen, chars):
    """
    Defines v1 of the character level RNN with LSTM units,
    each model will have its own function, no need to 
    parametrize at this time

    Inputs:
        maxlen: length of sequence to define input layer
        chars: set of characters, used to define input layer

    Output:
        model: Keras LSTM model, with 3 LSTM layers, and 2
               dropout layers
    """
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

    return model

def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def encode_sentence(sentence, maxlen, chars, char_indices):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
    return x

def generate_random_sentence(
                            model,
                            sentence_seed,
                            sentence_length,
                            diversity,
                            maxlen,
                            chars,
                            char_indices,
                            indices_char,
                            ):

    """
    Generates a random sentence based on the current model.
    You can specify the length of the sentence, the seed for the sentence
    and then also how to shift the probability distribution of the predictions

    Inputs:
        model: The LSTM model, pass any version
        sentence_seed: The seed to start the new random sentence
        sentence_length: How long of a sentence to generate
        diversity: Paramter to shift prediction probability distribution
        maxlen: Length of model input sequence
        chars: Set of characters from training data
        char_indices: Mapping characters to indices
        indices_char: Reverse mapping of the above

    Outputs:
        generated_sentence: The resulting random sentence from model predictions

    """

    generated_sentence = sentence_seed

    for i in range(sentence_length):
        encoded_sentence = encode_sentence(sentence_seed, 
                                           maxlen, 
                                           chars, 
                                           char_indices)
        preds = model.predict(encoded_sentence, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated_sentence += next_char
        sentence_seed = sentence_seed[1:] + next_char

    return generated_sentence

def train_model(model, X, y, batch_size):
    """
    Trains the model given training data, for a batch size and 
    a number of epochs, set epochs to 1 to be able to 
    test out model after every epoch, otherwise just set to
    how many ever times you want to iterate through the data

    Inputs:
        model: The LSTM model
        X: training data examples
        y: training data labels (next chars)
        batch_size: number of examples to updates weight with at a time
        num_epochs: How many times to cycle through training data

    Outputs:
        None as it updates the model in place

    """

    model.fit(X, y, batch_size, nb_epoch=1)

def make_model_name(model_version):
    current_date = time.strftime('%Y_%m_%d_%I_%M_%S')
    model_name = 'model_v' + str(model_version) + '_' + current_date
    return model_name

def save_model_info(model, notes, output_file):
    """
    This method saves down the architecture of the model so that it 
    can be referenced in the future. Also can add in any notes to the
    resulting text file to provide more details on a given model
    
    Inputs:
        model: The keras model
        model_name: name of the model based on timestamp and version
        notes: Additional information about the model
        in_gcp: flag for where to save the model
    
    Outputs:
        None, just saves the model
        
    """
    print('---------- Any additional notes: ---------- ', file=output_file)
    print('', file=output_file)
    print(notes, file=output_file)
    print('', file=output_file)
    print('---------- All model information: ---------- ', file = output_file)
    print('', file=output_file)
    print(json.dumps(json.loads(model.to_json()), indent=2, sort_keys=True), file = output_file)

    
def save_model_weights(model, model_name, in_gcp):
    if in_gcp == 1:
        model.save_weights(fp.goog_file_path + model_name + '_weights.hdf5')
    else: 
        model.save_weights('./' + model_name + '_weights.hdf5')

        
#%%  
def main():
    
    boyfriend_data = pd.read_csv('./boyfriend_lines.csv')
    in_gcp = sys.argv[1]
    save_weights = sys.argv[2]
    if in_gcp == 1:
        boyfriend_data = pd.read_csv(fp.goog_file_path + 'boyfriend_lines.csv')
    
    text, chars, char_indices, indices_char = convert_csv_to_text(boyfriend_data)
    
    maxlen = 7
    step = 1    
    
    training_data = convert_text_to_train(text, maxlen, step)
    X, y = vectorize_training_data(training_data, maxlen, step, chars, char_indices)
    model = get_model_v1(maxlen, chars)
    batch_size = 256
    num_epochs = 1
    
    sentence_seeds = ['I want ', 'I like ', 'I need ']
    sentence_length = 50
    diversities = [0.2, 0.5, 1.0, 1.2]
    
    model_name = make_model_name('3')

    output_file = open('./' + model_name + '_output.txt', 'wb')

    if in_gcp == 1:
        output_file = open(fp.goog_file_path + model_name + '_output.txt', 'wb')

    print('', file=output_file)
    print("---------- Training Info: ---------- ", file=output_file)
    print('', file=output_file)
    print("Number of examples: " + str(X.shape[0]), file=output_file)
    print("Length of examples: " + str(maxlen), file=output_file)
    print("Step size used: " + str(step), file=output_file)
    print("Character set size: " + str(len(chars)), file=output_file)
    print("Batch size: " + str(batch_size), file=output_file)
    print("Number of Iterations: " + str(num_epochs), file=output_file)
    print("Generated sentence length: " + str(sentence_length), file=output_file)
    print("Sentence Seeds: " + str(sentence_seeds), file=output_file)
    print("Diversities: "  + str(diversities), file=output_file)
    print('', file=output_file)

    notes = "Testing this out"
    save_model_info(model, notes, output_file)
   
    for i in range(num_epochs):
        print("----- Epoch: " + str(i))
        print("---------- Iteration: " + str(i) + " ---------- ", file=output_file)
        print("", file=output_file)
        train_model(model, X[0:1000], y[0:1000], batch_size)
        first_thousand_loss = model.test_on_batch(X[0:1000], y[0:1000])
        print("Loss after training: " + str(first_thousand_loss), file=output_file)
        print("", file=output_file)
        for seed in sentence_seeds:
            print("----- Sentence seed: " + seed + "----- ", file=output_file)
            print('', file=output_file)
            for diversity in diversities:
                print("----- Sentence seed: " + seed + "----- ")
                print("Diversity: " + str(diversity))
                print("Diversity: " + str(diversity), file=output_file)
                
                random_sentence = generate_random_sentence(
                                                           model,
                                                           seed,
                                                           sentence_length,
                                                           diversity,
                                                           maxlen,
                                                           chars,
                                                           char_indices,
                                                           indices_char
                                                           )
                print("Generated Sentence: " + random_sentence)
                print("Generated Sentence: " + random_sentence, file=output_file)
                print("", file=output_file)
                print()

    
    output_file.close()
    
    if save_weights == 1:
        save_model_weights(model, model_name, in_gcp)           
    
    
#%%  
    
    
if __name__ == '__main__':
    main()


"""
for iteration in range(1, 3):
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

model.save_weights(fp.goog_file_path + 'model_weights.hdf5')

print("Completed running script")

"""

