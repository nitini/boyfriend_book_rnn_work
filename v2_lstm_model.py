# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 23:04:37 2016

@author: nitini
"""
#%%
from __future__ import print_function
import numpy as np
import random
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
import json
import pandas as pd
import file_paths as fp
import sys


def pad_sequences(data, feat, fixed_length):
    padded_tweets = data.apply(lambda x: x[feat] + str('~' * (fixed_length - len(x[feat])))
                               if len(x[feat]) < fixed_length else x[feat][0:fixed_length],
                               axis=1)
    return padded_tweets
    
def shift_sequences(data, feat, seq_len):
    shifted_tweets = data[feat].apply(lambda x: str(x + ' ')[1:seq_len + 1])
    return shifted_tweets

 
def vectorize_sequences(sequences, chars, char_indices):
    """
    
    Take the training data in text form and convert it to a 3D tensor
    that represents the text in a form that can be inputted to the model

    Inputs:


    Outputs:

    """

    sequence_vectors = np.zeros((sequences.shape[0], 
                                 len(sequences.iloc[0]), 
                                 len(chars)))

    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            sequence_vectors[i, t, char_indices[char]] = 1

    return sequence_vectors
    

def get_chars(data):
    text = ''

    for i in range(data.shape[0]):
        text = text + ' ' + data.tweet.iloc[i]

    chars = set(text)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    return chars, char_indices, indices_char
    
def yield_batches(batch_size, num_samples):
    for i in range(0,num_samples, batch_size):
        yield i, i + batch_size
    


def build_model(infer, batch_size, seq_len, vocab_size, lstm_size, num_layers):
    if infer:
        batch_size = seq_len = 1
    model = Sequential()
    model.add(LSTM(lstm_size,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab_size),
                   stateful=True))

    model.add(Dropout(0.2))
    for l in range(num_layers - 1):
        model.add(LSTM(lstm_size, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def modify_prob_dist(a, temperature=1.0):
    """
    Samples an index from a given probability distribution,
    can modify that distribution by changing the temperature

    Inputs:
        a: Probability distribution to sample from
        temperature: Scalar to modify prbability distribution by,
                     setting temperature to 1 does nothing

    Outputs:
        val: Index from distribution that has the highest likelihood
        
    """
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return a


def sample(test_model, 
           weights_file, 
           char_indices,
           indices_char,
           temperature=1.0,
           sample_chars=40,
           primer_text='i want'):
    
    test_model.reset_states()
    test_model.load_weights(weights_file)
    sampled = [char_indices[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, len(char_indices)))
        batch[0, 0, char_indices[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, len(char_indices)))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        softmax = modify_prob_dist(softmax, temperature)

        sample = np.random.choice(range(len(char_indices)), p=softmax)
        #sample = np.argmax(np.random.multinomial(1,softmax,1))

        sampled.append(sample)
    
    return ''.join([indices_char[c] for c in sampled])
    

def sample_given_model(test_model,
                       char_indices, 
                       temperature=1.0,
                       sample_chars=40,
                       primer_text='i'):

    sampled = [char_indices[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, len(char_indices)))
        batch[0, 0, char_indices[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, len(char_indices)))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        softmax = modify_prob_dist(softmax, temperature)

        #sample = np.random.choice(range(len(char_indices)), p=softmax)
        sample = np.argmax(np.random.multinomial(1,softmax,1))

        sampled.append(sample)
    
    return ''.join([indices_char[c] for c in sampled])


def make_model_name(model_version):
    """
    Takes the current time and combines that with the model version to 
    create a name for the model, used to save various associated files

    Inputs:
        model_version: Version of the model to create name
    
    Outputs:
        model_name: resulting model name
    """
    current_date = time.strftime("%m_%d_%Y")
    model_name = current_date + '_' + 'model_v' + str(model_version)
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
    """
    Saves the model weights so training doesn't need to be done again,
    can save locally or in google cloud storage

    Inputs:
        model: The trained keras LSTM model
        model_name: Name for the model to save the weights
        in_gcp: Flag for where to save the weights file
    
    Outputs:
        No output is returned, model is saved and function exits

    """
    model_weights_file_name = './' + model_name + '_weights.hdf5'
    if in_gcp == 1:
        model_weights_file_name = fp.goog_file_path + model_name + '_weights.hdf5'
   
    model.save_weights(model_weights_file_name, overwrite=True)

    return model_weights_file_name

    
#%%
def main():
    
    bf_tweets = pd.read_csv('./old_tweet_data/old_tweet_data_unweighted.csv')
    
    in_gcp = int(sys.argv[1])
    
    if in_gcp == 1:
        bf_tweets = pd.read_csv(fp.goog_file_path + 'old_tweet_data_weighted.csv')
       
    chars, char_indices, indices_char = get_chars(bf_tweets)
    
    SEQ_LEN = 75
    BATCH_SIZE = 256
    VOCAB_SIZE = len(chars)
    LAYERS = 3
    LSTM_SIZE = 256
    NUM_SAMPLES = (bf_tweets.shape[0]  / BATCH_SIZE) * BATCH_SIZE
    EPOCHS = 75
    
    bf_tweets['tweet'] = pad_sequences(bf_tweets, 'tweet', SEQ_LEN)
    bf_tweets['shifted_tweet'] = shift_sequences(bf_tweets, 'tweet', SEQ_LEN)

    bf_tweets = bf_tweets.iloc[0:NUM_SAMPLES].copy()
    
    X_seq_vectors = vectorize_sequences(bf_tweets['tweet'], 
                                        chars, 
                                        char_indices)
    y_seq_vectors = vectorize_sequences(bf_tweets['shifted_tweet'],
                                        chars,
                                        char_indices)

    training_model = build_model(0, 
                                 BATCH_SIZE, 
                                 SEQ_LEN, 
                                 VOCAB_SIZE,
                                 LSTM_SIZE,
                                 LAYERS)
                                 
    primer_texts = ['my',
                    'his', 
                    'when']
    sample_length = 80
    diversities = [0.2, 0.5, 1.0, 1.2]

    model_name = make_model_name('5')
    output_file = open('./output_' + model_name + '.txt', 'wb')
    
    if in_gcp == 1:
        output_file = open(fp.goog_file_path + 'output_' + model_name + '.txt', 'wb')
        
    
    print('', file=output_file)
    print("---------- Training Info: ---------- ", file=output_file)
    print('', file=output_file)
    print("Number of examples: " + str(NUM_SAMPLES), file=output_file)
    print("Length of examples: " + str(SEQ_LEN), file=output_file)
    print("Character set size: " + str(len(chars)), file=output_file)
    print("Batch size: " + str(BATCH_SIZE), file=output_file)
    print("Number of epochs: " + str(EPOCHS), file=output_file)
    print("Number of batches per epoch: " + str(NUM_SAMPLES / BATCH_SIZE), file=output_file)
    print("Generated sentence length: " + str(sample_length), file=output_file)
    print("Sentence Seeds: " + str(primer_texts), file=output_file)
    print("Diversities: "  + str(diversities), file=output_file)
    print('', file=output_file)
    
    notes = "Unweighted Twitter data, semi-long sentence"
    save_model_info(training_model, notes, output_file)
    print('', file=output_file)
    
    print("Number of epochs: " + str(EPOCHS))
    sys.stdout.flush()
    print("Number of batches per epoch: " + str(NUM_SAMPLES / BATCH_SIZE))
    sys.stdout.flush()
    
    for epoch in range(EPOCHS):
        print("----- Epoch: " + str(epoch))
        sys.stdout.flush()
        print("---------- Epoch: " + str(epoch) + " ---------- ", file=output_file)
        
        print("", file=output_file)
        for i, (start, end) in enumerate(yield_batches(BATCH_SIZE, NUM_SAMPLES)):
            batch_X = X_seq_vectors[start:end,:,:]
            batch_y = y_seq_vectors[start:end,:,:]
            loss = training_model.train_on_batch(batch_X, batch_y)
            print("Batch " + str(i) + ' / ' + str(NUM_SAMPLES / BATCH_SIZE) + ' of Epoch ' + str(epoch))
            sys.stdout.flush()
            print('Loss on batch ' + str(i) + ':' + str(loss))
            sys.stdout.flush()

        first_batch_loss = training_model.test_on_batch(X_seq_vectors[0:BATCH_SIZE,:,:], y_seq_vectors[0:BATCH_SIZE,:,:])
        print("Loss on first batch after epoch " + str(epoch) + ": " + str(first_batch_loss), file=output_file)
        print("", file=output_file)

        model_weights_name = model_name
        model_weights_file_name = save_model_weights(training_model, model_weights_name, in_gcp)

        test_model = build_model(1,
                                 BATCH_SIZE,
                                 SEQ_LEN,
                                 VOCAB_SIZE,
                                 LSTM_SIZE,
                                 LAYERS)

        for primer in primer_texts:
            print("----- Sentence seed: " + primer + " ----- ", file=output_file)
            print('', file=output_file)
            for diversity in diversities:
                print("----- Sentence seed: " + primer + " ----- ")
                sys.stdout.flush()
                print("Diversity: " + str(diversity))
                sys.stdout.flush()
                print("Diversity: " + str(diversity), file=output_file)

                sampled_tweet = sample(test_model, 
                                       model_weights_file_name, 
                                       char_indices,
                                       indices_char,
                                       diversity,
                                       sample_length, 
                                       primer)

                print("Generated Tweet: " + sampled_tweet)
                sys.stdout.flush()
                print("Generated Tweet: " + sampled_tweet, file=output_file)
                print("", file=output_file)

if __name__ == '__main__':
    main()
    
    
