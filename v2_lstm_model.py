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
from keras.utils.data_utils import get_file
from keras.optimizers import RMSprop

def clear():
    print("\n"*600)

#%%


def get_nietzsche_data():
    path = get_file('nietzsche.txt', 
                    origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    
    text = open(path).read().lower().decode('ascii','ignore').encode('ascii','ignore')
    
    return text


def convert_text_to_data(text, seq_len):
    num_seqs = len(text) / seq_len
    text_data = pd.DataFrame(columns=['text'], index=range(num_seqs))
    for i in range(num_seqs):
        text_data.text.iloc[i] = text[i*seq_len:i*seq_len + seq_len]
    return text_data


def pad_sequences(data, feat, fixed_length):
    padded_tweets = data.apply(lambda x: str('~' * (fixed_length - len(x[feat]) - 1)) + '>' + x[feat]
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
    

def get_chars(data, feat):
    text = '~'

    for i in range(data.shape[0]):
        text = text + ' ' + data[feat].iloc[i]

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
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, sample_weight_mode='temporal')
    return model

def build_model_variable_length(batch_size,
                                seq_len,
                                vocab_size,
                                lstm_size,
                                num_layers,
                                model_weights_file):

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
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, sample_weight_mode='temporal')

    model.load_weights(model_weights_file)
    #model.reset_states()
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

        sample = np.random.choice(range(len(char_indices)), p=softmax)
        #sample = np.argmax(np.random.multinomial(1,softmax,1))

        sampled.append(sample)
    
    return ''.join([char_indices[c] for c in sampled])


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
    print('---------- All model information: ---------- ', file=output_file)
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


def check_terminate_training_early(loss_values):
    return 0


def train_batches(training_model, X, y, sample_weights, epoch, BATCH_SIZE, NUM_SAMPLES, loss_values, output_file):
    
    for i, (start, end) in enumerate(yield_batches(BATCH_SIZE, NUM_SAMPLES)):
        batch_X = X[start:end,:,:]
        batch_y = y[start:end,:,:]
        batch_sample_weights = sample_weights[start:end][:]
        loss = training_model.train_on_batch(batch_X, batch_y, sample_weight=batch_sample_weights)
        loss_values[i] = loss
        if check_terminate_training_early(loss_values) == 1:
            return 1

        #print("Batch " + str(i) + ' / ' + str(NUM_SAMPLES / BATCH_SIZE) + ' of Epoch ' + str(epoch), file=output_file)
        print("Batch " + str(i) + ' / ' + str(NUM_SAMPLES / BATCH_SIZE) + ' of Epoch ' + str(epoch))
        sys.stdout.flush()
        #print('Loss on batch ' + str(i) + ': ' + str(loss), file=output_file)
        print('Loss on batch ' + str(i) + ': ' + str(loss))
        sys.stdout.flush()  
    return 0, loss_values

def create_sample_weights(num_samples, max_seq_len, data, feat):
    sample_weights = np.ones((num_samples, max_seq_len))
    for i in range(sample_weights.shape[0]):
        padded_sequence = data[feat].iloc[i]
        for c, j in zip(padded_sequence, range(sample_weights.shape[1])):
            if c == '~':
                sample_weights[i,j] = 0

    return sample_weights

def train_one_sample_at_a_time(lyrics, feat):

    chars, char_indices, indices_char = get_chars(lyrics, feat)

    LAYERS = 3
    LSTM_SIZE = 512
    EPOCHS = 100
    BATCH_SIZE = 1
    VOCAB_SIZE = len(chars)

    build_first_model = 0

    model_name = make_model_name('8')

    output_file = open('./output_' + model_name + '.txt', 'wb')

    model_weights_file_name = ''

    primer_texts = ['my ',
                    'i ',
                    'you ']

    diversities = [0.2, 0.5, 1.0, 1.2]

    for epoch in range(EPOCHS):
        print("----- Epoch: " + str(epoch))
        sys.stdout.flush()
        print("---------- Epoch: " + str(epoch) + " ---------- ", file=output_file)

        for i in range(lyrics.shape[0]):

            print("Stanza " + str(i) + " / " + str(lyrics.shape[0]))
            print("Stanza " + str(i) + " / " + str(lyrics.shape[0]), file=output_file)

            SEQ_LEN = len(lyrics.stanza.iloc[i])
            NUM_SAMPLES = 1

            lyric = lyrics.iloc[i:i+1].copy()

            loss_values = np.empty(1 / 1)

            X_seq_vectors = vectorize_sequences(lyric['stanza'],
                                                chars,
                                                char_indices)

            y_seq_vectors = vectorize_sequences(lyric['stanza'],
                                                chars,
                                                char_indices)

            training_model = build_model(0,
                                         BATCH_SIZE,
                                         SEQ_LEN,
                                         VOCAB_SIZE,
                                         LSTM_SIZE,
                                         LAYERS
                                         )

            if build_first_model != 0:
                training_model = build_model_variable_length(BATCH_SIZE,
                                                             SEQ_LEN,
                                                             VOCAB_SIZE,
                                                             LSTM_SIZE,
                                                             LAYERS,
                                                             model_weights_file_name)
                build_first_model = 1

            sample_weights = create_sample_weights(NUM_SAMPLES, SEQ_LEN, lyric, 'stanza')

            terminate_flag, loss_values = train_batches(training_model,
                                                        X_seq_vectors,
                                                        y_seq_vectors,
                                                        sample_weights,
                                                        epoch,
                                                        BATCH_SIZE,
                                                        NUM_SAMPLES,
                                                        loss_values,
                                                        output_file)

            model_weights_file_name = save_model_weights(training_model,
                                                         model_name,
                                                         0)

            print("Epoch: " + str(epoch), file=output_file)
            print("Loss for " + str(SEQ_LEN) + " length stanza", file=output_file)
            print("Number of batches: " + str(NUM_SAMPLES / BATCH_SIZE), file=output_file)
            print("Number of samples: " + str(NUM_SAMPLES), file=output_file)
            print(np.array_str(loss_values), file=output_file)
            print("", file=output_file)

def train_variable_length_model_swapped():

    lyrics_200_400 = pd.read_pickle('./lyrics_pickles/lyrics_200_400.pkl')
    lyrics_800_1000 = pd.read_pickle('./lyrics_pickles/lyrics_800_1000.pkl')

    feat = 'stanza'

    #lyrics_full = pd.concat([lyrics_200_400, lyrics_800_1000], ignore_index=True)

    lyrics_full = pd.read_pickle('./lyrics_pickles/mnm_jyz_stanzas.pkl')

    chars, char_indices, indices_char = get_chars(lyrics_full,
                                                  feat)

    lyrics_less_100 = lyrics_full[lyrics_full.len_of_stanza < 100].copy()
    lyrics_100_300 = lyrics_full[(lyrics_full.len_of_stanza > 100) & (lyrics_full.len_of_stanza < 300)].copy()
    lyrics_300_600 = lyrics_full[(lyrics_full.len_of_stanza > 300) & (lyrics_full.len_of_stanza < 600)].copy()
    lyrics_600_800 = lyrics_full[(lyrics_full.len_of_stanza > 600) & (lyrics_full.len_of_stanza < 800)].copy()
    lyrics_800_1100 = lyrics_full[(lyrics_full.len_of_stanza > 800) & (lyrics_full.len_of_stanza < 1100)].copy()
    lyrics_more_1100 = lyrics_full[(lyrics_full.len_of_stanza > 1100)].copy()

    stanzas_batches = [lyrics_less_100,
                       lyrics_100_300,
                       lyrics_300_600,
                       lyrics_600_800,
                       lyrics_800_1100,
                       lyrics_more_1100]

    LAYERS = 3
    LSTM_SIZE = 512
    EPOCHS = 16
    BATCH_SIZE = 32
    VOCAB_SIZE = len(chars)

    build_first_model = 1

    model_name = make_model_name('7')

    output_file = open('./output_' + model_name + '.txt', 'wb')

    model_weights_file_name = ''

    primer_texts = ['my ',
                    'i ',
                    'you ']

    diversities = [0.2, 0.5, 1.0, 1.2]

    stanza_batches_counter = 0
    num_stanza_batches = len(stanza_batches)

    for lyrics in stanzas_batches:

        SEQ_LEN = int(np.max(lyrics.len_of_stanza))
        sample_length = SEQ_LEN
        lyrics['padded_stanza'] = pad_sequences(lyrics, feat, SEQ_LEN)
        lyrics['shifted_stanza'] = shift_sequences(lyrics, 'padded_stanza', SEQ_LEN)
        NUM_SAMPLES = (lyrics.shape[0] / BATCH_SIZE) * BATCH_SIZE

        loss_values = np.empty(NUM_SAMPLES / BATCH_SIZE)

        lyrics = lyrics.iloc[0:NUM_SAMPLES].copy()

        X_seq_vectors = vectorize_sequences(lyrics['padded_stanza'],
                                            chars,
                                            char_indices)

        y_seq_vectors = vectorize_sequences(lyrics['shifted_stanza'],
                                            chars,
                                            char_indices)

        training_model = build_model(0,
                                     BATCH_SIZE,
                                     SEQ_LEN,
                                     VOCAB_SIZE,
                                     LSTM_SIZE,
                                     LAYERS
                                     )

        if build_first_model != 0:
            training_model = build_model_variable_length(BATCH_SIZE,
                                                         SEQ_LEN,
                                                         VOCAB_SIZE,
                                                         LSTM_SIZE,
                                                         LAYERS,
                                                         model_weights_file_name)
            build_first_model = 1

        sample_weights = create_sample_weights(NUM_SAMPLES, SEQ_LEN, lyrics, 'padded_stanza')

        print(str(stanza_batches_counter) + ' of ' + str(num_stanza_batches) + " stanza batches", file=output_file)
        print("Padded " + str(SEQ_LEN) + " length stanzas", file=output_file)
        print("Number of batches: " + str(NUM_SAMPLES / BATCH_SIZE), file=output_file)
        print("Number of samples: " + str(NUM_SAMPLES), file=output_file)
        print("", file=output_file)

        for epoch in range(EPOCHS):
            print("----- Epoch: " + str(epoch))
            sys.stdout.flush()
            print("---------- Epoch: " + str(epoch) + " ---------- ", file=output_file)

            terminate_flag, loss_values = train_batches(training_model,
                                                        X_seq_vectors,
                                                        y_seq_vectors,
                                                        sample_weights,
                                                        epoch,
                                                        BATCH_SIZE,
                                                        NUM_SAMPLES,
                                                        loss_values,
                                                        output_file)

            print("Epoch: " + str(epoch), file=output_file)
            print(np.array_str(loss_values), file=output_file)
            print("", file=output_file)

        model_weights_file_name = save_model_weights(training_model,
                                                        model_name,
                                                        0)

        stanza_batches_counter += 1

def train_variable_length_model():

    lyrics_200_400 = pd.read_pickle('./lyrics_pickles/lyrics_200_400.pkl')
    lyrics_800_1000 = pd.read_pickle('./lyrics_pickles/lyrics_800_1000.pkl')

    feat = 'stanza'

    lyrics_full = pd.concat([lyrics_200_400, lyrics_800_1000], ignore_index=True)

    #figure out way to batch lyrics into stanza length intervals

    chars, char_indices, indices_char = get_chars(lyrics_full,
                                                  feat)

    LAYERS = 3
    LSTM_SIZE = 512
    EPOCHS = 50
    BATCH_SIZE = 32
    VOCAB_SIZE = len(chars)

    build_first_model = 0

    model_name = make_model_name('7')

    output_file = open('./output_' + model_name + '.txt', 'wb')

    model_weights_file_name = ''

    primer_texts = ['my ',
                    'i ',
                    'you ']

    diversities = [0.2, 0.5, 1.0, 1.2]

    for epoch in range(EPOCHS):

        print("----- Epoch: " + str(epoch))
        sys.stdout.flush()
        print("---------- Epoch: " + str(epoch) + " ---------- ", file=output_file)

        print("", file=output_file)

        stanza_batches_counter = 0
        num_stanza_batches = len([lyrics_200_400, lyrics_800_1000])

        for lyrics in [lyrics_200_400, lyrics_800_1000]:


            SEQ_LEN = int(np.max(lyrics.len_of_stanza))
            sample_length = SEQ_LEN
            lyrics['padded_stanza'] = pad_sequences(lyrics, feat, SEQ_LEN)
            lyrics['shifted_stanza'] = shift_sequences(lyrics, 'padded_stanza', SEQ_LEN)
            NUM_SAMPLES = (lyrics.shape[0] / BATCH_SIZE) * BATCH_SIZE

            loss_values = np.empty(NUM_SAMPLES / BATCH_SIZE)

            lyrics = lyrics.iloc[0:NUM_SAMPLES].copy()

            X_seq_vectors = vectorize_sequences(lyrics['padded_stanza'],
                                                chars,
                                                char_indices)

            y_seq_vectors = vectorize_sequences(lyrics['shifted_stanza'],
                                                chars,
                                                char_indices)

            training_model = build_model(0,
                                         BATCH_SIZE,
                                         SEQ_LEN,
                                         VOCAB_SIZE,
                                         LSTM_SIZE,
                                         LAYERS
                                         )

            if build_first_model != 0:
                training_model = build_model_variable_length(BATCH_SIZE,
                                                             SEQ_LEN,
                                                             VOCAB_SIZE,
                                                             LSTM_SIZE,
                                                             LAYERS,
                                                             model_weights_file_name)
                build_first_model = 1

            sample_weights = create_sample_weights(NUM_SAMPLES, SEQ_LEN, lyrics, 'padded_stanza')

            terminate_flag, loss_values = train_batches(training_model,
                                                        X_seq_vectors,
                                                        y_seq_vectors,
                                                        sample_weights,
                                                        epoch,
                                                        BATCH_SIZE,
                                                        NUM_SAMPLES,
                                                        loss_values,
                                                        output_file)

            print("Epoch: " + str(epoch), file=output_file)
            print(str(stanza_batches_counter) + ' of ' + str(num_stanza_batches) + " stanza batches", file=output_file)
            print("Losses for padded " + str(SEQ_LEN) + " length stanzas", file=output_file)
            print("Number of batches: " + str(NUM_SAMPLES / BATCH_SIZE), file=output_file)
            print("Number of samples: " + str(NUM_SAMPLES), file=output_file)
            print(np.array_str(loss_values), file=output_file)
            print("", file=output_file)

            stanza_batches_counter += 1

            model_weights_file_name = save_model_weights(training_model,
                                                         model_name,
                                                         0)

            if terminate_flag == 1:
                print("Loss has seemed to asymptote, terminating program", file=output_file)
                break

            test_model = build_model(1,
                                     BATCH_SIZE,
                                     SEQ_LEN,
                                     VOCAB_SIZE,
                                     LSTM_SIZE,
                                     LAYERS)

            for primer in primer_texts:
                #print("----- Sentence seed: " + primer + " ----- ", file=output_file)
                #print("----- Sample Length: " + str(sample_length) + " -----",file=output_file)
                #print('', file=output_file)
                for diversity in diversities:
                    #print("----- Sentence seed: " + primer + " ----- ")
                    sys.stdout.flush()
                    #print("Diversity: " + str(diversity))
                    sys.stdout.flush()
                    #print("Diversity: " + str(diversity), file=output_file)

                    sampled_text = sample(test_model,
                                          model_weights_file_name,
                                          char_indices,
                                          indices_char,
                                          diversity,
                                          sample_length,
                                          primer)

                    #print("Generated Text: " + sampled_text)
                    sys.stdout.flush()
                    #print("Generated Text: " + sampled_text, file=output_file)
                    #print("", file=output_file)


#%%
def main():

    #train_variable_length_model_swapped()

    lyrics = pd.read_pickle('./lyrics_pickles/mnm_lyrics.pkl')

    lyrics.sort_values(by='len_of_stanza',inplace=True)

    train_one_sample_at_a_time(lyrics, 'stanza')


    """

    SEQ_LEN = 70
    BATCH_SIZE = 512
    LAYERS = 3
    LSTM_SIZE = 128
    EPOCHS = 100
    
    text_data = pd.read_csv('./old_tweet_data/old_tweet_data_unweighted.csv')
    
    #text_data = convert_text_to_data(get_nietzsche_data(), SEQ_LEN)
    
    feat = 'tweet'
    
    in_gcp = int(sys.argv[1])
    
    if in_gcp == 1:
        text_data = pd.read_csv(fp.goog_file_path + 'old_tweet_data_weighted.csv')
       
    chars, char_indices, indices_char = get_chars(text_data, feat)
    
    NUM_SAMPLES = (text_data.shape[0] / BATCH_SIZE) * BATCH_SIZE
    VOCAB_SIZE = len(chars)
    
    text_data[feat] = pad_sequences(text_data, feat, SEQ_LEN)
    text_data['shifted_text'] = shift_sequences(text_data, feat, SEQ_LEN)

    text_data = text_data.iloc[0:NUM_SAMPLES].copy()
    
    X_seq_vectors = vectorize_sequences(text_data[feat], 
                                        chars, 
                                        char_indices)

    y_seq_vectors = vectorize_sequences(text_data['shifted_text'],
                                        chars,
                                        char_indices)

    training_model = build_model(0, 
                                 BATCH_SIZE, 
                                 SEQ_LEN, 
                                 VOCAB_SIZE,
                                 LSTM_SIZE,
                                 LAYERS)
                                 
    primer_texts = ['my ',
                    'i ', 
                    'you ']

    sample_length = 80
    diversities = [0.2, 0.5, 1.0, 1.2]

    model_name = make_model_name('7')
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
    print("Diversities: " + str(diversities), file=output_file)
    print('', file=output_file)
    
    notes = "Unweighted Twitter data, semi-long sentence"
    save_model_info(training_model, notes, output_file)
    print('', file=output_file)
    
    print("Number of epochs: " + str(EPOCHS))
    sys.stdout.flush()
    print("Number of batches per epoch: " + str(NUM_SAMPLES / BATCH_SIZE))
    sys.stdout.flush()
    
    loss_values = []    
    
    for epoch in range(EPOCHS):
        print("----- Epoch: " + str(epoch))
        sys.stdout.flush()
        print("---------- Epoch: " + str(epoch) + " ---------- ", file=output_file)
        
        print("", file=output_file)
        
        terminate_flag = train_batches(training_model,
                                       X_seq_vectors,
                                       y_seq_vectors,
                                       epoch,
                                       BATCH_SIZE,
                                       NUM_SAMPLES,
                                       loss_values)
        
        if terminate_flag == 1:
            print("Loss has seemed to asymptote, terminating program", file=output_file)
            break

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

                sampled_text = sample(test_model, 
                                       model_weights_file_name, 
                                       char_indices,
                                       indices_char,
                                       diversity,
                                       sample_length, 
                                       primer)

                print("Generated Text: " + sampled_text)
                sys.stdout.flush()
                print("Generated Text: " + sampled_text, file=output_file)
                print("", file=output_file)
    """

if __name__ == '__main__':
    main()
