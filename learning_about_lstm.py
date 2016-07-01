import sys
import numpy as np
import random
import time
 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

#%%

T = ['a','b','c','d','e','f','g','h',
     'i','j','k','l','m','n','o','p',
     'q','r','s','t','u','v','w','x',
     'y', 'z','1','2','3','4','5','6']

SEQ_LEN = 4
BATCH_SIZE = 
BATCH_CHARS  = len(T) / BATCH_SIZE
NUM_BATCHES = (len(T) / BATCH_SIZE) / SEQ_LEN
NUM_BATCHES_2 = len(range(0,BATCH_CHARS - SEQ_LEN - 1, SEQ_LEN))

x = [[0,0,0,0],[0,0,0,0]]

print 'Sequence: ', '  '.join(str(c) for c in T)
for i in range(0, BATCH_CHARS - SEQ_LEN + 1, SEQ_LEN):
    print 'BATCH', i/SEQ_LEN
    for batch_idx in range(BATCH_SIZE):
        start = batch_idx * BATCH_CHARS + i          
        print '\tsequence', batch_idx, 'of batch:',
        for j in range(SEQ_LEN):
            x[batch_idx][j] = T[start+j]
            print T[start+j],
    # here we would yield x (the batch) if this were an iterator
        print



allChars= ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
allChars *= 10
 
 
text = ''.join(allChars)
 
 
def chunks(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]
    

totalTimeSteps = 12

text = text[:len(text) // totalTimeSteps  * totalTimeSteps]

chars = set(text)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

batches = chunks(text,totalTimeSteps)
batchSize = len(batches)

featurelen = len(chars)
numOfPrevSteps = 1 # We are only looking at the most recent character each time.


X = np.zeros([batchSize, totalTimeSteps , featurelen])
for b in range(len(batches)):
    for r in range(totalTimeSteps):
        currentChar = text[r + b*totalTimeSteps]
        X[b][r][char_indices[currentChar]] = 1
        
print('Formatted Data ',X)

i = 0
for matrix in X:
    cl = ''
    for row in matrix:
        mi = list(row).index(max(row))
        c = indices_char[mi]
        cl = cl + c
    i += 1
    print('batch ',i,cl)
    
    
model = Sequential()
model.add(LSTM(512,
               return_sequences=True,
               batch_input_shape=(batchSize, 
                                  numOfPrevSteps, 
                                  featurelen), stateful=True))

model.add(Dropout(0.2))
model.add(LSTM(512 , return_sequences=False,stateful=True))
model.add(Dropout(0.2))
model.add(Dense( featurelen ))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.reset_states()




num_epochs = 100
for e in range(num_epochs):
    print('epoch - ',e+1)
    startTime = time.time()
    for i in range(0,totalTimeSteps-1):
        model.train_on_batch(X[:, numOfPrevSteps*i:(i+1)*numOfPrevSteps, :], 
                             np.reshape(X[:, (i+1)*numOfPrevSteps, :], 
                            (batchSize, featurelen)) )
                            
#%%

 bf_text_sample = 'I want my boyfriend to flaunt me around. I want my boyfriend to tell his friends that we have great sex often even if we havent done it that much in the past couple weeks. I want my boyfriend to be humble. I want my boyfriend to be confident and competitive but not blame the equipment of the game if he is losing.  I want my boyfriend to be athletic.  I want my boyfriend to be extra horny after playing sports.  I want my boyfriend to be cultured.  I want my boyfriend to know that I dont date casually.  I want my boyfriend to be someone I can imagine myself definitely marrying.  I want my boyfriend to be sensitive.  I want my boyfriend to appreciate beauty, art, and quiet.  I want my boyfriend to not stop when I am cumming.  I want my boyfriend to dress well, like manly in jeans and button down shirts but not button downs that are too tight.  I want my boyfriend to be up to date with pop culture.  I want my boyfriend to like rap music and not heavy metal.  I want my boyfriend to like music but not love it.  I want my boyfriend to be proud of me in public.  I want my boyfriend to be an avid reader.  I want my boyfriend to remind me of my dad, not sexually, but in all the good nurturing, protective ways.  I want my boyfriend to be friends with my friends.  I want my boyfriend to be taller than 5 11 because I dont want other men  towering over him.  I want my boyfriend to be older than me. I want my boyfriend to text me every morning when he wakes up.  I want my boyfriend to be strong, as in gym strong with muscles, but not too much. I want my boyfriend to be able to grow facial hair but not need it to look good or old.  I want my boyfriend to have high paying job, undecided about who should earn more because if money = power, then I want power too.  I want my boyfriend to be motivated to work and go to work and advance in his job.  I want my boyfriend to say he loves me often and mean it every time. I want my boyfriend to cuddle with me after sex. I want my boyfriend to be kind of aggressive with me during sex, the take charge kind not the hurt me kind.  I want my boyfriend to clean his penis before I give him head; if not that, then definitely not have peed within the past hour before my giving him head. I want my boyfriend to surprise me with gifts pretty often.  I want my boyfriend to buy me things.  I want my boyfriend to save money. I want my boyfriend to be good looking but not so good looking that he attracts too many women.  I want my boyfriend to only look at me. I want my boyfriend to not be scared to look at other girls but oaaaaaaaaaaaaaaaaaaaaa'

char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(bf_text_sample))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}                       
vocab_size = len(char_to_idx)
                            
SEQ_LEN = 64
BATCH_SIZE = 16
BATCH_CHARS = len(bf_text_sample) / BATCH_SIZE
LSTM_SIZE = 128
LAYERS = 3

NUM_BATCHES = (len(bf_text_sample) / BATCH_SIZE) / SEQ_LEN
NUM_BATCHES_2 = len(range(0,BATCH_CHARS - SEQ_LEN  - 1, SEQ_LEN))


def read_batches(text):
    T = np.asarray([char_to_idx[c] for c in bf_text_sample], dtype=np.int32)
    X = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    Y = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    X_string = ''
    Y_string = ''
    for i in range(0, BATCH_CHARS - SEQ_LEN - 1, SEQ_LEN):
        X[:] = 0
        Y[:] = 0
        for batch_idx in range(BATCH_SIZE):
            X_string = ''
            Y_string = ''
            start = batch_idx * BATCH_CHARS + i
            for j in range(SEQ_LEN):
                X_string += bf_text_sample[start+j]
                Y_string += bf_text_sample[start+j+1]
                X[batch_idx, j, T[start+j]] = 1
                Y[batch_idx, j, T[start+j+1]] = 1

        yield X, Y


def build_model(infer):
    if infer:
        batch_size = seq_len = 1
    else:
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
    model = Sequential()
    model.add(LSTM(LSTM_SIZE,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab_size),
                   stateful=True))

    model.add(Dropout(0.2))
    for l in range(LAYERS - 1):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model
    



print 'Building model.'
training_model = build_model(infer=False)
test_model = build_model(infer=True)
print '... done'


def sample(epoch, sample_chars=256, primer_text='And the '):
    test_model.reset_states()
    test_model.load_weights('/tmp/keras_char_rnn.%d.h5' % epoch)
    sampled = [char_to_idx[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, char_to_idx[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=softmax)
        sampled.append(sample)

    print ''.join([idx_to_char[c] for c in sampled])

for epoch in range(10):
    for i, (x, y) in enumerate(read_batches(bf_text_sample)):
        loss = training_model.train_on_batch(x, y)
        print epoch, i, loss

        if epoch % 10 == 0:
            training_model.save_weights('./keras_char_rnn.%d.h5' % epoch,
                                        overwrite=True)


            sample(epoch)












#%% since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 25
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = 1


def gen_cosine_amp(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing
    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(idx / (2 * np.pi * period))
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()
model.add(LSTM(50,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)
    model.fit(cos,
              expected_output,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False)
    model.reset_states()

print('Predicting')
predicted_output = model.predict(cos, batch_size=batch_size)








