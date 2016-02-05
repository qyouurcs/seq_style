#!/usr/bin/python
import numpy as np
import theano
import theano.tensor as T
import lasagne


from lasagne.utils import floatX

import pickle
import gzip
import random
from collections import Counter
import pdb

corpus = gzip.open('claims.txt.gz').read()

print corpus.split('\n')[0]

VOCABULARY = set(corpus)
VOCAB_SIZE = len(VOCABULARY)

CHAR_TO_IX = {c: i for i,c in enumerate(VOCABULARY)}
IX_TO_CHAR = {i: c for i,c in enumerate(VOCABULARY)}

CHAR_TO_ONEHOT = {c: np.eye(VOCAB_SIZE)[i] for i,c in enumerate(VOCABULARY)}


SEQUENCE_LENGTH = 52

BATCH_SIZE = 50

RNN_HIDDEN_SIZE = 200

train_corpus = corpus[:(len(corpus) * 9 // 10)]
val_corpus = corpus[(len(corpus) * 9 // 10):]


def data_batch_generator(corpus, size = BATCH_SIZE):
    startidx = np.random.randint(0, len(corpus) - SEQUENCE_LENGTH + 1, size = size)

    while True:
        items = np.array([ corpus[start:start + SEQUENCE_LENGTH + 1] for start in startidx])
        startidx = ( startidx + SEQUENCE_LENGTH) % (len(corpus) - SEQUENCE_LENGTH - 1)
        yield items

gen = data_batch_generator(corpus, size = 1)

print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))


def prep_batch_for_network(batch):
    x_seq = np.zeros((len(batch), SEQUENCE_LENGTH, VOCAB_SIZE), dtype = 'float32')
    y_seq = np.zeros((len(batch), SEQUENCE_LENGTH), dtype = 'int32')

    for i, item in enumerate(batch):
        for j in range(SEQUENCE_LENGTH):
            x_seq[i,j] = CHAR_TO_ONEHOT[item[j]]
            y_seq[i,j] = CHAR_TO_IX[item[j+1]]

    print 'x_seq.shape ', x_seq.shape
    print 'y_seq.shape ', y_seq.shape
    return x_seq, y_seq

### Now, we define the symbolic variables for input.

x_sym = T.tensor3()
y_sym = T.imatrix()

hid_init_sym = T.matrix()
hid2_init_sym = T.matrix()

l_input = lasagne.layers.InputLayer((None, SEQUENCE_LENGTH, VOCAB_SIZE))
l_rnn = lasagne.layers.GRULayer(l_input, num_units = RNN_HIDDEN_SIZE, grad_clipping = 5, hid_init = hid_init_sym,)
l_rnn2 = lasagne.layers.GRULayer(l_rnn, num_units = RNN_HIDDEN_SIZE, grad_clipping = 5, hid_init = hid2_init_sym,)

l_shp = lasagne.layers.ReshapeLayer(l_rnn2, (-1, RNN_HIDDEN_SIZE))

l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=VOCAB_SIZE, nonlinearity = lasagne.nonlinearities.softmax)

l_out = lasagne.layers.ReshapeLayer(l_decoder, (-1, SEQUENCE_LENGTH, VOCAB_SIZE))


hid_out, hid2_out, prob_out = lasagne.layers.get_output([l_rnn, l_rnn2, l_out], {l_input:x_sym})

hid_out = hid_out[:,-1]
hid2_out = hid2_out[:,-1]

def calc_cross_ent(net_output, targets):
    preds = T.reshape(net_output, (-1, VOCAB_SIZE))
    targets = T.flatten(targets)

    cost = T.nnet.categorical_crossentropy(preds, targets)
    return cost

loss = T.mean(calc_cross_ent(prob_out, y_sym))

MAX_GRAD_NORM = 15
all_params = lasagne.layers.get_all_params(l_out, trainable = True)

all_grads = T.grad(loss, all_params)
all_grads = [ T.clip (g, -5, 5) for g in all_grads ]
all_grads, norm = lasagne.updates.total_norm_constraint( all_grads, MAX_GRAD_NORM, return_norm = True)

updates = lasagne.updates.adam(all_grads, all_params, learning_rate = 0.002)

f_train = theano.function([x_sym, y_sym, hid_init_sym, hid2_init_sym], [ loss, norm, hid_out, hid2_out], updates = updates)

f_val = theano.function([x_sym, y_sym, hid_init_sym, hid2_init_sym], [loss, hid_out, hid2_out])

hid = np.zeros((BATCH_SIZE, RNN_HIDDEN_SIZE), dtype = 'float32')
hid2 = np.zeros((BATCH_SIZE, RNN_HIDDEN_SIZE), dtype = 'float32')

train_batch_gen = data_batch_generator(train_corpus)

for iteration in range(20000):
    x,y =  prep_batch_for_network(next(train_batch_gen))
    loss_train, norm, hid, hid2 = f_train(x,y, hid, hid2)
    pdb.set_trace()

    if iteration % 250 == 0:
        print('Iteration {}, loss_train: {}, norm:{}'.format(iteration, loss_train, norm))
 
param_values = lasagne.layers.get_all_param_values(l_out)

d = {'param values': param_values,
    'VOCABULARY': VOCABULARY, 
    'CHAR_TO_IX': CHAR_TO_IX,
    'IX_TO_CHAR': IX_TO_CHAR,
    }
pickle.dump(d, open('gru_2layer_trained.pkl','w'), protocol=pickle.HIGHEST_PROTOCOL)

