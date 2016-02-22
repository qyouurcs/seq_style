#!/usr/bin/python

import sys
import pickle
import random
import numpy as np

import theano
import theano.tensor as T
import lasagne
import ConfigParser

import pdb

from collections import Counter
from lasagne.utils import floatX

from data_provider import *

logging = climate.get_logger(__name__)
climate.enable_default_logging()


def load_vocab(vocab_fn):
    idx2word = {}
    word2idx = {}
    with open(vocab_fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            idx2word[int(parts[0])] = parts[1]
            word2idx[parts[1]] = int(parts[0])
    return idx2word, word2idx

def load_vocab_fea(word_vec_fn):
    word2vec_fea = {}
    with open(word_vec_fn,'r') as fid:
        for aline in fid:
            aline = aline.strip()
            parts = aline.split()
            if parts[0] in word2idx:
                vec_fea = np.array([ float(fea) for fea in parts[1:] ], dtype='float32')
                word2vec_fea[parts[0]] = vec_fea

    start_fea = np.zeros((word2vec_fea.values()[0].shape),dtype='float32')
    t_num_fea = start_fea.size
    # using a 1/n as features.
    # I think start token is special token. No idea, how to initialize it.  
    start_fea[:] = 1.0 / start_fea.size
    word2vec_fea['#START#'] = start_fea
    return word2vec_fea, t_num_fea

if __name__ == '__main__':

    cf = ConfigParser.ConfigParser()

    if len(sys.argv) < 2:
        logging.info("Usage: {0} <conf_fn> <model_fn>".format(sys.argv[0]))
        sys.exit()

    cf.read(sys.argv[1])
    model_fn = sys.argv[2]

    dataset = cf.get('INPUT', 'dataset')
    h_size=int(cf.get('INPUT','h_size'))
    e_size=int(cf.get('INPUT','e_size'))

    word_vec_fn = "."
    if cf.has_option('INPUT', 'word_vec_fn'):
        word_vec_fn = cf.get('INPUT', 'word_vec_fn')
    vocab_fn =  cf.get('INPUT', 'vocab_fn')

    save_dir=cf.get('OUTPUT', 'save_dir')

    d = pickle.load(open(model_fn))
    idx2word = d['vocab']
    params_loaded = d['param_vals']
 
    dp = getDataProvider(dataset)
    # Now, load the vocab.
    idx2word, word2idx = load_vocab(vocab_fn)
    word2vec_fea, t_num_fea = load_vocab_fea(word_vec_fn)
    logging.info('Total vocab has a total of %d words and feas %d', len(word2idx), len(word2vec_fea))

    SEQUENCE_LENGTH = 32
    MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3 # 1 for image, 1 for start token, 1 for end token
    BATCH_SIZE = 100
    CNN_FEATURE_SIZE = 1000
    EMBEDDING_SIZE = 256
    LEARNING_RATE = 0.0001
    save_fn = os.path.splitext(model_fn)[0] + '_finetune.pkl'

    logging.info("Saving model to {}".format(save_fn))
    batch_size = 256
    def vis_fea_len():
        pair = dp.sampleImageSentencePair()
        vis_fea_len = 0
        vis_fea_len = pair['image']['feat'].size
        return vis_fea_len
    vis_fea_len = vis_fea_len()
    def batch_train():
        batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]

        fea = np.zeros(( batch_size, SEQUENCE_LENGTH -1, t_num_fea), dtype='float32')
        label = np.zeros(( batch_size, SEQUENCE_LENGTH), dtype='int32') # label do not need the #START# token.
        mask = np.zeros(( batch_size, SEQUENCE_LENGTH - 1 ), dtype='int32')
        vis_fea = np.zeros((batch_size, vis_fea_len), dtype='float32')

        for i, pair in enumerate(batch):
            img_fn = pair['image']['filename']
            vis_fea[i,:] = np.squeeze(pair['image']['feat'][:])

            tokens = ['#START#']
            tokens.extend(pair['sentence']['tokens'][0:SEQUENCE_LENGTH-3])
            tokens.append('.')

            for j, w in enumerate(tokens):
                if w in word2idx:
                    fea[ i, j, :] = word2vec_fea[w]
                    mask[i, j] = 1.0
                    label[i, j] = word2idx[w]
        return fea, label, mask, vis_fea

    def batch_val():
        batch = [dp.sampleImageSentencePair('val') for i in xrange(batch_size)]

        fea = np.zeros(( batch_size, SEQUENCE_LENGTH -1, t_num_fea), dtype='float32')
        label = np.zeros(( batch_size, SEQUENCE_LENGTH), dtype='int32') # label do not need the #START# token.
        mask = np.zeros(( batch_size, SEQUENCE_LENGTH - 1), dtype='int32')
        vis_fea = np.zeros((batch_size, vis_fea_len), dtype='float32')

        for i, pair in enumerate(batch):
            img_fn = pair['image']['filename']
            vis_fea[i,:] = np.squeeze(pair['image']['feat'][:])

            tokens = ['#START#']
            tokens.extend(pair['sentence']['tokens'][0:SEQUENCE_LENGTH-2])
            tokens.append('.')

            for j,w in enumerate(tokens):
                if w in word2idx:
                    fea[ i, j, :] = word2vec_fea[w]
                    mask[i, j] = 1.0
                    label[i, j] = word2idx[w]
        return fea, label, mask, vis_fea

    # Now, we can start building the model
    l_input_sentence = lasagne.layers.InputLayer((batch_size, SEQUENCE_LENGTH - 1, t_num_fea))
    l_input_sentence_shp = lasagne.layers.ReshapeLayer(l_input_sentence, (-1, l_input_sentence.output_shape[2]))
    l_sentence_embedding = lasagne.layers.DenseLayer(l_input_sentence_shp, num_units = e_size,
                                                nonlinearity=lasagne.nonlinearities.identity)

    l_sentence_embedding_shp = lasagne.layers.ReshapeLayer(l_sentence_embedding,(l_input_sentence.output_shape[0],
        l_input_sentence.output_shape[1], e_size))
    l_input_cnn = lasagne.layers.InputLayer((batch_size, vis_fea_len))
    l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=e_size,
                                                nonlinearity=lasagne.nonlinearities.identity)
    
    l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))
    
    # the two are concatenated to form the RNN input with dim (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding_shp])
    
    l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
    l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                      num_units=e_size,
                                      unroll_scan=True,
                                      grad_clipping=5.)
    l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
    
    # the RNN output is reshaped to combine the batch and time dimensions
    # dim (BATCH_SIZE * SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, e_size))
    
    # decoder is a fully connected layer with one output unit for each word in the vocabulary
    l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(idx2word), nonlinearity=lasagne.nonlinearities.softmax)
    
    # finally, the separation between batch and time dimension is restored
    l_out = lasagne.layers.ReshapeLayer(l_decoder, (batch_size, SEQUENCE_LENGTH, len(idx2word)))
    
    # cnn feature vector
    x_cnn_sym = T.matrix()
    
    x_sentence_sym = T.tensor3()
    
    # mask defines which elements of the sequence should be predicted
    mask_sym = T.imatrix()
    
    # ground truth for the RNN output
    y_sentence_sym = T.imatrix()
    
    output = lasagne.layers.get_output(l_out, {
                    l_input_sentence: x_sentence_sym,
                    l_input_cnn: x_cnn_sym
    })
    
    
    def calc_cross_ent(net_output, mask, targets):
        # Helper function to calculate the cross entropy error
        preds = T.reshape(net_output, (-1, len(idx2word)))
        targets = T.flatten(targets)
        cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
        return cost
    
    loss = T.mean(calc_cross_ent(output, mask_sym, y_sentence_sym))
    
    MAX_GRAD_NORM = 15
    
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    lasagne.layers.set_all_param_values(l_out, params_loaded)

    all_grads = T.grad(loss, all_params)
    all_grads = [T.clip(g, -5, 5) for g in all_grads]
    all_grads, norm = lasagne.updates.total_norm_constraint(
        all_grads, MAX_GRAD_NORM, return_norm=True)
    
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=LEARNING_RATE)
    pdb.set_trace()
    
    f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
                              [loss, norm],
                              updates=updates
                             )
    
    f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], loss)
    
    for iteration in range(20000):
        fea, label, mask, vis_fea = batch_train()

        loss_train, norm = f_train(vis_fea, fea, mask, label)
        if not iteration % 250:
            logging.info('Iteration {}, loss_train: {}, norm: {}'.format(iteration, loss_train, norm))
            try:
                fea, label, mask, vis_fea = batch_val()
                loss_val = f_val(vis_fea, fea, mask, label)
                logging.info('Val loss: {}'.format(loss_val))
            except IndexError:
                continue        
    param_values = lasagne.layers.get_all_param_values(l_out)
    param_syms = lasagne.layers.get_all_params(l_out)
    param_strs = []
    for sym in param_syms:
        param_strs.append(str(sym))


    d = {'param_vals': param_values,
         'param_strs': param_strs,
         'vocab': idx2word,
        }
    pickle.dump(d, open(save_fn,'w'), protocol=pickle.HIGHEST_PROTOCOL)
