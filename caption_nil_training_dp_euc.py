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
        logging.info("Usage: {0} <conf_fn>".format(sys.argv[0]))
        sys.exit()

    cf.read(sys.argv[1])

    dataset = cf.get('INPUT', 'dataset')
    h_size=int(cf.get('INPUT','h_size'))
    e_size=int(cf.get('INPUT','e_size'))
    lm_model_fn=(cf.get('INPUT', 'lm_model_fn'))

    word_vec_fn = "."
    if cf.has_option('INPUT', 'word_vec_fn'):
        word_vec_fn = cf.get('INPUT', 'word_vec_fn')
    vocab_fn =  cf.get('INPUT', 'vocab_fn')
    save_dir=cf.get('OUTPUT', 'save_dir')
    d = pickle.load(open(lm_model_fn))
    params_loaded = d['param_vals']
    params_strs = d['param_strs']

    dict_params = {}
    for p,n in zip(params_loaded, params_strs):
        dict_params[n] = p

    dp = getDataProvider(dataset)
    # Now, load the vocab.
    idx2word, word2idx = load_vocab(vocab_fn)
    word2vec_fea, t_num_fea = load_vocab_fea(word_vec_fn)
    word2vec_fea_np = np.zeros((len(idx2word), t_num_fea), dtype = 'float32')
    for i in range(len(idx2word)):
        word2vec_fea_np[i,:] = word2vec_fea[idx2word[i]] / np.linalg.norm(word2vec_fea[idx2word[i]])
        #word2vec_fea_np[i,:] = word2vec_fea[idx2word[i]]

    logging.info('Total vocab has a total of %d words and feas %d', len(word2idx), len(word2vec_fea))

    SEQUENCE_LENGTH = 32
    MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3 # 1 for image, 1 for start token, 1 for end token
    EMBEDDING_SIZE = 256
    LEARNING_RATE = 0.001
    save_fn = 'lstm_coco_trained_{}_euc_softmax.pkl'.format(LEARNING_RATE)

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
    
    l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding_shp])
    
    l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
    l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                      num_units=e_size,
                                      unroll_scan=True,
                                      grad_clipping=5.)
    l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
    
    # the RNN output is reshaped to combine the batch and time dimensions
    l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, e_size))
    # Now, we need to calculate the distance between this layer as well
    l_out = lasagne.layers.DenseLayer(l_shp, num_units = t_num_fea)

    
    # cnn feature vector
    x_cnn_sym = T.matrix()
    
    x_sentence_sym = T.tensor3()
    
    # mask defines which elements of the sequence should be predicted
    mask_sym = T.imatrix()
    vocab_sym = T.matrix() # vocab of the glove features for the dictionary.
    
    # ground truth for the RNN output
    y_sentence_sym = T.imatrix()
    
    output = lasagne.layers.get_output(l_out, {
                    l_input_sentence: x_sentence_sym,
                    l_input_cnn: x_cnn_sym
    })

    def calc_euc_softmax(net_output, word2vec_fea):
        # Calc the distance.
        dist = ( net_output** 2).sum(1).reshape((net_output.shape[0], 1)) \
                + (word2vec_fea ** 2).sum(1).reshape((1, word2vec_fea.shape[0])) - 2 * net_output.dot(word2vec_fea.T) # n * vocab
        # Now, softmax.
        z = T.exp( - dist + dist.max(axis = 1, keepdims = True) )
        prob = z / z.sum(axis = 1, keepdims = True) # n * vocab
        prob = T.reshape(prob, (batch_size, SEQUENCE_LENGTH, len(idx2word)))
        return prob
    
    def calc_cross_ent(net_output, mask, targets):
        # Helper function to calculate the cross entropy error
        preds = T.reshape(net_output, (-1, len(idx2word)))
        targets = T.flatten(targets)
        cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
        return cost
    
    prob_output = calc_euc_softmax(output, vocab_sym)
    loss = T.mean(calc_cross_ent(prob_output, mask_sym, y_sentence_sym))
    
    MAX_GRAD_NORM = 15
    
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    for p in all_params:
        str_p = str(p)
        if str_p in dict_params and p.get_value().shape == dict_params[str_p].shape:
            shape_str = [ str(str_i) for str_i in p.get_value().shape ]
            shape_str = ' '.join(shape_str)
            logging.info('Find {}:{} and setting it using LM'.format(str_p, shape_str))
            p.set_value(dict_params[str_p])
    all_grads = T.grad(loss, all_params)
    all_grads = [T.clip(g, -5, 5) for g in all_grads]
    #all_grads, norm = lasagne.updates.total_norm_constraint(
    #    all_grads, MAX_GRAD_NORM, return_norm=True)
    
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=LEARNING_RATE)
    
    f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym, vocab_sym],
                              [loss ],
                              updates=updates
                             )
    
    f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym, vocab_sym], loss)
    
    for iteration in range(20000):
        fea, label, mask, vis_fea = batch_train()

        loss_train = f_train(vis_fea, fea, mask, label, word2vec_fea_np)
        if not iteration % 250:
            logging.info('Iteration {}, loss_train: {}'.format(iteration, loss_train ))
            try:
                fea, label, mask, vis_fea = batch_val()
                loss_val = f_val(vis_fea, fea, mask, label, word2vec_fea_np)
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
