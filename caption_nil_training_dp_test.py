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

def predict_captions_forward_batch_glove(img_fea, word2vec_fea, idx2word, batch_size, beam_size = 20, t_fea_num = 300):
    captions = []
    batch_of_beams = [ [(0.0, [0])] for i in range(batch_size)]

    nsteps = 0
    while True:
        beam_c = [[] for i in range(batch_size) ]
        idx_prevs = [ [] for i in range(batch_size)]
        idx_of_idx = [[] for i in range(batch_size)]
        idx_of_idx_len = [ ]

        max_b = -1
        cnt_ins = 0
        for i in range(batch_size):
            beams = batch_of_beams[i]
            for k, b in enumerate(beams):
                idx_prev = b[-1]
                if idx_prev[-1] == 1:
                    beam_c[i].append(b)
                    continue

                idx_prevs[i].append( idx_prev)
                idx_of_idx[i].append(k) # keep the idx for future track.
                idx_of_idx_len.append(len(idx_prev))
                cnt_ins += 1
                if len(idx_prev) > max_b:
                    max_b = len(idx_prev)
        if cnt_ins == 0:
            # we do not need the 20 steps, now we have find a total of $beam_size$ candidates. just break.
            break
        x_i = np.zeros((cnt_ins, max_b, t_fea_num ), dtype='float32')
        v_i = np.zeros((cnt_ins, img_fea.shape[1]), dtype='float32')

        idx_base = 0
        for j,idx_prev_j in enumerate(idx_prevs):
            for m, idx_prev in enumerate(idx_prev_j):
                for k in range(len(idx_prev)):
                    x_i[m + idx_base, k, :] = word2vec_fea[idx2word[idx_prev[k]]]
            v_i[idx_base:idx_base + len(idx_prev_j),:] = img_fea[j,:]
            idx_base += len(idx_prev_j)
        x_sym = np.zeros((x_i.shape[0], SEQUENCE_LENGTH -1, t_fea_num), dtype='float32')
        x_sym[:,0:x_i.shape[1],:] = x_i
        network_pred = f_pred(v_i, x_sym) 
        p = np.zeros((network_pred.shape[0], network_pred.shape[2]))
        for i in range(network_pred.shape[0]):
            p[i,:] = network_pred[i,idx_of_idx_len[i],:]
        l = np.log( 1e-20 + p)
        top_indices = np.argsort( -l, axis=-1)
        idx_base = 0
        for batch_i, idx_i in enumerate(idx_of_idx):
            for j,idx in enumerate(idx_i):
                row_idx = idx_base + j
                for m in range(beam_size):
                    wordix = top_indices[row_idx][m]
                    beam_c[batch_i].append((batch_of_beams[batch_i][idx][0] + l[row_idx][wordix], batch_of_beams[batch_i][idx][1] + [wordix]))
            idx_base += len(idx_i)
        for i in range(len(beam_c)):
            beam_c[i].sort(reverse = True) # descreasing order.
        for i, b in enumerate(beam_c):
            batch_of_beams[i] = beam_c[i][:beam_size]
        nsteps += 1
        if nsteps >= 20:
            break
    for beams in batch_of_beams:
        pred = [(b[0], b[1]) for b in beams ]
        captions.append(pred)
    return captions 


if __name__ == '__main__':

    cf = ConfigParser.ConfigParser()

    if len(sys.argv) < 3:
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
    params_loaded = d['param values']
    
    word2idx = {}
    for idx in idx2word:
        word2idx[idx2word[idx]] = idx
    
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

            j = 0
            for w in tokens:
                if w in word2idx:
                    fea[ i, j, :] = word2vec_fea[w]
                    if j > 0:
                        mask[i, j-1] = 1.0
                        label[i, j-1] = word2idx[w]
                    j += 1
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

            j = 0
            for w in tokens:
                if w in word2idx:
                    fea[ i, j, :] = word2vec_fea[w]
                    if j > 0:
                        mask[i, j-1] = 1.0
                        label[i, j-1] = word2idx[w]
                    j += 1
        return fea, label, mask, vis_fea

    # Now, we can start building the model
    l_input_sentence = lasagne.layers.InputLayer((None, SEQUENCE_LENGTH - 1, t_num_fea))
    l_input_sentence_shp = lasagne.layers.ReshapeLayer(l_input_sentence, (-1, l_input_sentence.output_shape[2]))
    l_sentence_embedding = lasagne.layers.DenseLayer(l_input_sentence_shp, num_units = e_size,
                                                nonlinearity=lasagne.nonlinearities.identity)
    l_sentence_embedding_shp = lasagne.layers.ReshapeLayer(l_sentence_embedding,(-1,
        l_input_sentence.output_shape[1], e_size))
    l_input_cnn = lasagne.layers.InputLayer((None, vis_fea_len))
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
    
    # the RNe output is reshaped to combine the batch and time dimensions
    # dim (BATCH_SIZE * SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, e_size))
    
    # decoder is a fully connected layer with one output unit for each word in the vocabulary
    l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(idx2word), nonlinearity=lasagne.nonlinearities.softmax)
    
    # finally, the separation between batch and time dimension is restored
    l_out = lasagne.layers.ReshapeLayer(l_decoder, (-1, SEQUENCE_LENGTH, len(idx2word)))
    
    # Now set the parameters.
    lasagne.layers.set_all_param_values(l_out, params_loaded)
    # cnn feature vector
    x_cnn_sym = T.matrix()
    
    x_sentence_sym = T.tensor3()
    
    output = lasagne.layers.get_output(l_out, {
                    l_input_sentence: x_sentence_sym,
                    l_input_cnn: x_cnn_sym
    })

    f_pred = theano.function([x_cnn_sym, x_sentence_sym], output)

    # Now, predict the captions. 


    def iter_test_imgs(max_images):
        for img in dp.iterImages('test', max_images=max_images):
            vis_fea = np.zeros((1, vis_fea_len), dtype='float32')
            img_fn = img['filename']
            return vis_fea, img 

    def predict(x_cnn):
        x_sentence = np.zeros((batch_size, SEQUENCE_LENGTH - 1, 300), dtype='float32')
        words = []
        i = 0
        while True:
            i += 1
            p0 = f_pred(x_cnn, x_sentence)
            pa = p0.argmax(-1)
            tok = pa[0][i]
            word = idx2word[tok]
            if word == '#END#' or i >= SEQUENCE_LENGTH - 1:
                return ' '.join(words)
            else:
                x_sentence[0][i] = tok
                if word != '#START#':
                    words.append(word)
    x_cnn,img = iter_test_imgs(-1)
    pdb.set_trace()
    x_cnn = np.tile(x_cnn, [batch_size,1])

    for _ in range(5):
        print(predict(x_cnn))
