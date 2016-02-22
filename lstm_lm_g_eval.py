from __future__ import print_function
import sys

import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne
import urllib2 #For downloading the sample text file. You won't need this if you are providing your own file.
import ConfigParser
import math
import pdb
from data_provider_no_vfea import *

import climate
logging = climate.get_logger(__name__)
climate.enable_default_logging()

def perplexity(p, y, mask):
    # calculate the perplexity of each sentence and then average.
    # p : batch * seq - 1 * vocab_size
    # y: batch * seq - 1
    # mask: batch * seq - 1 
    batch_size = p.shape[0]
    seq_len = p.shape[1]
    vocab_size = p.shape[2]

    PPL = np.zeros((batch_size,))
    for i in range(batch_size):
        ppl_i = 0
        len_i = 0
        for  j in range(seq_len):
            if mask[i][j] > 0:
                len_i += 1
                ppl_i += math.log(p[i][j][y[i][j]],2)
        ppl_i /= len_i
        PPL[i] = 2**(-ppl_i)

    return np.mean(PPL)

def load_vocab(vocab_fn):
    idx2word = {}
    word2idx = {}
    with open(vocab_fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            idx2word[int(parts[0])] = parts[1]
            word2idx[parts[1]] = int(parts[0])
    return idx2word, word2idx

def load_vocab_fea(word_vec_fn, word2idx):
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

def main():
    cf = ConfigParser.ConfigParser()
    if len(sys.argv) < 3:
        logging.info('Usage: {0} <conf_fn> <model_fn>'.format(sys.argv[0]))
        sys.exit()
    cf.read(sys.argv[1])
    model_fn = sys.argv[2]

    dataset = cf.get('INPUT', 'dataset')
    h_size = cf.get('INPUT', 'h_size').split(',')

    word_vec_fn = cf.get('INPUT', 'word_vec_fn')
    vocab_fn =  cf.get('INPUT', 'vocab_fn')

    optim = cf.get('INPUT', 'optim')
    LEARNING_RATE = float(cf.get('INPUT','lr'))
    NUM_EPOCHS = int(cf.get("INPUT","epochs"))
    BATCH_SIZE = int(cf.get("INPUT", "batch_size"))

    # Now, we load the model.
    d = pickle.load(open(model_fn))

    word2idx = d['word2idx']
    idx2word = d['idx2word']
    params_loaded = d['param_vals']

    dp = getDataProvider(dataset)
    idx2word, word2idx = load_vocab(vocab_fn)
    vocab_size = len(word2idx)

    word2vec_fea, t_num_fea = load_vocab_fea(word_vec_fn, word2idx)

    #Lasagne Seed for Reproducibility
    lasagne.random.set_rng(np.random.RandomState(1))
    
    # All gradients above this will be clipped
    GRAD_CLIP = 100
    
    # How often should we check the output?
    PRINT_FREQ = 1
    
    # Number of epochs to train the net
    
    # Batch Size
    MAX_SEQ_LENGTH = 32

    EVAL_FREQ = 10
    
    # New strategy, just omit out-of-dict words.
    def batch_train(dp, batch_size = BATCH_SIZE):

        batch = [dp.sampleSentence() for i in xrange(batch_size)]
        
        x = np.zeros((batch_size,MAX_SEQ_LENGTH, t_num_fea))
        y = np.zeros((batch_size,MAX_SEQ_LENGTH-1), dtype='int32')
        masks = np.zeros((batch_size, MAX_SEQ_LENGTH-1), dtype='int32')

        for i, sent in enumerate(batch):
            tokens = ['#START#']
            tokens.extend(sent['tokens'][0:MAX_SEQ_LENGTH-2])
            tokens.append('.')
            
            pos = 0
            for j,word in enumerate(tokens):
                if word in word2idx:
                    x[i,pos,:] = word2vec_fea[word]
                    if pos > 0:
                        y[i,pos-1] = word2idx[word]
                        masks[i,pos-1] = 1
                    pos += 1
        return x,y, masks
    
    def batch_val(dp, batch_size = BATCH_SIZE):

        batch = [dp.sampleSentence('val') for i in xrange(batch_size)]
        
        x = np.zeros((batch_size,MAX_SEQ_LENGTH, t_num_fea))
        y = np.zeros((batch_size,MAX_SEQ_LENGTH-1), dtype='int32')
        masks = np.zeros((batch_size, MAX_SEQ_LENGTH-1), dtype='int32')

        for i, sent in enumerate(batch):
            tokens = ['#START#']
            tokens.extend(sent['tokens'][0:MAX_SEQ_LENGTH-2])
            tokens.append('.')
            
            pos = 0
            for j,word in enumerate(tokens):
                if word in word2idx:
                    x[i,pos,:] = word2vec_fea[word]
                    if pos > 0:
                        y[i, pos - 1] = word2idx[word]
                        masks[i,pos - 1] = 1
                    pos += 1
        return x,y, masks

    logging.info("Building network ...")    
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_SEQ_LENGTH, t_num_fea))
    l_in_dropout = lasagne.layers.DropoutLayer(l_in, p = 0.5)

    h_prev = lasagne.layers.LSTMLayer(
            l_in_dropout, int(h_size[0]), grad_clipping = GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)
    h_cur = lasagne.layers.DropoutLayer(h_prev, p = 0.5)
    h_prev = h_cur
    for i in xrange(1,len(h_size)):
        h_cur = lasagne.layers.LSTMLayer(
                h_prev, int(h_size[i]), grad_clipping=GRAD_CLIP,
                nonlinearity=lasagne.nonlinearities.tanh)
        h_prev = lasagne.layers.DropoutLayer(h_cur, p = 0.5)
        h_cur = h_prev

    # The output of the sliced layer will then be of size (batch_size, SEQ_LENGH-1, N_HIDDEN)
    l_forward_slice = lasagne.layers.SliceLayer(h_cur, indices = slice(0, -1), axis = 1)
    logging.info('l_forward_slide shape {0}, {1},{2}'.format(l_forward_slice.output_shape[0],l_forward_slice.output_shape[1], l_forward_slice.output_shape[2]))

    l_forward_slice_rhp = lasagne.layers.ReshapeLayer(l_forward_slice, (-1, l_forward_slice.output_shape[2]))

    # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(l_forward_slice_rhp, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
    
    ############################
    # set the params.
    #
    lasagne.layers.set_all_param_values(l_out, params_loaded)

    logging.info('l_out shape {0}, {1}'.format(l_out.output_shape[0],l_out.output_shape[1]))

    # Theano tensor for the targets
    target_values = T.imatrix('target_output')
    mask_sym = T.imatrix('mask')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    network_output_tst = lasagne.layers.get_output(l_out, deterministic = True)

    l_out_rhp = lasagne.layers.ReshapeLayer(l_out,(l_forward_slice.output_shape[0], l_forward_slice.output_shape[1], vocab_size))
    network_output_rhp = lasagne.layers.get_output(l_out_rhp)
    network_output_rhp_tst = lasagne.layers.get_output(l_out_rhp, deterministic = True)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.

    def calc_cross_ent(net_output, mask_sym, targets):
        preds = T.reshape(net_output, (-1, len(word2idx)))
        targets = T.flatten(targets)
        cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask_sym).nonzero()]
        return cost

    cost_train = T.mean(calc_cross_ent(network_output, mask_sym, target_values))
    cost_test = T.mean(calc_cross_ent(network_output_tst, mask_sym, target_values))

    logging.info("Compiling functions ...")
    f_val = theano.function([l_in.input_var, target_values, mask_sym], cost_test, allow_input_downcast=True)

    probs_test = theano.function([l_in.input_var],network_output_rhp_tst,allow_input_downcast=True)

    logging.info("Testing...")
    data_size = dp.getSplitSize('test')
    logging.info('Total of {}'.format(data_size))
    
    avg_cost = 0
    ppl = 0

    x = np.zeros((BATCH_SIZE,MAX_SEQ_LENGTH, t_num_fea))
    y = np.zeros((BATCH_SIZE,MAX_SEQ_LENGTH-1), dtype='int32')
    masks = np.zeros((BATCH_SIZE, MAX_SEQ_LENGTH-1), dtype='int32')

    cnt = 0

    for sent in  dp.iterSentences('test'):
        tokens = ['#START#']
        tokens.extend(sent['tokens'][0:MAX_SEQ_LENGTH-2])
        tokens.append('.')
        pos = 0
        for j,word in enumerate(tokens):
            if word in word2idx:
                x[cnt,pos,:] = word2vec_fea[word]
                if pos > 0:
                    y[cnt, pos - 1] = word2idx[word]
                    masks[cnt,pos - 1] = 1
                pos += 1
       
        cnt += 1
        if cnt % BATCH_SIZE == 0:
            logging.info('Progress {}/{}'.format(j, data_size))
            avg_cost += f_val(x, y, masks)
            p = probs_test(x)
            ppl += perplexity(p,y,masks)
            cnt = 0

    if cnt > 0:
        #x = x[0:cnt,:]
        #y = y[0:cnt,:]
        #masks = masks[0:cnt,:]
        avg_cost += f_val(x, y, masks)
        p = probs_test(x)
        ppl += perplexity(p,y,masks)
    # We also need to eval on the val dataset. 

    logging.info('Done.')
    logging.info('avg_cost = {}'.format(avg_cost / data_size))
    logging.info('ppl = {}'.format(ppl / data_size))
if __name__ == '__main__':
    main()
