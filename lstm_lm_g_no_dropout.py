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
    if len(sys.argv) < 2:
        logging.info('Usage: {0} <conf_fn>'.format(sys.argv[0]))
        sys.exit()
    cf.read(sys.argv[1])
    dataset = cf.get('INPUT', 'dataset')
    h_size = cf.get('INPUT', 'h_size').split(',')

    word_vec_fn = cf.get('INPUT', 'word_vec_fn')
    vocab_fn =  cf.get('INPUT', 'vocab_fn')

    optim = cf.get('INPUT', 'optim')
    LEARNING_RATE = float(cf.get('INPUT','lr'))
    NUM_EPOCHS = int(cf.get("INPUT","epochs"))
    BATCH_SIZE = int(cf.get("INPUT", "batch_size"))

    save_dir=cf.get('OUTPUT', 'save_dir')
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    h_size_str = [str(h) for h in h_size]
    h_size_str = '_'.join(h_size_str)

    save_fn = os.path.join(save_dir, os.path.basename(sys.argv[1]) + '_' + optim  + '_' + h_size_str + '_no_dropout.pkl')

    dp = getDataProvider(dataset)
    idx2word, word2idx = load_vocab(vocab_fn)
    vocab_size = len(word2idx)

    word2vec_fea, t_num_fea = load_vocab_fea(word_vec_fn, word2idx)

    lasagne.random.set_rng(np.random.RandomState(1))
    GRAD_CLIP = 100
    MAX_SEQ_LENGTH = 32
    EVAL_FREQ = 10
    PRINT_FREQ = 10
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
   
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_SEQ_LENGTH, t_num_fea))
    #l_in_dropout = lasagne.layers.DropoutLayer(l_in, p = 0.5)
    l_in_dropout = l_in

    h_prev = lasagne.layers.LSTMLayer(
            l_in_dropout, int(h_size[0]), grad_clipping = GRAD_CLIP,
            nonlinearity=lasagne.nonlinearities.tanh)
    #h_cur = lasagne.layers.DropoutLayer(h_prev, p = 0.5)
    h_cur = h_prev
    h_prev = h_cur
    for i in xrange(1,len(h_size)):
        h_cur = lasagne.layers.LSTMLayer(
                h_prev, int(h_size[i]), grad_clipping=GRAD_CLIP,
                nonlinearity=lasagne.nonlinearities.tanh)
        #h_prev = lasagne.layers.DropoutLayer(h_cur, p = 0.5)
        h_prev = h_cur
        h_cur = h_prev

    # The output of the sliced layer will then be of size (batch_size, SEQ_LENGH-1, N_HIDDEN)
    l_forward_slice = lasagne.layers.SliceLayer(h_cur, indices = slice(0, -1), axis = 1)
    logging.info('l_forward_slide shape {0}, {1},{2}'.format(l_forward_slice.output_shape[0],l_forward_slice.output_shape[1], l_forward_slice.output_shape[2]))

    l_forward_slice_rhp = lasagne.layers.ReshapeLayer(l_forward_slice, (-1, l_forward_slice.output_shape[2]))

    # The sliced output is then passed through the softmax nonlinearity to create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(l_forward_slice_rhp, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)
    
    logging.info('l_out shape {0}, {1}'.format(l_out.output_shape[0],l_out.output_shape[1]))

    # Theano tensor for the targets
    target_values = T.imatrix('target_output')
    mask_sym = T.imatrix('mask')
    
    # lasagne.layers.get_output produces a variable for the output of the net

    l_out_rhp = lasagne.layers.ReshapeLayer(l_out,(l_forward_slice.output_shape[0], l_forward_slice.output_shape[1], vocab_size))
    network_output = lasagne.layers.get_output(l_out_rhp, deterministic = False)
    network_output_tst = lasagne.layers.get_output(l_out_rhp, deterministic = True)

    def calc_cross_ent(net_output, mask_sym, targets):
        preds = T.reshape(net_output, (-1, len(word2idx)))
        targets = T.flatten(targets)
        cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask_sym).nonzero()]
        return cost

    cost_train = T.mean(calc_cross_ent(network_output, mask_sym, target_values))
    cost_test = T.mean(calc_cross_ent(network_output_tst, mask_sym, target_values))

    all_params = lasagne.layers.get_all_params(l_out)
    # Compute AdaGrad updates for training
    logging.info("Computing updates ...")
    if optim == 'ada':
        updates = lasagne.updates.adagrad(cost_train, all_params, LEARNING_RATE)
    elif optim == 'adam':
        updates = lasagne.updates.adam(cost_train, all_params, LEARNING_RATE)
    elif optim == 'rmsprop':
        updates = lasagne.updates.rmsprop(cost_train, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    logging.info("Compiling functions ...")
    f_train = theano.function([l_in.input_var, target_values, mask_sym], cost_train, updates=updates, allow_input_downcast=True)
    f_val = theano.function([l_in.input_var, target_values, mask_sym], cost_test, allow_input_downcast=True)

    probs_train = theano.function([l_in.input_var],network_output_tst,allow_input_downcast=True)
    probs_test = theano.function([l_in.input_var],network_output_tst,allow_input_downcast=True)

    logging.info("Training ...")
    data_size = dp.getSplitSize('train')
    mini_batches_p_epo = int(math.floor(data_size / BATCH_SIZE))
    try:
        for epoch in xrange(NUM_EPOCHS):
            avg_cost = 0;

            for j in xrange(mini_batches_p_epo):
                x,y, mask = batch_train(dp)
                avg_cost += f_train(x, y, mask)
                if not(j % PRINT_FREQ):
                    p = probs_train(x)
                    ppl = perplexity(p,y,mask)
                    logging.info("Epoch {}, mini_batch = {}/{}, avg loss = {}, PPL = {}".format(epoch, j, mini_batches_p_epo, avg_cost / PRINT_FREQ, ppl))
                    avg_cost = 0
                if not(j % EVAL_FREQ):
                    x,y,mask = batch_val(dp)
                    val_cost = f_val(x, y, mask)
                    p = probs_test(x)
                    ppl = perplexity(p,y,mask)
                    logging.info("-----------------------------------------------------")
                    logging.warning("\tVAL average loss = {}, PPL = {}".format(val_cost, ppl))
            # We also need to eval on the val dataset. 
    except Exception:
        logging.warning("EXCEPTION")
        pass
    param_values = lasagne.layers.get_all_param_values(l_out)
    param_syms = lasagne.layers.get_all_params(l_out)
    param_strs = []
    for sym in param_syms:
        param_strs.append(str(sym))

    d = {'param_vals': param_values,
         'param_strs': param_strs,
            'word2idx':word2idx,
            'idx2word':idx2word}
    pickle.dump(d,  open(save_fn,'w'), protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Done with {}".format(save_fn))

if __name__ == '__main__':
    main()
