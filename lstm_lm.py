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
            if mask[i][j]:
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
    h_size = int(cf.get('INPUT', 'h_size'))

    word_vec_fn = cf.get('INPUT', 'word_vec_fn')
    vocab_fn =  cf.get('INPUT', 'vocab_fn')

    save_dir=cf.get('OUTPUT', 'save_dir')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    save_fn = os.path.join(save_dir, sys.argv[1] + '.pkl')

    dp = getDataProvider(dataset)
    idx2word, word2idx = load_vocab(vocab_fn)
    vocab_size = len(word2idx)

    word2vec_fea, t_num_fea = load_vocab_fea(word_vec_fn, word2idx)

    #Lasagne Seed for Reproducibility
    lasagne.random.set_rng(np.random.RandomState(1))
    
    LEARNING_RATE = .01
    
    # All gradients above this will be clipped
    GRAD_CLIP = 100
    
    # How often should we check the output?
    PRINT_FREQ = 1
    
    # Number of epochs to train the net
    NUM_EPOCHS = 50
    
    # Batch Size
    BATCH_SIZE = 128
    MAX_SEQ_LENGTH = 32

    EVAL_FREQ = 10
    
    
    def batch_train(dp, batch_size = BATCH_SIZE):

        batch = [dp.sampleSentence() for i in xrange(batch_size)]
        
        x = np.zeros((batch_size,MAX_SEQ_LENGTH, t_num_fea))
        y = np.zeros((batch_size,MAX_SEQ_LENGTH-1), dtype='int32')
        masks = np.zeros((batch_size, MAX_SEQ_LENGTH-1), dtype='int32')

        for i, sent in enumerate(batch):
            tokens = ['#START#']
            tokens.extend(sent['tokens'][0:MAX_SEQ_LENGTH-2])
            tokens.append('.')
            
            for j,word in enumerate(tokens):
                if word in word2idx:
                    x[i,j,:] = word2vec_fea[word]
                    if j > 0:
                        y[i,j-1] = word2idx[word]
                        masks[i,j-1] = 1
    
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
            
            for j,word in enumerate(tokens):
                if word in word2idx:
                    x[i,j,:] = word2vec_fea[word]
                    if j > 0:
                        y[i,j-1] = word2idx[word]
                        masks[i,j-1] = 1
    
        return x,y, masks
 
    logging.info("Building network ...")    
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, MAX_SEQ_LENGTH, t_num_fea))

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, h_size, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, h_size, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    # The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
    # Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
    # The output of the sliced layer will then be of size (batch_size, SEQ_LENGH-1, N_HIDDEN)
    l_forward_slice = lasagne.layers.SliceLayer(l_forward_2, indices = slice(0, -1), axis = 1)
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
    network_output = lasagne.layers.get_output(l_out)

    l_out_rhp = lasagne.layers.ReshapeLayer(l_out,(l_forward_slice.output_shape[0], l_forward_slice.output_shape[1], vocab_size))
    network_output_rhp = lasagne.layers.get_output(l_out_rhp)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.

    def calc_cross_ent(net_output, mask_sym, targets):
        preds = T.reshape(net_output, (-1, len(word2idx)))
        targets = T.flatten(targets)
        cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask_sym).nonzero()]
        return cost

    cost = T.mean(calc_cross_ent(network_output, mask_sym, target_values))

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute AdaGrad updates for training
    logging.info("Computing updates ...")
    #updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    updates = lasagne.updates.adam(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    logging.info("Compiling functions ...")
    f_train = theano.function([l_in.input_var, target_values, mask_sym], cost, updates=updates, allow_input_downcast=True)
    f_val = theano.function([l_in.input_var, target_values, mask_sym], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([l_in.input_var],network_output_rhp,allow_input_downcast=True)

    logging.info("Training ...")
    data_size = dp.getSplitSize('train')
    mini_batches_p_epo = int(math.floor(data_size / BATCH_SIZE))
    try:
        for epoch in xrange(NUM_EPOCHS):
            avg_cost = 0;

            for j in xrange(mini_batches_p_epo):
            #for _ in range(PRINT_FREQ):
                x,y, mask = batch_train(dp)
                avg_cost += f_train(x, y, mask)
                if not(j % PRINT_FREQ):
                    p = probs(x)
                    ppl = perplexity(p,y,mask)
                    logging.info("Epoch {}, mini_batch = %d/%d, avg loss = {}, PPL = {}".format(epoch, j, mini_batches_p_epo, avg_cost / PRINT_FREQ, ppl))
                    avg_cost = 0
                if not(j % EVAL_FREQ):
                    x,y, mask = batch_val(dp)
                    val_cost = f_val(x, y, mask)
                    p = probs(x)
                    ppl = perplexity(p,y,mask)
                    logging.info("-----------------------------------------------------")
                    logging.warning("\tVAL average loss = {}, PPL = {}".format(val_cost, ppl))
            # We also need to eval on the val dataset. 
    except KeyboardInterrupt:
        pass

    d = {'param_vals': param_values,
            'word2idx':word2idx,
            'idx2word':idx2word}
    pickle.dump(d,  open(save_fn,'w'), protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("Done with {}".format(save_fn))

if __name__ == '__main__':
    main()
