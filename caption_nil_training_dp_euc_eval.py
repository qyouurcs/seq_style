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

def coco_eval(ann_fn, json_fn, save_fn):
    coco = COCO(ann_fn)
    coco_res = coco.loadRes(json_fn)
    coco_evaluator = COCOEvalCap(coco, coco_res)
    # comment below line to evaluate the full validation or testing set. 
    coco_evaluator.params['image_id'] = coco_res.getImgIds()
    coco_evaluator.evaluate(save_fn)

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

def predict_captions_forward_batch_glove(img_fea, word2vec_fea, word2vec_fea_np, idx2word, batch_size, beam_size = 20, t_fea_num = 300):
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
        network_pred = f_pred(v_i, x_sym, word2vec_fea_np) 
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

    model_fn_pure = os.path.basename(model_fn)

    save_dir=cf.get('OUTPUT', 'save_dir') + model_fn_pure
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    d = pickle.load(open(model_fn))
    idx2word = d['vocab']
    params_loaded = d['param_vals']

    word2idx = {}
    for idx in idx2word:
        word2idx[idx2word[idx]] = idx
    
 
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
    batch_size = 100
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
    }, deterministic  = True)

    lasagne.layers.set_all_param_values(l_out, params_loaded)
    def calc_euc_softmax(net_output, word2vec_fea):
        # Calc the distance.
        dist = ( net_output** 2).sum(1).reshape((net_output.shape[0], 1)) \
                + (word2vec_fea ** 2).sum(1).reshape((1, word2vec_fea.shape[0])) - 2 * net_output.dot(word2vec_fea.T) # n * vocab
        # Now, softmax.
        z = T.exp( - dist + dist.max(axis = 1, keepdims = True) )
        prob = z / z.sum(axis = 1, keepdims = True) # n * vocab
        prob = T.reshape(prob, (-1, SEQUENCE_LENGTH, len(idx2word)))
        return prob
    
    prob_output = calc_euc_softmax(output, vocab_sym)
    
    f_pred = theano.function([x_cnn_sym, x_sentence_sym, vocab_sym], prob_output)
    def iter_test_imgs(max_images):
        for img in dp.iterImages('test', max_images=max_images):
            vis_fea = np.zeros((1, vis_fea_len), dtype='float32')
            vis_fea[0, :] = np.squeeze(img['feat'][:])
            img_fn = img['filename']
            yield vis_fea, img 
 
    # Now, it's time to do the beam search to generate captions.
    batch_vis_fea = np.zeros((batch_size, vis_fea_len), dtype='float32')
    batch_imgs = []
    batch_cnt = 0
    all_references = []
    all_candidates = []
    all_logprobs = []
    img_ids = []

    log = os.path.join(save_dir,'log.log')
    wfid = open(log,'w')
    #max_images = -1
    max_images = dp.getSplitSize('test','images')
    all_beam_search = []
    result_list = []
    beam_size = 3
    for i,(vis_fea, img_iter) in enumerate(iter_test_imgs(max_images)):
        references = [' '.join(x['tokens']) for x in img_iter['sentences']]  # as list of lists of tokens
        all_references.append(references)
        batch_vis_fea[i%batch_size,:] = vis_fea
        batch_imgs.append(img_iter)

        if not (i+1) %  batch_size:
            batch_cnt += 1
            start_num = (batch_cnt - 1) * batch_size + 1
            end_num= min(max_images, batch_cnt * batch_size)
            logging.info('batch %d-%d/%d:', start_num, end_num, max_images)
            batch_captions = predict_captions_forward_batch_glove(batch_vis_fea, word2vec_fea,word2vec_fea_np, idx2word, batch_size, beam_size)
            for caption, img in zip (batch_captions, batch_imgs):
                top_prediction = caption[0]
                # ix 0 is the END token, skip that
                candidate = ' '.join([idx2word[ix] for ix in top_prediction[1] if ix > 0])
                print >>wfid, 'PRED: (%f) %s' % (top_prediction[0], candidate)
                all_candidates.append(candidate)
                all_logprobs.append(str(top_prediction[0]))
                cur_img = {}
                cur_img['image_id'] = img['cocoid']
                cur_img['caption'] = candidate
                result_list.append(cur_img)
                img_ids.append(img['filename'])

                img['gen_beam_search_10'] = []
                for score, tokens in caption:
                    img['gen_beam_search_10'].append([idx2word[ix] for ix in tokens if ix > 0])
                all_beam_search.append(img)

            batch_imgs = []
    # Now set the parameters.
    if max_images % batch_size:
        num_imgs = max_images -  batch_cnt * batch_size
        start_num = (batch_cnt ) * batch_size + 1
        logging.info('batch %d-%d/%d:', start_num, max_images, max_images)
        batch_captions = predict_captions_forward_batch_glove(batch_vis_fea, word2vec_fea, word2vec_fea_np, idx2word, batch_size, beam_size)
        batch_captions = batch_captions[:num_imgs]
        for caption, img in zip (batch_captions, batch_imgs):
            top_prediction = caption[0]
            # ix 0 is the END token, skip that
            candidate = ' '.join([idx2word[ix] for ix in top_prediction[1] if ix > 0])
            print >>wfid, 'PRED: (%f) %s' % (top_prediction[0], candidate)
            cur_img = {}
            cur_img['image_id'] = img['cocoid']
            cur_img['caption'] = candidate
            result_list.append(cur_img)
            img_ids.append(img['filename'])

            all_candidates.append(candidate)
            all_logprobs.append(str(top_prediction[0]))

             # Now all beam_search.
            img['gen_beam_search_10'] = []
            for score, tokens in caption:
                img['gen_beam_search_10'].append([idx2word[ix] for ix in tokens if ix > 0])
            all_beam_search.append(img)


    json_fn = os.path.join(save_dir, 'captions.json')
    with open(json_fn, 'w') as json_fid:
        json.dump(result_list, json_fid)

    np_all_bs = np.asarray(all_beam_search)
    # start the eval code.
    join_str = [ str(img_id) + ' ' + str(score) for img_id, score in zip(img_ids, all_logprobs) ]
    open(os.path.join(save_dir,'4_visual.txt'), 'w').write('\n'.join(join_str))
    open(os.path.join(save_dir,'output'), 'w').write('\n'.join(all_candidates))
    for q in xrange(5):
        open(os.path.join(save_dir,'reference' + repr(q)), 'w').write('\n'.join([x[q] for x in all_references]))

    coco_eval(ann_fn, json_fn)
