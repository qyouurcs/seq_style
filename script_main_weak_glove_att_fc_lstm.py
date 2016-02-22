#!/usr/bin/python
import ConfigParser
import time
import sys
import os
import numpy as np
from data_provider import *
import theanets
import climate
import theano as T
import random

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

def load_fix_dict(word_fix_fn):
    fix_dict = {}
    if os.path.isfile(word_fix_fn):
        with open(word_fix_fn) as fid:
            for word in fid:
                word = word.strip().split()
                fix_dict[word[0]] = ' '.join(word[1:])
    words_ = fix_dict.keys()
    for w in words_:
        mapped_w = fix_dict[w].split()
        mapped = True
        for mp in mapped_w:
            if mp not in word2idx:
                mapped = False
        if w not in word2idx and not mapped:
            del fix_dict[w]
    return fix_dict

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

def load_key_words(keywords_train_fn, keywords_val_fn):
    dict_train_key_words = {}
    dict_val_key_words = {}
    with open(keywords_train_fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            words = aline.split()
            dict_train_key_words[ os.path.basename(words[0])] = words[1:]

    with open(keywords_val_fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            words = aline.split()
            dict_val_key_words[ os.path.basename(words[0])] = words[1:]
    return dict_train_key_words, dict_val_key_words

if __name__ == '__main__':

    cf = ConfigParser.ConfigParser()
    if len(sys.argv) < 3:
        print 'Usage: {0} <conf_fn> <n_words> [rnd_kws = 0]'.format(sys.argv[0])
        sys.exit()
    #np.set_printoptions(threshold=np.nan)
    cf.read(sys.argv[1])
    n_words = int(sys.argv[2])
    rnd_kws = 0
    if len(sys.argv) >= 4:
        rnd_kws = int(sys.argv[3])
    dataset = cf.get('INPUT', 'dataset')
    h_size=int(cf.get('INPUT','h_size'))
    e_size=int(cf.get('INPUT','e_size'))
    h_l1=float(cf.get('INPUT','h_l1'))
    h_l2=float(cf.get('INPUT','h_l2'))
    l1=float(cf.get('INPUT','l1'))
    l2=float(cf.get('INPUT','l2'))
    dropout = float(cf.get('INPUT', 'dropout'))

    model_fn = cf.get('INPUT', 'model_fn')
    keywords_train_fn = cf.get('INPUT', 'keywords_train')
    keywords_val_fn = cf.get('INPUT', 'keywords_val')
    word_vec_fn = "."
    if cf.has_option('INPUT', 'word_vec_fn'):
        word_vec_fn = cf.get('INPUT', 'word_vec_fn')
    word_fix_fn = '.'
    if cf.has_option('INPUT', 'word_fix_mapping'):
        word_fix_fn = cf.get('INPUT', 'word_fix_mapping')
    vocab_fn =  cf.get('INPUT', 'vocab_fn')
    save_dir=cf.get('OUTPUT', 'save_dir')

    dp = getDataProvider(dataset)
    # Now, load the vocab.
    idx2word, word2idx = load_vocab(vocab_fn)

    # now check fix_dict:
    # fix vocab.
    fix_dict = load_fix_dict(word_fix_fn) 
    word2vec_fea, t_num_fea = load_vocab_fea(word_vec_fn)

    logging.info('Total vocab has a total of %d words and feas %d', len(word2idx), len(word2vec_fea))
    
    dict_train_key_words, dict_val_key_words = load_key_words(keywords_train_fn, keywords_val_fn)

    batch_size = 256
    def vis_fea_len():
        pair = dp.sampleImageSentencePair()
        vis_fea_len = 0
        vis_fea_len = pair['image']['feat'].size
        return vis_fea_len

    def batch_train():
        batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
        # May need to optimize here.
        # need to find out the longest length of a sentence.
        max_len = 0
        vis_fea_len = 0
        for i, pair in enumerate(batch):
            len_i = 0
            for token in pair['sentence']['tokens']:
                if token in fix_dict:
                    len_i += len(fix_dict[token].split())
                else:
                    len_i += 1
            #if max_len < len(pair['sentence']['tokens']):
            if max_len < len_i:
                max_len = len_i
            vis_fea_len = pair['image']['feat'].size

        max_len += 2 # add both start and end token.
        fea = np.zeros(( max_len, batch_size, t_num_fea), dtype='float32')
        label = np.zeros(( max_len - 1, batch_size), dtype='int32') # label do not need the #START# token.
        mask = np.zeros(( max_len - 1, batch_size), dtype='float32')
        vis_fea = np.zeros((batch_size, vis_fea_len), dtype='float32')
        words_fea = np.zeros((n_words, batch_size, t_num_fea), dtype='float32')
        ref_mask = np.zeros((words_fea.shape[0:2]), dtype = 'float32')

        for i, pair in enumerate(batch):
            #gtix = [word2idx[w] for w in pair['sentence']['tokens'] if w in word2idx]
            # one hot representation.
            img_fn = pair['image']['filename']
            key_words = dict_train_key_words[img_fn]
            if rnd_kws:
                random.shuffle(key_words)

            vis_fea[i,:] = np.squeeze(pair['image']['feat'][:])
            tokens = ['#START#']
            tokens.extend(pair['sentence']['tokens'])
            tokens.append('.')
            #for j, w in enumerate(pair['sentence']['tokens']):
            j = 0
            for w in tokens:
                if w in fix_dict:
                    ws = fix_dict[w].split()
                    for w in ws:
                        fea[j, i, :] = word2vec_fea[w]
                        if j > 0:
                            mask[j-1, i] = 1.0
                            label[j-1,i] = word2idx[w]
                        j += 1
                elif w in word2idx:
                    fea[ j, i, :] = word2vec_fea[w]
                    if j > 0:
                        mask[j-1, i] = 1.0
                        label[j-1,i] = word2idx[w]
                    j += 1
            #logging.info('Max len for this batch %d',max_len)
            j = 0
            for w in key_words:
                if w in fix_dict:
                    ws = fix_dict[w].split()
                    for w in ws:
                        words_fea[j, i, :] = word2vec_fea[w]
                        ref_mask[j,i] = 1.0
                        j+= 1
                        if j >= n_words:
                            break
                elif w in word2idx:
                    words_fea[j, i, :] = word2vec_fea[w]
                    ref_mask[j, i] = 1.0
                    j+= 1
                if j >= n_words:
                    break
            #if j < n_words:
            #    logging.info('Not enough words provided for %s:%d', img_fn, j)
        return [fea, label, mask, vis_fea, words_fea, ref_mask]

    def batch_val():
        batch = [dp.sampleImageSentencePair('val', dict_val_key_words) for i in xrange(batch_size)]
        max_len = 0
        vis_fea_len = 0
        for i, pair in enumerate(batch):
            len_i = 0
            for token in pair['sentence']['tokens']:
                if token in fix_dict:
                    len_i += len(fix_dict[token].split())
                else:
                    len_i += 1
            #if max_len < len(pair['sentence']['tokens']):
            if max_len < len_i:
                max_len = len_i
            vis_fea_len = pair['image']['feat'].size
        max_len += 2
        fea = np.zeros(( max_len, batch_size, t_num_fea), dtype='float32')
        label = np.zeros(( max_len-1, batch_size), dtype='int32')
        mask = np.zeros(( max_len-1, batch_size), dtype='float32')
        vis_fea = np.zeros((batch_size,vis_fea_len), dtype='float32')
        words_fea = np.zeros((n_words, batch_size, t_num_fea), dtype='float32')
        ref_mask = np.zeros((words_fea.shape[0:2]), dtype = 'float32')

        for i, pair in enumerate(batch):
            #gtix = [word2idx[w] for w in pair['sentence']['tokens'] if w in word2idx]
            # one hot representation.
            img_fn = pair['image']['filename']
            key_words = dict_val_key_words[img_fn]
            if rnd_kws:
                random.shuffle(key_words)

            vis_fea[i,:] = np.squeeze(pair['image']['feat'][:])
            tokens = ['#START#']
            tokens.extend(pair['sentence']['tokens'])
            tokens.append('.')
            #for j, w in enumerate(pair['sentence']['tokens']):
            j = 0
            for w in tokens:
                if w in fix_dict:
                    ws = fix_dict[w].split()
                    for w in ws:
                        fea[j, i, :] = word2vec_fea[w]
                        if j > 0:
                            mask[j-1, i] = 1.0
                            label[j-1,i] = word2idx[w]
                        j += 1
                elif w in word2idx:
                    fea[ j, i, :] = word2vec_fea[w]
                    if j > 0:
                        mask[j-1, i] = 1.0
                        label[j-1,i] = word2idx[w]
                    j += 1
            j = 0
            for w in key_words:
                if w in fix_dict:
                    ws = fix_dict[w].split()
                    for w in ws:
                        words_fea[j, i, :] = word2vec_fea[w]
                        ref_mask[j, i] = 1.0
                        j+= 1
                        if j >= n_words:
                            break
                elif w in word2idx:
                    words_fea[j, i, :] = word2vec_fea[w]
                    ref_mask[j, i] = 1.0
                    j+= 1
                if j >= n_words:
                    break

            #if j < n_words:
            #    logging.info('Not enough words provided for %s:%d', img_fn, j )
        #logging.info('Max len for this batch %d',max_len)
        return [fea, label, mask, vis_fea, words_fea, ref_mask]

    input_size = len(idx2word)
    vis_fea_len = vis_fea_len()
    def layer_lstm(n):
        return dict(form = 'lstm', size = n)

    def layer_input_emb_att(n,v,i, k):
        #return dict(size = n, v_size = v, input_size = i, n_words = k)
        return dict(size = n, v_size = v, input_size = i, ref='ref')

    def layer_lstm_att(n):
        return dict(form = 'lstmatt', size = n)


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    time_str = time.strftime("%d-%b-%Y-%H%M%S", time.gmtime())
    save_prefix = os.path.join(save_dir, os.path.splitext(os.path.basename(sys.argv[1]))[0] + '_' + str(dropout))
    save_fn = save_prefix + '_' + time_str + '_' + dataset + '.pkl'
    logging.info('will save model to %s', save_fn)

    if os.path.isfile(model_fn):
        e = theanets.Experiment(model_fn)
    else:
        e = theanets.Experiment(
            theanets.recurrent.Classifier,
            layers=(
                layer_input_emb_att(e_size, vis_fea_len, t_num_fea, n_words), 
                layer_lstm_att(h_size), 
                (input_size, 'softmax')),
            weighted=True,
            embedding=True,
            weak = True
        )
        e.train(
            batch_train,
            batch_val,
            algorithm='rmsprop',
            learning_rate=0.0001,
            momentum=0.9,
            max_gradient_elem=10,
            input_noise=0.0,
            train_batches=30,
            valid_batches=3,
            hidden_l1 = h_l1,
            hidden_l2 = h_l2,
            weight_l1 = l1,
            weight_l2 = l2,
            batch_size=batch_size,
            dropout=dropout,
            save_every = 200
        )

    e.train(
        batch_train,
        batch_val,
        algorithm='rmsprop',
        learning_rate=0.00001,
        momentum=0.9,
        max_gradient_elem=10,
        input_noise=0.0,
        train_batches=30,
        valid_batches=3,
        hidden_l1 = h_l1,
        hidden_l2 = h_l2,
        weight_l1 = l1,
        weight_l2 = l2,
        batch_size=batch_size,
        dropout=dropout,
        save_every = 200
    )
    e.save(save_fn)
