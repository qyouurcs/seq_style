import sys
from data_provider import *
import time

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Usage: {0} <word_vec_fn> <vocab_fn>".format(sys.argv[0])
        sys.exit()
    
    word_vec_fn = sys.argv[1]
    vocab_fn = sys.argv[2]

    save_fn = os.path.splitext(vocab_fn)[0] + '_word2vec.txt'

    vocab_dict = {}
    with open(vocab_fn) as fid:
        for aline in fid:
            w = aline.strip()
            vocab_dict[w] = 1
    # add end token.
    vocab_dict['.'] = 1

    word2vec_dict = {}

    with open(word_vec_fn,'r') as fid:
        for aline in fid:
            aline = aline.strip()
            parts = aline.split()
            if parts[0] in vocab_dict:
                word2vec_dict[parts[0]] = 1
    idx2word = {}
    # period at the end of the sentence. make first dimension be end token
    idx2word[0] = '#START#'
    idx2word[1] = '.'
    word2idx = {}
    word2idx['#START#'] = 0  # make first vector be the start token
    word2idx['.'] = 1 # make the 2nd vector be the end token
    idx = 2 
    for w in word2vec_dict:
        if w not in word2idx:
            word2idx[w] = idx
            idx2word[idx] = w
            idx += 1

    with open(save_fn, 'w') as fid:
        for i in range(len(idx2word)):
            print >>fid, i, idx2word[i]

    print 'Done with vocab', save_fn
