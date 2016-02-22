#!/usr/bin/python

import sys
import pdb
import os
import climate
import time
import nltk.data 
import nltk.tokenize

logging = climate.get_logger(__name__)
climate.enable_default_logging()


def preProBuildWordVocab(sentence_iterator, word_count_threshold):
    t0 = time.time()
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        tokens = nltk.tokenize.word_tokenize(sent)
        for w in tokens:
            w = w.lower()
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold and not w.isdigit()]
    logging.info('filtered words from %d to %d in %.2fs' % (len(word_counts), len(vocab), time.time() - t0))
    logging.info('Total of {} sentences'.format(nsents))

    ixtoword = {}
    ixtoword[0] = '.'
    ixtoword[1] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0  
    wordtoix['#END#'] = 1  
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    return wordtoix, ixtoword

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <article_dir> [threshod=5] [igonore_1st_colum=True]'.format(sys.argv[0])
        sys.exit()
    
    art_dir = sys.argv[1]
    word_count_threshold = 5
    ignore_1st_col = True

    if len(sys.argv) >= 3:
        word_count_threshold = int(sys.argv[2])
    if len(sys.argv) >= 4:
        igonore_1st_col = sys.argv[3] == 'False' or sys.argv[3] == 'false' 

    save_fn = art_dir.replace('.','')
    save_fn = save_fn.replace('/','_')
    save_fn = save_fn + '_' + str(word_count_threshold) + '_vocab.txt'

    s_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    def yield_sentences():
        for root, subdirs, fns in os.walk(art_dir, followlinks=True):
            for fn in fns:
                f_fn = os.path.join(root, fn)
                with open(f_fn,'r') as fid:
                    for aline in fid:
                        psg = aline.split(' ')
                        psg = ' '.join( psg[1:])
                        sentences = s_detector.tokenize(psg.strip())
                        for s in sentences:
                            yield s

    word2idx, idx2word = preProBuildWordVocab(
        yield_sentences(), word_count_threshold)
    
    with open(save_fn,'w') as fid:
        for idx in idx2word:
            print>>fid, idx2word[idx]
    print 'Done with', save_fn
