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


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <article_dir> [igonore_1st_colum=True]'.format(sys.argv[0])
        sys.exit()
    
    art_dir = sys.argv[1]

    save_fn = art_dir.replace('.','')
    save_fn = save_fn.replace('/','_')
    save_fn = save_fn + '_sentences.txt'

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

    with open(save_fn,'w') as fid:
        for sent in yield_sentences():
            print>>fid, sent
    print 'Done with', save_fn
