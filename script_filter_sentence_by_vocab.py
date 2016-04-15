#!/usr/bin/python

import sys
import os

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <sentence_fn> <vocab_fn>'.format(sys.argv[0])
        sys.exit()

    sent_fn = sys.argv[1]
    vocab_fn = sys.argv[2]

    dict_vocab = {}
    with open(vocab_fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            if parts[1] not in dict_vocab:
                dict_vocab[parts[1]] = 1

    save_fn = sent_fn + os.path.basename(vocab_fn)
    
    wfid = open(save_fn,'w')
    with open(sent_fn,'r') as fid:
        for sent in fid:
            sent = sent.lower()
            parts = sent.strip().split()
            s = []
            for w in parts:
                if w in dict_vocab:
                    s.append(w)

            wfid.write(' '.join(s) + '\n')

    wfid.close()
    print 'Done with', save_fn
