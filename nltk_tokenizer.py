#!/usr/bin/python
import sys
import os
import nltk

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: {0} <txt_fn>'.format(sys.argv[0])
        sys.exit()
    
    if sys.argv[1][-1] == '/':
        txt_fn = sys.argv[1][0:-1]
    else:
        txt_fn = sys.argv[1]

    save_fn = txt_fn + '.tok'
    with open(sys.argv[1],'r') as fid, \
            open(save_fn,'w') as wfid:
        for aline in fid:
            aline = aline.strip()
            tok = nltk.word_tokenize(aline)
            if len(tok) < 5:
                continue # ignore those sentences with less than 5 words.
            tok_str = ' '.join(tok)
            wfid.write(tok_str + '\n')
    
    print 'Done with', save_fn
