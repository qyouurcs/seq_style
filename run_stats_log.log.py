#!/usr/bin/python

import os
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <log.log> <vocab>'.format(sys.argv[0])
        sys.exit()
    dict_vocab = {}
    with open(sys.argv[2]) as fid:
        for aline in fid:
            parts = aline.strip().split()
            if parts[1] not in dict_vocab:
                dict_vocab[parts[1]] = 1

    dict_word_cnt = {}

    with open(sys.argv[1]) as fid:
        for aline in fid:
            aline = aline.split(')')[1]
            parts = aline.strip().split()
            for w in parts:
                if w not in dict_word_cnt:
                    dict_word_cnt[w] = 0
                dict_word_cnt[w] += 1

    list_cnt = sorted(dict_word_cnt.items(), key = lambda x: x[1], reverse = True)

    print "total of ", len(list_cnt)
    for cnt in list_cnt:
        print cnt[0],cnt[1]

    for w in dict_word_cnt:
        if w not in dict_vocab:
            print w

