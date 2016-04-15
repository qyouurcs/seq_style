#!/usr/bin/python

import os
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {} <main_vocab> <ref_small_vocab>'.format(sys.argv[0])
        sys.exit()

    main_vocab = sys.argv[1]
    ref_vocab = sys.argv[2]
    
    dict_main = {}
    with open(main_vocab) as fid:
        for aline in fid:
            parts = aline.strip().split()

            if parts[1] not in dict_main:
                dict_main[parts[1]] = 1
    dict_ref = {}
    with open(ref_vocab) as fid:
        for aline in fid:
            parts = aline.strip().split()

            if parts[1] not in dict_ref:
                dict_ref[parts[1]] = 1

    cnt = 0
    for w in dict_ref:
        if w not in dict_main:
            print w,
            cnt += 1

    print
    print 'total', cnt

