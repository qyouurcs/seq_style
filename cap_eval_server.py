#!/usr/bin/python

import sys
import pdb
import os
import climate
import time
import json

logging = climate.get_logger(__name__)
climate.enable_default_logging()
local_eval='./coco-caption/'
if  os.path.isdir(local_eval):
    sys.path.append(local_eval)
# for coco
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def coco_eval(ann_fn, json_fn):
    coco = COCO(ann_fn)
    coco_res = coco.loadRes(json_fn)
    coco_evaluator = COCOEvalCap(coco, coco_res)
    # comment below line to evaluate the full validation or testing set. 
    coco_evaluator.params['image_id'] = coco_res.getImgIds()
    coco_evaluator.evaluate()

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Usage: {0} <captions.json> <dataset> [split=val]'.format(sys.argv[0])
        sys.exit()
    json_fn = sys.argv[1]
    dataset = sys.argv[2]
    split = 'val'
    # annotation fn 
    ann_fn = './data/{0}/annotations/captions_{1}2014.json'.format(dataset, split)
    logging.info('annotation fn = %s', ann_fn)
    if os.path.isfile(json_fn):
        logging.info('find json file %s, loading it', json_fn)
        # we do not need to perform any op.
        coco_eval(ann_fn, json_fn)
        sys.exit()
