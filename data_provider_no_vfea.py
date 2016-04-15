import json
import os
import random
import sys
import scipy.io
import codecs
import pdb
import h5py
import numpy as np
from collections import defaultdict

import climate
logging = climate.get_logger(__name__)
climate.enable_default_logging()


class BasicDataProvider:

    def __init__(self, dataset):
        print 'Initializing data provider for dataset %s...' % (dataset, )

        # !assumptions on folder structure
        self.dataset_root = os.path.join('data', dataset)
        self.image_root = os.path.join('data', dataset, 'imgs')

        # load the dataset into memory
        dataset_path = os.path.join(self.dataset_root, 'dataset.json')
        print 'BasicDataProvider: reading %s' % (dataset_path, )
        self.dataset = json.load(open(dataset_path, 'r'))
        logging.info('dataset len %d', len(self.dataset['images']))
        tdataset = []
        if dataset.startswith('coco'):
            exclude = 'COCO_train2014_000000167126.jpg'
            idx = 0
            for img in self.dataset['images']:
                if img['filename'] == exclude:
                    break
                else:
                    tdataset.append(img)
                idx += 1
            tdataset.extend(self.dataset['images'][idx + 1:])
            self.dataset['images'] = tdataset
            logging.info('dataset len %d', len(self.dataset['images']))

        self.split = defaultdict(list)
        for img in self.dataset['images']:
            self.split[img['split']].append(img)

        for split in self.split:
            logging.info('imgs %d in %s', len(self.split[split]), split)

    def _getImage(self, img):
        """ create an image structure for the driver """

        # lazily fill in some attributes
        if not 'local_file_path' in img:
            img['local_file_path'] = os.path.join(self.image_root, img['filename'])
        if not 'feat' in img:  # also fill in the features
            # NOTE: imgid is an integer, and it indexes into features
            fn = os.path.basename(img['filename'])
        return img

    def _getSentence(self, sent):
        """ create a sentence structure for the driver """
        # NOOP for now
        return sent

    # PUBLIC FUNCTIONS

    def getSplitSize(self, split, ofwhat='sentences'):
        """ return size of a split, either number of sentences or number of images """
        if ofwhat == 'sentences':
            return sum(len(img['sentences']) for img in self.split[split])
        else:  # assume images
            return len(self.split[split])

    def getSplitSizeAllVal(self):

        return len(self.dataset['images']) - len(self.split['train'])

    def sampleImageSentencePair(self, split='train', within=None):
        """ sample image sentence pair from a split """
        images = self.split[split]

        while True:
            img = random.choice(images)
            sent = random.choice(img['sentences'])
            out = {}
            out['image'] = self._getImage(img)
            out['sentence'] = self._getSentence(sent)
            if within and out['image']['filename'] not in within:
                continue
            return out
    
    def sampleSentence(self, split = 'train', max_images = -1):
        images = self.split[split]
        while True:
            img = random.choice(images)
            sent = random.choice(img['sentences'])
            return sent

    def iterImageSentencePair(self, split='train', max_images=-1):
        for i, img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images:
                break
            for sent in img['sentences']:
                out = {}
                out['image'] = self._getImage(img)
                out['sentence'] = self._getSentence(sent)
                yield out

    def iterImageSentencePairBatch(
            self,
            split='train',
            max_images=-1,
            max_batch_size=100):
        batch = []
        for i, img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images:
                break
            for sent in img['sentences']:
                out = {}
                out['image'] = self._getImage(img)
                out['sentence'] = self._getSentence(sent)
                batch.append(out)
                if len(batch) >= max_batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def iterSentences(self, split='train'):
        for img in self.split[split]:
            for sent in img['sentences']:
                yield self._getSentence(sent)

    def retrieveSentences(self, img_fn):
        img_fn = img_fn
        split = 'val'

        for img in self.split[split]:
            for sent in img['sentences']:
                yield self._getSentence(sent)


    def iterSentencesPerImg(self, split='train'):
        for img in self.split[split]:
            sents = {}
            fn = img['filename']
            sents[fn] = []
            for sent in img['sentences']:
                sents[fn].append(self._getSentence(sent))
            yield sents

    def iterImages(self, split='train', shuffle=False, max_images=-1):
        imglist = self.split[split]
        ix = range(len(imglist))
        if shuffle:
            random.shuffle(ix)
        if max_images > 0:
            ix = ix[:min(len(ix), max_images)]  # crop the list
        for i in ix:
            yield self._getImage(imglist[i])

    def iterImagesAll(self, shuffle=False, max_images=-1):
        for split in self.split:
            imglist = self.split[split]
            ix = range(len(imglist))
            if shuffle:
                random.shuffle(ix)
            if max_images > 0:
                ix = ix[:min(len(ix), max_images)]  # crop the list
            for i in ix:
                yield self._getImage(imglist[i])



    def iterImagesAllVal(self, shuffle=False, max_images=-1):
        for split in self.split:
            if split == 'train':
                continue
            imglist = self.split[split]
            ix = range(len(imglist))
            if shuffle:
                random.shuffle(ix)
            if max_images > 0:
                ix = ix[:min(len(ix), max_images)]  # crop the list
            for i in ix:
                yield self._getImage(imglist[i])


def getDataProvider(dataset):
    """ we could intercept a special dataset and return different data providers """
    assert dataset in ['flickr8k', 'flickr30k', 'coco', 'coco_places', 'coco_mrnn', 'coco_mrnn_rf', 'coco_inception','coco_mrnn_inception', 'coco_inception_crop', 'coco_coco_places'], 'dataset %s unknown' % (dataset, )
    return BasicDataProvider(dataset)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <split>'.format(sys.argv[0])
        sys.exit()

    split = sys.argv[1]
    dp = getDataProvider('coco_inception')

    for sent in dp.iterSentences(split):
        print sent['raw']
