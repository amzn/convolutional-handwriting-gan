# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_transform
import os
import sys

class TextDataset(BaseDataset):
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--collate', action='store_false', default=True,
                            help='use regular collate function in data loader')
        parser.add_argument('--aug_dataroot', type=str, default=None,
                            help='augmentation images file location, default is None (no augmentation)')
        parser.add_argument('--aug', action='store_true', default=False,
                            help='use augmentation (currently relevant for OCR training)')
        return parser

    def __init__(self, opt, target_transform=None):

        BaseDataset.__init__(self, opt)

        self.env = lmdb.open(
            os.path.abspath(opt.dataroot),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (opt.dataroot))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
            self.nSamples = nSamples

        if opt.aug and opt.aug_dataroot is not None:
            self.env_aug = lmdb.open(
                os.path.abspath(opt.aug_dataroot),
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

            with self.env_aug.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
                self.nSamples = self.nSamples + nSamples
                self.nAugSamples = nSamples

        self.transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.target_transform = target_transform
        if opt.collate:
            self.collate_fn = TextCollator(opt)
        else:
            self.collate_fn = RegularCollator(opt)

        self.labeled = opt.labeled

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        envAug = False
        if hasattr(self, 'env_aug'):
            if index>=self.nAugSamples:
                index = index-self.nAugSamples
            else:
                envAug = True
        index += 1
        with eval('self.env'+'_aug'*envAug+'.begin(write=False)') as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            item = {'img': img, 'img_path': img_key, 'idx':index}

            if self.labeled:
                label_key = 'label-%09d' % index
                label = txn.get(label_key.encode('utf-8'))

                if self.target_transform is not None:
                    label = self.target_transform(label)
                item['label'] = label


            if hasattr(self,'Z'):
                z = self.Z[index-1]
                item['z'] = z

        return item


class TextCollator(object):
    def __init__(self, opt):
        self.resolution = opt.resolution
    def __call__(self, batch):

        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'img_path':img_path, 'idx':indexes}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item


class RegularCollator(object):
    def __init__(self, opt):
        self.resolution = opt.resolution
    def __call__(self, batch):
        img_path = [item['img_path'] for item in batch]
        imgs = torch.stack([item['img'] for item in batch])
        item = {'img': imgs, 'img_path':img_path}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item