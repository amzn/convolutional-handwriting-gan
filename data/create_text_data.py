# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import os
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import sys
from PIL import Image
import random
import io
import xmltodict
import html
from sklearn.decomposition import PCA
import math
from tqdm import tqdm
from itertools import compress

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)


def find_rot_angle(idx_letters):
    idx_letters = np.array(idx_letters).transpose()
    pca = PCA(n_components=2)
    pca.fit(idx_letters)
    comp = pca.components_
    angle = math.atan(comp[0][0]/comp[0][1])
    return math.degrees(angle)

def read_data_from_folder(folder_path):
    image_path_list = []
    label_list = []
    pics = os.listdir(folder_path)
    pics.sort(key=lambda i: len(i))
    for pic in pics:
        image_path_list.append(folder_path + '/' + pic)
        label_list.append(pic.split('_')[0])
    return image_path_list, label_list


def read_data_from_file(file_path):
    image_path_list = []
    label_list = []
    f = open(file_path)
    while True:
        line1 = f.readline()
        line2 = f.readline()
        if not line1 or not line2:
            break
        line1 = line1.replace('\r', '').replace('\n', '')
        line2 = line2.replace('\r', '').replace('\n', '')
        image_path_list.append(line1)
        label_list.append(line2)

    return image_path_list, label_list


def show_demo(demo_number, image_path_list, label_list):
    print('\nShow some demo to prevent creating wrong lmdb data')
    print('The first line is the path to image and the second line is the image label')
    for i in range(demo_number):
        print('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))

def create_img_label_list(top_dir,dataset, mode, words, author_number, remove_punc):
    root_dir = os.path.join(top_dir, dataset)
    output_dir = root_dir + (dataset=='IAM')*('/words'*words + '/lines'*(not words))
    image_path_list, label_list = [], []
    author_id = 'None'
    if dataset=='CVL':
        root_dir = os.path.join(root_dir, 'cvl-database-1-1')
        if words:
            images_name = 'words'
        else:
            images_name = 'lines'
        if mode == 'tr' or mode == 'val':
            mode_dir = ['trainset']
        elif mode == 'te':
            mode_dir = ['testset']
        elif mode == 'all':
            mode_dir = ['testset', 'trainset']
        idx = 1
        for mod in mode_dir:
            images_dir = os.path.join(root_dir, mod, images_name)
            for path, subdirs, files in os.walk(images_dir):
                for name in files:
                    if (mode == 'tr' and idx >= 10000) or (
                            mode == 'val' and idx < 10000) or mode == 'te' or mode == 'all' or mode == 'tr_3te':
                        if os.path.splitext(name)[0].split('-')[1] == '6':
                            continue
                        label = os.path.splitext(name)[0].split('-')[-1]
                        if 'ä' in label or 'ü' in label or label=='':
                            continue
                        imagePath = os.path.join(path, name)
                        label_list.append(label)
                        image_path_list.append(imagePath)
                    idx += 1

    elif dataset=='IAM':
        labels_name = 'original'
        if mode=='all':
            mode = ['te', 'va1', 'va2', 'tr']
        elif mode=='valtest':
            mode=['te', 'va1', 'va2']
        else:
            mode = [mode]
        if words:
            images_name = 'wordImages'
        else:
            images_name = 'lineImages'
        images_dir = os.path.join(root_dir, images_name)
        labels_dir = os.path.join(root_dir, labels_name)
        full_ann_files = []
        im_dirs = []
        line_ann_dirs = []
        image_path_list, label_list = [], []
        for mod in mode:
            part_file = os.path.join(root_dir, 'original_partition', mod + '.lst')
            with open(part_file)as fp:
                for line in fp:
                    name = line.split('-')
                    if int(name[-1][:-1]) == 0:
                        anno_file = os.path.join(labels_dir, '-'.join(name[:2]) + '.xml')
                        full_ann_files.append(anno_file)
                        im_dir = os.path.join(images_dir, name[0], '-'.join(name[:2]))
                        im_dirs.append(im_dir)

        if author_number >= 0:
            full_ann_files = [full_ann_files[author_number]]
            im_dirs = [im_dirs[author_number]]
            author_id = im_dirs[0].split('/')[-1]

        lables_to_skip = ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']
        for i, anno_file in enumerate(full_ann_files):
            with open(anno_file) as f:
                try:
                    line = f.read()
                    annotation_content = xmltodict.parse(line)
                    lines = annotation_content['form']['handwritten-part']['line']
                    if words:
                        lines_list = []
                        for j in range(len(lines)):
                            lines_list.extend(lines[j]['word'])
                        lines = lines_list
                except:
                    print('line is not decodable')
                for line in lines:
                    try:
                        label = html.unescape(line['@text'])
                    except:
                        continue
                    if remove_punc and label in lables_to_skip:
                        continue
                    id = line['@id']
                    imagePath = os.path.join(im_dirs[i], id + '.png')
                    image_path_list.append(imagePath)
                    label_list.append(label)

    elif dataset=='RIMES':
        if mode=='tr':
            images_dir = os.path.join(root_dir, 'orig','training_WR')
            gt_file = os.path.join(root_dir, 'orig',
                               'groundtruth_training_icdar2011.txt')
        elif mode=='te':
            images_dir = os.path.join(root_dir, 'orig', 'testdataset_ICDAR')
            gt_file = os.path.join(root_dir, 'orig',
                                       'ground_truth_test_icdar2011.txt')
        elif mode=='val':
            images_dir = os.path.join(root_dir, 'orig', 'valdataset_ICDAR')
            gt_file = os.path.join(root_dir, 'orig',
                                       'ground_truth_validation_icdar2011.txt')
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        image_path_list = [os.path.join(images_dir, line.split(' ')[0]) for line in lines if len(line.split(' ')) > 1]

        label_list = [line.split(' ')[1][:-1] for line in lines if len(line.split(' ')) > 1]

    return image_path_list, label_list, output_dir, author_id

def createDataset(image_path_list, label_list, outputPath, mode, author_id, remove_punc, resize, imgH, init_gap, h_gap, charminW, charmaxW, discard_wide, discard_narr, labeled):
    assert (len(image_path_list) == len(label_list))
    nSamples = len(image_path_list)

    outputPath = outputPath + (resize=='charResize') * ('/h%schar%sto%s/'%(imgH, charminW, charmaxW)) + (resize=='keepRatio') * ('/h%s/'%(imgH)) \
                 + (resize=='noResize') * ('/noResize/') + (author_id!='None') * ('single_authors/'+author_id+'/' ) \
                 + mode + (resize!='noResize') * (('_initGap%s'%(init_gap)) * (init_gap>0) + ('_hGap%s'%(h_gap)) * (h_gap>0) \
                 + '_NoDiscard_wide' * (not discard_wide) + '_NoDiscard_wide' * (not discard_narr))+'_unlabeld' * (not labeled) +\
                 (('IAM' in outputPath) and remove_punc) *'_removePunc'
    print(outputPath)
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    for i in tqdm(range(nSamples)):
        imagePath = image_path_list[i]
        label = label_list[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        try:
            im = Image.open(imagePath)
        except:
            continue
        if resize in ['charResize', 'keepRatio']:
            width, height = im.size
            new_height = imgH - (h_gap * 2)
            len_word = len(label)
            width = int(width * imgH / height)
            new_width = width
            if resize=='charResize':
                if (width/len_word > (charmaxW-1)) or (width/len_word < charminW) :
                    if discard_wide and width/len_word > 3*((charmaxW-1)):
                        print('%s has a width larger than max image width' % imagePath)
                        continue
                    if discard_narr and (width / len_word) < (charminW/3):
                        print('%s has a width smaller than min image width' % imagePath)
                        continue
                    else:
                        new_width = len_word * random.randrange(charminW, charmaxW)

            # reshape the image to the new dimensions
            im = im.resize((new_width, new_height))
            # append with 256 to add left, upper and lower white edges
            init_w = int(random.normalvariate(init_gap, init_gap / 2))
            new_im = Image.new("RGB", (new_width+init_gap, imgH), color=(256,256,256))
            new_im.paste(im, (abs(init_w), h_gap))
            im = new_im

        imgByteArr = io.BytesIO()
        im.save(imgByteArr, format='tiff')
        wordBin = imgByteArr.getvalue()
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        cache[imageKey] = wordBin
        if labeled:
            cache[labelKey] = label
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)

def createDict(label_list, top_dir, dataset, mode, words, remove_punc):
    lex_name = dataset+'_' + mode + (dataset in ['IAM','RIMES'])*('_words' * words) + (dataset=='IAM') * ('_removePunc' * remove_punc)
    all_words = '-'.join(label_list).split('-')
    unique_words = []
    words = []
    for x in tqdm(all_words):
        if x!='' and x!=' ':
            words.append(x)
            if x not in unique_words:
                unique_words.append(x)
    print(len(words))
    print(len(unique_words))
    with open(os.path.join(top_dir, 'Lexicon', lex_name+'_stratified.txt'), "w") as file:
        file.write("\n".join(unique_words))
    file.close()
    with open(os.path.join(top_dir, 'Lexicon', lex_name + '_NOTstratified.txt'), "w") as file:
        file.write("\n".join(words))
    file.close()

def printAlphabet(label_list):
    # get all unique alphabets - ignoring alphabet longer than one char
    all_chars = ''.join(label_list)
    unique_chars = []
    for x in all_chars:
        if x not in unique_chars and len(x) == 1:
            unique_chars.append(x)

    # for unique_char in unique_chars:
    print(''.join(unique_chars))

if __name__ == '__main__':
    create_Dict = True # create a dictionary of the generated dataset
    dataset = 'IAM'     #CVL/IAM/RIMES/gw
    mode = 'tr'        # tr/te/val/va1/va2/all
    labeled = True
    top_dir = 'Datasets'
    # parameter relevant for IAM/RIMES:
    words = True        # use words images, otherwise use lines
    #parameters relevant for IAM:
    author_number = -1  # use only images of a specific writer. If the value is -1, use all writers, otherwise use the index of this specific writer
    remove_punc = True  # remove images which include only one punctuation mark from the list ['.', '', ',', '"', "'", '(', ')', ':', ';', '!']

    resize = 'charResize'  # charResize|keepRatio|noResize - type of resize,
                        # char - resize so that each character's width will be in a specific range (inside this range the width will be chosen randomly),
                        # keepRatio - resize to a specific image height while keeping the height-width aspect-ratio the same.
                        # noResize - do not resize the image
    imgH = 32           # height of the resized image
    init_gap = 0        # insert a gap before the beginning of the text with this number of pixels
    charmaxW = 17       # The maximum character width
    charminW = 16       # The minimum character width
    h_gap = 0           # Insert a gap below and above the text
    discard_wide = True # Discard images which have a character width 3 times larger than the maximum allowed character size (instead of resizing them) - this helps discard outlier images
    discard_narr = True # Discard images which have a character width 3 times smaller than the minimum allowed charcter size.

    image_path_list, label_list, outputPath, author_id = create_img_label_list(top_dir,dataset, mode, words, author_number, remove_punc)
    # in a previous version we also cut the white edges of the image to keep a tight rectangle around the word but it
    # seems in all the datasets we use this is already the case so I removed it. If there are problems maybe we should add this back.
    createDataset(image_path_list, label_list, outputPath, mode, author_id, remove_punc, resize, imgH, init_gap, h_gap, charminW, charmaxW, discard_wide, discard_narr, labeled)
    if create_Dict:
        createDict(label_list, top_dir, dataset, mode, words, remove_punc)
    printAlphabet(label_list)