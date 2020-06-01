"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT

General-purpose script to generate lmdb with ScrabbleGAN generated images.

You need to specify the dataset ('--dataname'), the path to save the lmdb ('--results_dir'), the number of images to
synthesize can be a few values to generate a few lmdbs of different sizes,('--n_synth')  and the model name ('--name').

Example:
python generate_wordsLMDB.py --dataname IAMcharH32rmPunct --results_dir ./lmdb_files/IAM_concat --n_synth 100,200 --name model_name

See options/base_options.py and options/train_options.py for more training options.

"""

import os, stat
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import pandas as pd
import lmdb
import io
from data.create_text_data import writeCache, find_rot_angle
from PIL import Image
from tqdm import tqdm
from joblib import cpu_count, Parallel, delayed
import torch
from util.util import prepare_z_y
import six

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    exception_chars = ['ï', 'ü', '.', '_', 'ö', ',', 'ã', 'ñ']
    if opt.lex.endswith('.tsv'):
        lex = pd.read_csv(opt.lex, sep='\t')['lemme']
        lex = [word.split()[-1] for word in lex if
                    (pd.notnull(word) and all(char not in word for char in exception_chars))]
    elif opt.lex.endswith('.txt'):
        with open(opt.lex, 'rb') as f:
            lex = f.read().splitlines()
        lex_updated = []
        for word in lex:
            try:
                word = word.decode("utf-8")
            except:
                continue
            if len(word) < 20:
                lex_updated.append(word)
        lex = lex
    min_factor, max_factor = 1, 1

    n_synth = opt.n_synth.split(',')
    env_paths = [opt.results_dir + n + 'k' for n in opt.n_synth.split(',')]
    if not os.path.exists('/'.join(opt.results_dir.split('/')[:-1])):
        os.makedirs('/'.join(opt.results_dir.split('/')[:-1]))
    env = [lmdb.open(env_path, map_size=1099511627776) for env_path in env_paths]
    n_synth = [int(i)*1000 for i in n_synth]
    max_n_synth = max(n_synth)

    cache = {}
    cnt = 1
    print(model.netG.blocks[0][0].conv1.weight[0,0,:,:])
    print(model.netG.blocks[0][0].bn1.bias.weight[0,:10])
    print(model.netG.blocks[0][0].bn1.gain.weight[0,:10])
    print(model.netG.linear.bias[:10])
    print(opt.model)
    def GenImg(words=None, z=None):
        model.forward(words, z)
        im = model.fake.data.cpu().numpy().squeeze(0).squeeze(0) * 255
        im = Image.fromarray(im).convert('RGB')
        imgByteArr = io.BytesIO()
        im.save(imgByteArr, format='tiff')
        wordBin = imgByteArr.getvalue()
        return wordBin, model.words[0].decode('utf-8')

    def GenImgs(words=None, z=None, nsamples=5, device=0):
        model.netG.to(device)
        model.z, model.label_fake = prepare_z_y(opt.batch_size, opt.dim_z, len(model.lex),
                                              device=device, fp16=opt.G_fp16)
        model.device = device
        if words is None:
            words = nsamples*[words]
            z = nsamples*[z]
        words_encoded = []
        wordBins = []
        for i in tqdm(range(len(words))):
            wordBin, word = GenImg(words[i], z[i])
            words_encoded.append(word)
            wordBins.append(wordBin)
        return wordBins, words_encoded

    if opt.no_concat_dataset:
        cnt_orig = 0
    else:
        env_orig = lmdb.open(
            os.path.abspath(opt.dataroot),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        cnt_orig = env_orig.stat()['entries'] // 2
        with env_orig.begin() as txn:
            for index in tqdm(range(1, cnt_orig + 1)):
                img_key = 'image-%09d' % index
                imgbuf = txn.get(img_key.encode('utf-8'))
                label_key = 'label-%09d' % index
                label = txn.get(label_key.encode('utf-8'))
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf)
                imgByteArr = io.BytesIO()
                img.save(imgByteArr, format='tiff')
                img = imgByteArr.getvalue()
                imageKey = 'image-%09d' % cnt
                labelKey = 'label-%09d' % cnt
                cache[imageKey] = img
                cache[labelKey] = label
                if cnt % 1000 == 0:
                    for ev in env:
                        writeCache(ev, cache)
                    cache = {}
                cnt += 1
        for ev in env:
            writeCache(ev, cache)

    for iter in range((max_n_synth//1000000)+1):
        if iter==0:
            n_imgs = max_n_synth % 1000000
        else:
            n_imgs = 1000000
        n_jobs = torch.cuda.device_count()
        kwargs_gen = (dict(device=i, nsamples=int(n_imgs/n_jobs)) for i in range(n_jobs))
        data = Parallel(n_jobs=n_jobs)(delayed(GenImgs)(**kwargs) for kwargs in kwargs_gen)


        for d in data:
            for i in tqdm(range(len(d[0]))):
                imageKey = 'image-%09d' % cnt
                labelKey = 'label-%09d' % cnt
                cache[imageKey] = d[0][i]
                cache[labelKey] = d[1][i]
                if (cnt-cnt_orig) % 1000 == 0:
                    for n in range(len(n_synth)):
                        if n_synth[n]>=(cnt-cnt_orig):
                            writeCache(env[n], cache)
                    cache = {}
                cnt += 1


    for i in range(len(n_synth)):
        nSamples = cnt_orig + n_synth[i]
        cache['num-samples'] = str(nSamples)
        writeCache(env[i], cache)
        env[i].close()
        os.chmod(env_paths[i], 0o555)


