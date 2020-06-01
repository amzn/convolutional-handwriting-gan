# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from .base_options import BaseOptions
import numpy as np

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--clamp', type=float, default=np.inf, help='clamp values of z during test time or image generation.')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.add_argument('--path_lex', type=str, default='./datasets/Lexique/english_words.txt', help='path to lexicon')
        parser.add_argument('--n_synth', type=str, default='100 200 400', help='number of images to synthesize in each'
                                                                               ' lmbd file (each file will inmclude '
                                                                               'the train set and the number of images '
                                                                               'multiplied by 1000)')
        parser.add_argument('--no_concat_dataset', action='store_true', help='do not concat to original dataset when generating '
                                                                          'the dataset with gan images')

        self.isTrain = False
        return parser
