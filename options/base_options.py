# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import argparse
import os
from util import util
import torch
import models
import data
import data.alphabets as alphabets
import data.dataset_catalog as dataset_catalog

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        ### General ###
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models, if it is the default empty string, it will be determined by the other parameters')
        parser.add_argument('--name_prefix', type=str, default='', help='prefix to add to the automatically set experiment name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8096, help='visdom port of the web display')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')

        ### Dataset/Dataloader stuff ###
        parser.add_argument('--dataname', type=str, default='RIMEScharH32W16',
            help='dataset name, determines the path to the dataset according to data/dataset_catalog.py')
        parser.add_argument('--dataset_mode', type=str, default='text',
                            help='chooses how datasets are loaded. [folderClass]')
        parser.add_argument('--train', action='store_false', default=True, help='dataset mode')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--preprocess', type=str, default='no_preprocess',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--imgH', type=int, default=32,
                            help='height of the image')
        parser.add_argument(
            '--num_workers', type=int, default=8,
            help='Number of dataloader workers; consider using less for HDF5 '
                 '(default: %(default)s)')
        parser.add_argument(
            '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
            help='Pin data into memory through dataloader? (default: %(default)s)')
        parser.add_argument(
            '--no_shuffle', action='store_true', default=False,
            help='Do not shuffle the data (default: %(default)s)')
        parser.add_argument(
            '--load_in_mem', action='store_true', default=False,
            help='Load all data into memory? (default: %(default)s)')
        parser.add_argument(
            '--use_multiepoch_sampler', action='store_true', default=False,
            help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')
        parser.add_argument('--load_size', type=int, default=32, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=32, help='then crop to this size')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        parser.add_argument('--flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--alphabet', type=str, default='alphabet', help='alphabet')
        parser.add_argument('--labeled', action='store_false',
                            help='use labels for data')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')


        ### Model stuff ###
        parser.add_argument(
            '--model', type=str, default='ScrabbleGAN',
            help='Name of the model module (default: %(default)s)')
        parser.add_argument(
            '--G_param', type=str, default='SN',
            help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
                 ' or None (default: %(default)s)')
        parser.add_argument(
            '--D_param', type=str, default='SN',
            help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
                 ' or None (default: %(default)s)')
        parser.add_argument(
            '--G_ch', type=int, default=64,
            help='Channel multiplier for G (default: %(default)s)')
        parser.add_argument(
            '--D_ch', type=int, default=64,
            help='Channel multiplier for D (default: %(default)s)')
        parser.add_argument(
            '--G_depth', type=int, default=1,
            help='Number of resblocks per stage in G? (default: %(default)s)')
        parser.add_argument(
            '--bottom_width', type=int, default=4,
            help='The initial width dimension in G (default: %(default)s)')
        parser.add_argument(
            '--bottom_height', type=int, default=4,
            help='The initial height dimension in G (default: %(default)s)')
        parser.add_argument(
            '--D_depth', type=int, default=1,
            help='Number of resblocks per stage in D? (default: %(default)s)')
        parser.add_argument(
            '--D_thin', action='store_false', dest='D_wide', default=True,
            help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
        parser.add_argument(
            '--G_shared',  action='store_true', default=False,
            help='Use shared embeddings in G? (default: %(default)s)')
        parser.add_argument(
            '--shared_dim', type=int, default=0,
            help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
                 '(default: %(default)s)')
        parser.add_argument(
            '--dim_z', type=int, default=128,
            help='Noise dimensionality: %(default)s)')
        parser.add_argument(
            '--z_var', type=float, default=1.0,
            help='Noise variance: %(default)s)')
        parser.add_argument(
            '--no_hier', action='store_true', default=False,
            help='Do not use hierarchical z in G? (default: %(default)s)')
        parser.add_argument(
            '--cross_replica', action='store_true', default=False,
            help='Cross_replica batchnorm in G?(default: %(default)s)')
        parser.add_argument(
            '--mybn', action='store_true', default=False,
            help='Use my batchnorm (which supports standing stats?) %(default)s)')
        parser.add_argument(
            '--G_nl', type=str, default='relu',
            help='Activation function for G (default: %(default)s)')
        parser.add_argument(
            '--D_nl', type=str, default='relu',
            help='Activation function for D (default: %(default)s)')
        parser.add_argument(
            '--G_attn', type=str, default='64',
            help='What resolutions to use attention on for G (underscore separated) '
                 '(default: %(default)s)')
        parser.add_argument(
            '--D_attn', type=str, default='64',
            help='What resolutions to use attention on for D (underscore separated) '
                 '(default: %(default)s)')
        parser.add_argument(
            '--resolution', type=int, default='16',
            help='size of images generated'
                 '(default: %(default)s)')
        parser.add_argument(
            '--norm_style', type=str, default='bn',
            help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
                 'ln [layernorm], gn [groupnorm] (default: %(default)s)')
        parser.add_argument('--bn_linear', type=str, default='SN', help='type of layer used in cbn before BN')
        parser.add_argument('--one_hot', action='store_true', default=False,
            help='use labels as one hot vector(default: %(default)s)')
        parser.add_argument(
            '--first_layer', action='store_true', default=False,
            help='Use the class only in the first linear layer (not in the batch norms) (default: %(default)s)')
        parser.add_argument(
            '--z_dist', type=str, default='normal',
            help='distribution of the z (style) vector (normal, poisson, gamma) (default: %(default)s)')
        parser.add_argument('--single_writer', action='store_true', help='finetune the GAN on a single writer using only '
                                                                      'a single noise vector')
        parser.add_argument('--one_hot_k', type=int, default=1, help='use one-hot k encoding (the filter of each class '
                                                                     'has k length). if the value is 1 use a one hot '
                                                                     'encoding modulated by z ((instead of [0...0,1,0...0], '
                                                                     '[0...0, z1...zn,0...0]) ')
        parser.add_argument('--lex', type=str,
                            default='',
                            help='location of lexicon file')
        parser.add_argument('--randChars', action='store_true', default=False,
            help='randomly choose characters out of the alphabet to generate words'
                 '(default: %(default)s)')
        parser.add_argument('--capitalize', action='store_true', default=False,
            help='Capitilize the first letter of the word chosen to generate in 50% of the cases. '
                 '(default: %(default)s)')

        ### OCR model parameters ###
        parser.add_argument('--hidden_size_OCR', type=int, default=256, help= 'hidden size lstm in the ocr network'
                                                                                '(default: %(default)s)')
        parser.add_argument('--OCR_output_nc', type=int, default=512, help= 'feature extractor output size in the OCR net'
                                                                                '(default: %(default)s)')
        parser.add_argument('--use_rnn', action='store_true', default=False, help='Use rnn in the OCR network. (default: %(default)s)')
        parser.add_argument('--num_layers_OCR', type=int, default=1, help='number of lstm layers in the OCR network')



        ### Model init stuff ###
        parser.add_argument(
            '--seed', type=int, default=0,
            help='Random seed to use; affects both initialization and '
                 ' dataloading. (default: %(default)s)')
        parser.add_argument(
            '--G_init', type=str, default='N02',
            help='Init style to use for G (default: %(default)s)')
        parser.add_argument(
            '--D_init', type=str, default='N02',
            help='Init style to use for D(default: %(default)s)')
        parser.add_argument('--OCR_init', type=str, default='N02',
                            help='Init style to use for OCR net (N02, ortho, glorot, xavier) or weight file location (default: %(default)s)')
        parser.add_argument(
            '--skip_init', action='store_true', default=False,
            help='Skip initialization, ideal for testing when ortho init was used '
                 '(default: %(default)s)')


        ### Batch size, parallel, and precision stuff ###
        parser.add_argument(
            '--parallel', action='store_true', default=False,
            help='Train with multiple GPUs (default: %(default)s)')
        parser.add_argument(
            '--batch_size', type=int, default=8,
            help='Default overall batchsize (default: %(default)s)')
        parser.add_argument(
            '--G_batch_size', type=int, default=0,
            help='Batch size to use for G; if 0, same as D (default: %(default)s)')
        parser.add_argument(
            '--num_accumulations', type=int, default=1,
            help='Number of passes to accumulate gradients over '
                 '(default: %(default)s)')
        parser.add_argument(
            '--num_D_steps', type=int, default=2,
            help='Number of D steps per G step (default: %(default)s)')
        parser.add_argument(
            '--split_D', action='store_true', default=False,
            help='Run D twice rather than concatenating inputs? (default: %(default)s)')
        parser.add_argument(
            '--num_epochs', type=int, default=100,
            help='Number of epochs to train for (default: %(default)s)')
        parser.add_argument(
            '--G_fp16', action='store_true', default=False,
            help='Train with half-precision in G? (default: %(default)s)')
        parser.add_argument(
            '--D_fp16', action='store_true', default=False,
            help='Train with half-precision in D? (default: %(default)s)')
        parser.add_argument(
            '--D_mixed_precision', action='store_true', default=False,
            help='Train with half-precision activations but fp32 params in D? '
                 '(default: %(default)s)')
        parser.add_argument(
            '--G_mixed_precision', action='store_true', default=False,
            help='Train with half-precision activations but fp32 params in G? '
                 '(default: %(default)s)')
        parser.add_argument(
            '--accumulate_stats', action='store_true', default=False,
            help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
        parser.add_argument(
            '--num_standing_accumulations', type=int, default=16,
            help='Number of forward passes to use in accumulating standing stats? '
                 '(default: %(default)s)')

        ### Numerical precision and SV stuff ###
        parser.add_argument(
            '--adam_eps', type=float, default=1e-8,
            help='epsilon value to use for Adam (default: %(default)s)')
        parser.add_argument(
            '--BN_eps', type=float, default=1e-5,
            help='epsilon value to use for BatchNorm (default: %(default)s)')
        parser.add_argument(
            '--SN_eps', type=float, default=1e-8,
            help='epsilon value to use for Spectral Norm(default: %(default)s)')
        parser.add_argument(
            '--num_G_SVs', type=int, default=1,
            help='Number of SVs to track in G (default: %(default)s)')
        parser.add_argument(
            '--num_D_SVs', type=int, default=1,
            help='Number of SVs to track in D (default: %(default)s)')
        parser.add_argument(
            '--num_G_SV_itrs', type=int, default=1,
            help='Number of SV itrs in G (default: %(default)s)')
        parser.add_argument(
            '--num_D_SV_itrs', type=int, default=1,
            help='Number of SV itrs in D (default: %(default)s)')

        ### Ortho reg stuff ###
        parser.add_argument(
            '--G_ortho', type=float, default=0.0,  # 1e-4 is default for BigGAN
            help='Modified ortho reg coefficient in G(default: %(default)s)')
        parser.add_argument(
            '--D_ortho', type=float, default=0.0,
            help='Modified ortho reg coefficient in D (default: %(default)s)')
        parser.add_argument(
            '--toggle_grads', action='store_true', default=True,
            help='Toggle D and G''s "requires_grad" settings when not training them? '
                 ' (default: %(default)s)')


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults
        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        output_opt, _ = parser.parse_known_args()

        # determine alphabet and dataroot according to dataname:
        output_opt.dataroot = dataset_catalog.datasets[output_opt.dataname]
        if self.isTrain:
            if output_opt.unlabeled_dataname is not None:
                output_opt.unlabeled_dataroot = dataset_catalog.datasets[output_opt.unlabeled_dataname]
        alphabet_dict = dataset_catalog.alphabet_dict
        lex_dict = dataset_catalog.lex_dict
        for name in alphabet_dict.keys():
            if name in opt.dataname:
                output_opt.alphabet = getattr(alphabets, alphabet_dict[name])
                if output_opt.lex == '':
                    output_opt.lex = lex_dict[name]
                    lex_str = ''
                else:
                    lex_str = '_lex_'+os.path.splitext(output_opt.lex.split('/')[-1])[0]
        # save to the disk
        if output_opt.name=='':
            # name is constructed from the following:
            # 1. path to the datasets - with underscore instead of / beginning from main dataset name which is taken from the list IAM/RIMES/CVL/gw
            # 2. lexicon if it's not the default one for this dataset
            # 3. resolution
            # 4. batch size
            # if changed from default:
            # 5. dim_z
            # 6. no Hierarchial Z
            # 7. one_hot_k
            # 8. if different OCR from default is used, the structure of the OCR (TPS, feature extractor and prediction layer)
            # 9. if an rnn is used, '_useRNN' is added
            # 10. noGB or GB alpha different from 1
            # 11. semi supervised parameters
            # 12. Single writer parameters
            # 13. Not optimizing G
            # 14. Use reconstruction loss instead of adversarial loss and which one
            # 15. Use only rec onstruction loss (alpha=Inf)
            output_opt.name += output_opt.name_prefix + '_' +output_opt.dataname + lex_str + output_opt.capitalize * '_CapitalizeLex' + '_GANres%s'%output_opt.resolution + '_bs%s'%output_opt.batch_size
            if output_opt.dim_z != 128:
                output_opt.name += '_dimZ%s'%output_opt.dim_z
            if output_opt.no_hier:
                output_opt.name += '_noHier'
            if output_opt.one_hot_k > 1:
                output_opt.name += '_oneHot%s'%output_opt.one_hot_k
            if output_opt.use_rnn:
                output_opt.name += '_useRNN'

            if self.isTrain:
                if output_opt.no_grad_balance:
                    output_opt.name += '_noGB'
                elif output_opt.gb_alpha != 1:
                    output_opt.name += '_GB%s'%output_opt.gb_alpha
                if output_opt.unlabeled_dataname is not None:
                    output_opt.name += '_SemiSupervised_'+output_opt.unlabeled_dataname
                    if output_opt.disjoint:
                        output_opt.name += '_disjoint'
                if output_opt.single_writer:
                    output_opt.name += '_SingleWriter'
                    if output_opt.optimize_z:
                        output_opt.name += 'OptimizeZ'
                if output_opt.not_optimize_G:
                    output_opt.name += '_NotOptimizeG'
                if output_opt.onlyOCR:
                    output_opt.name += '_OnlyOCRLoss'

        output_opt.len_vocab = len(output_opt.alphabet)
        return output_opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

