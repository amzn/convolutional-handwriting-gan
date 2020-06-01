# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--percent_labeled', type=int, default=100, help='percentage of labeled data')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--num_critic_train', type=int, default=4, help='number of critic train (only one for the generator)')
        parser.add_argument('--saved_model', type=str, default='',
                            help='path to model in checkpoints directory. takes latest model')
        parser.add_argument('--mask_loss', action='store_true', help='use masking in the loss terms according to the '
                                                                     'length of each word (already used by default in '
                                                                     'the CTC, this flag adds the masking to the other '
                                                                     'losses).')
        # val and test datasets to test during training
        parser.add_argument('--test_dataroot', type=str, default='datasets/RIMEStest_prepared_height32varyingWidth/',
                            help='augmentation images file location')
        parser.add_argument('--val_dataroot', type=str, default='datasets/RIMESval_prepared_height32varyingWidth/',
                            help='augmentation images file location')
        # unlabeled data for semi-supervised training
        parser.add_argument('--unlabeled_dataname', type=str, default=None,
                            help='unlabeled images dataset name for semi supervised training of the GAN, the path is determined according to the dictionary in data/dataset_catalog.py')
        # semi supervised training
        parser.add_argument('--disjoint', action='store_true', help='disjoint training of the OCR and the discriminator '
                                                                    '(the discriminator is trained only on the '
                                                                    'unlbaled dataset and the OCR is trained onlt on '
                                                                    'the labeled dataset)')
        parser.add_argument('--onlyOCR', action='store_true', help="use only OCR loss during training")
        parser.add_argument('--optimize_z', action='store_true', help="optimize z during training (relevant only for single writer optimization)")
        parser.add_argument('--not_optimize_G', action='store_true', help="don't optimize G during training")
        parser.add_argument('--reconst_loss', type=str, default='mse',
                            help='mse|l1|d_features')
        # gradients
        parser.add_argument('--no_grad_balance', action='store_true',
                            help='use gradient balancing on the fake image between the ocr and the adversarial/reconstruction loss')
        parser.add_argument('--gb_alpha', type=float, default=1,
                            help='coefficient for gradient balancing - larger alpha means stronger balancing (default: %(default)s)')
        parser.add_argument('--clip_grad', type=float, default=0.0, help='clip grad for D and G if 0 do nothing')

        ### Optimizer stuff ###
        parser.add_argument(
            '--G_lr', type=float, default=2e-4,
            help='Learning rate to use for Generator (default: %(default)s)')
        parser.add_argument(
            '--D_lr', type=float, default=2e-4,
            help='Learning rate to use for Discriminator (default: %(default)s)')
        parser.add_argument(
            '--G_B1', type=float, default=0.0,
            help='Beta1 to use for Generator (default: %(default)s)')
        parser.add_argument(
            '--D_B1', type=float, default=0.0,
            help='Beta1 to use for Discriminator (default: %(default)s)')
        parser.add_argument(
            '--G_B2', type=float, default=0.999,
            help='Beta2 to use for Generator (default: %(default)s)')
        parser.add_argument(
            '--D_B2', type=float, default=0.999,
            help='Beta2 to use for Discriminator (default: %(default)s)')
        parser.add_argument('--OCR_B2', type=float, default=0.999,
                            help='Beta2 to use for OCRnet (default: %(default)s)')
        parser.add_argument('--OCR_B1', type=float, default=0.0, help='Beta1 to use for OCRnet (default: %(default)s)')
        parser.add_argument('--OCR_lr', type=float, default=2e-4,
                            help='Learning rate to use for OCRnet (default: %(default)s)')
        self.isTrain = True
        return parser
