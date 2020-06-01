"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT

General-purpose semi-supervised training script for ScrabbleGAN.

You need to specify the labeled dataset ('--dataname'), unlabeled dataset ('--unlabeled_dataname'),
('--disjoint') if the training is disjoint (the recognizer sees only the labeled data and the discriminator sees only the unlabeled data)
 and experiment name prefix ('--name_prefix').


It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
python train_semi_supervised.py --dataname IAMcharH32W16rmPunct --unlabeled_dataname CVLtrH32 --disjoint


See options/base_options.py and options/train_options.py for more training options.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import seed_rng
from util.util import prepare_z_y, get_curr_data
import torch
from itertools import cycle

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    seed_rng(opt.seed)
    torch.backends.cudnn.benchmark = True
    opt.labeled = True
    dataset_labeled = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.labeled = False
    opt.dataroot = opt.unlabeled_dataroot
    dataset_unlabeled = create_dataset(opt)
    dataset_size = max(len(dataset_labeled),len(dataset_unlabeled))    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    opt.iter = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        if len(dataset_unlabeled) > len(dataset_labeled):
            zip_dataset = zip(cycle(dataset_labeled), dataset_unlabeled)
        else:
            zip_dataset = zip(dataset_labeled, cycle(dataset_unlabeled))

        for i, (data_labeled, data_unlabeled) in enumerate(zip_dataset):  # inner loop within one epoch
            opt.iter = i
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size*opt.num_accumulations
            epoch_iter += opt.batch_size*opt.num_accumulations

            if opt.num_critic_train == 1:
                curr_data = data_labeled
                model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                model.optimize_G()
                if opt.disjoint:
                    model.optimize_OCR()
                else:
                    model.optimize_D_OCR()
                model.optimize_G_step()
                model.optimize_D_OCR_step()

                curr_data = data_unlabeled
                model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                model.optimize_G()
                model.optimize_D()
                model.optimize_G_step()
                model.optimize_D_OCR_step()
            else:
                if (i % opt.num_critic_train) == 0:
                    curr_data = data_labeled
                    model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                    model.optimize_G()
                    model.optimize_G_step()

                    curr_data = data_unlabeled
                    model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                    model.optimize_G()
                    model.optimize_G_step()

                curr_data = data_labeled
                model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                if opt.disjoint:
                    model.optimize_OCR()
                else:
                    model.optimize_D_OCR()
                model.optimize_D_OCR_step()

                curr_data = data_unlabeled
                model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                model.optimize_D()
                model.optimize_D_OCR_step()

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / (opt.batch_size*opt.num_accumulations)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            torch.cuda.empty_cache()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
