"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: MIT

General-purpose training script for ScrabbleGAN.

You need to specify the dataset ('--dataname') and experiment name prefix ('--name_prefix').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
python train.py --name_prefix demo --dataname RIMEScharH32W16 --capitalize --display_port 8192

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
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # Seed RNG
    seed_rng(opt.seed)
    torch.backends.cudnn.benchmark = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.single_writer:
        opt.G_init='N02'
        opt.D_init='N02'
        model.netG.init_weights()
        model.netD.init_weights()
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    opt.iter = 0
    # seed_rng(opt.seed)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            opt.iter = i
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size*opt.num_accumulations
            epoch_iter += opt.batch_size*opt.num_accumulations

            if opt.num_critic_train == 1:
                counter = 0
                for accumulation_index in range(opt.num_accumulations):
                    curr_data = get_curr_data(data, opt.batch_size, counter)
                    model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                    model.optimize_G()
                    model.optimize_D_OCR()
                    counter += 1
                model.optimize_G_step()
                model.optimize_D_OCR_step()
            else:
                if (i % opt.num_critic_train) == 0:
                    counter = 0
                    for accumulation_index in range(opt.num_accumulations):
                        curr_data = get_curr_data(data, opt.batch_size, counter)
                        model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                        model.optimize_G()
                        counter += 1
                    model.optimize_G_step()
                counter = 0
                for accumulation_index in range(opt.num_accumulations):
                    curr_data = get_curr_data(data, opt.batch_size, counter)
                    model.set_input(curr_data)  # unpack data from dataset and apply preprocessing
                    model.optimize_D_OCR()
                    counter += 1
                model.optimize_D_OCR_step()
                # print(model.netG.linear.bias[:10])
                # print('G',model.loss_G, 'D', model.loss_D, 'Dreal',model.loss_Dreal, 'Dfake', model.loss_Dfake,
                #       'OCR_real', model.loss_OCR_real, 'OCR_fake', model.loss_OCR_fake, 'grad_fake_OCR', model.loss_grad_fake_OCR, 'grad_fake_adv', model.loss_grad_fake_adv)



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

            for i in opt.gpu_ids:
                with torch.cuda.device('cuda:%f' % (i)):
                    torch.cuda.empty_cache()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            if opt.single_writer:
                torch.save(model.z[0], os.path.join(opt.checkpoints_dir, opt.name, str(epoch)+'_z.pkl'))
                torch.save(model.z[0], os.path.join(opt.checkpoints_dir, opt.name, 'latest_z.pkl'))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
