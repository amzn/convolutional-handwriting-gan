# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

from .ScrabbleGAN_baseModel import *

activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),}

class ScrabbleGANModel(ScrabbleGANBaseModel):

    def __init__(self, opt):
        ScrabbleGANBaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.netD = Discriminator(**vars(opt))
        if len(opt.gpu_ids) > 0:
            self.netD.to(opt.gpu_ids[0])
            if len(opt.gpu_ids) > 1:
                self.netD = torch.nn.DataParallel(self.netD, device_ids=opt.gpu_ids, output_device=opt.gpu_ids[0]).cuda()

        print(self.netD)
        if self.isTrain:
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.D_lr, betas=(opt.D_B1, opt.D_B2), weight_decay=0, eps=opt.adam_eps)
            self.optimizers.append(self.optimizer_D)
            self.optimizer_D.zero_grad()

