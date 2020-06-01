# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from . import BigGAN_layers as layers
from .sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d
from util.util import to_device, load_network
from .networks import init_weights

# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}
    arch[512] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
                 'upsample': [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 'resolution': [8, 16, 32, 64, 128, 256, 512],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 10)}}
    arch[256] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
                 'upsample': [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 'resolution': [8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 9)}}
    arch[128] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                 'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                 'upsample': [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 'resolution': [8, 16, 32, 64, 128],
                 'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                               for i in range(3, 8)}}
    arch[64] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [(2, 2), (2, 2), (2, 2), (2, 2)],
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)}}

    arch[63] = {'in_channels': [ch * item for item in [16, 16, 8, 4]],
                'out_channels': [ch * item for item in [16, 8, 4, 2]],
                'upsample': [(2, 2), (2, 2), (2, 2), (2,1)],
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 7)},
                'kernel1': [3, 3, 3, 3],
                'kernel2': [3, 3, 1, 1]
                }

    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [(2, 2), (2, 2), (2, 2)],
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)}}

    arch[32] = {'in_channels': [ch * item for item in [4, 4, 4]],
                'out_channels': [ch * item for item in [4, 4, 4]],
                'upsample': [(2, 2), (2, 2), (2, 2)],
                'resolution': [8, 16, 32],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                              for i in range(3, 6)},
                'kernel1': [3, 3, 3],
                'kernel2': [3, 3, 1]
                }

    arch[129] = {'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
                      'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1, 1]],
                      'upsample': [(2,2), (2,2), (2,2),  (2,2),  (2,2),  (1,2), (1,2)],
                      'resolution': [8, 16, 32, 64, 128, 256, 512],
                      'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 10)}}

    arch[33] = {'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
                      'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
                      'upsample': [(2,2), (2,2), (2,2),  (1,2),  (1,2)],
                      'resolution': [8, 16, 32, 64, 128],
                      'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 8)}}

    arch[31] = {'in_channels': [ch * item for item in [16, 16, 4, 2]],
                'out_channels': [ch * item for item in [16, 4, 2, 1]],
                'upsample': [(2,2), (2,2), (2,2), (1,2)],
                'resolution': [8, 16, 32, 64],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 7)},
                'kernel1':[3, 3, 3, 3],
                'kernel2': [3, 1, 1, 1]}

    arch[16] = {'in_channels': [ch * item for item in [8, 4, 2]],
                'out_channels': [ch * item for item in [4, 2, 1]],
                'upsample': [(2,2), (2,2), (2,1)],
                'resolution': [8, 16, 16],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 6)},
                'kernel1':[3, 3, 3],
                'kernel2': [3, 3, 1]}

    arch[17] = {'in_channels': [ch * item for item in [8, 4, 2]],
                'out_channels': [ch * item for item in [4, 2, 1]],
                'upsample': [(2,2), (2,2), (2,1)],
                'resolution': [8, 16, 16],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 6)},
                'kernel1':[3, 3, 3],
                'kernel2': [3, 3, 1]}

    arch[20] = {'in_channels': [ch * item for item in [8, 4, 2]],
                'out_channels': [ch * item for item in [4, 2, 1]],
                'upsample': [(2,2), (2,2), (2,1)],
                'resolution': [8, 16, 16],
                'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')])
                                    for i in range(3, 6)},
                'kernel1':[3, 3, 3],
                'kernel2': [3, 1, 1]}

    return arch


class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, bottom_height=4,resolution=128,
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, no_hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 BN_eps=1e-5, SN_eps=1e-12, G_fp16=False,
                 G_init='ortho', skip_init=False,
                 G_param='SN', norm_style='bn',gpu_ids=[], bn_linear='embed', input_nc=3,
                 one_hot=False, first_layer=False, one_hot_k=1,
                 **kwargs):
        super(Generator, self).__init__()
        self.name = 'G'
        # Use class only in first layer
        self.first_layer = first_layer
        # gpu-ids
        self.gpu_ids = gpu_ids
        # Use one hot vector representation for input class
        self.one_hot = one_hot
        # Use one hot k vector representation for input class if k is larger than 0. If it's 0, simly use the class number and not a k-hot encoding.
        self.one_hot_k = one_hot_k
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial width dimensions
        self.bottom_width = bottom_width
        # The initial height dimension
        self.bottom_height = bottom_height
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = not no_hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]
        self.bn_linear = bn_linear

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.dim_z = self.z_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_embedding = nn.Embedding

        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        if self.bn_linear=='SN':
            bn_linear = functools.partial(self.which_linear, bias=False)
        if self.G_shared:
            input_size = self.shared_dim + self.z_chunk_size
        elif self.hier:
            if self.first_layer:
                input_size = self.z_chunk_size
            else:
                input_size = self.n_classes + self.z_chunk_size
            self.which_bn = functools.partial(layers.ccbn,
                                              which_linear=bn_linear,
                                              cross_replica=self.cross_replica,
                                              mybn=self.mybn,
                                              input_size=input_size,
                                              norm_style=self.norm_style,
                                              eps=self.BN_eps)
        else:
            input_size = self.n_classes
            self.which_bn = functools.partial(layers.bn,
                                              cross_replica=self.cross_replica,
                                              mybn=self.mybn,
                                              eps=self.BN_eps)




        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                       else layers.identity())
        # First linear layer
        # The parameters for the first linear layer depend on the different input variations.
        if self.first_layer:
            if self.one_hot:
                self.linear = self.which_linear(self.dim_z // self.num_slots + self.n_classes,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            else:
                self.linear = self.which_linear(self.dim_z // self.num_slots + 1,
                                                self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            if self.one_hot_k==1:
                self.linear = self.which_linear((self.dim_z // self.num_slots) * self.n_classes,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
            if self.one_hot_k>1:
                self.linear = self.which_linear(self.dim_z // self.num_slots + self.n_classes*self.one_hot_k,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))


        else:
            self.linear = self.which_linear(self.dim_z // self.num_slots,
                                        self.arch['in_channels'][0] * (self.bottom_width * self.bottom_height))
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            if 'kernel1' in self.arch.keys():
                padd1 = 1 if self.arch['kernel1'][index]>1 else 0
                padd2 = 1 if self.arch['kernel2'][index]>1 else 0
                conv1 = functools.partial(layers.SNConv2d,
                                                kernel_size=self.arch['kernel1'][index], padding=padd1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
                conv2 = functools.partial(layers.SNConv2d,
                                                kernel_size=self.arch['kernel2'][index], padding=padd2,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
                self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv1=conv1,
                                               which_conv2=conv2,
                                               which_bn=self.which_bn,
                                               activation=self.activation,
                                               upsample=(functools.partial(F.interpolate,
                                                                           scale_factor=self.arch['upsample'][index])
                                                         if index < len(self.arch['upsample']) else None))]]
            else:
                self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv1=self.which_conv,
                                               which_conv2=self.which_conv,
                                               which_bn=self.which_bn,
                                               activation=self.activation,
                                               upsample=(functools.partial(F.interpolate, scale_factor=self.arch['upsample'][index])
                                                         if index < len(self.arch['upsample']) else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], input_nc))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self = init_weights(self, G_init)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            zs = torch.split(z, self.z_chunk_size, 1)
            z = zs[0]
            if len(y.shape)<2:
                y = y.unsqueeze(1)
            if self.first_layer:
                ys = zs[1:]
            else:
                ys = [torch.cat([y.type(torch.float32), item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.blocks)
        # This is the change we made to the Big-GAN generator architecture.
        # The input goes into classes go into the first layer only.
        if self.first_layer:
            # Each characters filter is modulated by the noise vector
            if self.one_hot_k==1:
                z = z.unsqueeze(1).repeat(1, y.shape[1], y.shape[2]) * torch.repeat_interleave(y, z.shape[1], 2)
            # Each character's filter is a one-hot k (for N char alphabet -
            # the entire vector is N*k long and the k values in the specific character location are equal to 1.
            # The filters are concatenated to the noise vector.
            elif self.one_hot_k>1:
                y = torch.repeat_interleave(y, self.one_hot_k, 2)
                z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)
            # only the noise vector is used as an input
            else:
                z = torch.cat((z.unsqueeze(1).repeat(1, y.shape[1], 1), y), 2)

        # First linear layer
        h = self.linear(z)
        # Reshape - when y is not a single class value but rather an array of classes, the reshape is needed to create
        # a separate vertical patch for each input.
        if self.first_layer:
            # correct reshape
            h = h.view(h.size(0), h.shape[1] * self.bottom_width, self.bottom_height, -1)
            h = h.permute(0, 3, 2, 1)

        else:
            h = h.view(h.size(0), -1, self.bottom_width, self.bottom_height)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))



# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64', input_nc=3, ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[128] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[64] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[63] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8]],
                'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 7)}}
    arch[32] = {'in_channels': [input_nc] + [item * ch for item in [4, 4, 4]],
                'out_channels': [item * ch for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 16, 16, 16],
                'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                              for i in range(2, 6)}}
    arch[129] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                 'downsample': [True] * 6 + [False],
                 'resolution': [128, 64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 8)}}
    arch[33] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[31] = {'in_channels': [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
                 'downsample': [True] * 5 + [False],
                 'resolution': [64, 32, 16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 10)}}
    arch[16] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}

    arch[17] = {'in_channels': [input_nc] + [ch * item for item in [1, 4]],
                 'out_channels': [item * ch for item in [1, 4, 8]],
                 'downsample': [True] * 3,
                 'resolution': [16, 8, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}


    arch[20] = {'in_channels': [input_nc] + [ch * item for item in [1, 8, 16]],
                 'out_channels': [item * ch for item in [1, 8, 16, 16]],
                 'downsample': [True] * 3 + [False],
                 'resolution': [16, 8, 4, 4],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 5)}}
    return arch


class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                 D_kernel_size=3, D_attn='64', n_classes=1000,
                 num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                 D_init='ortho', skip_init=False, D_param='SN', gpu_ids=[],bn_linear='embed', input_nc=3, one_hot=False, **kwargs):
        super(Discriminator, self).__init__()
        self.name = 'D'
        # gpu_ids
        self.gpu_ids = gpu_ids
        # one_hot representation
        self.one_hot = one_hot
        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16
        # Architecture
        self.arch = D_arch(self.ch, self.attention, input_nc)[resolution]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
            if bn_linear=='SN':
                self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
            # We use a non-spectral-normed embedding here regardless;
            # For some reason applying SN to G's embedding seems to randomly cripple G
            self.which_embedding = nn.Embedding
        if one_hot:
            self.which_embedding = functools.partial(layers.SNLinear,
                                                         num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                         eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]
        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self = init_weights(self, D_init)

    def forward(self, x, y=None, **kwargs):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        if y is not None:
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out

    def return_features(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        block_output = []
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
                block_output.append(h)
        # Apply global sum pooling as in SN-GAN
        # h = torch.sum(self.activation(h), [2, 3])
        return block_output

class Encoder(Discriminator):
    def __init__(self, opt, output_dim, **kwargs):
        super(Encoder, self).__init__(**vars(opt))
        self.output_layer = nn.Sequential(self.activation,
                                          nn.Conv2d(self.arch['out_channels'][-1], output_dim, kernel_size=(4,2), padding=0, stride=2))

    def forward(self, x):
        # Stick x into h for cleaner for loops without flow control
        h = x
        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        out = self.output_layer(h)
        return out

class BiDiscriminator(nn.Module):
    def __init__(self, opt):
        super(BiDiscriminator, self).__init__()
        self.infer_img = Encoder(opt, output_dim=opt.nimg_features)
        # self.infer_z = nn.Sequential(
        #     nn.Conv2d(opt.dim_z, 512, 1, stride=1, bias=False),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Dropout2d(p=self.dropout),
        #     nn.Conv2d(512, opt.nz_features, 1, stride=1, bias=False),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Dropout2d(p=self.dropout)
        # )
        self.infer_joint = nn.Sequential(
            nn.Conv2d(opt.dim_z+opt.nimg_features, 1024, 1, stride=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

    def forward(self, x, z, **kwargs):
        output_x = self.infer_img(x)
        # output_z = self.infer_z(z)
        if len(z.shape)==2:
            z = z.unsqueeze(2).unsqueeze(2).repeat((1,1,1,output_x.shape[3]))
        output_features = self.infer_joint(torch.cat([output_x, z], dim=1))
        output = self.final(output_features)
        return output

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                split_D=False):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            G_z = self.G(z, self.G.shared(gy))
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        # Split_D means to run D once with real data and once with fake,
        # rather than concatenating along the batch dimension.
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            else:
                if return_G_z:
                    return D_fake, G_z
                else:
                    return D_fake
        # If real data is provided, concatenate it with the Generator's output
        # along the batch dimension for improved efficiency.
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            # Get Discriminator output
            D_out = self.D(D_input, D_class)
            if x is not None:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])  # D_fake, D_real
            else:
                if return_G_z:
                    return D_out, G_z
                else:
                    return D_out

