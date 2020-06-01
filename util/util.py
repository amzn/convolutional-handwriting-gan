"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from torch.autograd import Variable

def random_word(len_word, alphabet):
    # generate a word constructed from len_word characters where each character is randomly chosen from the alphabet.
    char = np.random.randint(low=0, high=len(alphabet), size=len_word)
    word = [alphabet[c] for c in char]
    return ''.join(word)

def load_network(net, save_dir, epoch):
    """Load all the networks from the disk.

    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    load_filename = '%s_net_%s.pth' % (epoch, net.name)
    load_path = os.path.join(save_dir, load_filename)
    # if you are using PyTorch newer than 0.4 (e.g., built from
    # GitHub source), you can remove str() on self.device
    state_dict = torch.load(load_path)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    net.load_state_dict(state_dict)
    return net

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)

def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)

def multiple_replace(string, rep_dict):
    for key in rep_dict.keys():
        string = string.replace(key, rep_dict[key])
    return string

def get_curr_data(data, batch_size, counter):
    curr_data = {}
    for key in data:
        curr_data[key] = data[key][batch_size*counter:batch_size*(counter+1)]
    return curr_data

# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)

# turn tensor of classes to tensor of one hot tensors:
def make_one_hot(labels, len_labels, n_classes):
    one_hot = torch.zeros((labels.shape[0], labels.shape[1], n_classes),dtype=torch.float32)
    for i in range(len(labels)):
        one_hot[i,np.array(range(len_labels[i])), labels[i,:len_labels[i]]-1]=1
    return one_hot

# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, len_text_fake, len_text, mask_loss):
    mask_real = torch.ones(dis_real.shape).to(dis_real.device)
    mask_fake = torch.ones(dis_fake.shape).to(dis_fake.device)
    if mask_loss and len(dis_fake.shape)>2:
        for i in range(len(len_text)):
            mask_real[i, :, :, len_text[i]:] = 0
            mask_fake[i, :, :, len_text_fake[i]:] = 0
    loss_real = torch.sum(F.relu(1. - dis_real * mask_real))/torch.sum(mask_real)
    loss_fake = torch.sum(F.relu(1. + dis_fake * mask_fake))/torch.sum(mask_fake)
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake, len_text_fake, mask_loss):
    mask_fake = torch.ones(dis_fake.shape).to(dis_fake.device)
    if mask_loss and len(dis_fake.shape)>2:
        for i in range(len(len_text_fake)):
            mask_fake[i, :, :, len_text_fake[i]:] = 0
    loss = -torch.sum(dis_fake*mask_fake)/torch.sum(mask_fake)
    return loss

def loss_std(z, lengths, mask_loss):
    loss_std = torch.zeros(1).to(z.device)
    z_mean = torch.ones((z.shape[0], z.shape[1])).to(z.device)
    for i in range(len(lengths)):
        if mask_loss:
            if lengths[i]>1:
                loss_std += torch.mean(torch.std(z[i, :, :, :lengths[i]], 2))
                z_mean[i,:] = torch.mean(z[i, :, :, :lengths[i]], 2).squeeze(1)
            else:
                z_mean[i, :] = z[i, :, :, 0].squeeze(1)
        else:
            loss_std += torch.mean(torch.std(z[i, :, :, :], 2))
            z_mean[i,:] = torch.mean(z[i, :, :, :], 2).squeeze(1)
    loss_std = loss_std/z.shape[0]
    return loss_std, z_mean

# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t())
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes & not in blacklist
      if len(param.shape) < 2 or param in blacklist:
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t())
               - torch.eye(w.shape[0], device=w.device), w))
      param.grad.data += strength * grad.view(param.shape)


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        seed_rng(kwargs['seed'])
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif self.dist_type == 'poisson':
            self.lam = kwargs['var']
        elif self.dist_type == 'gamma':
            self.scale = kwargs['var']


    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
        elif self.dist_type == 'poisson':
            type = self.type()
            device = self.device
            data = np.random.poisson(self.lam, self.size())
            self.data = torch.from_numpy(data).type(type).to(device)
        elif self.dist_type == 'gamma':
            type = self.type()
            device = self.device
            data = np.random.gamma(shape=1, scale=self.scale, size=self.size())
            self.data = torch.from_numpy(data).type(type).to(device)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


def to_device(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        if len(gpu_ids)>1:
            net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
            # net = torch.nn.DistributedDataParallel(net)
    return net


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda',
                fp16=False, z_var=1.0, z_dist='normal', seed=0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution(z_dist, mean=0, var=z_var, seed=seed)
    z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses, seed=seed)
    y_ = y_.to(device, torch.int64)
    return z_, y_


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
