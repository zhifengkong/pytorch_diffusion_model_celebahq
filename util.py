import os
import re

import numpy as np
import torch

def calc_t_emb(ts, t_emb_dim):
    """
    Embed time steps into a higher dimension space
    """
    assert t_emb_dim % 2 == 0

    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.cuda()
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)
    
    return t_emb

def flatten(v):
    """
    Flatten a list of lists/tuples
    """
    return [x for y in v for x in y]

def rescale(x):
    """
    Rescale a tensor to 0-1
    """
    return (x - x.min()) / (x.max() - x.min())

def find_max_epoch(path, ckpt_name):
    """
    Find max epoch in path, formatted ($ckpt_name)_$epoch.pkl, such as unet_ckpt_30.pkl
    """
    files =  os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= len(ckpt_name) + 5:
            continue
        if f[:len(ckpt_name)] == ckpt_name and f[-4:]  == '.pkl':
            number = f[len(ckpt_name)+1:-4]
            try:
                epoch = max(epoch, int(number))
            except:
                continue
    return epoch

def print_size(net):
    """
    Print the number of parameters of a network
    """
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)

def std_normal(size):
    """
    Generate a standard Gaussian of a given size
    """
    return torch.normal(0, 1, size=size).cuda()

def sampling(net, size, T, Alpha, Alpha_bar, Sigma):
    """
    Perform the complete sampling step according to p(x_0|x_T)
    """
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 4
    print('begin sampling, total steps = %s' % T)

    x = std_normal(size)
    with torch.no_grad():
        for t in range(T-1,-1,-1):
            if t % 100 == 0:
                print('reverse step:', t)
            ts = (t * torch.ones((size[0], 1))).cuda()
            epsilon_theta = net((x,ts,))
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)
    return x

def training_loss(net, loss_fn, T, X, Alpha_bar):
    """
    Compute the loss_fn (default is \ell_2) loss of (epsilon - epsilon_theta)
    """
    B, C, H, W = X.shape
    ts = torch.randint(T, size=(B,1,1,1)).cuda()
    z = std_normal(X.shape)
    xt = torch.sqrt(Alpha_bar[ts]) * X + torch.sqrt(1-Alpha_bar[ts]) * z
    epsilon_theta = net((xt, ts.view(B,1),))
    return loss_fn(epsilon_theta, z)

    