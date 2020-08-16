import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import calc_t_emb, flatten

def swish(x):
    return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, n_channels, with_conv=True):
        super(Upsample, self).__init__()
        self.with_conv = with_conv
        self.n_channels = n_channels

        self.conv = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=1, padding=1)

    def forward(self, x):
        up = nn.Upsample(scale_factor=2, mode='nearest')
        B, C, H, W = x.shape
        assert C == self.n_channels
        x = up(x)
        assert x.shape == (B, C, 2*H, 2*W)
        if self.with_conv:
            x = self.conv(x)
            assert x.shape == (B, C, 2*H, 2*W)
        return x

class Downsample(nn.Module):
    def __init__(self, n_channels, with_conv=True):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        self.n_channels = n_channels

        self.conv = nn.Conv2d(self.n_channels, self.n_channels, 3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.n_channels
        if self.with_conv:
            x = self.conv(x)
        else:
            down = nn.AvgPool2d(2)
            x = down(x)
        assert x.shape == (B, C, H // 2, W // 2)
        return x

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super(residual_block, self).__init__()
        self.t_emb_dim = t_emb_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.fc_t = nn.Linear(self.t_emb_dim, self.out_channels)

        if self.in_channels != self.out_channels:
            self.conv_x = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)

        self.gn_in = nn.GroupNorm(32, self.in_channels)
        self.gn_out = nn.GroupNorm(32, self.out_channels)

    def forward(self, data_in):
        x, t_emb = data_in

        B, C, H, W = x.shape
        assert C == self.in_channels
        h = x

        # 1st conv
        h = swish(self.gn_in(h))
        h = self.conv1(h)

        # add timestep embedding (broadcast across pixels)
        part_t = self.fc_t(swish(t_emb))
        part_t = part_t.view([B, self.out_channels, 1, 1])
        h = h + part_t

        # 2nd conv
        h = swish(self.gn_out(h))
        h = self.conv2(h)

        # modify input n_channels
        if self.in_channels != self.out_channels:
            x = self.conv_x(x)

        assert x.shape == h.shape
        return x + h

class self_attn(nn.Module):
    def __init__(self, in_dim):
        super(self_attn, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma * out + x
        return out

class UNet(nn.Module):
    def __init__(self, n_channels=3, t_emb_dim=128, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.t_emb_dim = t_emb_dim

        # unet channels at each layer
        self.ch_mult = (1,1,2,2,4,4)
        u_channels = [128]
        for i in range(len(self.ch_mult)):
            n = 128 * self.ch_mult[i]
            for _ in range(3):
                u_channels.append(n)

        # timestep embedding fc layers
        self.fc_t1 = nn.Linear(self.t_emb_dim, 4*self.t_emb_dim)
        self.fc_t2 = nn.Linear(4*self.t_emb_dim, 4*self.t_emb_dim)

        # in conv
        self.inc = nn.Conv2d(self.n_channels, 128, 3, padding=1)

        # down sampling (res + res + (attn when resolution=16) + down) * 6
        # no down layer at last
        self.down = [[] for _ in self.ch_mult]
        ind = 0
        for i in range(len(self.ch_mult)):
            for j in range(2):
                self.down[i].append(residual_block(u_channels[ind], u_channels[ind+1], 4*self.t_emb_dim))
                ind += 1
                if i == 4:
                    self.down[i].append(self_attn(u_channels[ind]))
            if i != len(self.ch_mult) - 1:
                self.down[i].append(Downsample(u_channels[ind]))
                ind += 1
        self.down = nn.ModuleList(flatten(self.down))
        
        # middle layers
        self.res_mid_1 = residual_block(u_channels[-1], u_channels[-1], 4*self.t_emb_dim)
        self.self_attn_mid = self_attn(u_channels[-1])
        self.res_mid_2 = residual_block(u_channels[-1], u_channels[-1], 4*self.t_emb_dim)
        
        # up sampling (res + res + res + (attn when resolution=16) + up) * 6
        # no up layer at last
        self.up = [[] for _ in self.ch_mult]
        ind += 1
        for i in range(len(self.ch_mult))[::-1]:
            for j in range(3):
                self.up[i].append(residual_block(u_channels[ind]+u_channels[ind-1], u_channels[ind-1], 4*self.t_emb_dim))
                ind -= 1
                if i == 4:
                    self.up[i].append(self_attn(u_channels[ind]))
            if i != 0:
                self.up[i].append(Upsample(u_channels[ind]))
            self.up[i] = self.up[i][::-1]
        self.up = nn.ModuleList(flatten(self.up))
        assert ind == 0

        # out conv
        self.outc = nn.Conv2d(128, self.n_channels, kernel_size=1)
        

    def forward(self, data_in):
        x, ts = data_in
        B, C, H, W = x.shape
        assert C == self.n_channels

        # timestep embedding
        t_emb = calc_t_emb(ts, self.t_emb_dim)
        t_emb = self.fc_t1(t_emb)
        t_emb = swish(t_emb)
        t_emb = self.fc_t2(t_emb)

        x0 = self.inc(x)

        # downsampling
        hs = [x0]
        ind = 0
        for i in range(len(self.ch_mult)):
            for j in range(2):
                h = self.down[ind]((hs[-1], t_emb))  # j-th res layer
                ind += 1
                if i == 4:
                    h = self.down[ind](h)  # self attention layer
                    ind += 1
                hs.append(h)
            if i != len(self.ch_mult) - 1:
                h = self.down[ind](hs[-1])  # down sampling layer
                hs.append(h)
                ind += 1

        # middle layers
        h = self.res_mid_1((h, t_emb))
        h = self.self_attn_mid(h)
        h = self.res_mid_2((h, t_emb))

        # upsampling
        ind = -1
        for i in range(len(self.ch_mult)):
            for j in range(3):
                h = torch.cat([h, hs.pop()], dim=1)
                h = self.up[ind]((h, t_emb))  # j-th res layer
                ind -= 1
                if i == 1:
                    h = self.up[ind](h)  # self attention layer
                    ind -= 1
            if i != len(self.ch_mult)-1:
                h = self.up[ind](h)  # up sampling layer
                ind -= 1
        assert not hs

        # final conv
        h = self.outc(swish(h))
        assert h.shape == x.shape
        return h
