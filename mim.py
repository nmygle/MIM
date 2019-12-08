import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvLSTM(nn.Module):
    def __init__(self, in_ch, h_ch, kernel_size=3, bias=True):
        super(ConvLSTM, self).__init__()

        self.in_ch = in_ch
        self.h_ch = h_ch
        self._forget_bias = 1.0
        pad = int((kernel_size - 1) / 2)
        self.s_cc = nn.Conv2d(h_ch, 4 * h_ch,
                        kernel_size=kernel_size, stride=1, padding=pad, bias=bias)
        self.t_cc = nn.Conv2d(h_ch, 4 * h_ch,
                        kernel_size=kernel_size, stride=1, padding=pad, bias=bias)
        self.x_cc = nn.Conv2d(in_ch, 4 * h_ch,
                        kernel_size=kernel_size, stride=1, padding=pad, bias=bias)
        self.last = nn.Conv2d(2 * h_ch, h_ch, 1, 1, 0)

    def set_shape(self, shape, device):
        self.shape = shape
        self.device = device

    def init_state(self):
        return Variable(torch.zeros((self.shape[0], self.h_ch, self.shape[2], self.shape[3]))).to(self.device)

    def forward(self, x, h, c, m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()

        i_s, g_s, f_s, o_s = torch.split(self.s_cc(h), self.h_ch, dim=1)
        i_t, g_t, f_t, o_t = torch.split(self.t_cc(h), self.h_ch, dim=1)
        i_x, g_x, f_x, o_x = torch.split(self.s_cc(h), self.h_ch, dim=1)

        i  = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g  = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f  = torch.sigmoid(f_x + f_t + self._forget_bias)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o  = torch.sigmoid(o_x + o_t + o_s)

        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g

        cell = torch.cat([new_c, new_m], 1)
        cell = self.last(cell)

        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m


class MIMS(nn.Module):
    def __init__(self, h_ch, h=128, w=128, k_size=3, tln=False):
        super(MIMS, self).__init__()
        pad = (k_size - 1) // 2
        if tln:
            self.conv_h = nn.Squential(
                                nn.Conv2d(h_ch, h_ch * 4, k_size, 1, pad),
                                nn.LayerNorm(h_ch * 4, h, w))
            self.conv_x = nn.Squential(
                                nn.Conv2d(h_ch, h_ch * 4, k_size, 1, pad),
                                nn.LayerNorm(h_ch * 4, h, w))
        else:
            self.conv_h = nn.Conv2d(h_ch, h_ch * 4, k_size, 1, pad)
            self.conv_x = nn.Conv2d(h_ch, h_ch * 4, k_size, 1, pad)

        self.h_ch = h_ch
        self._forget_bias = 1.0
        self.ct_weight = nn.init.normal_(nn.Parameter(
                                torch.Tensor(torch.zeros(h_ch*2, h, w))), 0.0, 1.0)
        self.oc_weight = nn.init.normal_(nn.Parameter(
                                torch.Tensor(torch.zeros(h_ch, h, w))), 0.0, 1.0)

    def set_shape(self, shape, device):
        self.shape = shape
        self.device = device

    def init_state(self):
        return Variable(torch.zeros((self.shape[0], self.h_ch, self.shape[2], self.shape[3]))).to(self.device)

    def forward(self, x, h_t, c_t):
        if h_t is None:
            h_t = self.init_state()
        if c_t is None:
            c_t = self.init_state()

        i_h, g_h, f_h, o_h = torch.split(self.conv_h(h_t), self.h_ch, dim=1)

        ct_activation = torch.matmul(c_t.repeat([1,2,1,1]), self.ct_weight)
        i_c, f_c = torch.split(ct_activation, self.h_ch, dim=1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            i_x, g_x, f_x, o_x = torch.split(self.conv_x(x), self.h_ch, dim=1)
            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        o_c = torch.matmul(c_new, self.oc_weight)
        
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new


class MIMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h=128, w=128, k_size=3, bias=True):
        super(MIMBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convlstm_c = None
        self._forget_bias = 1.0

        pad = int((k_size - 1) / 2)
        self.t_cc = nn.Conv2d(out_ch, 3 * out_ch,
                        kernel_size=k_size, stride=1, padding=pad, bias=bias)
        self.s_cc = nn.Conv2d(out_ch, 4 * out_ch,
                        kernel_size=k_size, stride=1, padding=pad, bias=bias)
        self.x_cc = nn.Conv2d(in_ch, 4 * out_ch,
                        kernel_size=k_size, stride=1, padding=pad, bias=bias)
        self.mims = MIMS(out_ch, h, w)
        self.last = nn.Conv2d(2 * out_ch, out_ch, 1, 1, 0)

    def set_shape(self, shape, device):
        self.shape = shape
        self.device = device
        self.mims.set_shape(shape, device)

    def init_state(self):
        return Variable(torch.zeros((self.shape[0], self.out_ch, self.shape[2], self.shape[3]))).to(self.device)

    def forward(self, x, diff_h, h, c, m):
        if h is None:
            h = self.init_state()
        if c is None:
            c = self.init_state()
        if m is None:
            m = self.init_state()
        if diff_h is None:
            diff_h = torch.zeros_like(h)

        i_t, g_t, o_t = torch.split(self.t_cc(h), self.out_ch, dim=1)
        i_s, g_s, f_s, o_s = torch.split(self.s_cc(m), self.out_ch, dim=1)
        i_x, g_x, f_x, o_x = torch.split(self.x_cc(x), self.out_ch, dim=1)

        i  = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g  = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o  = torch.sigmoid(o_x + o_t + o_s)

        new_m = f_ * m + i_ * g_
        c, self.convlstm_c = self.mims(diff_h, c, self.convlstm_c)
        new_c = c + i * g
        cell = self.last(torch.cat([new_c, new_m], 1))
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m


if __name__ == "__main__":
    x = torch.zeros(1, 1, 128, 128).cuda()
    #model = ConvLSTM(1, 32).cuda()
    #model = MIMN(1).cuda()
    model = MIMBlock(1, 32).cuda()
    y, c, m = model(x, None, None, None, None)
    print(y.shape, c.shape, m.shape)

