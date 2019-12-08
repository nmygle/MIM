import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mim import ConvLSTM, MIMBlock, MIMS


class MIM(nn.Module):
    def __init__(self, h=128, w=128, num_hidden=[32,32]):
        super(MIM, self).__init__()

        in_ch = 1
        out_ch = 1
        self.num_layers = len(num_hidden)
        self.total_length = 48
        self.input_length = 24

        # stationarity
        self.stlstm_layer = nn.ModuleList([])
        for i in range(self.num_layers):
            if i < 1:
                self.stlstm_layer.append(ConvLSTM(in_ch, num_hidden[i]))
            else:
                self.stlstm_layer.append(MIMBlock(num_hidden[i-1], num_hidden[i], h, w))

        # non-stationarity
        self.stlstm_layer_diff = nn.ModuleList([])
        for i in range(self.num_layers - 1):
            self.stlstm_layer_diff.append(MIMS(num_hidden[i+1], h, w))

        self.top = nn.Conv2d(num_hidden[-1], out_ch, 1, 1, 0)


    def set_shape(self, shape, device):
        for i in range(self.num_layers):
            self.stlstm_layer[i].set_shape(shape, device)

        for i in range(self.num_layers - 1):
            self.stlstm_layer_diff[i].set_shape(shape, device)


    def forward(self, images):
        self.set_shape(images.shape, images.device)

        st_memory = None
        cell_state = [None] * self.num_layers
        hidden_state = [None] * self.num_layers
        cell_state_diff = [None] * (self.num_layers - 1)
        hidden_state_diff = [None] * (self.num_layers - 1)

        gen_images = []
        for time_step in range(self.total_length - 1):
            print("time:", time_step)
            # input
            if time_step < self.input_length:
                x_gen = images[:, [time_step]]
            else:
                gamma = 1.0
                x_gen = gamma * images[:, [time_step]] + (1-gamma) * x_gen
            preh = hidden_state[0]

            # 1st layer(convlstm)
            hidden_state[0], cell_state[0], st_memory = self.stlstm_layer[0](
                    x_gen, hidden_state[0], cell_state[0], st_memory)
            # higher layer(mim)
            for i in range(1, self.num_layers):
                print("  layer:", i)
                if time_step == 0:
                    _, _ = self.stlstm_layer_diff[i-1](torch.zeros_like(hidden_state[i-1]), None, None)
                else:
                    diff = hidden_state[i-1] - preh if i == 1 else hidden_state[i-2]
                    hidden_state_diff[i-1], cell_state_diff[i-1] = self.stlstm_layer_diff[i-1](
                        diff, hidden_state_diff[i-1], cell_state_diff[i-1])
                preh = hidden_state[i]

                hidden_state[i], cell_state[i], st_memory = self.stlstm_layer[i](
                    hidden_state[i-1], hidden_state_diff[i-1], hidden_state[i], cell_state[i], st_memory)
            x_gen = self.top(hidden_state[-1])
            gen_images.append(x_gen)
        return torch.stack(gen_images)

if __name__ == "__main__":
    w = 128
    num_hidden = [64, 64]
    mim = MIM(w, w, num_hidden).cuda()
    images = torch.zeros(1, 48, w, w).cuda()
    y = mim(images)
    print(y.shape)

