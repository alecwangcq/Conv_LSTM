from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn


def unpack(dic, keys):
    # if there exists pair like <key, None> in dic, this function
    # won't work.
    vals = [dic.get(key) for key in keys]
    vals.remove(None)
    return vals


def pack(vals, keys):
    dic = dict()
    for idx, key in enumerate(keys):
        dic[key] = vals[idx]
    return dic


def print_dict(dic, keys, prefix=''):
    for key in keys:
        print (prefix + ': ', key, dic[key])


class ConvLSTMCell(nn.Module):

    def __init__(self, configs):
        _KEYS = ConvLSTMCell.get_init_keys()
        # ['num_layers', 'h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG']
        self.num_layers, self.h_c, self.active_func, self.in_c, self.in_h, \
            self.in_w, self.kernel_size, DEBUG = unpack(configs, _KEYS)

        if DEBUG:
            print_dict(configs, _KEYS, 'ConvLSTMCell.__init__')

        self.conv2d = nn.Conv2d(in_channels=self.in_c + self.h_c,
                                out_channels=4 * self.h_c,
                                kernel_size=self.kernel_size,
                                padding=(self.kernel_size-1)//2)

        self.wci = nn.Parameter(torch.zeros(self.in_c, self.in_w, self.in_h))
        self.bi = nn.Parameter(torch.zeros())

    @staticmethod
    def get_init_keys():
        return ['num_layers', 'h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG']

    def forward(self, data, configs=None):
        # x: batch_size * in_c * in_w * in_h
        # states: h, c
        # h: batch_size * h_c * in_w * in_h
        # c: the same shape as h
        _KEYS = ['x', 'states']
        x, states = unpack(data, _KEYS)
        h, c = states

        concat_hx = torch.cat([x, h], 1)
        conv_hx = self.conv2d(concat_hx)
        ai, af, ao, ag = torch.split(conv_hx, self.h_c, dim=1)










