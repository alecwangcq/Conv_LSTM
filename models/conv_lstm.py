from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from misc.utils import pack, unpack, print_dict, to_var



class ConvLSTMCell(nn.Module):

    def __init__(self, configs):
        super(ConvLSTMCell, self).__init__()
        _KEYS = ConvLSTMCell.get_init_keys()
        # ['num_layers', 'h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG']
        self.h_c, self.active_func, self.in_c, self.in_h, \
            self.in_w, self.kernel_size, DEBUG = unpack(configs, _KEYS)

        if DEBUG:
            print_dict(configs, _KEYS, 'ConvLSTMCell.__init__')

        self.conv2d = nn.Conv2d(in_channels=self.in_c + self.h_c,
                                out_channels=4 * self.h_c,
                                kernel_size=self.kernel_size,
                                padding=(self.kernel_size-1)//2)

        self.w_ci = nn.Parameter(torch.zeros(self.h_c, self.in_h, self.in_w))
        self.w_cf = nn.Parameter(torch.zeros(self.h_c, self.in_h, self.in_w))
        self.w_co = nn.Parameter(torch.zeros(self.h_c, self.in_h, self.in_w))
        self.init_weights()

    def init_weights(self):
        scale = 0.1
        self.w_ci.data.uniform_(-scale, scale)
        self.w_cf.data.uniform_(-scale, scale)
        self.w_co.data.uniform_(-scale, scale)

    @staticmethod
    def get_init_keys():
        return ['h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG']

    def forward(self, data, configs=None):
        # x: batch_size * in_c * in_w * in_h
        # states: h, c
        # h: batch_size * h_c * in_w * in_h
        # c: the same shape as h
        _KEYS = ['x', 'states']
        x, states = unpack(data, _KEYS)
        h, c = states
        active_func = self.active_func
        concat_hx = torch.cat([x, h], 1)
        # print(concat_hx.size())
        conv_hx = self.conv2d(concat_hx)
        ai, af, ac, ao = torch.split(conv_hx, self.h_c, dim=1)
        # print (ai.size(), af.size(), ac.size(), ao.size())
        # print (c.size(), self.w_ci.size())
        ic, fc, oc = self.w_ci*c, self.w_cf*c, self.w_co*c
        next_i = nn.Sigmoid()(ai + ic)
        next_f = nn.Sigmoid()(af + fc)
        next_c = next_f*c + next_i*active_func(ac)
        next_o = nn.Sigmoid()(ao + oc)
        next_h = next_o*active_func(next_c)

        return next_h, next_c

    def init_hidden(self, batch_size, cuda=False):
        return (to_var((torch.zeros(batch_size, self.h_c, self.in_h, self.in_w))),
                to_var(torch.zeros(batch_size, self.h_c, self.in_h, self.in_w)))


class ConvLSTM(nn.Module):

    def __init__(self, configs):
        super(ConvLSTM, self).__init__()
        # 'h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG'
        _KEYS= ['num_layers', 'cell_configs']
        num_layers, cell_configs = unpack(configs, _KEYS)
        cells = [ConvLSTMCell(cell_configs[idx]) for idx in xrange(num_layers)]
        self.cell_list = nn.ModuleList(cells)
        self.num_layers = num_layers
        self.cell_config = cell_configs[0]

    @staticmethod
    def get_init_keys():
        _KEYS = ['num_layers', 'cell_configs']
        return _KEYS

    def forward(self, data, configs=None):
        _KEYS = ['x', 'states']
        x, states = unpack(data, _KEYS)
        batch_size = len(states)
        if x is None:
            x_c, x_h, x_w = self.cell_config['in_c'], self.cell_config['in_w'], self.cell_config['in_h']
            x = to_var(torch.zeros(batch_size, 1, x_c, x_h, x_w))
        # x: batch_size, time_steps, channels, height, width
        time_steps = x.size(1)
        next_states = []
        cell_list = self.cell_list
        current_input = [x[:, t] for t in xrange(time_steps)]
        for l in xrange(self.num_layers):
            h0, c0 = states[l]
            for t in xrange(time_steps):
                data = pack([current_input[t],(h0, c0)], ['x', 'states'])
                print (data)
                h, c = cell_list[l](data)
                next_states.append((h, c))
                current_input[t] = h
                states[l] = (h, c)

        return states

    def init_hidden(self, batch_size, cuda=False):
        init_states = []  # this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, cuda))
        return init_states


class ConvRNNCell(nn.Module):
    def __init__(self, configs):
        super(ConvRNNCell, self).__init__()
        _KEYS = ConvRNNCell.get_init_keys()
        # ['num_layers', 'h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG']
        self.h_c, self.active_func, self.in_c, self.in_h, \
        self.in_w, self.kernel_size, DEBUG = unpack(configs, _KEYS)

        if DEBUG:
            print_dict(configs, _KEYS, 'ConvRNNCell.__init__')

        self.conv2d = nn.Conv2d(in_channels=self.in_c + self.h_c,
                                out_channels= self.h_c,
                                kernel_size=self.kernel_size,
                                padding=(self.kernel_size - 1) // 2)


    @staticmethod
    def get_init_keys():
        return ['h_c', 'active_func', 'in_c', 'in_h', 'in_w', 'kernel_size', 'DEBUG']

    def forward(self, data, configs=None):
        # x: batch_size * in_c * in_w * in_h
        # states: h, c
        # h: batch_size * h_c * in_w * in_h
        # c: the same shape as h
        _KEYS = ['x', 'states']
        x, states = unpack(data, _KEYS)
        h = states
        active_func = self.active_func

        concat_hx = torch.cat([x, h], 1)
        conv_hx = self.conv2d(concat_hx)
        next_h = active_func(conv_hx)

        return next_h

    def init_hidden(self, batch_size, cuda=False):
        return to_var(torch.zeros(batch_size, self.h_c, self.in_h, self.in_w))


def test_lstm_cell():
    h_c = 16
    active_func = nn.Tanh()
    in_c = 1
    in_h = 32
    in_w = 32
    kernel_size = 3
    DEBUG = True

    configs = pack([h_c, active_func, in_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    model = ConvLSTMCell(configs)

    batch_size = 4
    input = Variable(torch.randn(batch_size, in_c, in_h, in_w))
    data = pack([input, model.init_hidden(batch_size, False)], ['x', 'states'])
    # print (data)
    out = model(data)
    print(out)


def test_lstm():

    h_c = 2
    active_func = nn.Tanh()
    in_c = 1
    in_h = 4
    in_w = 4
    kernel_size = 3
    time_steps = 5
    DEBUG = True

    num_layers = 3
    cell_conf_l0 = pack([h_c, active_func, in_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_conf_l1 = pack([h_c, active_func, h_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_conf_l2 = pack([h_c, active_func, h_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_configs = [cell_conf_l0, cell_conf_l1, cell_conf_l2]

    configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())
    model = ConvLSTM(configs)

    batch_size = 3
    input = Variable(torch.randn(batch_size, time_steps, in_c, in_h, in_w))
    data = pack([input, model.init_hidden(batch_size, False)], ['x', 'states'])
    # print (data)
    out = model(data)
    print(out)

if __name__ == '__main__':
    test_lstm_cell()
    test_lstm()




