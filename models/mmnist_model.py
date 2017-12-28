import torch
import torch.nn as nn
from models.conv_lstm import ConvLSTMCell, ConvLSTM
from misc.utils import pack, unpack, print_dict, to_var

class MMNIST_ConvLSTM(nn.Module):

    def __init__(self, configs):
        super(MMNIST_ConvLSTM, self).__init__()
        _KEYS = ['encoder_configs', 'reconstruct_configs', 'predict_configs']
        en_conf, rec_conf, pred_conf = unpack(configs, _KEYS)
        self.encoder = ConvLSTM(en_conf)
        self.reconstructor = Generator(rec_conf)
        self.predictor = Generator(pred_conf)

    def forward(self, data, configs=None):
        x_train, x_predict, states = unpack(data, ['x_train', 'x_predict', 'states'])
        use_gt, max_steps = unpack(configs, ['use_gt', 'max_steps'])
        # x: batch_size * time_steps * channels * height * width
        # states: batch_size * channels' * height * width
        batch_size = x_train.size(0)
        time_steps = x_train.size(1)

        encoder = self.encoder
        reconstructor = self.reconstructor
        predictor = self.predictor

        # encoding stages
        en_data = pack([x_train, encoder.init_hidden(batch_size)], ['x', 'states'])
        states = encoder(en_data)

        # reconstruct
        r_res = []
        r_x = None
        r_states = states
        for t in xrange(time_steps):
            r_data = pack([r_x, r_states], ['x', 'states'])
            r_out, r_states = reconstructor(r_data)
            r_res.append(r_out)
            r_x = x_train[:, t].unsqueeze(1) if use_gt else r_out.unsqueeze(1)

        # predict
        time_steps = x_predict.size(1) if use_gt else max_steps
        p_res = []
        p_x = None
        p_states = states
        # print ('start from p')
        for t in xrange(time_steps):
            p_data = pack([p_x, p_states], ['x', 'states'])
            p_out, p_states = predictor(p_data)
            p_res.append(p_out)
            p_x = x_predict[:, t].unsqueeze(1) if use_gt else p_out.unsqueeze(1)

        return torch.cat(r_res, 1), torch.cat(p_res, 1)

    @staticmethod
    def get_init_keys():
        _KEYS = ['encoder_configs', 'reconstruct_configs', 'predict_configs']
        return _KEYS


class Generator(ConvLSTM):

    def __init__(self, configs):
        super(Generator, self).__init__(configs)
        num_layers, h_c = configs['num_layers'], configs['cell_configs'][0]['h_c']
        h_c_sum = 0
        for conf in configs['cell_configs']:
            h_c_sum += conf['h_c']
        in_channels = h_c_sum
        out_channels = 1
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1)

    def forward(self, data, configs=None):
        states = super(Generator, self).forward(data)
        # states: list, len(states) == num_layers
        # states[i] = (h, c)
        # h: batch_size * channels * height * width
        hs = [states[i][0] for i in xrange(self.num_layers)]
        hs = torch.cat(hs, 1)
        output = self.conv(hs)
        # batch_size * 1 * height * width
        return output, states



if __name__ == '__main__':
    _KEYS = ['encoder_configs', 'reconstruct_configs', 'predict_configs']
    h_c = 2
    active_func = nn.Tanh()
    in_c = 1
    in_h = 4
    in_w = 4
    kernel_size = 3
    time_steps = 5
    DEBUG = True
    batch_size = 3

    num_layers = 3
    cell_conf_l0 = pack([h_c, active_func, in_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_conf_l1 = pack([h_c, active_func, h_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_conf_l2 = pack([h_c, active_func, h_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_configs = [cell_conf_l0, cell_conf_l1, cell_conf_l2]

    encoder_configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())
    reconstruct_configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())
    predict_configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())

    model_configs = pack([encoder_configs, reconstruct_configs, predict_configs], _KEYS)

    model = MMNIST_ConvLSTM(model_configs)

    x_train = to_var(torch.randn(batch_size, time_steps, in_c, in_h, in_w))
    x_predict = to_var(torch.randn(batch_size, time_steps, in_c, in_h, in_w))

    data = pack([x_train, x_predict, None], ['x_train', 'x_predict', 'states'])
    configs = pack([True, 6], ['use_gt', 'max_steps'])

    x, y = model(data, configs)
    print (x.size())
    print (y.size())