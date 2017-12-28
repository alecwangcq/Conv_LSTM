from .hparams import HParams
from .register import register
import pickle
import torch.nn as nn
from models.conv_lstm import ConvLSTMCell, ConvLSTM
from misc.utils import pack


@register("MMNIST_CONV_LSTM")
def MMNIST_CONV_LSTM(extra_info):
    _KEYS = ['encoder_configs', 'reconstruct_configs', 'predict_configs']
    h_c = 16
    active_func = nn.Tanh()
    in_c = 1
    in_h = 64
    in_w = 64
    kernel_size = 5
    DEBUG = True

    num_layers = 3
    cell_conf_l0 = pack([h_c, active_func, in_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_conf_l1 = pack([h_c, active_func, h_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_conf_l2 = pack([h_c, active_func, h_c, in_h, in_w, kernel_size, DEBUG], ConvLSTMCell.get_init_keys())
    cell_configs = [cell_conf_l0, cell_conf_l1, cell_conf_l2]

    encoder_configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())
    reconstruct_configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())
    predict_configs = pack([num_layers, cell_configs], ConvLSTM.get_init_keys())

    model_info = pack([encoder_configs, reconstruct_configs, predict_configs], _KEYS)
    model_info['name'] = 'MMNIST_CONV_LSTM'

    trainloader_info={
        'file_addr':'./data/mmnist_train.npy',
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 2
    }

    valloader_info={
        'file_addr':'./data/mmnist_val.npy',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 2
    }

    testloader_info = {
        'file_addr': './data/mmnist_test.npy',
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 2
    }
    seed = 666
    folder_name = 'mmnist_convLSTM'
    main_info={
        'num_epochs': 60,
        'halve_every': 10,
        'log_dir': './logs/%s'%folder_name,
        'save_dir': './checkpoints/%s'%folder_name
    }

    optimizer_info = {
        'lr': 1e-4,
        'optim_alg': 'Adam'
    }

    hparams = HParams(trainloader_info=trainloader_info,
                      valloader_info=valloader_info,
                      testloader_info=testloader_info,
                      model_info=model_info,
                      optimizer_info=optimizer_info,
                      main_info=main_info,
                      seed=seed)
    return hparams


