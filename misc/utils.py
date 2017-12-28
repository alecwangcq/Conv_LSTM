import torch
import torch.optim as optim
from torch.autograd import Variable

def unpack(dic, keys):
    vals = [dic.get(key) for key in keys]
    return vals


def pack(vals, keys):
    dic = dict()
    for idx, key in enumerate(keys):
        dic[key] = vals[idx]
    return dic


def print_dict(dic, keys, prefix=''):
    for key in keys:
        print (prefix + ': ', key, '-> ', dic[key])


def to_var(x, requires_grad=False):
    x = Variable(x, requires_grad=requires_grad)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_optimizer(configs, parameters):
    _KEY = ['lr', 'optim_alg']
    OPTIM = {'SGD': optim.SGD, 'Adam': optim.Adam}
    optim_alg = configs['optim_alg']
    lr = configs['lr']
    return OPTIM[optim_alg](parameters, lr)

def adjust_learning_rate(optimizer, epoch, halveEvery=10):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    if (epoch + 1)%halveEvery ==0:
        print 'Adjust learning rate to: ', lr/2.0
        lr = lr/2.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr