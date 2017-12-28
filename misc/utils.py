import torch
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