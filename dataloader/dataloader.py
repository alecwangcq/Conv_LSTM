import numpy as np
import torch
import torch.utils.data as data
from misc.utils import unpack, pack

class MovingMNISTDataset(data.Dataset):

    def __init__(self, file_addr):
        data = np.load(file_addr)
        # of shape 20*n*64*64
        data = 1.0 * (data > 0)
        data = torch.from_numpy(data)
        data = data.unsqueeze(2)
        self.data = data

    def __getitem__(self, index):
        return self.data[:, index, :, :, :]

    def __len__(self):
        return self.data.size(1)

    def collate_fn(self, data):
        data = torch.cat(data, 1)
        input = data[0:10]
        predict = data[10:]
        input = input.transpose(0, 1)
        predict = predict.transpose(0, 1)
        # batch_size, time_steps, num_channels, height, width
        return input.unsqueeze(2).float(), predict.unsqueeze(2).float()


def get_dataloader(configs):
    _KEYS=['file_addr', 'batch_size', 'shuffle', 'num_workers']
    file_addr, batch_size, shuffle, num_workers = unpack(configs, _KEYS)
    dataset = MovingMNISTDataset(file_addr)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers, collate_fn=dataset.collate_fn)
    return dataloader


if __name__ == '__main__':
    file_addr = '/u/cqwang/work/bases/dataset/mmnist_val.npy'
    batch_size = 5
    shuffle = True
    num_workers = 2
    configs = pack([file_addr, batch_size, shuffle, num_workers],
                   ['file_addr', 'batch_size', 'shuffle', 'num_workers'])
    dataloader = get_dataloader(configs)
    for idx, (input, predict) in enumerate(dataloader):
        print (input.size())
        print (predict.size())
        print (predict[0])
        break