import torch
import argparse

from models.mmnist_model import MMNIST_ConvLSTM
from dataloader.dataloader import get_dataloader
from hparams.register import *
from misc.logger import create_logger
from misc.saver import Saver
from misc.criterions import cross_entropy
from misc.utils import get_optimizer, adjust_learning_rate, pack, to_var

logger = None
saver = None
evaluator = None


def train(dataloader, model, optimizer, criterion, train_info):
    model.train()
    num_epochs = train_info['num_epochs']
    epoch = train_info['epoch']
    clip = train_info['clip']
    total_steps = len(dataloader)
    for idx, (x_train, x_predict) in enumerate(dataloader):
        optimizer.zero_grad()
        x_train = to_var(x_train)
        x_predict = to_var(x_predict)
        data = pack([x_train, x_predict, None], ['x_train', 'x_predict', 'states'])
        configs = pack([False, 10], ['use_gt', 'max_steps'])
        reconstruct, predict = model(data, configs)
        r_loss = criterion(reconstruct, x_train)
        p_loss = criterion(predict, x_predict)
        loss = r_loss + p_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        logger.info('Epoch [%d/%d], Step [%d/%d], Reconstruct Loss: %5.4f, Predict Loss: %5.4f, Total: %5.4f' % (
            epoch, num_epochs, idx + 1, total_steps, r_loss.data[0], p_loss.data[0], loss.data[0]))

def val(dataloader, model, val_info, criterion):
    model.eval()
    num_epochs = val_info['num_epochs']
    epoch = val_info['epoch']
    total_steps = len(dataloader)
    total_loss = 0
    for idx, (x_train, x_predict) in enumerate(dataloader):
        x_train = to_var(x_train)
        x_predict = to_var(x_predict)
        data = pack([x_train, x_predict, None], ['x_train', 'x_predict', 'states'])
        configs = pack([False, 10], ['use_gt', 'max_steps'])
        reconstruct, predict = model(data, configs)
        r_loss = criterion(reconstruct, x_train)
        p_loss = criterion(predict, x_predict)
        loss = r_loss + p_loss
        logger.info('[Val] Epoch [%d/%d], Step [%d/%d], Reconstruct Loss: %5.4f, Predict Loss: %5.4f, Total: %5.4f' % (
            epoch, num_epochs, idx + 1, total_steps, r_loss.data[0], p_loss.data[0], loss.data[0]))
        total_loss += loss.data[0]

    return total_loss / total_steps


def test(dataloader, model, test_info):
    pass


def main(hparams):
    model_info = hparams.model_info
    train_info = hparams.trainloader_info
    val_info = hparams.valloader_info
    test_info = hparams.testloader_info
    optimizer_info = hparams.optimizer_info
    main_info = hparams.main_info

    # initialize model and dataloader
    model = MMNIST_ConvLSTM(model_info)
    model = model.cuda()
    for name, param in model.named_parameters():
        print (name, param.size())
    num_epochs = main_info['num_epochs']
    # learning rate scheduler
    halve_every = main_info['halve_every']

    train_loader = get_dataloader(train_info)
    val_loader = get_dataloader(val_info)
    test_loader = get_dataloader(test_info)
    optimizer = get_optimizer(optimizer_info, model.parameters())

    criterion = cross_entropy

    for epoch in xrange(num_epochs):
        if train_loader is not None:
            adjust_learning_rate(optimizer, epoch, halve_every)
            traininfo = {'epoch': epoch, 'num_epochs':num_epochs, 'clip':main_info['clip']}
            train(train_loader, model, optimizer, criterion, traininfo)
        if val_loader is not None:
            valinfo = {'epoch': epoch, 'num_epochs': num_epochs}
            loss = val(val_loader, model, valinfo, criterion)
            res = saver.save(model, optimizer, loss, epoch)
            if res:
                logger.info('\033[96m' + '[Best model]:' + '\033[0m: on Validation set!')
        if test_loader is not None:
            # TODO
            pass


def make_global_parameters(hparams):
    torch.manual_seed(hparams.seed)

    global logger
    main_info = hparams.main_info
    model_info = hparams.model_info
    log_dir = main_info['log_dir']
    logger = create_logger(log_dir, model_info['name'])
    logger.info(hparams._items)

    global saver
    save_path = main_info['save_dir']
    saver = Saver(1e30, 'ENTROPY', hparams, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Main Module')
    parser.add_argument('--hparams', type=str, default="MMNIST_CONV_LSTM",
                        help='choose hyper-parameter set')
    args = parser.parse_args()
    extra_info = None

    hparams = get_hparams(args.hparams)(extra_info)
    make_global_parameters(hparams)
    main(hparams)