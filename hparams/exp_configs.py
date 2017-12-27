from .hparams import HParams
from .register import register
import pickle


@register("CAPTION_GNN_CONTRASTIVE")
def CAPTION_GNN_CONTRASTIVE(extra_info):
    root = './data'
    dataset = 'refcocog'
    split_by = 'google'
    name = 'CAP_GNN'
    vocab = pickle.load(open('./data/refcocog/vocab/google/vocab.pkl', 'r'))
    trainloader_info = {
        'name': name,
        'root': root,
        'dataset': dataset,
        'split_by': split_by,
        'split': 'train',
        'vocab': vocab,
        'batch_size': 20,
        'shuffle': True,
        'num_workers': 2,
        'transform': None
    }

    valloader_info = {
        'name': name,
        'root': root,
        'dataset': dataset,
        'split_by': split_by,
        'split': 'val',
        'vocab': vocab,
        'batch_size': 10,
        'shuffle': False,
        'num_workers': 2,
        'transform': None
    }

    testloader_info = None
    model_info = {
        'name': name,
        'gnn_hid_dim': 1024,
        'vis_dim': 512,
        'vis_num': 50,
        'image_size': 224,
        'n_channels': 1,
        'dr_decoder': 0.5,
        'dr_embedding': 0.5,
        'vocab_size': 4255,
        'embed_dim': 512,
        'lstm_hid_dim': 1024,
        'lstm_fea_dim': 512,
        'time_steps': 4,
        'hid_init_method': 'average',
    }
    criterion_info = {
        'name': 'CVPR16_MMI',
        'lamb': 1,
        'margin': 1
    }

    optimizer_info = {
        'lr': 1e-4,
        'optim_alg': 'Adam'
    }
    seed = 666
    folder_name = 'CAPTION_GNN_CONTRASTIVE'
    main_info = {'name': name,
                 'num_epochs': 50,
                 'halve_every': 30,
                 'save_dir': './checkpoints/%s'%folder_name,
                 'log_dir': './logs/%s'%folder_name,
                 'with_neg': False,
                 'checkpoint': './checkpoints/%s/checkpoint.pth.tar'%folder_name,
                 'test': False,
                 # 'save': './vis/images/CAP_neginit_GNN_512.pth'
                 }
    hparams = HParams( trainloader_info=trainloader_info,
                       valloader_info=valloader_info,
                       testloader_info=testloader_info,
                       model_info=model_info,
                       criterion_info=criterion_info,
                       optimizer_info=optimizer_info,
                       main_info=main_info,
                       seed=seed)
    return hparams



