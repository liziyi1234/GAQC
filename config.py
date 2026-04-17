import torch


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def EM360IQA_config():
    config = Config({
        # model setting
        'num_vps': 8,                       # number of viewports in a sequence.
        'img_channels': 3,
        'img_size': 1024,
        'dim': 64,                          # dimension after Stem module.
        'depths': (2, 2, 5, 3),             # number of maxvit block in each stage.
        'channels': (128, 256, 512, 512),     # channels in each stage.
        'num_heads': (2, 4, 8, 16),          # number of head in each stage.
        
        'mlp_ratio': 3,
        'drop_rate': 0.,
        'pos_drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.,              # droppath rate in encoder block.
        'kernel_size': 7,
        'layer_scale': None,
        'dilations': None,
        'qkv_bias': True,
        'qk_scale': None,
        'select_rate': 0.5,                 # the rate of select feature from all viewport features.
        'num_classes': 4,
        'hidden_dim': 1152,                   
        
        
        # resource setting

        
        'image_path': '/mnt/10T/liziyi/LargeKernel/JUFE-10K/resized_dis_1024',
        'info_csv_path': '/mnt/10T/liziyi/LargeKernel/JUFE-10K/JUFE-10k_mos.csv',
        

             
        'save_ckpt_path': '/mnt/10T/liziyi/GAQC/pt/GAQC_JUFE-10K_1024_1024',
        'load_ckpt_path': '',
        'tensorboard_path': '',
        # train setting
        'seed': 47,
        'model_name': 'GAQC',
        'dataset_name': 'JUFE-10K_1024_1024',
        'epochs':50,
        'batch_size':  8,
        'num_workers': 8,
        'lr': 1e-4,
        'lrf': 0.01,
        #'weight_decay': 5e-4,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'p': 1,
        'q': 2,
        'q': 2,
        'use_tqdm': False,
        'use_tensorboard': False,
        'batch_print': False,
        'device': torch.device("cuda:7" if torch.cuda.is_available() else "cpu"),
    })  
        
    return config