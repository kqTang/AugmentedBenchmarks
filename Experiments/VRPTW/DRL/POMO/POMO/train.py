##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE


#aasdsad
##########################################################################################
# Path Config

import os
import sys

import torch.nn as nn
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTrainer import CVRPTrainer as Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm
from CVRPEnv import VRPTWDataset
import torch
import pdb
import numpy as np
import argparse
##########################################################################################
# parameters




##########################################################################################
# main
def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)
def main(opts):
    if opts.mutiGPU:
        torch.distributed.init_process_group(backend='nccl')
        
    env_params['problem_size'] = opts.size
    # env_params['pomo_size'] = opts.size
    trainer_params['train_episodes'] = 10*trainer_params['train_batch_size']
    trainer_params['train_batch_size'] = opts.batch_size
    dataname = opts.dataname
    if DEBUG_MODE:
        _set_debug_mode()
    logger_params['log_file']['desc'] = dataname[5:-4]
    create_logger(**logger_params)
    _print_config()
    # training_dataset = VRPTWDataset(size=100, num_samples=trainer_params['epochs']*trainer_params['train_batch_size'],solomon_train=True)
    # training_dataset = VRPTWDataset(size=100, num_samples=90,solomon_train=True)
    print(dataname)
    # if opts.mutiGPU:
    #     training_dataset = torch.load(
    #         dataname, map_location=lambda storage, loc: storage.cuda(CUDA_DEVICE_NUM))['data']
    # else:
    training_dataset = torch.load(dataname)['data']
    if opts.mutiGPU:
        train_sample = torch.utils.data.distributed.DistributedSampler(
            training_dataset)
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      mutiGPU=opts.mutiGPU)
    
    copy_all_src(trainer.result_folder)
    
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    device = torch.device('cuda', CUDA_DEVICE_NUM)
    epoch = 1
    n_turns = 3 if opts.size <= 100 else int(3*opts.size/100)
    for i in range(n_turns):
        # np.random.shuffle(training_dataset.data)
        # print()
        # for j in range(5):
        # pdb.set_trace()
        if opts.mutiGPU:
            for batch in tqdm(DataLoader(training_dataset, batch_size=trainer_params['train_batch_size'], sampler=train_sample), disable=False):
                # np.random.shuffle(training_dataset.data)
                batch = move_to(batch, device)
                # pdb.set_trace()
                trainer.run_one_epoch(epoch,batch,opts.hard)
                epoch = epoch + 1     
        else:
            for batch in tqdm(DataLoader(training_dataset, batch_size=trainer_params['train_batch_size']), disable=False):
                # np.random.shuffle(training_dataset.data)
                batch = move_to(batch, device)
                # pdb.set_trace()
                trainer.run_one_epoch(epoch, batch, opts.hard)
                epoch = epoch + 1

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataname", type=str,  default='data.pth', help="Name of the results file to write")
    parser.add_argument("--hard",  action='store_true',  help="hard or soft time_window")
    parser.add_argument("--mutiGPU",  action='store_true',
                        help="mutiGPU or single-GPU")
    parser.add_argument("--size", type=int,  default=100,help="customer size")
    parser.add_argument("--batch_size", type=int,  default=140, help="batch size")
    parser.add_argument("--type", type=str,
                        default='origin', help=['origin','bool','center','differ2b0'])
    opts = parser.parse_args()
    if opts.mutiGPU:
        CUDA_DEVICE_NUM = int(os.environ["LOCAL_RANK"])
    else:
        CUDA_DEVICE_NUM = 0
    env_params = {
        'problem_size': opts.size,
        'pomo_size': opts.size,
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'type': opts.type,
        'cuda_device_num': CUDA_DEVICE_NUM
    }

    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6
        },
        'scheduler': {
            'milestones': [8001, 8051],
            'gamma': 0.1
        }
    }

    trainer_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'epochs': 1000,
        'train_episodes': 10 * opts.batch_size,
        'train_batch_size': opts.batch_size,
        'prev_model_path': None,
        'logging': {
            'model_save_interval': 100,
            'img_save_interval': 500,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_cvrp_100.json'
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json'
            },
        },
        'model_load': {
            'enable': False,  # enable loading pre-trained model
            # 'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
            # 'epoch': 2000,  # epoch version of pre-trained model to laod.

        }
    }

    logger_params = {
        'log_file': {
            'desc': 'train_cvrp_n100_with_instNorm',
            'filename': 'run_log'
        }
    }
    main(opts)
