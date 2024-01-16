##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config
import math
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester import CVRPTester as Tester

from torch.utils.data import DataLoader
from tqdm import tqdm
from CVRPEnv import VRPTWDataset
import torch
import numpy as np
import pdb
##########################################################################################
# parameters
import argparse



##########################################################################################
# main
def split(var, size):
    var['loc']=var['loc'][:,:-size,:]
    var['demand'] = var['demand'][:,:-size]
    var['enter_time'] = var['enter_time'][:, :-size]
    var['leave_time'] = var['leave_time'][:, :-size]
    var['service_duration'] = var['service_duration'][:, :-size]
    return var

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)
def main(opts):
    env_params['problem_size'] = opts.size
    env_params['pomo_size'] = opts.size
    # pdb.set_trace()
    if opts.solomon:
        training_dataset = VRPTWDataset(size=100, num_samples=1, solomon=True,
                                      train_type=opts.train_type, file_pre=opts.file_pre, filename=opts.filename)
        num_files = len(opts.filename)
        test_batch_size = tester_params['test_batch_size']
        tester_params['test_batch_size'] = tester_params['test_batch_size'] * num_files
    elif opts.homberger:
        training_dataset = VRPTWDataset(size=opts.size, num_samples=1, homberger=True,file_pre=opts.file_pre, filename=opts.filename)
        num_files = len(opts.filename)
        test_batch_size = tester_params['test_batch_size']
        tester_params['test_batch_size'] = tester_params['test_batch_size'] * num_files
    else:
        # pdb.set_trace()
        training_dataset = torch.load(opts.filename[0])['data']
        num_files = opts.batch_size
        tester_params['test_batch_size'] = opts.batch_size 
        
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    tester_params['model_load']['path'] = opts.model_load_path

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)
    # pdb.set_trace()

    torch.cuda.set_device(CUDA_DEVICE_NUM)
    device = torch.device('cuda', CUDA_DEVICE_NUM)
    # pdb.set_trace()


    # print(test_batch_size)

    for batch in tqdm(DataLoader(training_dataset, batch_size=num_files)):
        # pdb.set_trace()
        batch = move_to(batch, device)
        
        
        if opts.solomon:
            size = 100
            batch = {'loc': batch['loc'][None,...].repeat(test_batch_size,1, 1,1).reshape(tester_params['test_batch_size'],size,2),  # batch,size,2
                    'depot': batch['depot'][None,...].repeat(test_batch_size,1, 1).reshape(tester_params['test_batch_size'],2), # batch,2
                    'demand': batch['demand'][None,...].repeat(test_batch_size,1, 1).reshape(tester_params['test_batch_size'],size), # batch,size
                    'enter_time': batch['enter_time'][None,...].repeat(test_batch_size,1, 1).reshape(tester_params['test_batch_size'],size+1), # batch,size+1
                    'leave_time': batch['leave_time'][None,...].repeat(test_batch_size,1, 1).reshape(tester_params['test_batch_size'],size+1), # batch,size+1
                    'service_duration': batch['service_duration'][None,...].repeat(test_batch_size,1, 1).reshape(tester_params['test_batch_size'],size), # batch,size
                     # batch,size+1
                     'bool': batch['bool'][None, ...].repeat(test_batch_size, 1, 1).reshape(tester_params['test_batch_size'], size+1),
                    }
        # for i in range(1, 31):
        checkpoint = torch.load(opts.model_load_path, map_location=device)

        score = tester.run(batch, num_files, hard=opts.hard,
                            checkpoint=checkpoint, solomon=opts.solomon)
        with open("temp_{}.txt".format(opts.type), "w") as f:
            for j in range(score.shape[0]):
                f.writelines(str(score[j].item()))
                f.writelines(',')

            f.writelines("\n")


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", type=str, nargs='+', default=['Solomon.txt', 'Solomon2.txt', 'Solomon3.txt'], help="Name of the results file to write")
    parser.add_argument("--file_pre", type=str,  default='./data/RC1', help="file_pre")

    parser.add_argument("--train_type", type=str,  default='c1', help="which type of solomon benchmark to test")

    parser.add_argument("--model_load_path", type=str,  default='./pt/c1/checkpoint-3000.pt', help="which type of solomon benchmark to test")
    parser.add_argument("--hard",  action='store_true',  help="varying")
    parser.add_argument("--type", type=str,
                        default='origin', help=['origin', 'bool', 'center', 'differ2b0'])
    parser.add_argument("--solomon",  action='store_true',
                        help="test solomon benchmark or not")
    parser.add_argument("--homberger",  action='store_true',
                        help="test homberger benchmark or not")
    parser.add_argument("--size", type=int,  default=100,
                        help="customer size")
    parser.add_argument("--batch_size", type=int,  default=512,
                        help="customer size")
    opts = parser.parse_args()
    env_params = {
        'problem_size':  opts.size,
        'pomo_size':  opts.size,
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'softmax',
        'type': opts.type,
        'cuda_device_num': CUDA_DEVICE_NUM,
    }


    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_load': {
            # 'path': './result/saved_CVRP100_model',  # directory path of pre-trained model and log files saved.
            'path': './',
            'epoch': '1750-C2',  # epoch version of pre-trained model to laod.
        },
        'test_episodes': 1000,
        'test_batch_size': opts.batch_size,
        'augmentation_enable': True,
        'aug_factor': 1,
        'aug_batch_size': opts.batch_size,
        'test_data_load': {
            'enable': False,
            'filename': '../vrp100_test_seed1234.pt'
        },
    }
    # if tester_params['augmentation_enable']:
    # tester_params['test_batch_size'] = tester_params['aug_batch_size']


    logger_params = {
        'log_file': {
            'desc': 'test_cvrp100',
            'filename': 'log.txt'
        }
    }
    main(opts)
