import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from CVRPEnv import VRPTWDataset
import pdb
import argparse
import time
start = time.time()
parser = argparse.ArgumentParser()

parser.add_argument("--train_type", type=str,  default='c2', help="Name of the results file to write")
parser.add_argument("--solomon_train",  action='store_true',  help="solomon_train")
parser.add_argument("--trainortest", type=str,
                    default='train', help="train or test")
parser.add_argument("--num_samples",  type=int,  default=140000,
                    help="num_samples")
parser.add_argument("--size",  type=int,  default=100,
                    help="customer size")
opts = parser.parse_args()
# print(opts.center_arrive)

training_dataset = VRPTWDataset(size=opts.size,
                              num_samples=opts.num_samples,
                              solomon_train=opts.solomon_train,
                              train_type=opts.train_type)
np.random.shuffle(training_dataset.data)
state = {'data':training_dataset}
if opts.size>100:
    torch.save(state, './pth/data_'+opts.train_type +
           '_%s_%s.pth' % (opts.trainortest, opts.size))
else:
    torch.save(state, './pth/data_'+opts.train_type +
           '_%s.pth' % opts.trainortest)
print("time cost for generating data:", time.time()-start)
# training_dataset_load = VRPTWDataset(size=100, num_samples=90,solomon_train=True)
# print(len(training_dataset_load.data))
# training_dataset_load.data = torch.load('./data.pth')['data']
# print(len(training_dataset_load.data), time.time()-start)
# pdb.set_trace()
# # training_dataset_load = torch.load(state,'./data.pth')
# training_dataloader = DataLoader(training_dataset_load, batch_size=200, num_workers=1)

# for bat in tqdm(DataLoader(training_dataset_load, batch_size=200), disable=False):
#     print(bat['loc'].shape)
