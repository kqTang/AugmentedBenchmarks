import argparse
import os
import numpy as np
from VRPTWDataset import VRPTWDataset
import pickle
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def generate_cvrptw_data(size, dataset_size, type, solomon, solomon_train):

    return VRPTWDataset(size=size, num_samples=dataset_size, train_type=type, solomon=solomon, solomon_train=solomon_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--type", default="all")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name",  default='cvrptw', type=str,
                         help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='cvrptw',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=1024, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"
    a = os.getcwd()
    b = os.path.dirname(__file__)
    if opts.problem == 'cvrptw':
        datadir = os.path.join(opts.data_dir, 'cvrptw')
        i = 0
        if opts.type == "all":
            TYPE=['c1','c2','r1','r2','rc1','rc2']
        else:
            TYPE=[opts.type]
        for type in TYPE:
            solomon = False
            solomon_train = True
            capacity = {
                'c1': 200.,
                'c2': 700.,
                'r1': 200.,
                'r2': 1000.,
                'rc1': 200.,
                'rc2': 1000.,
            }.get(type, None)
            dataset = generate_cvrptw_data(
                size=100, dataset_size=opts.dataset_size, type=type, solomon=solomon, solomon_train=solomon_train)
            i += 1
            csvfile = open(os.path.dirname(__file__)+'/cvrptw_{}_testDER_seed4321.csv'.format(type),
                        mode='w', newline='')

            fieldnames = ['x', 'y', 'demand', 'e', 'l', 'service_time']
            # data = torch.load('./pth/data_c1_2_test_200.pth')['data']
            # 创建 DictWriter 对象

            write = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 写入表头
            # write.writeheader()
            # 写入数据
            for batch in DataLoader(dataset, batch_size=1024):
                # pdb.set_trace()
                write.writerow({'x': batch['depot'][0, 0].item(), 'y': batch['depot'][0, 1].item(), 'demand': 0,
                                'e': 0, 'l': batch['leave_time'][0, 0].item(), 'service_time': 0})
                for i in tqdm(range(1024)):
                    for j in range(batch['loc'].shape[1]):
                        write.writerow({'x': batch['loc'][i, j, 0].item(), 'y': batch['loc'][i, j, 1].item(), 'demand': int(batch['demand'][i, j].item()*capacity),
                                        'e': batch['enter_time'][i, j+1].item(), 'l': batch['leave_time'][i, j+1].item(), 'service_time': batch['service_duration'][i, j].item()})
