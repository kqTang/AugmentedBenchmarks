from torch.utils.data import Dataset
import numpy as np
import pdb
import torch

import scipy.stats as st
import os
import sys


def read(filename, capacity, sub_num_samples, size):

    solomon = np.genfromtxt(filename, dtype=[
        np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])

    x = [x[1] for x in solomon]
    x = torch.from_numpy(np.array(x))

    y = [x[2] for x in solomon]
    y = torch.from_numpy(np.array(y))

    enter_time = [x[4] for x in solomon]
    enter_time = torch.from_numpy(np.array(enter_time))

    leave_time = [x[5] for x in solomon]
    leave_time = torch.from_numpy(np.array(leave_time))

    service_duration = [x[6] for x in solomon]
    service_duration = torch.from_numpy(np.array(service_duration))

    demand = [x[3] for x in solomon]
    demand = torch.from_numpy(np.array(demand))/capacity

    loc = torch.cat((x[1:, None], y[1:, None]), dim=-1)
    depot = torch.cat((x[0:1], y[0:1]), dim=-1)

    low = (loc - depot.unsqueeze(-2)).norm(p=2, dim=-1).floor()

    high = (leave_time[0] - (loc - depot.unsqueeze(-2)
                             ).norm(p=2, dim=-1) - service_duration[1]).floor()

    center = torch.rand(sub_num_samples, size)*((high-low)[None, :]).expand(sub_num_samples, -1) + \
        low[None, :].expand(sub_num_samples, -1)
    return x[:size+1], y[:size+1], demand[:size+1], enter_time[:size+1], leave_time[:size+1],\
        center[:, :size], low[:size], high[:size], service_duration[1:size+1]

class VRPTWDataset(Dataset):

    def __init__(self,
                 file_pre="./data/RC1",
                 filename="Solomon.txt"):
        super(VRPTWDataset, self).__init__()
        
        self.data_set = []
        # pdb.set_trace()
        self.data = []

        capacity = 200.
        for file in filename:
            homberger = np.genfromtxt(file_pre+file, dtype=[
                np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])
            x_random = [x[1] for x in homberger]
            x_random = torch.from_numpy(np.array(x_random))

            y_random = [x[2] for x in homberger]
            y_random = torch.from_numpy(np.array(y_random))

            demand_random = [x[3] for x in homberger]
            demand_random = torch.from_numpy(
                np.array(demand_random))/capacity

            enter_time_random = [x[4] for x in homberger]
            # enter_time_random = [x[4] for x in homberger1]
            enter_time_random = torch.from_numpy(
                np.array(enter_time_random))

            # leave_time_random = [x[5] for x in homberger]
            leave_time_random = [x[5] for x in homberger]
            leave_time_random = torch.from_numpy(
                np.array(leave_time_random))
            loc = torch.cat(
                (x_random[1:, None], y_random[1:, None]), dim=-1)
            depot = torch.cat((x_random[0:1], y_random[0:1]), dim=-1)

            service_duration = [x[6] for x in homberger]
            service_duration = torch.from_numpy(np.array(service_duration))
            low = (loc - depot.unsqueeze(-2)).norm(p=2, dim=-1).floor()

            high = (leave_time_random[0] - (loc - depot.unsqueeze(-2)
                                            ).norm(p=2, dim=-1) - service_duration[1]).floor()
            # if enter==0, enter=low
            # enter_time_random[1:] = torch.where(
            #     enter_time_random[1:] > low, enter_time_random[1:], low)

            leave_time_random[1:] = torch.where(
                leave_time_random[1:] < high, leave_time_random[1:], high)
            # assert (enter_time_random[1:] >= low).all(
            # ), 'enter_time out of limit'
            # assert (leave_time_random > enter_time_random).all() & (
            #     leave_time_random[1:] <= high).all(), 'leave_time out of limit'
            
        # 先随机生成num_samples组数据

            self.data = self.data+[
                {'loc': torch.cat((x_random[1:, None], y_random[1:, None]), dim=-1),  # size,2
                    # 1,2
                    'depot': torch.cat((x_random[0:1], y_random[0:1]), dim=-1),
                    'demands': demand_random,  # size
                    # size
                    'enter_time': enter_time_random,
                    'leave_time': leave_time_random,
                    'service_duration': service_duration[1:]
                    }]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

