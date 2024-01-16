from torch.utils.data import Dataset
import numpy as np
import pdb
import torch

import scipy.stats as st
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
from dataclasses import dataclass
from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold

temp_i = 0


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    enter_time: torch.Tensor = None
    # shape: (batch, problem)
    leave_time: torch.Tensor = None
    # shape: (batch, problem)
    service_duration: torch.Tensor = None
    bool: torch.Tensor = None


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


def split_percentage(center,
                     percentage_size,
                     width_half,
                     width_half_percentage,
                     low,
                     high,
                     percentage_select=[25, 50, 75, 100]):
    low = low[None, :].expand_as(center)
    high = high[None, :].expand_as(center)
    size = high.size(-1)

    width_half = width_half.clamp_(min=5)
    width_half_percentage = width_half_percentage.clone().clamp_(min=5)
    print("size:", size)

    enter_time_all = (center[percentage_size:] -
                      width_half).clamp_(min=0).floor()
    leave_time_all = (center[percentage_size:]+width_half).floor()

    # 优先保证时间窗宽度，然后再保证时间窗中间值，即当中间值-半宽度<low的时候，保证enter>=low和半宽度不变，使中间值随之变化
    '''
    Prioritize ensuring the width of the time window, and then ensure the middle value of the time window. 
    When the middle value - half width<low, ensure that the middle>=low and half width remain unchanged, 
    so that the middle value changes accordingly
    '''
    enter_time_all = torch.where(
        leave_time_all > high[percentage_size:],
        high[percentage_size:] - (2*width_half),
        enter_time_all
    )
    leave_time_all = torch.where(
        enter_time_all < low[percentage_size:],
        low[percentage_size:] + (2*width_half),
        leave_time_all
    )
    enter_time_all = torch.where(
        enter_time_all > low[percentage_size:], enter_time_all, low[percentage_size:])
    leave_time_all = torch.where(
        leave_time_all < high[percentage_size:], leave_time_all, high[percentage_size:])

    enter_time_percentage = (
        center[:percentage_size]-width_half_percentage).clamp_(min=0).floor()
    leave_time_percentage = (
        center[:percentage_size]+width_half_percentage).floor()

    enter_time_percentage = torch.where(
        leave_time_percentage > high[:percentage_size],
        high[:percentage_size] - (2*width_half_percentage),
        enter_time_percentage
    )
    leave_time_percentage = torch.where(
        enter_time_percentage < low[:percentage_size],
        low[:percentage_size] + (2*width_half_percentage),
        leave_time_percentage
    )
    enter_time_percentage = torch.where(
        enter_time_percentage > low[:percentage_size], enter_time_percentage, low[:percentage_size])
    leave_time_percentage = torch.where(
        leave_time_percentage < high[:percentage_size], leave_time_percentage, high[:percentage_size])

    percentage = torch.cat(
        [enter_time_percentage[:, None, :], leave_time_percentage[:, None, :]], dim=1)
    enter_time_low = torch.zeros_like(enter_time_percentage)
    # enter_time_low = low[:percentage_size]
    leave_time_high = high[:percentage_size]
    low_high = torch.cat(
        [enter_time_low[:, None, :], leave_time_high[:, None, :]], dim=1)

    # The next 4 lines generate percentage
    per_size = torch.FloatTensor(np.array([np.random.choice(
        percentage_select, percentage_size)/100.])).squeeze(0)[:, None, None].expand(-1, 2, size)
    rand = torch.rand(percentage[:, 0:1, :].size()).expand(-1, 2, -1)
    percentage = torch.where(rand < per_size, percentage, low_high)

    # perceBool is a boolean array show whether a customer is in the time window constraint. If in, then = 1, otherwise = 0
    perceBool = torch.where(rand < per_size, torch.ones_like(
        percentage), torch.zeros_like(percentage))
    assert (perceBool[:, 0, :] == perceBool[:, 1, :]
            ).all(), 'perceBool out of limit'
    perceBool = torch.cat(
        [perceBool[:, 0, :], torch.ones_like(enter_time_all)], dim=0)

    enter_time_percentage = percentage[:, 0, :].floor()
    leave_time_percentage = percentage[:, 1, :].floor()

    # cat percentage with others
    enter_time_all = torch.cat((enter_time_percentage, enter_time_all), dim=0)
    leave_time_all = torch.cat((leave_time_percentage, leave_time_all), dim=0)
    # pdb.set_trace()
    # assert (enter_time_all >= low).all(), 'enter_time out of limit'
    assert (enter_time_all < high).all(), 'enter_time out of limit'
    assert (leave_time_all > enter_time_all).all() & (
        leave_time_all <= high).all(), 'leave_time out of limit'
    return enter_time_all, leave_time_all, perceBool


class VRPTWDataset(Dataset):

    def __init__(self, size=100,
                 num_samples=1000000,
                 solomon=False,
                 solomon_train=False,
                 train_type="c1",
                 file_pre="./SolomonBenchmark/RC1/",
                 filename="Solomon.txt",
                 *args,
                 **kwargs):
        super(VRPTWDataset, self).__init__()
        torch.manual_seed(1234)
        # pdb.set_trace()
        self.data = []
        if solomon_train:
            print("solomon_train:", True)
            if train_type == "c1":
                print("generate similar data of cluster_1")
                capacity = 200.
                x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
                    read("./SolomonBenchmark/C1/Solomon1.txt", capacity, num_samples,
                         size)

                # 3/8 beta
                percentage_size = int(3*num_samples/8)
                a, b, loc, scale = 4.06, 5.95, 16.05, 35.34
                width_half_percentage = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                                     size=[percentage_size, size])).to(torch.float)
                # 1/8 beta
                a, b, loc, scale = 3.66, 5.33, 33.51, 67.03
                width_half_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 gamma
                a, loc, scale = 1.52, 12.49, 43.03
                width_half_2 = torch.from_numpy(st.gamma.rvs(a=a, loc=loc, scale=scale,
                                                             size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 beta
                a, b, loc, scale = 3.73, 5.23, 66.20, 133.38
                width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)

                # 2/8 90,180
                width_half_4 = torch.FloatTensor(np.array([np.random.choice(
                    [90, 180], int(num_samples/4))])).squeeze(0)[:, None].expand(-1, size)

                # pdb.set_trace()
                width_half = torch.cat(
                    (width_half_1, width_half_2, width_half_3, width_half_4), dim=0)
                percentage_select = [25, 50, 75, 100]
                enter_time_all, leave_time_all, perceBool = split_percentage(
                    center, percentage_size, width_half, width_half_percentage, low, high, percentage_select)

            if train_type == "c2":
                print("generate similar data of Solomon_cluster_2")
                capacity = 700.
                x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
                    read("./SolomonBenchmark/C2/Solomon1.txt", capacity, num_samples,
                         size)
                # 3/8 constant width 80, but 25-100%
                percentage_size = int(num_samples/2)
                width_half_percentage = torch.FloatTensor(
                    [80])[None, :].expand(percentage_size, size)

                # 3/8 constant width 80 & 160 & 320
                width_half_1 = torch.FloatTensor(np.array([np.random.choice(
                    [160, 320], int(num_samples/4))])).squeeze(0)[:, None].expand(-1, size)
                # 1/8 Beta
                a, b, loc, scale = 3.67, 5.20, 133.29, 266.06
                width_half_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 Beta
                a, b, loc, scale = 0.86, 1.41, 88.50, 547.94
                width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)

                width_half = torch.cat(
                    (width_half_1, width_half_2, width_half_3), dim=0)

                enter_time_all, leave_time_all, perceBool = split_percentage(
                    center, percentage_size, width_half, width_half_percentage, low, high)

            elif train_type == "r1":
                print("generate similar data of Solomon_random_1")
                capacity = 200.
                x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
                    read("./SolomonBenchmark/R1/Solomon1.txt", capacity, num_samples,
                         size)
                # 1/2 5,15
                percentage_size = int(num_samples/2)
                width_half_percentage = torch.FloatTensor(np.array([np.random.choice(
                    [5, 15], percentage_size)])).squeeze(0)[:, None].expand(-1, size)
                # 1/8 genextreme
                c, loc, scale = 0.23, 27.77, 4.35
                width_half_1 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                                  size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 beta
                a, b, loc, scale = 1.23, 1.82, 11.32, 79.54
                width_half_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 beta
                a, b, loc, scale = 0.77, 1.25, 9.50, 88.05
                width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 genextreme
                c, loc, scale = 0.24, 55.60, 8.57
                width_half_4 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                                  size=[int(num_samples/8), size])).to(torch.float)
                width_half = torch.cat((
                    width_half_1, width_half_2, width_half_3, width_half_4), dim=0)

                enter_time_all, leave_time_all, perceBool = split_percentage(
                    center, percentage_size, width_half, width_half_percentage, low, high)

            elif train_type == "r2":
                print("generate similar data of Solomon_random_2")
                capacity = 1000.
                x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
                    read("./SolomonBenchmark/R2/Solomon1.txt", capacity, num_samples,
                         size)

                # 3/8 genextreme
                c, loc, scale = 0.22, 51.24, 17.33
                width_half_percentage_1 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                                             size=[int(3*num_samples/8), size])).to(torch.float)

                # 1/4 constant width 120, 25%-100%
                width_half_percentage_2 = torch.FloatTensor(
                    [120])[None, :].expand(int(num_samples/4), size)

                # 1/8 beta
                a, b, loc, scale = 1.30, 2.27, 44.52, 359.15
                width_half_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 beta
                a, b, loc, scale = 0.90, 1.76, 36.50, 457.83
                width_half_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)
                # 1/8 genextreme
                c, loc, scale = 0.22, 222.49, 34.74
                width_half_3 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                                  size=[int(num_samples/8), size])).to(torch.float)

                width_half = torch.cat(
                    (width_half_1, width_half_2, width_half_3), dim=0)

                width_half_percentage = torch.cat(
                    (width_half_percentage_1, width_half_percentage_2), dim=0)
                percentage_size = int(5*num_samples/8)

                enter_time_all, leave_time_all, perceBool = split_percentage(
                    center, percentage_size, width_half, width_half_percentage, low, high)

            elif train_type == "rc1":
                print("generate similar data of Solomon_random_cluster_1")
                capacity = 200.
                x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
                    read("./SolomonBenchmark/RC1/Solomon1.txt", capacity, num_samples,
                         size)
                # 1/2 constant width 15, but 25-100%
                percentage_size = int(num_samples/2)
                width_half_percentage = torch.FloatTensor(
                    [15])[None, :].expand(percentage_size, size)

                # 1/8 mix beta, constant 5, and constant 60
                a, b, loc, scale = 1.94, 87.21, 8.89, 663.77
                width_half_1_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                              size=[int(num_samples/16), size])).to(torch.float)
                width_half_1_2 = torch.FloatTensor(
                    [5])[None, :].expand(int(num_samples/32), size)
                width_half_1_3 = torch.FloatTensor(
                    [60])[None, :].expand(int(num_samples/32), size)
                width_half_1 = torch.cat(
                    (width_half_1_1, width_half_1_2, width_half_1_3), dim=0)  # cat all 1 to 3
                np.random.shuffle(width_half_1.view(-1).numpy())

                # 1/8 constant width 30
                width_half_2 = torch.FloatTensor(
                    [30])[None, :].expand(int(num_samples/8), size)

                # 1/8 mix beta and beta
                a, b, loc, scale = 2.88, 8.24, 19.28, 40.81
                width_half_3_1 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                              size=[int(num_samples/16), size])).to(torch.float)
                a, b, loc, scale = 12.26, 10.26, 16.42, 78.39
                width_half_3_2 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                              size=[int(num_samples/16), size])).to(torch.float)
                width_half_3 = torch.cat(
                    (width_half_3_1, width_half_3_2), dim=0)
                np.random.shuffle(width_half_3.view(-1).numpy())

                # 1/8 beta
                a, b, loc, scale = 9.90, 5.49, -27.18, 129.57
                width_half_4 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)


                width_half = torch.cat(
                    (width_half_1, width_half_2, width_half_3, width_half_4), dim=0)

                enter_time_all, leave_time_all, perceBool = split_percentage(
                    center, percentage_size, width_half, width_half_percentage, low, high)

            elif train_type == "rc2":
                print("generate similar data of Solomon_random_cluster_2")
                capacity = 1000.
                x, y, demand, enter_time, leave_time, center, low, high, service_duration = \
                    read("./SolomonBenchmark/RC2/Solomon1.txt", capacity, num_samples,
                         size)
                # 1/2 constant width 60, 25%-100%
                percentage_size = int(num_samples/2)
                width_half_percentage = torch.FloatTensor(
                    [60])[None, :].expand(percentage_size, size)

                # 1/8 mix 30, 240 and dweibull
                width_half_1_1 = torch.FloatTensor(np.array([np.random.choice(
                    [30, 240], int(num_samples/16))])).squeeze(0)[:, None].expand(-1, size)
                c, loc, scale = 2.05, 92.65, 31.63
                width_half_1_2 = torch.from_numpy(st.dweibull.rvs(c=c, loc=loc, scale=scale,
                                                                  size=[int(num_samples/16), size])).to(torch.float)
                width_half_1 = torch.cat(
                    (width_half_1_1, width_half_1_2), dim=0)
                np.random.shuffle(width_half_1.view(-1).numpy())

                # 1/8
                width_half_2 = torch.FloatTensor(
                    [120])[None, :].expand(int(num_samples/8), size)

                # 1/8 beta
                a, b, loc, scale = 1.30, 2.27, 44.52, 359.15
                width_half_3 = torch.from_numpy(st.beta.rvs(a=a, b=b, loc=loc, scale=scale,
                                                            size=[int(num_samples/8), size])).to(torch.float)

                # 1/8 genextreme
                c, loc, scale = 0.22, 222.48, 34.73
                width_half_4 = torch.from_numpy(st.genextreme.rvs(c=c, loc=loc, scale=scale,
                                                                  size=[int(num_samples/8), size])).to(torch.float)

                width_half = torch.cat(
                    (width_half_1, width_half_2, width_half_3, width_half_4), dim=0)

                enter_time_all, leave_time_all, perceBool = split_percentage(
                    center, percentage_size, width_half, width_half_percentage, low, high)

        elif solomon:
            self.data = []
            capacity = {
                'c1': 200.,
                'c2': 700.,
                'r1': 200.,
                'r2': 1000.,
                'rc1': 200.,
                'rc2': 1000.,
            }.get(train_type, None)
            for file in filename:
                # pdb.set_trace()
                solomon = np.genfromtxt(file_pre+file, dtype=[
                                        np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])
                x_random = [x[1] for x in solomon]
                x_random = torch.from_numpy(np.array(x_random))

                y_random = [x[2] for x in solomon]
                y_random = torch.from_numpy(np.array(y_random))

                demand_random = [x[3] for x in solomon]
                demand_random = torch.from_numpy(
                    np.array(demand_random))/capacity

                enter_time_random = [x[4] for x in solomon]
                enter_time_random = torch.from_numpy(
                    np.array(enter_time_random))

                leave_time_random = [x[5] for x in solomon]
                leave_time_random = torch.from_numpy(
                    np.array(leave_time_random))

                service_duration = [x[6] for x in solomon]
                service_duration = torch.from_numpy(np.array(service_duration))
                perceBool = torch.where(enter_time_random == torch.zeros_like(
                    enter_time_random), torch.zeros_like(enter_time_random), torch.ones_like(enter_time_random))
            # 先随机生成num_samples组数据

                self.data = self.data+[
                    {'loc': torch.cat((x_random[1:, None], y_random[1:, None]), dim=-1),  # size,2
                     # 1,2
                     'depot': torch.cat((x_random[0:1], y_random[0:1]), dim=-1),
                     'demand': demand_random[1:],  # size
                     # size
                     'enter_time': enter_time_random,
                     'leave_time': leave_time_random,
                     'service_duration': service_duration[1:],
                     'bool': perceBool
                     }]
        if self.data == []:
             self.data = [
                {'loc': torch.cat((x[1:, None], y[1:, None]), dim=-1),  # size,2
                 'depot': torch.cat((x[0:1], y[0:1]), dim=-1),  # 2
                 'demand': demand[1:],  # size
                 'enter_time': torch.cat((enter_time[0:1], enter_time_all[i]), dim=0),
                 'leave_time': torch.cat((leave_time[0:1], leave_time_all[i]), dim=0),
                 'service_duration': service_duration,
                 'bool': torch.cat((torch.zeros_like(leave_time[0:1]), perceBool[i]), dim=0)
                 } for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)

    cur_coord: torch.Tensor = None
    # shape: (batch, pomo,1,2)

    cur_time: torch.Tensor = None
    # shape: (batch, pomo)
    unusual: torch.Tensor = None
    # shape: (batch, pomo)
    # service_duration: torch.Tensor = None
    # shape: (batch, pomo)
    lengths: torch.Tensor = None
    # shape: (batch, pomo)
    # visited_ninf_flag: torch.Tensor = None
    # tw_constraint: torch.Tensor = None
    # demand_too_large: torch.Tensor = None
    # route_duration_all: torch.Tensor = None
    # leave_time: torch.Tensor = None
    # enter_time: torch.Tensor = None
    # high: torch.Tensor = None
    # distance: torch.Tensor = None


class CVRPEnv:
    def __init__(self, device='cuda', **env_params):

        # Const @INIT
        ####################################
        self.temp_i = 0
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.enter_time = None
        self.leave_time = None
        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.device = device
        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_problems(self, batch_size, batch, aug_factor=1):
        self.batch_size = batch_size
        self.aug_factor = aug_factor
        # pdb.set_trace()
        if not self.FLAG__use_saved_problems:
            # depot_xy, node_xy, node_demand = get_random_problems(
            #     batch_size, self.problem_size)
            depot_xy = batch['depot'][:, None, :]
            node_xy = batch['loc']
            node_demand = batch['demand']
            enter_time = batch['enter_time']
            leave_time = batch['leave_time']
            service_duration = batch['service_duration']
            bool = batch['bool']
            # pdb.set_trace()
            # depot_xy = batch['depot']

            # node_xy, node_demand, enter_time, leave_time = get_random_problems(
            #     batch_size, self.problem_size)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                enter_time = enter_time.repeat(8, 1)
                leave_time = leave_time.repeat(8, 1)
                service_duration = service_duration.repeat(8, 1)
            else:
                raise NotImplementedError
        # pdb.set_trace()
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(
            size=(self.batch_size, 1), device=self.device)
        # print(self.batch_size)
        # pdb.set_trace()
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[
            :, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None, :].expand(
            self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.enter_time = enter_time
        self.reset_state.leave_time = leave_time
        self.reset_state.service_duration = service_duration
        self.reset_state.bool = bool

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.cur_coord = depot_xy.unsqueeze(
            1).expand(-1, self.pomo_size, -1, -1)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long, device=self.device)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(
            size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(
            size=(self.batch_size, self.pomo_size), device=self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size+1), device=self.device)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(
            size=(self.batch_size, self.pomo_size, self.problem_size+1), device=self.device)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(
            size=(self.batch_size, self.pomo_size), dtype=torch.bool, device=self.device)
        # shape: (batch, pomo)
        self.lengths = torch.zeros(
            size=(self.batch_size, self.pomo_size), device=self.device)
        self.unusual = torch.zeros(
            size=(self.batch_size, self.pomo_size), device=self.device)
        self.cur_time = torch.zeros(
            size=(self.batch_size, self.pomo_size), device=self.device)
        # self.service_duration=torch.ones(size=(self.batch_size, self.pomo_size))*90

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.lengths = self.lengths
        self.step_state.unusual = self.unusual
        self.step_state.cur_time = self.cur_time
        # self.step_state.service_duration=self.service_duration

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, batch, train=False, hard=False):

        # selected.shape: (batch, pomo)
        # selected = torch.ones_like(selected) * 400
        # Dynamic-1
        ####################################
        self.selected_count += 1
        selected = selected.to(self.device)
        self.current_node = selected
        # shape: (batch, pomo)

        '''计算每步骤的当前时刻,以及length'''
        cur_coord = self.depot_node_xy[:, None, :, :].\
            expand(-1, self.pomo_size, -1, -1).\
            gather(
                dim=2, index=self.current_node[:, :, None, None].expand(-1, -1, -1, 2))
        route_duration = (  # batch x pomo
            cur_coord - self.step_state.cur_coord).norm(p=2, dim=-1).squeeze(-1)
        # pdb.set_trace()
        arrive_time = torch.max(  # batch x pomo
            (self.step_state.cur_time + route_duration),
            batch['enter_time'][:, None, :].repeat(self.aug_factor, 1, 1).expand(self.batch_size, self.pomo_size, -1).\
            gather(dim=2, index=self.current_node[:, :, None]).squeeze(-1)
        )

        # pdb.set_trace()
        lengths = self.step_state.lengths + (route_duration*10).floor()/10 +\
            (arrive_time -
             batch['leave_time'][:, None, :].repeat(self.aug_factor, 1, 1).expand(self.batch_size, self.pomo_size, -1).
             gather(dim=2, index=self.current_node[:, :, None]).squeeze(-1)
             ).clamp(min=0) * 0.5

        unusual = self.step_state.unusual+(arrive_time -
                                           batch['leave_time'][:, None, :].repeat(self.aug_factor, 1, 1).expand(self.batch_size, self.pomo_size, -1).
                                           gather(
                                               dim=2, index=self.current_node[:, :, None]).squeeze(-1)
                                           ).clamp(min=0) * 0.5
        # print("lengths:",lengths,"select:",selected,"route_duration:",route_duration,"arrive_time:",arrive_time)
        # pdb.set_trace()
        cur_time = torch.where(
            selected.squeeze(-1) == 0,
            torch.zeros_like(self.step_state.cur_time),
            arrive_time +
            torch.cat([torch.zeros_like(self.reset_state.service_duration[:, 0:1]),
                      self.reset_state.service_duration], dim=1).gather(dim=1, index=self.current_node)
        )

        self.step_state.cur_time = cur_time
        self.step_state.cur_coord = cur_coord
        self.step_state.lengths = lengths
        self.step_state.unusual = unusual

        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(
            self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(
            dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot
        # mask visited
        # self.temp_i = self.temp_i + 1
        # x = self.BATCH_IDX
        # y = self.POMO_IDX
        # print(self.temp_i, end=' ')
        # pdb.set_trace()
        # selected = torch.ones_like(selected) * 400
        self.visited_ninf_flag.scatter_(dim=2, index=selected[..., None], src=torch.ones_like(
            self.visited_ninf_flag)*float('-inf'))
        # if self.temp_i != 4:
        # self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        # depot is considered unvisited, unless you are AT the depot
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + \
            round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        if hard:
            # mask if cur_time + route_duration(for all) > leave_time
            # pdb.set_trace()
            route_duration_all = (self.step_state.cur_coord.clone().expand(-1, -1, self.depot_node_xy.size(-2), -1) -
                                  self.depot_node_xy[:, None, :, :].clone().expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1)
            tw_constraint = ((cur_time[..., None].expand_as(route_duration_all) + route_duration_all.clone()) >
                             batch['leave_time'][:, None, :].expand_as(route_duration_all))
            demand_too_large = demand_too_large + tw_constraint
            # self.ninf_mask[tw_constraint] = float('-inf')# mask points with too large demand
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        # the follow states are used to debug:
        # self.step_state.visited_ninf_flag = self.visited_ninf_flag
        # self.step_state.tw_constraint = tw_constraint
        # self.step_state.demand_too_large = demand_too_large
        # self.step_state.route_duration_all = route_duration_all
        # self.step_state.leave_time = batch['leave_time']
        # self.step_state.enter_time = batch['enter_time']
        # self.step_state.high = batch['high']
        # self.step_state.distance = batch['distance']

        # returning values
        done = self.finished.all()

        if done:
            reward = -self._get_travel_distance(batch)  # note the minus sign!
        else:
            reward = None
        if train:
            return self.step_state, reward, done, unusual
        else:
            return self.step_state, reward, done, unusual, self.selected_node_list

    def _get_travel_distance(self, batch):
        # pdb.set_trace()
        route_duration = (
            self.depot_node_xy[:, None, 0:1,
                               :].expand(-1, self.pomo_size, -1, -1)
            - self.step_state.cur_coord
        ).norm(p=2, dim=-1).squeeze(-1)
        # pdb.set_trace()
        penalty = (self.step_state.cur_time + route_duration -
                   batch['leave_time'][:, None, 0].repeat(self.aug_factor, 1).expand(self.batch_size, self.pomo_size)).clamp(min=0) * 0.5
        # pdb.set_trace()
        return self.step_state.lengths + route_duration + penalty
    # def _get_travel_distance(self):
    #     gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
    #     # shape: (batch, pomo, selected_list_length, 2)
    #     all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
    #     # shape: (batch, pomo, problem+1, 2)

    #     ordered_seq = all_xy.gather(dim=2, index=gathering_index)
    #     # shape: (batch, pomo, selected_list_length, 2)

    #     rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    #     segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
    #     # shape: (batch, pomo, selected_list_length)

    #     travel_distances = segment_lengths.sum(2)
    #     # shape: (batch, pomo)
    #     return travel_distances
