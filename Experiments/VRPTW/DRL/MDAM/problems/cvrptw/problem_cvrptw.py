from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.cvrptw.state_cvrptw import StateCVRPTW
from utils.beam_search import beam_search
import numpy as np
import scipy.stats as st
import pdb


class CVRPTW(object):

    NAME = 'cvrptw'  # Capacitated Vehicle Routing Problem

    # (w.l.o.g. vehicle capacity is 1, demands should be scaled)
    VEHICLE_CAPACITY = 1.0

    @staticmethod
    def get_costs(dataset, pi):
        # pdb.set_trace()
        batch_size, graph_size = dataset['demand'].size()
        ids = torch.arange(batch_size, dtype=torch.int64,
                           device=dataset['demand'].device)[:, None]
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand']
                                [:, :1], -CVRPTW.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            # This will reset/make capacity negative if i == 0, e.g. depot visited
            used_cap += d[:, i]
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRPTW.VEHICLE_CAPACITY +
                    1e-5).all(), "Used more than capacity"

        

        # Gather dataset in order of tour
        loc_with_depot = torch.cat(
            (dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(
            *pi.size(), loc_with_depot.size(-1)))
        
        # time window constriant check
        cur_time = torch.zeros_like(dataset['demand'][:, 0])        
        for i in range(1,pi.size(1)):            
            cur_node = pi[:, i].unsqueeze(-1)
            route_duration = (d[:, i] - d[:, i-1]).norm(p=2, dim=-1)
            cur_time += route_duration
            
            assert (cur_time <= dataset['leave_time'][ids, cur_node].squeeze(-1) +
                    1e-5).all(), "Out of time windows"
            
            cur_time = torch.max(
                cur_time, dataset['enter_time'][ids, cur_node].squeeze(-1))
            cur_time += torch.cat([torch.zeros_like(dataset['service_duration'][:, 0:1]),
                                   dataset['service_duration']], dim=1)[ids, cur_node].squeeze(-1)
            
            # reset to 0 when go back to depot
            cur_time = torch.where(
                pi[:, i] == 0, torch.zeros_like(cur_time), cur_time)
            

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            # Last to depot, will be 0 if depot is last
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)
        )*dataset['horizon'][0].item(), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRPTW.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


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
# pdb.set_trace()
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

    def __init__(self, size=50,
                 num_samples=1000000,
                 solomon=False,
                 solomon_train=False,
                 train_type="c1",
                 filename=None,
                 file_pre="./SolomonBenchmark/RC1",
                 test_filename="Solomon.txt",
                 *args,
                 **kwargs):
        super(VRPTWDataset, self).__init__()
        torch.manual_seed(1234)
        # pdb.set_trace()
        self.data = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = data
        elif solomon_train:
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
            for file in test_filename:
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
                    {'loc': torch.cat((x_random[1:, None], y_random[1:, None]), dim=-1)/leave_time_random[0:1],  # size,2
                     # 1,2
                     'depot': torch.cat((x_random[0:1], y_random[0:1]), dim=-1)/leave_time_random[0:1],
                     'demand': demand_random[1:],  # size
                     # size
                     'enter_time': enter_time_random/leave_time_random[0:1],
                     'leave_time': leave_time_random/leave_time_random[0:1],
                     'service_duration': service_duration[1:]/leave_time_random[0:1],
                     'bool': perceBool,
                     'horizon': leave_time_random[0:1]
                     }]
        if self.data == []:
            self.data = [
                {'loc': torch.cat((x[1:, None], y[1:, None]), dim=-1)/leave_time[0:1],  # size,2
                 # 2
                 'depot': torch.cat((x[0:1], y[0:1]), dim=-1)/leave_time[0:1],
                 'demand': demand[1:],  # size
                 'enter_time': torch.cat((enter_time[0:1], enter_time_all[i]), dim=0)/leave_time[0:1],
                 'leave_time': torch.cat((leave_time[0:1], leave_time_all[i]), dim=0)/leave_time[0:1],
                 'service_duration': service_duration/leave_time[0:1],
                 'bool': torch.cat((torch.zeros_like(leave_time[0:1]), perceBool[i]), dim=0),
                 'horizon': leave_time[0:1]
                 } for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
