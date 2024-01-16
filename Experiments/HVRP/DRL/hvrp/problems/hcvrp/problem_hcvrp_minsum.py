from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from problems.hcvrp.state_hcvrp import StateHCVRP
from utils.beam_search import beam_search
import copy


class HCVRP(object):
    NAME = 'hcvrp'  # Capacitated Vehicle Routing Problem

    # VEHICLE_CAPACITY = [20., 25., 30., 35., 40.]

    @staticmethod
    def get_costs(dataset, obj, pi, veh_list, tours):  # pi is a solution sequence, [batch_size, num_veh, tour_len]
        if obj == 'min-max':
            SPEED = [1, 1, 1, 1, 1, 1]
        if obj == 'min-sum':
            # SPEED = [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]
            SPEED = [1, 1, 1, 1, 1, 1]

        batch_size, graph_size = dataset['demand'].size()
        # num_veh = len(HCVRP.VEHICLE_CAPACITY)
        num_veh = 6

        # # Check that tours are valid, i.e. contain 0 to n -1, [batch_size, num_veh, tour_len]
        sorted_pi = pi.data.sort(1)[0]
        # Sorting it should give all zeros at front and then 1...n
        assert (torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
                sorted_pi[:, -graph_size:]
                ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        demand_with_depot = torch.cat(  # [batch_size, graph_size]
            (
                torch.full_like(dataset['demand'][:, :1], 0),  # pickup problem, set depot demand to -capacity
                dataset['demand']
            ),
            1
        )
        # pi: [batch_size, tour_len]
        d = demand_with_depot.gather(1, pi)

        used_cap = torch.zeros_like(dataset['demand'][:, 0:num_veh])  # batch_size, 3

        for i in range(pi.size(-1)):  # tour_len
            used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] += d[:,
                                                                                         i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            used_cap[used_cap[torch.arange(batch_size), veh_list[torch.arange(batch_size), i]] < 0] = 0
            # used_cap[(tour_1[:, i] == 0), 0] = 0
            # assert (used_cap[torch.arange(batch_size), 0] <=
            #         HCVRP.VEHICLE_CAPACITY[0] + 1e-5).all(), "Used more than capacity 1"
            # used_cap[(tour_2[:, i] == 0), 1] = 0
            # assert (used_cap[torch.arange(batch_size), 1] <=
            #         HCVRP.VEHICLE_CAPACITY[1] + 1e-5).all(), "Used more than capacity 2"
            # used_cap[(tour_3[:, i] == 0), 2] = 0
            # assert (used_cap[torch.arange(batch_size), 2] <=
            #         HCVRP.VEHICLE_CAPACITY[2] + 1e-5).all(), "Used more than capacity 3"
            # used_cap[(tour_4[:, i] == 0), 3] = 0
            # assert (used_cap[torch.arange(batch_size), 3] <=
            #         HCVRP.VEHICLE_CAPACITY[3] + 1e-5).all(), "Used more than capacity 4"
            # used_cap[(tour_5[:, i] == 0), 4] = 0
            # assert (used_cap[torch.arange(batch_size), 4] <=
            #         HCVRP.VEHICLE_CAPACITY[4] + 1e-5).all(), "Used more than capacity 5"
            assert (used_cap <= dataset['capacity'] + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        # (batch_size, graph_size, 2)
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # batch_size, graph_size+1, 2

        dis = [loc_with_depot.gather(1, tours[i][..., None].expand(*tours[i].size(), loc_with_depot.size(-1))) for i in range(len(tours))]
        total_dis = [(((dis[i][:, 1:] - dis[i][:, :-1]).norm(p=2, dim=2).sum(1) + (dis[i][:, 0] - dataset['depot']).norm(p=2, dim=1) + (dis[i][:, -1] - dataset['depot']).norm(p=2, dim=1)) / SPEED[0]).unsqueeze(-1) for i in range(len(tours))]

        total_dis = torch.stack(total_dis, -1)

        if obj == 'min-max':
            return torch.max(total_dis, dim=1)[0], None
        if obj == 'min-sum':
            return torch.sum(total_dis, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = HCVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),  # scale demand
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'capacity': torch.tensor(capacity, dtype=torch.float)
    }


class HCVRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None):
        super(HCVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)  # (N, size+1, 2)

            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                20: {0:[20., 30., 40., 70., 120., 0.],
                     1:[60., 80., 150., 0., 0., 0.]},
                50: {0:[20., 30., 40., 70., 120.,200.],
                     1:[120., 160., 300., 0., 0., 0.],
                     2:[50., 100., 160., 0., 0., 0.],
                     3:[40., 80., 140., 0., 0., 0.],
                     },
                75: {0:[50., 120., 200., 350., 0., 0.],
                     1:[20., 50., 100., 150., 250., 400.]},
                100: {0:[100., 200., 300., 0., 0., 0.],
                     1:[60., 140., 200., 0., 0., 0.]}
            }
            COST = {
                20: {0:[20., 35., 50., 120., 225., 0.],
                     1:[1000., 1500., 3000., 0., 0., 0.]},
                50: {0:[20., 35., 50., 120., 225.,400.],
                     1:[1000., 1500., 3500., 0., 0., 0.],
                     2:[100., 250., 450., 0., 0., 0.],
                     3:[100., 200., 400., 0., 0., 0.],
                     },
                75: {0:[25., 80., 150., 350., 0., 0.],
                     1:[10., 35., 100., 180., 400., 800.]},
                100: {0:[500., 1200., 2100., 0., 0., 0.],
                     1:[100., 300., 500., 0., 0., 0.]}
            }
            item = {
                20:[0,1],
                50:[0,1,2,3],
                75:[0,1],
                100:[0,1]
            }.get(size)
            # capa = torch.zeros((size, CAPACITIES[size]))
            depot_list = [20., 30., 35., 40.]
            DEMAND = torch.from_numpy(st.beta.rvs(a=2.0055793317300266, b=4.169106615704806, loc=0.04861715821901606, scale=49.075732304600734,
                                                            size=[num_samples, size])).to(torch.float)
            X = torch.from_numpy(st.uniform.rvs(loc=2, scale=68, size=[num_samples, size])).to(torch.float)
            Y = torch.from_numpy(st.uniform.rvs(loc=3, scale=74, size=[num_samples, size])).to(torch.float)
            
            self.data = [
                {
                    'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': DEMAND[i],
                    'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                    'capacity': torch.Tensor(CAPACITIES[size][random.choice(item)]),
                    'cost': torch.Tensor(COST[size][random.choice(item)])
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)  # num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data



