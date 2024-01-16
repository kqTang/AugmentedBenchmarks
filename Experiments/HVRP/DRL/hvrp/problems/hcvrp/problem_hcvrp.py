from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from problems.hcvrp.state_hcvrp import StateHCVRP
from utils.beam_search import beam_search
import copy
import random
import scipy.stats as st
import pdb
import time
class HCVRP(object):
    NAME = 'hcvrp'  # Capacitated Vehicle Routing Problem

    # VEHICLE_CAPACITY = [20., 25., 30., 35., 40.]
    @staticmethod
    def get_costs(dataset, obj, pi, veh_list, tours, instances, num_veh):  # pi is a solution sequence, [batch_size, num_veh, tour_len]

        batch_size, graph_size = dataset['demand'].size()
        # num_veh = len(HCVRP.VEHICLE_CAPACITY)

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
            tour = torch.cat([t.unsqueeze(1) for t in tours], dim=1) # batch_size, len(tours[0]), tours[0].size(-1)
            # tour = torch.cat((torch.zeros_like(tour[...,0:1]),tour,torch.zeros_like(tour[...,0:1])),dim=-1)
            # import pdb
            # pdb.set_trace()
            used_cap[tour[...,i]==0] = 0
            
            # if not (used_cap <= dataset['capacity'] + 1e-5).all():
            #     
            assert (used_cap <= dataset['capacity'] + 1e-5).all(), "Used more than capacity"
            
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

        # Gather dataset in order of tour
        # (batch_size, graph_size, 2)
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)  # batch_size, graph_size+1, 2
        
        # (batch_size, len(tours), graph_size+1, 2)
        dis = loc_with_depot.unsqueeze(1).expand(batch_size, len(tours), graph_size + 1, 2).gather(2,tour.unsqueeze(-1).expand(*tour.size(), 2)) # batch_size, len(tours), tour.size(-1), 2
        # dataset['depot']= [batch_size, 2]
        total_dis = (dis[:,:, 1:] - dis[:,:, :-1]).norm(p=2, dim=-1).sum(-1)

        tour_d = tour[...,:-1] - tour[...,1:]
        if instances != 'Taillard':
            # (number of zeros in tour - 1) * fixed cost
            fixed_cost=((((tour[...,:-1] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1)) * dataset['fixed_cost'])
            # print(fixed_cost)
        if instances=='Gloden': #if in Taillrd, use number and a_cost; elif in Choia use fixedcost and a_cost, else in Gloden, use only fixed cost
            return torch.sum(total_dis+fixed_cost, dim=1), None
        elif instances=='Choia&Tcha':                
            return torch.sum(total_dis*dataset['a_cost']+fixed_cost, dim=1), None
        elif instances=='Taillard':                
            
            number_limit = dataset['number']  # Number limit of each vehicle type
            used_number = ((tour[..., 1:] == 0) & (tour_d != 0)).to(dtype=torch.bool).sum(-1)  # Used number of each vehicle type
            exceed_number = torch.clamp(used_number - number_limit, min=0)  # Compute exceed number
            exceed_penalty = exceed_number * 100  # Exceed number times 100
            return torch.sum(total_dis*dataset['a_cost'] + exceed_penalty, dim=1), None  # Added exceed_penalty to the first return

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


def make_instance(args,instances):
    if instances == 'Choia&Tcha':        
        veh = args['veh']
        loc = args['loc']
        depot = args['depot']
        demand = args['q']
        # pdb.set_trace()
        return {
                'loc': torch.Tensor(loc),
                # Uniform 1 - 9, scaled by capacities
                'demand': torch.Tensor(demand),
                'depot': torch.Tensor(depot),
                'capacity': torch.Tensor([veh_i[1] for veh_i in veh] ),
                'fixed_cost': torch.Tensor([veh_i[2] for veh_i in veh] ),
                'a_cost': torch.Tensor([veh_i[3] for veh_i in veh] ),
            }
    elif instances == 'Gloden':
        veh = args['veh']
        loc = args['loc']
        depot = args['depot']
        demand = args['q']        
        return {
                'loc': torch.Tensor(loc),
                # Uniform 1 - 9, scaled by capacities
                'demand': torch.Tensor(demand),
                'depot': torch.Tensor(depot),
                'capacity': torch.Tensor([veh_i[1] for veh_i in veh] ),
                'fixed_cost': torch.Tensor([veh_i[2] for veh_i in veh] )
            }
    if instances == 'Taillard':        
        veh = args['veh']
        loc = args['loc']
        depot = args['depot']
        demand = args['q']
        # pdb.set_trace()
        return {
                'loc': torch.Tensor(loc),
                # Uniform 1 - 9, scaled by capacities
                'demand': torch.Tensor(demand),
                'depot': torch.Tensor(depot),
                'capacity': torch.Tensor([veh_i[1] for veh_i in veh] ),
                'number': torch.Tensor([veh_i[0] for veh_i in veh] ),
                'a_cost': torch.Tensor([veh_i[3] for veh_i in veh] )
            }
                


class HCVRPDataset(Dataset):

    def __init__(self, filename=None, size=50, instances='Gloden',num_veh=3,num_samples=10000, offset=0, distribution=None):
        super(HCVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                Dataset = pickle.load(f)  # (N, size+1, 2)
            # pdb.set_trace()
            self.data = [make_instance(args,instances) for args in Dataset[offset:offset + num_samples]]
            

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                20: {5:[20., 30., 40., 70., 120.],
                     3:[60., 80., 150.]},
                50: {6:[20., 30., 40., 70., 120.,200.],
                     3:{
                     1:[120., 160., 300.],
                     2:[50., 100., 160.],
                     3:[40., 80., 140.]}
                     },
                75: {4:[50., 120., 200., 350.],
                     6:[20., 50., 100., 150., 250., 400.]},
                100: {3:{
                        0:[100., 200., 300.],
                        1:[60., 140., 200.]}}
            }
            COST = {
                20: {5:[20., 35., 50., 120., 225.],
                     3:[1000., 1500., 3000.]},
                50: {6:[20., 35., 50., 120., 225.,400.],
                     3:{
                     1:[1000., 1500., 3500.],
                     2:[100., 250., 450.],
                     3:[100., 200., 400.]
                     }
                     },
                75: {4:[25., 80., 150., 350.],
                     6:[10., 35., 100., 180., 400., 800.]},
                100: {3:{
                        0:[500., 1200., 2100.],
                        1:[100., 300., 500.]}}
            }
            a_COST = {
                20: {5:[1., 1.1, 1.2, 1.7, 2.5],
                     3:[1., 1.1, 1.4]
                     },
                50: {6:[1., 1.1, 1.2, 1.7, 2.5, 3.2],
                     3:{
                            1:[1., 1.1, 1.4],
                            2:[1., 1.6, 2.],
                            3:[1., 1.6, 2.1]}
                     },
                75: {4:[1., 1.2, 1.5, 1.8],
                     6:[1., 1.3, 1.9, 2.4, 2.9, 3.2]},
                100: {3:{
                        0:[1., 1.4, 1.7],
                        1:[1., 1.7, 2.]}}
            }
            NUMBER = {
                50: {6:[4, 2, 4, 4, 2, 1],
                     3:{
                            1:[4, 2, 1],
                            2:[4, 3, 2],
                            3:[2, 4, 3]
                     }
                     },
                75: {4:[4, 4, 2, 1],
                     6:[4, 4, 2, 2, 1, 1]},
                100: {3:{
                        0:[4, 3, 3],
                        1:[6, 4, 3]
                     }}
            }            
            item = {
                50:[1,2,3],
                100:[0,1]
            }.get(size)
            # capa = torch.zeros((size, CAPACITIES[size]))
            depot_list = [20., 30., 35., 40.]
            DEMAND = torch.from_numpy(st.beta.rvs(a=2.0055793317300266, b=4.169106615704806, loc=0.04861715821901606, scale=49.075732304600734,
                                                            size=[num_samples, size])).to(torch.float)
            X = torch.from_numpy(st.uniform.rvs(loc=2, scale=68, size=[num_samples, size])).to(torch.float)
            Y = torch.from_numpy(st.uniform.rvs(loc=3, scale=74, size=[num_samples, size])).to(torch.float)
            if instances == 'Gloden':
                if (size not in [50,100]) | (num_veh==6):
                    self.data = [
                        {
                            'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                            # Uniform 1 - 9, scaled by capacities
                            'demand': DEMAND[i],
                            'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                            'capacity': torch.Tensor(CAPACITIES[size][num_veh]),
                            'fixed_cost': torch.Tensor(COST[size][num_veh])
                        }
                        for i in range(num_samples)
                    ]
                else:
                    self.data = [
                        {
                            'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                            # Uniform 1 - 9, scaled by capacities
                            'demand': DEMAND[i],
                            'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                            'capacity': torch.Tensor(CAPACITIES[size][num_veh][random.choice(item)]),
                            'fixed_cost': torch.Tensor(COST[size][num_veh][random.choice(item)])
                        }
                        for i in range(num_samples)
                    ]                    
            elif instances == 'Choia&Tcha':
                if (size not in [50,100]) | (num_veh==6):
                    self.data = [
                        {
                            'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                            # Uniform 1 - 9, scaled by capacities
                            'demand': DEMAND[i],
                            'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                            'capacity': torch.Tensor(CAPACITIES[size][num_veh]),
                            'fixed_cost': torch.Tensor(COST[size][num_veh]),
                            'a_cost': torch.Tensor(a_COST[size][num_veh]),
                        }
                        for i in range(num_samples)
                    ]
                else:
                    self.data = [
                        {
                            'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                            # Uniform 1 - 9, scaled by capacities
                            'demand': DEMAND[i],
                            'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                            'capacity': torch.Tensor(CAPACITIES[size][num_veh][random.choice(item)]),
                            'fixed_cost': torch.Tensor(COST[size][num_veh][random.choice(item)]),
                            'a_cost': torch.Tensor(a_COST[size][num_veh][random.choice(item)]),
                        }
                        for i in range(num_samples)
                    ]                    
            elif instances == 'Taillard':
                if (size not in [50,100]) | (num_veh==6):
                    # pdb.set_trace()
                    self.data = [
                        {
                            'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                            # Uniform 1 - 9, scaled by capacities
                            'demand': DEMAND[i],
                            'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                            'capacity': torch.Tensor(CAPACITIES[size][num_veh]),
                            'number': torch.Tensor(NUMBER[size][num_veh]),
                            'a_cost': torch.Tensor(a_COST[size][num_veh]),
                        }
                        for i in range(num_samples)
                    ]
                else:
                    self.data = [
                        {
                            'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                            # Uniform 1 - 9, scaled by capacities
                            'demand': DEMAND[i],
                            'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                            'capacity': torch.Tensor(CAPACITIES[size][num_veh][random.choice(item)]),
                            'number': torch.Tensor(NUMBER[size][num_veh][random.choice(item)]),
                            'a_cost': torch.Tensor(a_COST[size][num_veh][random.choice(item)]),
                        }
                        for i in range(num_samples)
                    ]                
            elif instances == 'DLP':
                # pdb.set_trace()
                DLPDataset = torch.load('./data/hcvrp/DLP.pt')
                customer_key = DLPDataset['loc_dict'].keys()
                veh_key = DLPDataset['veh_dict'].keys() #对应顾客总需求
                
                self.data = []
                start_time = time.time()
                for i in range(num_samples):
                    customer_choice = random.choice([key for key in customer_key if size  < key ])
                    # 顾客点序号
                    numbers_selected = random.sample(range(1, int(customer_choice) + 1), size)
                    pdb.set_trace()
                    selected_demands = np.array(DLPDataset['q_dict'][customer_choice])[np.array(numbers_selected, dtype=int) - 1]
                    indices = [0] + numbers_selected
                    # 选择距离矩阵
                    selected_loc = DLPDataset['loc_dict'][customer_choice][np.ix_(indices, indices)]
                    
                    # 选择车辆
                    sum(selected_demands)
                    selected_demands = DLPDataset['q_dict'][customer_choice][numbers_selected]
                    
                    veh_choice = random.choice([key for key in veh_key if sum(selected_demands)/2  < key < 2*sum(selected_demands) ])
                    selected_veh = DLPDataset['veh_dict'][veh_choice]
                    self.data.append(
                        {
                        'distance': torch.tensor(selected_loc),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': torch.tensor(selected_demands),
                        'number': torch.Tensor([veh[0] for veh in selected_veh] + [0] * (6 - len(selected_veh))),
                        'capacity': torch.Tensor([veh[1] for veh in selected_veh] + [0] * (6 - len(selected_veh))),                        
                        'fixed_cost': torch.Tensor([veh[2] for veh in selected_veh] + [0] * (6 - len(selected_veh))),                        
                        'a_cost': torch.Tensor([veh[3] for veh in selected_veh] + [0] * (6 - len(selected_veh))),
                        }
                    )
                during_time = time.time() - start_time
                print(num_samples,during_time)
                pdb.set_trace()
                self.data = [
                    {
                        'loc': torch.cat([X[i].unsqueeze(1),Y[i].unsqueeze(1)],dim=1),
                        # Uniform 1 - 9, scaled by capacities
                        'demand': DEMAND[i],
                        'depot': torch.tensor([random.choice(depot_list),random.choice(depot_list)]),
                        'capacity': torch.Tensor(CAPACITIES[size][random.choice(item)]),
                        'number': torch.Tensor(NUMBER[size][random.choice(item)]),
                        'a_cost': torch.Tensor(a_COST[size][random.choice(item)]),
                    }
                    for i in range(num_samples)
                ]
        self.len = len(self.data)  # num_samples
        # self.size = len(self.data)  # num_samples
     
    def __len__(self):
        return self.len
        # return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # index of sampled data


