from problems import VRP_Dataset
import torch
import pdb
import numpy as np
from problems.generate_solomon_train_data import generate_solomon_train_data

def get_all_solomon():
    x, y, d, e, l, dur = [], [], [], [], [], []
    File_pre = ['./SolomonBenchmark/C1/',
                './SolomonBenchmark/C2/',
                './SolomonBenchmark/R1/',
                './SolomonBenchmark/R2/',
                './SolomonBenchmark/RC1/',
                './SolomonBenchmark/RC2/']
    Test_filename = [
        ['Solomon1.txt', 'Solomon2.txt', 'Solomon3.txt', 'Solomon4.txt', 'Solomon5.txt',
         'Solomon6.txt', 'Solomon7.txt', 'Solomon8.txt', 'Solomon9.txt'],
        ['Solomon1.txt', 'Solomon2.txt', 'Solomon3.txt', 'Solomon4.txt',
         'Solomon5.txt', 'Solomon6.txt', 'Solomon7.txt', 'Solomon8.txt'],
        ['Solomon1.txt', 'Solomon2.txt', 'Solomon3.txt', 'Solomon4.txt', 'Solomon5.txt', 'Solomon6.txt',
         'Solomon7.txt', 'Solomon8.txt', 'Solomon9.txt', 'Solomon10.txt', 'Solomon11.txt', 'Solomon12.txt'],
        ['Solomon1.txt', 'Solomon2.txt', 'Solomon3.txt', 'Solomon4.txt', 'Solomon5.txt', 'Solomon6.txt',
         'Solomon7.txt', 'Solomon8.txt', 'Solomon9.txt', 'Solomon10.txt', 'Solomon11.txt'],
        ['Solomon1.txt', 'Solomon2.txt', 'Solomon3.txt', 'Solomon4.txt',
         'Solomon5.txt', 'Solomon6.txt', 'Solomon7.txt', 'Solomon8.txt'],
        ['Solomon1.txt', 'Solomon2.txt', 'Solomon3.txt', 'Solomon4.txt',
         'Solomon5.txt', 'Solomon6.txt', 'Solomon7.txt', 'Solomon8.txt']
    ]
    i = 0
    for type in ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']:

        file_pre = File_pre[i]
        test_filename = Test_filename[i]
        # pdb.set_trace()
        solomon = True
        solomon_train = False
        i += 1
        capacity = {
            'c1': 200.,
            'c2': 700.,
            'r1': 200.,
            'r2': 1000.,
            'rc1': 200.,
            'rc2': 1000.,
        }.get(type, None)
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
            x.append(x_random)
            y.append(y_random)
            d.append(demand_random)
            e.append(enter_time_random)
            l.append(leave_time_random)
            dur.append(service_duration)
    x = torch.stack(x)
    y = torch.stack(y)
    d = torch.stack(d)
    e = torch.stack(e)
    l = torch.stack(l)
    dur = torch.stack(dur)
    return x, y, d, e, l, dur

class VRPTW_Dataset(VRP_Dataset):
    CUST_FEAT_SIZE = 6

    @classmethod
    def generate_solomon_train(cls,
                         train_type='c1',
                         batch_size=56,
                         cust_count=100,
                         veh_count=25,
                         veh_capa=200,
                         veh_speed=1,
                         min_cust_count=None,
                         cust_loc_range=(0, 101),
                         cust_dem_range=(5, 41),
                         horizon=480,
                         cust_dur_range=(10, 31),
                         tw_ratio=0.5,  # (有的有时间窗，有的没有时间窗，这里是50%时间窗)
                         cust_tw_range=(30, 91)
                         ):
        
        x = None
        if train_type != 'all':
            x, y, d, e, l, dur = generate_solomon_train_data(
                train_type=train_type, num_samples=batch_size, size=cust_count)
        else:
            for train_type in ['c1', 'c2', 'r1', 'r2', 'rc1', 'rc2']:
                x_tmp, y_tmp, d_tmp, e_tmp, l_tmp, dur_tmp = generate_solomon_train_data(
                    train_type=train_type, num_samples=int(batch_size/6), size=cust_count)
                if x is None:
                    x, y, d, e, l, dur = x_tmp, y_tmp, d_tmp, e_tmp, l_tmp, dur_tmp
                else:
                    x, y, d, e, l, dur = torch.cat((x, x_tmp), 0), torch.cat((y, y_tmp)), torch.cat(
                        (d, d_tmp)), torch.cat((e, e_tmp)), torch.cat((l, l_tmp)), torch.cat((dur, dur_tmp))
        if x.shape[0]<batch_size:
            x, y, d, e, l, dur = torch.cat((x[0:batch_size-x.shape[0], ...], x)), torch.cat((y[0:batch_size-x.shape[0], ...],y)), torch.cat((d[0:batch_size-x.shape[0], ...],d)), torch.cat((e[0:batch_size-x.shape[0], ...],e)), torch.cat((l[0:batch_size-x.shape[0], ...],l)), torch.cat((dur[0:batch_size-x.shape[0], ...],dur))

        locs = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim=2)
        dems = d.unsqueeze(2) * veh_capa

        customers = torch.cat((locs[:, 1:], dems[:, 1:], e.unsqueeze(
            2)[:, 1:], l.unsqueeze(2)[:, 1:], dur[:, 1:].unsqueeze(2)), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:, :, :2] = locs[:, 0:1]
        depot_node[:, :, 4] = l[:, 0:1]
        # pdb.set_trace()
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(
                min_cust_count+1, cust_count+2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(
                cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset
    @classmethod
    def generate_solomon_test(cls,
                 batch_size=56,
                 cust_count=100,
                 veh_count=25,
                 veh_capa=200,
                 veh_speed=1,
                 min_cust_count=None,
                 cust_loc_range=(0, 101),
                 cust_dem_range=(5, 41),
                 horizon=480,
                 cust_dur_range=(10, 31),
                 tw_ratio=0.5,  # (有的有时间窗，有的没有时间窗，这里是50%时间窗)
                 cust_tw_range=(30, 91)
                 ):
        x, y, d, e, l, dur = get_all_solomon()
        locs = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), dim=2)
        dems = d.unsqueeze(2) * veh_capa
        # pdb.set_trace()
        customers = torch.cat((locs[:, 1:], dems[:,1:], e.unsqueeze(2)[:,1:], l.unsqueeze(2)[:,1:], dur[:,1:].unsqueeze(2)), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:, :, :2] = locs[:, 0:1]
        depot_node[:, :, 4] = l[:,0:1]
        # pdb.set_trace()
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(
                min_cust_count+1, cust_count+2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(
                cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset
    
    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 100,
            veh_count = 25,
            veh_capa = 200,
            veh_speed = 1,
            min_cust_count = None,
            cust_loc_range = (0,101),
            cust_dem_range = (5,41),
            horizon = 480,
            cust_dur_range = (10,31),
            tw_ratio = 0.5,#(有的有时间窗，有的没有时间窗，这里是50%时间窗)
            cust_tw_range = (30,91)
            ):
        size = (batch_size, cust_count, 1)

        # Sample locs        x_j, y_j ~ U(0, 100)
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype = torch.float)
        # Sample dems             q_j ~ U(5,  40)
        dems = torch.randint(*cust_dem_range, size, dtype = torch.float)
        # Sample serv. time       s_j ~ U(10, 30)
        durs = torch.randint(*cust_dur_range, size, dtype = torch.float)

        # Sample TW subset            ~ B(tw_ratio)
        if isinstance(tw_ratio, float):
            has_tw = torch.empty(size).bernoulli_(tw_ratio)
        elif len(tw_ratio) == 1:
            has_tw = torch.empty(size).bernoulli_(tw_ratio[0])
        else: # tuple of float
            ratio = torch.tensor(tw_ratio)[torch.randint(0, len(tw_ratio), (batch_size,), dtype = torch.int64)]
            has_tw = ratio[:,None,None].expand(*size).bernoulli()

        # Sample TW width        tw_j = H if not in TW subset
        #                        tw_j ~ U(30,90) if in TW subset
        tws = (1 - has_tw) * torch.full(size, horizon) \
                + has_tw * torch.randint(*cust_tw_range, size, dtype = torch.float)

        tts = (locs[:,None,0:1,:] - locs[:,1:,None,:]).pow(2).sum(-1).pow(0.5) / veh_speed
        # Sample ready time       e_j = 0 if not in TW subset
        #                         e_j ~ U(a_j, H - max(tt_0j + s_j, tw_j))
        rdys = has_tw * (torch.rand(size) * (horizon - torch.max(tts + durs, tws)))
        rdys.floor_()

        # Regroup all features in one tensor
        customers = torch.cat((locs[:,1:], dems, rdys, rdys + tws, durs), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        depot_node[:,:,4] = horizon
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is None:
            cust_mask = None
        else:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype = torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset

    def normalize(self):
        print(self.veh_capa)
        loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        loc_scl -= loc_off
        t_scl = self.nodes[:,0,4].max().item()
        self.loc_scl = loc_scl

        self.nodes[:,:,:2] -= loc_off
        self.nodes[:,:,:2] /= loc_scl
        self.nodes[:,:, 2] /= self.veh_capa
        self.nodes[:,:,3:] /= t_scl

        self.veh_capa = 1
        self.veh_speed *= t_scl / loc_scl

        return loc_scl, t_scl
