import os
import math
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import re
# Change with the Cost Matrix of your problem or
# consider using it as an argument
fname_tsp = "test"
user_comment = "a comment by the user"

# Change these directories based on where you have
# a compiled executable of the LKH TSP Solver
tsplib_dir = '/TSPLIB/'
lkh_cmd = 'LKH'
pwd = os.path.dirname(os.path.abspath(__file__))


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args)
                         for args in data[offset:offset+num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.,
                200: 70.
            }

            self.data = [
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def make_dataset(*args, **kwargs):
	return VRPDataset(*args, **kwargs)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


def writeTSPLIBfile_FE(opts, fname_tsp, coordinate, demand, user_comment):

	name_line = 'NAME : ' + fname_tsp + '\n'
	# type_line = 'TYPE: VRP' + '\n'
	comment_line = 'COMMENT : ' + user_comment + '\n'
	tsp_line = 'TYPE : ' + 'CVRP' + '\n'
	dimension_line = 'DIMENSION : ' + str(opts.graph_size+1) + '\n'
	edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EUC_2D' + '\n'
	CAPACITIES = {
            10: 200,
            20: 300,
            50: 400,
            100: 500,
            200: 700
        }.get(opts.graph_size, None)
	capacity = 'CAPACITY : ' + str(CAPACITIES) + '\n'
	node_coord_section = 'NODE_COORD_SECTION' + '\n'
	eof_line = 'EOF\n'
	Cost_Matrix_STRline = []
	for i in range(0, opts.graph_size+1):
		cost_matrix_strline = str(i+1) + ' ' + str(int(coordinate[i, 0].numpy(
		)*1000)) + ' ' + str(int(coordinate[i, 1].numpy()*1000)) + '\n'
		Cost_Matrix_STRline.append(cost_matrix_strline)
	Demand_Matrix_STRline = []
	demand_matrix_strline = '1' + ' ' + '0' + '\n'
	Demand_Matrix_STRline.append(demand_matrix_strline)
	for i in range(0, opts.graph_size):
		demand_matrix_strline = str(i+2) + ' ' + \
            str(int(demand[0, i].numpy()*CAPACITIES)) + '\n'
	Demand_Matrix_STRline.append(demand_matrix_strline)
	fileID = open((pwd + tsplib_dir + fname_tsp + '.vrp'), "w")
	# print(name_line)
	fileID.write(name_line)
	fileID.write(comment_line)
	fileID.write(tsp_line)
	fileID.write(dimension_line)
	fileID.write(edge_weight_type_line)
	fileID.write(capacity)
	fileID.write(node_coord_section)
	for i in range(0, len(Cost_Matrix_STRline)):
		fileID.write(Cost_Matrix_STRline[i])
	fileID.write('DEMAND_SECTION' + '\n')
	for i in range(0, len(Demand_Matrix_STRline)):
		fileID.write(Demand_Matrix_STRline[i])
	fileID.write('DEPOT_SECTION' + '\n' + ' ' + '1' + '\n' + ' ' + '-1' + '\n')
	fileID.write(eof_line)
	fileID.close()

	# fileID2 = open((pwd + tsplib_dir + fname_tsp + '.par'), "w")

	# problem_file_line = 'PROBLEM_FILE = ' + pwd + tsplib_dir + fname_tsp + '.vrp' + '\n' # remove pwd + tsplib_dir
	# optimum_line = 'OPTIMUM = 378032' + '\n'
	# move_type_line = 'MOVE_TYPE = 5' + '\n'
	# patching_c_line = 'PATCHING_C = 3' + '\n'
	# patching_a_line = 'PATCHING_A = 2' + '\n'
	# runs_line = 'RUNS = 10' + '\n'
	# tour_file_line = 'TOUR_FILE = ' + fname_tsp + '.txt' + '\n'

	# fileID2.write(problem_file_line)
	# fileID2.write(optimum_line)
	# fileID2.write(move_type_line)
	# fileID2.write(patching_c_line)
	# fileID2.write(patching_a_line)
	# fileID2.write(runs_line)
	# fileID2.write(tour_file_line)
	# fileID2.close()
	return fileID


def main(opts):

	dataset = make_dataset(filename=opts.dataset_path,
	                       num_samples=opts.val_size, offset=opts.offset)
	dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
	COST = []
	TIME = []
	for batch in tqdm(dataloader, disable=opts.no_progress_bar):
		#1，整合depot+loc
		coordinate = torch.cat((batch['depot'], batch['loc'].squeeze(0)), dim=0)
		demand = batch['demand']
		#2，生成par和vrp文件
		writeTSPLIBfile_FE(opts, fname_tsp, coordinate, demand, user_comment)
		#3，调用LKH命令，并收集 cost 和 time 数据
		f = os.popen('./LKH ./TSPLIB/test.par', 'r')
		lines = f.read()
		cost = re.findall("Cost.min = (\d+(?:.\d+)?)", lines)
		time = re.findall("Time.total = (\d+(?:.\d+)?)", lines)
		COST.append(int(cost[0]))
		# print(cost[0])
		TIME.append(float(time[0]))
	Cost_mean = sum(COST)/len(COST)/1000.
	Time_total = sum(TIME)
	print('Cost_mean: ', Cost_mean, '    ', 'Time_total: ', Time_total)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", type=str, default="./vrp/vrp20_test_seed1234.pkl",
	                    help="Filename of the dataset(s) to evaluate")
	parser.add_argument('--val_size', type=int, default=1000,
	                    help='Number of instances used for reporting validation performance')
	parser.add_argument('--graph_size', type=int, default=20,
	                    help="The size of the problem graph")
	parser.add_argument('--offset', type=int, default=0,
	                    help='Offset where to start in dataset (default 0)')
	parser.add_argument('--eval_batch_size', type=int, default=1,
	                    help="Batch size to use during (baseline) evaluation")
	parser.add_argument('--no_progress_bar', action='store_true',
	                    help='Disable progress bar')
	opts = parser.parse_args()
	main(opts)
