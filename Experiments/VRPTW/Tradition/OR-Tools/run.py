"""Vehicles Routing Problem (VRP) with Time Windows."""

from vrptwdataset import VRPTWDataset
import time
from torch.utils.data import Dataset, DataLoader
import argparse
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import pdb
from tqdm import tqdm
def distance(a, b):
    # pdb.set_trace()
    return int(math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)) 


def create_data_model(batch, capacity, service_time,size):
    """Stores the data for the problem."""
    size = batch['loc'].size(-2)
    data = {}

    data['time_matrix'] = [[distance(coordinate[i], coordinate[j]) if (i*j*(i-j)) == 0 else distance(coordinate[i], coordinate[j]) + service_time
        for j in range(coordinate.size(0))] 
        for i in range(coordinate.size(0))]

    data['time_windows'] = [(int(batch['enter_time'][0,i].numpy()), int(batch['leave_time'][0,i].numpy())) for i in range(size)]

    data['demands'] = [int(batch['demands'][0,i].numpy()) for i in range(batch['demands'].size(-1))]

    data['num_vehicles'] = {
        200:50,
        400:100,
        600:150,
        1000:250
    }.get(size,None)
    
    data['vehicle_capacities'] = [
        capacity for i in range(data['num_vehicles'])]

    data['depot'] = 0
    return data


def print_solution(data, manager, routing, solution, service_time, solomon=False):
    """Prints solution on console."""
    # if solomon:
    # pdb.set_trace()
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    pi=[]
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = service_time * 2
        route_load = 0
        while not routing.IsEnd(index):
            pi.append(manager.IndexToNode(index))
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id) - service_time
        pi.append(0)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        total_distance += route_distance
        total_load += route_load
    # if solomon:
    # print('Total distance of all routes: {}m'.format(total_distance))
    #     print('Total load of all routes: {}'.format(total_load))
    return pi


# def print_solution(data, manager, routing, solution):
#     """Prints solution on console."""
#     print(f"Objective: {solution.ObjectiveValue()}")
#     time_dimension = routing.GetDimensionOrDie("Time")
#     total_time = 0
#     pi=[]
#     for vehicle_id in range(data["num_vehicles"]):
#         index = routing.Start(vehicle_id)
#         plan_output = f"Route for vehicle {vehicle_id}:\n"
#         while not routing.IsEnd(index):
#             time_var = time_dimension.CumulVar(index)
#             pi.append(manager.IndexToNode(index))
#             plan_output += (
#                 f"{manager.IndexToNode(index)}"
#                 f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
#                 " -> "
#             )
#             index = solution.Value(routing.NextVar(index))
#         time_var = time_dimension.CumulVar(index)
#         plan_output += (
#             f"{manager.IndexToNode(index)}"
#             f" Time({solution.Min(time_var)},{solution.Max(time_var)})\n"
#         )
#         plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
#         print(plan_output)
#         total_time += solution.Min(time_var)
#     print(f"Total time of all routes: {total_time}min")
#     return pi

def main(batch, capacity, service_time, solomon=False,size=100):
    """Solve the VRP with time windows."""
    # Instantiate the data problem.
    # pdb.set_trace()
    data = create_data_model(batch, capacity, service_time,size)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
        
    # Add Time Windows constraint.
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        300000,  # allow waiting time
        300000,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == data['depot']:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
    # Add time window constraints for each vehicle start node.
    depot_idx = data['depot']
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data['time_windows'][depot_idx][0],
            data['time_windows'][depot_idx][1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        return print_solution(data, manager, routing, solution, service_time, solomon)

def split(var, size):
    var['loc']=var['loc'][:,:-size,:]
    var['demand'] = var['demand'][:,:-size]
    var['enter_time'] = var['enter_time'][:, :-size]
    var['leave_time'] = var['leave_time'][:, :-size]
    var['service_duration'] = var['service_duration'][:, :-size]
    return var
def get_cost(loc, pi):
    d = loc.gather(dim=0, index=pi.unsqueeze(-1).repeat(1,2))
    return (d[1:,:]-d[:-1,:]).norm(p=2,dim=-1).sum(-1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--filename", type=str, nargs='+', default=[
                        'Solomon1.txt'], help="Name of the results file to write")
    parser.add_argument("--file_pre", type=str,
                        default='./data/C1/', help="file_pre")
    parser.add_argument("--solomon_train_type", type=str,
                        default='c1', help="which type of solomon benchmark to test")
    parser.add_argument("--dataset_path", type=str, default="./vrp/vrp100_test_seed1234.pkl",
                        help="Filename of the dataset(s) to evaluate")
    parser.add_argument("--solomon", action='store_true',
                        help="test solomon benchmark or not")
    parser.add_argument("--size", type=int,
                        default=100, help="file_pre")
    opts = parser.parse_args()

    training_dataset = VRPTWDataset(
        file_pre=opts.file_pre,
        filename=opts.filename)

    COST = []
    TIME = []
    total_distance = 0
    start = time.time()
    num_files = len(opts.filename)
    capacity = 200
    service_time = 90
    cost = []
    for batch in tqdm(DataLoader(training_dataset, batch_size=1, shuffle=False)):
        # if not opts.solomon:
        #     if opts.size<100:
        #         batch = split(batch, 100-opts.size)
        #     batch['demands'] = torch.cat((torch.zeros(1,1),batch['demand']),dim=1)*700
        coordinate = torch.cat(
            (batch['depot'], batch['loc'].squeeze(0)), dim=-2)
        # print(coordinate.transpose(-1, -2))
        pi = main(batch, capacity, service_time, size=opts.size)
        pi=torch.tensor(pi)
        cost.append(get_cost(coordinate, pi).item())
        # pdb.set_trace()
        
    # if not opts.solomon:
    print("cost :", cost)
