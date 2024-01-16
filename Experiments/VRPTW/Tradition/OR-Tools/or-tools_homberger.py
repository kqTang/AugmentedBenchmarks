import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from itertools import groupby
from torch.utils.data import DataLoader
import numpy as np
# from vrp_data import make_dataset
from vrptwdataset import VRPTWDataset
import argparse
from tqdm import tqdm
import re
import pdb
from matplotlib import pyplot as plt
# from vrpPlot import plot_vehicle_routes


def create_data_model(dataset):
    """Stores the data for the problem."""
    data = {}

    graph_size = dataset['loc'].size(1)
    data["distance_matrix"] = (dataset['loc'].expand(graph_size, -1, -1) -
                               dataset['loc'].reshape(graph_size, 1, -1).expand(-1, graph_size, -1)).to(dtype=torch.float64).norm(p=2, dim=-1).numpy().astype(int).tolist()
    data["demands"] = dataset['demand'][0].numpy().tolist()
    # print(data["demands"])
    data["vehicle_capacities"] = [1000 for i in range(int(graph_size/5)+1)]
    data["num_vehicles"] = int(graph_size/5) + 1
    data["depot"] = 0
    return data


def get_routes(solution, routing, manager):
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes += route
    routes = [x[0] for x in groupby(routes)]
    return routes

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    # print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        # print(plan_output)
        total_distance += route_distance
        total_load += route_load
    return total_distance
    # print(f"Total distance of all routes: {total_distance}m")
    # print(f"Total load of all routes: {total_load}")



def main(opts):
    dataset = VRPTWDataset(file_pre=opts.file_pre, filename=opts.filename)
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    COST = []
    TIME = []
    j = 0
    with open('./figure/result.txt', 'a') as ff:
        for bat in tqdm(dataloader, disable=opts.no_progress_bar):
            bat['loc'] = (bat['loc']*1000).to(torch.long)
            bat['demand'] = (
                torch.cat((torch.zeros(1, 1), bat['demand']), dim=1)*1000).to(torch.long)
            bat['depot'] = (bat['depot']*1000).to(torch.long)
            bat_copy = {
                'loc': torch.cat((bat['depot'].unsqueeze(1), bat['loc']), dim=1),
                'demand': bat['demand'].clone()
            }
            """Solve the CVRP problem."""
            # Instantiate the data problem.
            data = create_data_model(bat_copy)

            # Create the routing index manager.
            manager = pywrapcp.RoutingIndexManager(
                len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
            )

            # Create Routing Model.
            routing = pywrapcp.RoutingModel(manager)

            # Create and register a transit callback.
            def distance_callback(from_index, to_index):
                """Returns the distance between the two nodes."""
                # Convert from routing variable Index to distance matrix NodeIndex.
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data["distance_matrix"][from_node][to_node]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)

            # Define cost of each arc.
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # Add Capacity constraint.
            def demand_callback(from_index):
                """Returns the demand of the node."""
                # Convert from routing variable Index to demands NodeIndex.
                from_node = manager.IndexToNode(from_index)
                return data["demands"][from_node]

            demand_callback_index = routing.RegisterUnaryTransitCallback(
                demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                data["vehicle_capacities"],  # vehicle maximum capacities
                True,  # start cumul to zero
                "Capacity",
            )

            # Setting first solution heuristic.
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.FromSeconds(1)

            # Solve the problem.
            solution = routing.SolveWithParameters(search_parameters)

            # Print solution on console.
            if solution:
                total_distance=print_solution(data, manager, routing, solution)
                routes=get_routes(solution, routing, manager)
                ff.write('{}\n'.format(
                    round(total_distance/1000, 2)))
                dataPlot = {
                    "loc":
                    bat['loc'][0]/1000,

                    "demand":
                    bat['demand'][0,1:]/1000,

                    "depot":
                    bat['depot'][0]/1000
                }
                # fig, ax = plt.subplots(figsize=(10, 10))
                # demand_scale = 1
                # fig.savefig(
                #     './figure/{}/{}.png'.format(opts.graph_size, j), dpi=300)
                # pdb.set_trace()
            j += 1
        ff.write('\n')
    ff.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, nargs='+', default=['Solomon.txt', 'Solomon2.txt', 'Solomon3.txt'], help="Name of the results file to write")
    parser.add_argument("--file_pre", type=str,  default='./data/RC1', help="file_pre")
    parser.add_argument('--val_size', type=int, default=10,
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
