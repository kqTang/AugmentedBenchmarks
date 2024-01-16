

def test_epoch(args, test_env, learner):
    learner.eval()
    _, _, rs = learner(test_env)
    costs = -torch.stack(rs).sum(dim=0).squeeze(-1)
    for i in range(56):
        # import pdb
        # pdb.set_trace()
        costs[i] = costs[i]*test_env.loc_scl
        print(costs[i].item())
    mean = costs.mean()
    std = costs.std()

    print(
        "Cost on test dataset: {:5.2f} +- {:5.2f} ".format(mean, std))
    return mean.item(), std.item()

def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available()
                       and not args.no_cuda else "cpu")
    Environment = {
        "vrp": VRP_Environment,
        "vrptw": VRPTW_Environment,
        "svrptw": SVRPTW_Environment,
        "sdvrptw": SDVRPTW_Environment
    }.get(args.problem_type)
    env_params = [args.pending_cost]
    gen_params = [
        args.customers_count,
        args.vehicles_count,
        args.veh_capa,
        args.veh_speed,
        args.min_cust_count,
        args.loc_range,
        args.dem_range
    ]
    
    Dataset = {
        "vrp": VRP_Dataset,
        "vrptw": VRPTW_Dataset,
        "svrptw": VRPTW_Dataset,
        "sdvrptw": SDVRPTW_Dataset
    }.get(args.problem_type)
    test_data = Dataset.generate_solomon_test(
        56,
        *gen_params
    )
    test_data.normalize()
    test_env = Environment(test_data, None, None, *env_params)
    test_env.nodes = test_env.nodes.to(dev)
    learner = AttentionLearner(
        Dataset.CUST_FEAT_SIZE,
        Environment.VEH_STATE_SIZE,
        args.model_size,
        args.layer_count,
        args.head_count,
        args.ff_size,
        args.tanh_xplor,
        args.input_type
    )
    
    learner.to(dev)
    checkpoint = torch.load(args.load_path)
    learner.load_state_dict(checkpoint["model"])
    test_epoch(args, test_env, learner)

if __name__ == "__main__":
    import os
    import sys
    o_path = os.getcwd()
    sys.path.append(o_path)
    from utils import *
    from dep import *
    from externals import *
    from baselines import *
    from problems import *
    from layers import reinforce_loss
    import torch
    from _learner import AttentionLearner
    main(parse_args())
