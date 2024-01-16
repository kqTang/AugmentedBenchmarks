from argparse import ArgumentParser
import sys
import json

CONFIG_FILE = None
VERBOSE = False
NO_CUDA = False
SEED = None

PROBLEM = "vrptw"
CUST_COUNT = 100
VEH_COUNT = 25
VEH_CAPA = 200
VEH_SPEED = 1
HORIZON = 480
MIN_CUST_COUNT = None
LOC_RANGE = (0,101)
DEM_RANGE = (5,41)
DUR_RANGE = (10,31)
TW_RATIO = (0.25,0.5,0.75,1.0)
TW_RANGE = (30,91)
DEG_OF_DYN = (0.1,0.25,0.5,0.75)
APPEAR_EARLY_RATIO = (0.0,0.5,0.75,1.0)

PEND_COST = 2
PEND_GROWTH = None
LATE_COST = 1
LATE_GROWTH = None
SPEED_VAR = 0.1
LATE_PROB = 0.05
SLOW_DOWN = 0.5
LATE_VAR = 0.2

MODEL_SIZE = 128
LAYER_COUNT = 3
HEAD_COUNT = 8
FF_SIZE = 512
TANH_XPLOR = 10

EPOCH_COUNT = 100
ITER_COUNT = 1000
MINIBATCH_SIZE = 200
# MINIBATCH_SIZE = 640
BASE_LR = 0.0001
LR_DECAY = None
MAX_GRAD_NORM = 2
GRAD_NORM_DECAY = None
LOSS_USE_CUMUL = False

BASELINE = "critic"
ROLLOUT_COUNT = 3
ROLLOUT_THRESHOLD = 0.05
CRITIC_USE_QVAL = False
CRITIC_LR = 0.001
CRITIC_DECAY = None

TEST_BATCH_SIZE = 128

OUTPUT_DIR = None
RESUME_STATE = None
CHECKPOINT_PERIOD = 5


def write_config_file(args, output_file):
    with open(output_file, 'w') as f:
        json.dump(vars(args), f, indent = 4)


def parse_args(argv = None):
    parser = ArgumentParser()
    #     
    parser.add_argument("--input_type",  type=str, default='origin')
    parser.add_argument("--train_type",  type=str, default='r1')
    parser.add_argument("--use_tool_to_test_solomon",
                        action="store_true", default=False)
    parser.add_argument("--LKH_ENABLED",
                        action="store_true", default=False)
    parser.add_argument("--use_solomon_train",
                        action="store_true", default=False)


    parser.add_argument("--config_file", "-f", type = str, default = CONFIG_FILE)
    parser.add_argument("--verbose", "-v", action = "store_true", default = VERBOSE)
    parser.add_argument("--no_cuda", action = "store_true", default = NO_CUDA)
    parser.add_argument("--rng_seed", type = int, default = SEED)

    group = parser.add_argument_group("Data generation parameters")
    group.add_argument("--problem_type", "-p", type = str,
            choices = ["vrp", "vrptw", "svrptw", "sdvrptw"], default = PROBLEM)
    group.add_argument("--customers_count", "-n", type = int, default = CUST_COUNT)
    group.add_argument("--vehicles_count", "-m", type = int, default = VEH_COUNT)
    group.add_argument("--veh_capa", type = int, default = VEH_CAPA)
    group.add_argument("--veh_speed", type = int, default = VEH_SPEED)
    group.add_argument("--horizon", type = int, default = HORIZON)
    group.add_argument("--min_cust_count", type = int, default = MIN_CUST_COUNT)
    group.add_argument("--loc_range", type = int, nargs = 2, default = LOC_RANGE)
    group.add_argument("--dem_range", type = int, nargs = 2, default = DEM_RANGE)
    group.add_argument("--dur_range", type = int, nargs = 2, default = DUR_RANGE)
    group.add_argument("--tw_ratio", type = float, nargs = '*', default = TW_RATIO)
    group.add_argument("--tw_range", type = int, nargs = 2, default = TW_RANGE)
    group.add_argument("--deg_of_dyna", type = float, nargs = '*', default = DEG_OF_DYN)
    group.add_argument("--appear_early_ratio", type = float, nargs = '*', default = APPEAR_EARLY_RATIO)

    group = parser.add_argument_group("VRP Environment parameters")
    group.add_argument("--pending_cost", type = float, default = PEND_COST)
    group.add_argument("--pend_cost_growth", type = float, default = PEND_GROWTH)
    group.add_argument("--late_cost", type = float, default = LATE_COST)
    group.add_argument("--late_cost_growth", type = float, default = LATE_GROWTH)
    group.add_argument("--speed_var", type = float, default = SPEED_VAR)
    group.add_argument("--late_prob", type = float, default = LATE_PROB)
    group.add_argument("--slow_down", type = float, default = SLOW_DOWN)
    group.add_argument("--late_var", type = float, default = LATE_VAR)

    group = parser.add_argument_group("Model parameters")
    group.add_argument("--model_size", "-s", type = int, default = MODEL_SIZE)
    group.add_argument("--layer_count", type = int, default = LAYER_COUNT)
    group.add_argument("--head_count", type = int, default = HEAD_COUNT)
    group.add_argument("--ff_size", type = int, default = FF_SIZE)
    group.add_argument("--tanh_xplor", type = float, default = TANH_XPLOR)

    group = parser.add_argument_group("Training parameters")
    group.add_argument("--epoch_count", "-e", type = int, default = EPOCH_COUNT)
    group.add_argument("--iter_count", "-i", type = int, default = ITER_COUNT)
    group.add_argument("--batch_size", "-b", type = int, default = MINIBATCH_SIZE)
    group.add_argument("--learning_rate", "-r", type = float, default = BASE_LR)
    group.add_argument("--rate_decay", "-d", type = float, default = LR_DECAY)
    group.add_argument("--max_grad_norm", type = float, default = MAX_GRAD_NORM)
    group.add_argument("--grad_norm_decay", type = float, default = GRAD_NORM_DECAY)
    group.add_argument("--loss_use_cumul", action = "store_true", default = LOSS_USE_CUMUL)

    group = parser.add_argument_group("Baselines parameters")
    group.add_argument("--baseline_type", type = str,
            choices = ["none", "nearnb", "rollout", "critic"], default = BASELINE)
    group.add_argument("--rollout_count", type = int, default = ROLLOUT_COUNT)
    group.add_argument("--rollout_threshold", type = float, default = ROLLOUT_THRESHOLD)
    group.add_argument("--critic_use_qval", action = "store_true", default = CRITIC_USE_QVAL)
    group.add_argument("--critic_rate", type = float, default = CRITIC_LR)
    group.add_argument("--critic_decay", type = float, default = CRITIC_DECAY)

    group = parser.add_argument_group("Testing parameters")
    group.add_argument("--test_batch_size", type = int, default = TEST_BATCH_SIZE)

    group = parser.add_argument_group("Checkpointing")
    group.add_argument("--output_dir", "-o", type = str, default = OUTPUT_DIR)
    group.add_argument("--checkpoint_period", "-c", type = int, default = CHECKPOINT_PERIOD)
    group.add_argument("--resume_state", type = str, default = RESUME_STATE)
    group.add_argument("--load_path", type=str)

    args = parser.parse_args(argv)
    if args.config_file is not None:
        with open(args.config_file) as f:
            parser.set_defaults(**json.load(f))

    return parser.parse_args(argv)
