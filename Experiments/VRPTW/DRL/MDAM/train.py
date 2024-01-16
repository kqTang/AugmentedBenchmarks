import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
import pdb

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts, validate=True)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts, validate=False):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    # model.eval()

    def eval_model_bat(bat):
        if validate:
            from write import write
            with torch.no_grad():
                cost, _, pis = model(move_to(bat, opts.device), return_pi = True)
            cost, cost_min_index = torch.min(cost, 1)
            pis2 = []
            for i in range(cost.shape[0]):
                pis2.append(pis[cost_min_index[i]][i])        
            if cost.shape[0]<=12:
                write(cost, pis2)
        else:
            with torch.no_grad():
                cost, _ = model(move_to(bat, opts.device), return_pi=False)
            cost, _ = torch.min(cost, 1)
        # cost = cost[:,0]
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    if opts.problem == "cvrptw":
        training_dataset = baseline.wrap_dataset(problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.epoch_size,
            solomon_train=True,
            train_type=opts.train_type))
    else:
        training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # if epoch == 0:
    #     n_EG = 20000
    # elif epoch == 1:
    #     n_EG = 16
    # else:
    #     n_EG = opts.n_EG
    n_EG = opts.n_EG
    optimizer.zero_grad()
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    costs, log_likelihood, reinforce_loss = model(x, opts, baseline, bl_val, n_EG=n_EG, return_kl=True)
    costs, _ = torch.min(costs, 1)
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    if grad_norms[0][0]!=grad_norms[0][0]:
        optimizer.zero_grad()
        print ("nan detected")
        return
    optimizer.step()
    # print ((cost-bl_val).mean(),log_likelihood.mean(),loss.mean(),grad_norms)
    # exit()
    # Logging
    if step % int(opts.log_step) == 0:
    #    print (n_wins, loss_kl.item())
        log_values(costs, grad_norms, epoch, batch_id, step,
                   log_likelihood, log_likelihood.mean(), 0, tb_logger, opts)
