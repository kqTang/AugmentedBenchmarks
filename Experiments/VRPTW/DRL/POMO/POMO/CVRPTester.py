import pdb
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *

import xlwt
import torch
import pdb


def write(data, pi):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet('data')  # epochs,score,loss
    # 设置字体格式
    Font0 = xlwt.Font()
    Font0.name = "Times New Roman"
    Font0.colour_index = 2
    Font0.bold = True  # 加粗
    style0 = xlwt.XFStyle()
    for i in range(pi.size(0)):
        # pdb.set_trace()
        if i < 100:
            sht1.write(i + 1, 0, data['loc'][0,i,0].item(), style0)  # 顾客点的坐标x与y
            sht1.write(i + 1, 1, data['loc'][0, i, 1].item(), style0)
        sht1.write(i + 1, 2, pi[i].item(), style0)  # 顾客点的需求
    xls.save('./xls/pi/rc2.xls')

class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            # torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(device=device,**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        # model_load = tester_params['model_load']
        # # checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        # print("path:",'{path}'.format(**model_load))
        # checkpoint_fullname = '{path}'.format(**model_load)
        # checkpoint = torch.load(checkpoint_fullname, map_location=device)
        # # pdb.set_trace()
        # # from collections import OrderedDict

        # self.model.load_state_dict(
        #     {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

        # # new_state_dict = OrderedDict()

        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # for k, v in checkpoint.items():
        #     name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        # self.model.load_state_dict(new_state_dict)
        # utility
        self.time_estimator = TimeEstimator()

    def run(self, batch, num_files, hard, checkpoint, solomon=False):

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator.reset()

        # score_AM = AverageMeter()
        # aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        # test_num_episode = self.tester_params['test_episodes']
        # episode = 0

        # while episode < test_num_episode:

        # remaining = test_num_episode - episode
        # pdb.set_trace()
        batch_size = self.tester_params['test_batch_size']

        score, unusual = self._test_one_batch(
            batch_size, batch, num_files, hard=hard)
        
        # score_AM.update(score, batch_size)
        # aug_score_AM.update(aug_score, batch_size)

        # episode += batch_size

        ############################
        # Logs
        ############################
        # elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
        # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
            # episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

        # all_done = (episode == test_num_episode)

        # if all_done:
        self.logger.info(" *** Test Done *** ")
        # print(score.size())
        if solomon:
            for i in range(score.size(0)):
                self.logger.info(" NO-AUG SCORE: {:.4f}, penalty: {:.4f}".format(score[i],unusual[i]))
        else:
            self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(score.mean().item()))
        return score

    def _test_one_batch(self, batch_size, batch, num_files, hard):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, batch, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, self.device)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            
            selected, _ = self.model(state)
            # print(selected,end=",")
            # shape: (batch, pomo)
            # selectlist: (batch, pomo, 122)
            # reward:(batch, pomo)
            state, reward, done, unusual, selectlist = self.env.step(
                selected, batch, hard=hard)
        # pdb.set_trace()
            # selectlist
        # sigle solomon
        # pdb.set_trace()
        
        _, order1 = reward.view(int(batch_size/num_files),num_files,-1).max(0)
        min, order2 = reward.view(int(batch_size/num_files),num_files,-1).max(0)[0].max(-1)
        # pdb.set_trace()
        # min = min.reshape(10,-1,min.size(-1))
        unusual=unusual.view(int(batch_size/num_files),num_files,-1).gather(dim=0,index = order1[None,...]).squeeze(0).gather(dim=-1,index=order2[...,None]).squeeze(-1)
        # pdb.set_trace()
        order3 = order1.gather(dim=-1, index=order2[..., None]).squeeze(-1)
        pi=selectlist.view(int(batch_size/num_files), num_files,
                           selectlist.size(-2), selectlist.size(-1))[order3, torch.arange(num_files), order2]
        # write(batch,pi[0])
        # print(min, selectlist.view(selectlist.size(0) *
        #   selectlist.size(1), selectlist.size(-1))[order][:, None])
        # if -min.item() < 828:
        #     print(min,selectlist.view(selectlist.size(0)*selectlist.size(1),selectlist.size(-1))[order][:,None])
        #     pdb.set_trace()
        return -min,unusual
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # selectlist = selectlist.reshape(aug_factor, batch_size, self.env.pomo_size,selectlist.size(-1))

        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().max()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().max()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
