from runner import Runner
# from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], 'multiagent-particle-envs'))
import argparse
from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy
import multiagent.scenarios as scenarios
from functools import reduce
from tensorboardX import SummaryWriter

from datetime import datetime, timedelta
import time
import json
import numpy as np
import torch
import random


if __name__ == '__main__':
    iter_end_step = []
    for i in range(0,10,1):
        # i = random.randint(0, 1000)
        torch.manual_seed(i)  # 为CPU设置随机种子
        torch.cuda.manual_seed(i)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(i)  # 为所有GPU设置随机种子
        random.seed(i)
        np.random.seed(i)
        os.environ['PYTHONHASHSEED'] = str(i)

        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        # env = StarCraft2Env(map_name=args.map,
        #                     step_mul=args.step_mul,
        #                     difficulty=args.difficulty,
        #                     game_version=args.game_version,
        #                     replay_dir=args.replay_dir)
        scenario = scenarios.load(args.scenario).Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                            shared_viewer=False)  # 2

        policies = [RandomPolicy(env, i) for i in range(env.n)]
        obs_n = env.reset()
        # env_info = env.get_env_info()
        args.n_actions = env.action_space[0].n
        args.n_agents = env.n
        obs_shape = [len(list(obs_n[0]))]
        args.state_shape = reduce(lambda x, y: x * y, obs_shape)  # For modified env (3*3*5)  #  把list内int相乘降维为int
        args.obs_shape = args.state_shape
        args.episode_limit = 25
        runner = Runner(env, args)

        cur_time = datetime.now() + timedelta(hours=0)
        args.log_dir = "logs-vdn_graph_new-6agents-v9-newest/" + cur_time.strftime("[%m-%d]%H.%M.%S") + "_seed_" + str(i)
        writer = SummaryWriter(logdir=args.log_dir)

        if args.learn:
            end_step = runner.run(i, writer)
            iter_end_step.append(end_step)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
