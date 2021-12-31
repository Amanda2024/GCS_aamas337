# import gfootball.env as football_env
import time, pprint, json, os, importlib, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
# import pdb

from actor import *
from learner import *
from evaluator import evaluator
from datetime import datetime, timedelta
import random


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"] + "/args.json", "w")
    f.write(args_info)
    f.close()


def copy_models(dir_src, dir_dst):  # src: source, dst: destination
    # retireve list of models
    l_cands = [f for f in os.listdir(dir_src) if os.path.isfile(os.path.join(dir_src, f)) and 'model_' in f]
    l_cands = sorted(l_cands, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"models to be copied: {l_cands}")
    for m in l_cands:
        shutil.copyfile(os.path.join(dir_src, m), os.path.join(dir_dst, m))
    print(f"{len(l_cands)} models copied in the given directory")

def judge_finish(data_all_queue, processes, arg_dict):
    while True:
        if(data_all_queue.get() > arg_dict["tensorboard_steps"]):
            for p in processes:
                p.terminate()
            break

def _get_critic_input_shape(args):
    # state
    input_shape = args['input_shape']  # 48
    # obs
    input_shape += args['input_shape']  # 30
    # agent_id
    input_shape += args['n_agents']  # 3
    # 所有agent的当前动作和上一个动作
    input_shape += args['n_actions'] * args['n_agents'] * 2  # 54

    return input_shape


def main(arg_dict):
    for i in range(0, 10, 1):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        np.random.seed(seed=i)
        random.seed(i)
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        cur_time = datetime.now() + timedelta(hours=0)
        arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S") + "_seed_" + str(i)
        save_args(arg_dict)
        if arg_dict["trained_model_path"] and 'kaggle' in arg_dict['env']:
            copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir'])

        np.set_printoptions(precision=3)  # #设置浮点精度为三位数
        np.set_printoptions(suppress=True)  # 使数组打印更漂亮
        pp = pprint.PrettyPrinter(indent=4)
        torch.set_num_threads(1)

        # pdb.set_trace()  #debug

        fe = importlib.import_module("encoders." + arg_dict["encoder"])
        fe = fe.FeatureEncoder()
        arg_dict["feature_dims"] = fe.get_feature_dims()  # arg_dict["feature_dims"]: {'player': 29, 'ball': 18, 'left_team': 7, 'left_team_closest': 7, 'right_team': 7, 'right_team_closest': 7}

        model = importlib.import_module("models.agents." + arg_dict["model"])
        actor_ = importlib.import_module("models.agents." + arg_dict["graph_actor"])
        critic = importlib.import_module("models.agents." + arg_dict["graph_critic"])
        mixer = importlib.import_module("models.mixers." + arg_dict["mixer_net"])

        cpu_device = torch.device('cpu')
        input_shape = arg_dict["state_shape"]  ## 获取输入维度
        if arg_dict["last_action"]:
            input_shape += arg_dict["n_actions"]
        if arg_dict["reuse_network"]:
            input_shape += arg_dict["n_agents"]
        if arg_dict["father_action"]:
            input_shape += arg_dict["n_agents"] * arg_dict["n_actions"]

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        center_model = model.RNNAgent(input_shape, arg_dict)
        center_actor_model = actor_.Actor_graph(arg_dict)
        center_critic_model = critic.Critic_graph(arg_dict)
        if arg_dict['mixer'] == 'qmix':
            center_mixer = mixer.QMixNet(arg_dict)
        elif arg_dict['mixer'] == 'vdn':
            center_mixer = mixer.VDNMixer()


        if arg_dict["trained_model_path"]:
            checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
            optimization_step = checkpoint['optimization_step']
            center_model.load_state_dict(checkpoint['model_state_dict'])
            center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            arg_dict["optimization_step"] = optimization_step
            print("Trained model", arg_dict["trained_model_path"], "suffessfully loaded")
        else:
            optimization_step = 0

        model_dict = {
            'optimization_step': optimization_step,
            'model_state_dict': center_model.state_dict(),
            # 'follower_model_state_dict': center_follower_model.state_dict(),
            'optimizer_state_dict': center_model.optimizer.state_dict(),
            # 'follower_optimizer_state_dict': center_follower_model.optimizer.state_dict(),
        }

        path = arg_dict["log_dir"] + f"/model_{optimization_step}"
        torch.save(model_dict, path)

        torch.multiprocessing.set_sharing_strategy('file_system')  #### ganga

        center_model.share_memory()  # 进程间通信
        center_actor_model.share_memory()
        center_mixer.share_memory()
        center_critic_model.share_memory()
        data_queue = mp.Queue()  # <multiprocessing.queues.Queue object at 0x7fba9d4ee0b8>
        signal_queue = mp.Queue()
        summary_queue = mp.Queue()
        data_all_queue = mp.Queue()

        processes = []
        # # p = mp.Process(target=learner, args=(
        # # center_model, center_follower_model, center_mixer, data_queue, signal_queue, summary_queue,
        # # arg_dict))  # target：调用对象/目标函数  args：调用对象的位置参数元组/按位置给目标函数传参
        p = mp.Process(target=learner, args=(center_model, center_actor_model, center_mixer, center_critic_model, data_queue, data_all_queue, signal_queue, summary_queue, arg_dict))  # target：调用对象/目标函数  args：调用对象的位置参数元组/按位置给目标函数传参
        p.start()
        processes.append(p)

        for rank in range(arg_dict["num_processes"]):
            # if arg_dict["env"] == "11_vs_11_kaggle":
            if arg_dict["env"] == "11_vs_11_kaggle":
                p = mp.Process(target=actor_self, args=(
                rank, center_model, center_actor_model, data_queue, signal_queue, summary_queue, arg_dict))
            else:
                p = mp.Process(target=actor, args=(rank, center_model, center_actor_model, data_queue, signal_queue, summary_queue, arg_dict))
            p.start()
            processes.append(p)

        # if "env_evaluation" in arg_dict:
        #     p = mp.Process(target=evaluator, args=(center_model, center_follower_model, signal_queue, summary_queue, arg_dict))
        #     p.start()
        #     processes.append(p)
        p = mp.Process(target=judge_finish,
                       args=(data_all_queue, processes, arg_dict))  # target：调用对象/目标函数  args：调用对象的位置参数元组/按位置给目标函数传参
        p.start()
        processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    arg_dict = {
        "env": "academy_run_pass_and_shoot_with_keeper_vs_6",
        # "env": "academy_3_vs_1_with_keeper",
        "w_vdn": True,
        "w_to_use": 0.75,
        ### coma_critic-----
        "n_actions": 19,
        "n_agents": 3,
        "td_lambda": 0.8,
        "input_shape": 165,#136#
        "state_shape": 143,#115
        "obs_shape" : 143,
        "critic_dim": 128,
        "lr_critic": 1e-3,
        "coma_critic_dim": 405,
        # "state_shape":,
        #
        # "env": "11_vs_11_kaggle",
        # "11_vs_11_kaggle" : environment used for self-play training
        # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
        "num_processes": 30,  # should be less than the number of cpu cores in your workstation.
        # "batch_size": 2,   #ori:32
        # "buffer_size": 1,  # ori:6
        # "rollout_len": 5, # ori:30
        "batch_size": 32,  # ori:32
        "buffer_size": 6,  # ori:6
        "rollout_len": 30,  # ori:30
        "rich_data_scale": 1,

        "lstm_size": 64,  # 64
        "action_dim": 19,
        "k_epoch": 3,

        "optimizer": 'RMS',
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "lmbda": 0.96,
        "entropy_coef": 0.0001,
        "grad_clip": 10.0,
        "eps_clip": 0.1,
        "episode_limit": 210,

        "summary_game_window": 100,
        "model_save_interval": 100000,  # number of gradient updates bewteen saving model #ori:300000
        "target_update_interval": 50,  # ori:200
        "tensorboard_steps": 50000,

        "trained_model_path": None,  # use when you want to continue traning from given model.
        "latest_ratio": 0.5,  # works only for self_play training.
        "latest_n_model": 10,  # works only for self_play training.
        "print_mode": False,

        "encoder": "encoder_basic",
        "rewarder": "rewarder_basic",
        # "encoder": "encoder_highpass",
        # "rewarder": "rewarder_highpass",
        "model": "rnn_agent",
        "follower_model": "follower_agent",
        "coma_critic": "coma_critic_agent",
        "algorithm": "vdn_graph",
        "mixer": "vdn",
        "mixer_net": "vdn_net",

        ########### actor_graph
        "graph_actor": "action_graph_2",
        "graph_critic": "action_graph_2",
        "n_xdims": 165,
        "nhead": 3,
        "num_layers": 4,
        "decoder_hidden_dim": 64,
        "node_num": 3,
        "critic_hidden_dim": 64,
        "father_action": True,
        "alpha": 0.99,

        # epsilon-greedy for action prob
        "epsilon_": 0.5,
        "anneal_epsilon_": 0.00064,
        "min_epsilon_": 0.02,
        "epsilon_anneal_scale_": 'episode',
        ##########################

        "env_evaluation": 'academy_run_pass_and_shoot_with_keeper_vs_6',
    # for evaluation of self-play trained agent (like validation set in Supervised Learning),

        "epsilon": 0.2,
        "epsilon_anneal_scale": 'step',
        "min_epsilon": 0.05,
        "anneal_epsilon": (0.2 - 0.05) / 50000,

        "last_action": True,
        "reuse_network": True,
        # "last_action": False,
        # "reuse_network": False,

        ### # arguments of  qmix
        # network
        "qmix_hidden_dim": 32,
        "two_hyper_layers": False,
        "hyper_hidden_dim": 64,
        "lr": 5e-4,

    }

    main(arg_dict)

# p center_model
# Model(
#   (fc_player): Linear(in_features=29, out_features=64, bias=True)
#   (fc_ball): Linear(in_features=18, out_features=64, bias=True)
#   (fc_left): Linear(in_features=7, out_features=48, bias=True)
#   (fc_right): Linear(in_features=7, out_features=48, bias=True)
#   (fc_left_closest): Linear(in_features=7, out_features=48, bias=True)
#   (fc_right_closest): Linear(in_features=7, out_features=48, bias=True)
#   (conv1d_left): Conv1d(48, 36, kernel_size=(1,), stride=(1,))
#   (conv1d_right): Conv1d(48, 36, kernel_size=(1,), stride=(1,))
#   (fc_left2): Linear(in_features=360, out_features=96, bias=True)
#   (fc_right2): Linear(in_features=396, out_features=96, bias=True)
#   (fc_cat): Linear(in_features=416, out_features=256, bias=True)
#   (norm_player): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#   (norm_ball): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
#   (norm_left): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
#   (norm_left2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
#   (norm_left_closest): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
#   (norm_right): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
#   (norm_right2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
#   (norm_right_closest): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
#   (norm_cat): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
#   (lstm): LSTM(256, 256)
#   (fc_pi_a1): Linear(in_features=256, out_features=164, bias=True)
#   (fc_pi_a2): Linear(in_features=164, out_features=12, bias=True)
#   (norm_pi_a1): LayerNorm((164,), eps=1e-05, elementwise_affine=True)
#   (fc_pi_m1): Linear(in_features=256, out_features=164, bias=True)
#   (fc_pi_m2): Linear(in_features=164, out_features=8, bias=True)
#   (norm_pi_m1): LayerNorm((164,), eps=1e-05, elementwise_affine=True)
#   (fc_v1): Linear(in_features=256, out_features=164, bias=True)
#   (norm_v1): LayerNorm((164,), eps=1e-05, elementwise_affine=True)
#   (fc_v2): Linear(in_features=164, out_features=1, bias=False)
# )
