import gfootball.env as football_env
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


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"]+"/args.json","w")
    f.write(args_info)
    f.close()
    

def copy_models(dir_src, dir_dst): # src: source, dst: destination
    # retireve list of models
    l_cands = [f for f in os.listdir(dir_src) if os.path.isfile(os.path.join(dir_src, f)) and 'model_' in f]
    l_cands = sorted(l_cands, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"models to be copied: {l_cands}")
    for m in l_cands:
        shutil.copyfile(os.path.join(dir_src, m), os.path.join(dir_dst, m))
    print(f"{len(l_cands)} models copied in the given directory")
    
def main(arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cur_time = datetime.now() + timedelta(hours = 9)
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    save_args(arg_dict)
    if arg_dict["trained_model_path"] and 'kaggle' in arg_dict['env']: 
        copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir'])

    np.set_printoptions(precision=3) # #设置浮点精度为三位数
    np.set_printoptions(suppress=True) # 使数组打印更漂亮
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)

    # pdb.set_trace()  #debug

    fe = importlib.import_module("encoders." + arg_dict["encoder"])
    fe = fe.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()  #arg_dict["feature_dims"]: {'player': 29, 'ball': 18, 'left_team': 7, 'left_team_closest': 7, 'right_team': 7, 'right_team_closest': 7}
    
    model = importlib.import_module("models.agents" + arg_dict["model"])
    cpu_device = torch.device('cpu')
    center_model = model.QValueAgent(arg_dict)
    imported_algo = importlib.import_module("algos" + arg_dict["algorithm"])
    algo = imported_algo.QLearner(arg_dict, center_model)
    center_mixer = algo.mixer

    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
        optimization_step = checkpoint['optimization_step']
        center_model.load_state_dict(checkpoint['model_state_dict'])
        center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        arg_dict["optimization_step"] = optimization_step
        print("Trained model", arg_dict["trained_model_path"] ,"suffessfully loaded") 
    else:
        optimization_step = 0

    model_dict = {
        'optimization_step': optimization_step,
        'model_state_dict': center_model.state_dict(),
        'optimizer_state_dict': algo.optimizer.state_dict(),
    }

    path = arg_dict["log_dir"]+f"/model_{optimization_step}.tar"
    torch.save(model_dict, path)
        
    center_model.share_memory() #进程间通信？
    center_mixer.share_memory()
    data_queue = mp.Queue() # <multiprocessing.queues.Queue object at 0x7fba9d4ee0b8>
    signal_queue = mp.Queue()
    summary_queue = mp.Queue()
    
    processes = [] 
    p = mp.Process(target=learner, args=(center_model, center_mixer, data_queue, signal_queue, summary_queue, arg_dict))  #  target：调用对象/目标函数  args：调用对象的位置参数元组/按位置给目标函数传参
    p.start()
    processes.append(p)
    for rank in range(arg_dict["num_processes"]):
        # if arg_dict["env"] == "11_vs_11_kaggle":
        if arg_dict["env"] == "11_vs_11_kaggle":
                p = mp.Process(target=actor_self, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        else:
            p = mp.Process(target=actor, args=(rank, center_model, data_queue, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)
    
    if "env_evaluation" in arg_dict:
        p = mp.Process(target=evaluator, args=(center_model, signal_queue, summary_queue, arg_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
    

if __name__ == '__main__':

    arg_dict = {
        "env": "academy_run_pass_and_shoot_with_keeper",
        # "env": "11_vs_11_kaggle",
        # "11_vs_11_kaggle" : environment used for self-play training
        # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
        "num_processes": 1,  # should be less than the number of cpu cores in your workstation.
        "batch_size": 32,   #ori:32
        "buffer_size": 6,  # ori:6
        "rollout_len": 60, # ori:30

        "lstm_size" : 256,
        "k_epoch" : 3,
        "learning_rate" : 0.0001,
        "gamma" : 0.993,
        "lmbda" : 0.96,
        "entropy_coef" : 0.0001,
        "grad_clip" : 3.0,
        "eps_clip" : 0.1,

        "summary_game_window" : 10, 
        "model_save_interval" : 100000,  # number of gradient updates bewteen saving model
        "target_update_interval": 200,

        "trained_model_path" : None, # use when you want to continue traning from given model.
        "latest_ratio" : 0.5, # works only for self_play training. 
        "latest_n_model" : 10, # works only for self_play training. 
        "print_mode" : False,

        "encoder" : "encoder_basic",
        "rewarder" : "rewarder_basic",
        "model" : "rnn_agent",
        "algorithm" : "qmix",

        "env_evaluation":'academy_run_pass_and_shoot_with_keeper'  # for evaluation of self-play trained agent (like validation set in Supervised Learning)
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
