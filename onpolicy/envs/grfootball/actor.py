import gfootball.env as football_env
import time, pprint, importlib, random, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from os import listdir
from os.path import isfile, join
import numpy as np

from datetime import datetime, timedelta

import pdb

def state_to_tensor(state_dict, h_in): # state_dict:{'player':(29,),'ball':(18,),'left_team':(10,7),'left_closest':(7,),'right_team':(11,7),'player':(7,)}
    # pdb.set_trace() #debug

    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0) # 在第0维增加一个维度；[[   state_dict["player"]  ]] #shape(1,1,29)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0) #shape(1,1,18)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0) # shape(1,1,10,7)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0) # shape(1,1,7)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0) # shape(1,1,11,7)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0) # shape(1,1,7)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(0) # shape(1,1,12)  tensor([[[1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.]]])

    state_dict_tensor = {
      "player" : player_state,
      "ball" : ball_state,
      "left_team" : left_team_state,
      "left_closest" : left_closest_state,
      "right_team" : right_team_state,
      "right_closest" : right_closest_state,
      "avail" : avail,
      # "hidden" : h_in # ([1,2,256], [1,2,256])
    }
    return state_dict_tensor


def get_action(a_prob, m_prob):  # a_prob(1,1,12)   m_prob(1,1,8)

    a = Categorical(a_prob).sample().item()  # 采样
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0
    if a == 0:
        real_action = a
        prob = prob_selected_a
    elif a == 1:
        m = Categorical(m_prob).sample().item()
        need_m = 1
        real_action = m + 1
        prob_selected_m = m_prob[0][0][m].item()
        prob = prob_selected_a * prob_selected_m
    else:
        real_action = a + 7
        prob = prob_selected_a

    assert prob != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a, m, prob_selected_a, prob_selected_m)

    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m

def get_action2(a_prob, m_prob):  # a_prob(1,2,12)   m_prob(1,2,8)
    a_prob1 = torch.chunk(a_prob, dim=1, chunks=2)
    m_prob1 = torch.chunk(m_prob, dim=1, chunks=2)
    real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = [],[],[],[],[],[],[]
    for idx, t_chunk in enumerate(a_prob1):
        real_action1, a1, m1, need_m1, prob1, prob_selected_a1, prob_selected_m1 = get_action(a_prob1[idx], m_prob1[idx])
        real_action.append(real_action1)
        a.append(a1)
        m.append(m1)
        need_m.append(need_m1)
        prob.append(prob1)
        prob_selected_a.append(prob_selected_a1)
        prob_selected_m.append(prob_selected_m1)

    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m



def actor(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())
    
    env = football_env.create_environment(env_name=arg_dict["env"], number_of_left_players_agent_controls=3, representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    n_epi = 0
    rollout = []
    while True: # episode loop
        env.reset()   
        done = False
        steps, score, tot_reward, win = 0, [0,0], [0,0], 0
        n_epi += 1
        h_out = (torch.zeros([1, 2, arg_dict["lstm_size"]], dtype=torch.float),
                 torch.zeros([1, 2, arg_dict["lstm_size"]], dtype=torch.float))  ##
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        obs = env.observation()
        
        while not done:  # step loop
            init_t = time.time()
            
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t


            h_in = h_out
            state_dict1 = fe.encode(obs[0])
            state_dict_tensor1 = state_to_tensor(state_dict1, h_in)
            state_dict2 = fe.encode(obs[1])
            state_dict_tensor2 = state_to_tensor(state_dict2, h_in)
            state_dict = [state_dict1, state_dict2]
            state_dict_tensor = {}

            for k,v in state_dict_tensor1.items():
                state_dict_tensor[k] = torch.cat((state_dict_tensor1[k], state_dict_tensor2[k]), 1)
            state_dict_tensor['hidden'] = h_in  #((1,2,256),(1,2,256))

                    # t1 = torch.cat((state_dict_tensor1['hidden'][0], state_dict_tensor2['hidden'][0]), 1)
                    # t2 = torch.cat((state_dict_tensor1['hidden'][1], state_dict_tensor2['hidden'][1]), 1)
                    # state_dict_tensor['hidden'] = (t1, t2)
                    # t1 = torch.cat((h_in[0], h_in[0]), 1)
                    # t2 = torch.cat((h_in[1], h_in[1]), 1)


            # state_dict = fe.encode(obs[0])
            # state_dict_tensor = state_to_tensor(state_dict, h_in)

            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)  # (1,2,12), (1,2,8)
            forward_t += time.time()-t1 
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action2(a_prob, m_prob)  #mod: a_prob：[1,2,12]   m_prob:[1,2,8]
            # [16,9], [9,2] [0,0] [0,0] ... ... ...

            prev_obs = obs
            obs, rew, done, info = env.step(real_action)
            done = [done, done]
            fin_r0 = rewarder.calc_reward(rew[0], prev_obs[0], obs[0])
            fin_r1 = rewarder.calc_reward(rew[1], prev_obs[1], obs[1])
            fin_r = [fin_r0, fin_r1]
            state_prime_dict0 = fe.encode(obs[0])
            state_prime_dict1 = fe.encode(obs[1])
            state_prime_dict = [state_prime_dict0, state_prime_dict1]
            
            (h1_in, h2_in) = h_in
            # h_out_list = []
            # for idx, t_chunk in enumerate(torch.chunk(h_out, dim=1, chunks=2)):
            #     h_out_list.append(t_chunk)

            (h1_out, h2_out) = h_out

            state_dict[0]["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_dict[1]["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict[0]["hidden"] = (h1_out.numpy(), h2_out.numpy())
            state_prime_dict[1]["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            # score += rew
            # tot_reward += fin_r
            score = list(map(lambda x: x[0] + x[1], zip(score, rew)))
            tot_reward = list(map(lambda x: x[0] + x[1], zip(tot_reward, fin_r)))
            
            if arg_dict['print_mode']:
                print_status(steps,a,m,prob_selected_a,prob_selected_m,prev_obs,obs,fin_r,tot_reward)
            loop_t += time.time()-init_t
            
            if done:
                if score[0] > 0 or score[1] > 0:
                    win = 1
                print("score",score,"total reward",tot_reward)
                summary_data = (win, score, tot_reward, steps, 0, loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)

def select_opponent(arg_dict): # 50%的对手来自于最新的10个模型，其余的来自于从整个池中统一随机抽样
    onlyfiles_lst = [f for f in listdir(arg_dict["log_dir"]) if isfile(join(arg_dict["log_dir"], f))]
    model_num_lst = []
    for file_name in onlyfiles_lst:
        if file_name[:6] == "model_":
            model_num = file_name[6:]
            model_num = model_num[:-4]
            model_num_lst.append(int(model_num))
    model_num_lst.sort()
            
    coin = random.random()
    if coin < arg_dict["latest_ratio"]:  # latest_ratio=0.5
        if len(model_num_lst) > arg_dict["latest_n_model"]:  # latest_n_model=10
            opp_model_num = random.randint(len(model_num_lst)-arg_dict["latest_n_model"],len(model_num_lst)-1)
        else:
            opp_model_num = len(model_num_lst)-1
    else:
        opp_model_num = random.randint(0,len(model_num_lst)-1)
        
    model_name = "/model_"+str(model_num_lst[opp_model_num])+".tar"
    opp_model_path = arg_dict["log_dir"] + model_name
    return opp_model_num, opp_model_path
                
                
def actor_self(actor_num, center_model, data_queue, signal_queue, summary_queue, arg_dict):
    print("Actor process {} started".format(actor_num))
    cpu_device = torch.device('cpu')
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models." + arg_dict["model"])
    
    fe = fe_module.FeatureEncoder()
    model = imported_model.Model(arg_dict)
    model.load_state_dict(center_model.state_dict())
    opp_model = imported_model.Model(arg_dict)

    env = football_env.create_environment(env_name=arg_dict["env"], number_of_left_players_agent_controls=3, representation="raw", \
                                          stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, \
                                          render=False)

    n_epi = 0
    rollout = []
    while True: # episode loop
        opp_model_num, opp_model_path = select_opponent(arg_dict)
        checkpoint = torch.load(opp_model_path, map_location=cpu_device)
        opp_model.load_state_dict(checkpoint['model_state_dict'])
        print("Current Opponent model Num:{}, Path:{} successfully loaded".format(opp_model_num, opp_model_path))
        del checkpoint
        
        env.reset()   
        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        n_epi += 1
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))  # ([1,1,256], [1,1,256])
        opp_h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float), 
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float)) # ([1,1,256], [1,1,256])
        
        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        [obs, opp_obs] = env.observation()  # obs, opp_obs均为长度是23的字典，以下是字典元素的shape
        # {
        #     'ball': (3,),
        #     'ball_direction': (3,),
        #     'ball_rotation': (3,),
        #     'ball_owned_team': -1,
        #     'ball_owned_player': -1,
        #
        #     'left_team': (11, 2),
        #     'left_team_direction': (11, 2),
        #     'left_team_tired_factor': (11,),
        #     'left_team_yellow_card': (11,),
        #     'left_team_active': (11,),
        #     'left_team_roles': (11,),
        #
        #     'right_team': (11, 2),
        #     'right_team_direction': (11, 2),
        #     'right_team_tired_factor': (11,),
        #     'right_team_yellow_card': (11,),
        #     'right_team_active': (11,),
        #     'right_team_roles': (11,),
        #
        #     'score': [0, 0],
        #     'steps_left': 3001,
        #     'game_mode': 0,
        #
        #     'active': 6,
        #     'designated': 6,
        #     'sticky_actions': (10,)
        # }
        
        while not done:  # step loop
            init_t = time.time()
            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
            wait_t += time.time() - init_t
            
            h_in = h_out  # （ [1,1,256]  ， [1,1,256] ）
            opp_h_in = opp_h_out
            state_dict = fe.encode(obs)
            state_dict_tensor = state_to_tensor(state_dict, h_in)
            # state_dict ，state_dict_tensor均为 长为8的字典，后者比前者多两维
            # {'player': (29,),
            #  'ball': (18,),
            #  'left_team': (10, 7),
            #  'left_closest': (7,),
            #  'right_team': (11, 7),
            #  'right_closest': (7,),
            #  'avail': (12,),
            #  'hidden': ((1, 1, 256), (1, 1, 256))
            #  }
            opp_state_dict = fe.encode(opp_obs)
            opp_state_dict_tensor = state_to_tensor(opp_state_dict, opp_h_in)
            # opp_state_dict ，opp_state_dict_tensor均为 长为7的字典，后者比前者多两维
            # {'player': (29,),
            #  'ball': (18,),
            #  'left_team': (10, 7),
            #  'left_closest': (7,),
            #  'right_team': (11, 7),
            #  'right_closest': (7,),
            #  'avail': (12,),
            #  }

            t1 = time.time()
            with torch.no_grad():
                a_prob, m_prob, _, h_out = model(state_dict_tensor)  # (1,1,12), (1,1,8), _, ((1,1,256),(1,1,256))
                opp_a_prob, opp_m_prob, _, opp_h_out = opp_model(opp_state_dict_tensor)
            forward_t += time.time()-t1

            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)
            # 0          0  0    0       0.1549  0.1549          0
            opp_real_action, _, _, _, _, _, _ = get_action(opp_a_prob, opp_m_prob)
            # 16
            prev_obs = obs  # state_dict_tensor 使用来创建模型的 ；  obs 是用来环境step的  ；  state_dict_tensor 是 obs 的FeatureEncoder
            [obs, opp_obs], [rew, _], done, info = env.step([real_action, opp_real_action])
            fin_r = rewarder.calc_reward(rew, prev_obs, obs)
            state_prime_dict = fe.encode(obs)

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            transition = (state_dict, a, m, fin_r, state_prime_dict, prob, done, need_m)
            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())

            steps += 1
            score += rew
            tot_reward += fin_r
            
            if arg_dict['print_mode']:
                print_status(steps,a,m,prob_selected_a,prob_selected_m,prev_obs,obs,fin_r,tot_reward)
            
            loop_t += time.time()-init_t

            if done:
                if score > 0:
                    win = 1
                print("score {}, total reward {:.2f}, opp num:{}, opp:{} ".format(score,tot_reward,opp_model_num, opp_model_path))
                summary_data = (win, score, tot_reward, steps, str(opp_model_num), loop_t/steps, forward_t/steps, wait_t/steps)
                summary_queue.put(summary_data)                

