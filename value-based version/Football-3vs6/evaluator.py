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
import copy

from datetime import datetime, timedelta

import pdb


def state_to_tensor(state_dict, h_in):  # state_dict:{'player':(29,),'ball':(18,),'left_team':(10,7),'left_closest':(7,),'right_team':(11,7),'player':(7,)}
    # pdb.set_trace() #debug

    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)  # 在第0维增加一个维度；[[   state_dict["player"]  ]] #shape(1,1,29)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,18)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,10,7)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,7)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,11,7)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,7)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,12)  tensor([[[1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.]]])

    state_dict_tensor = {
        "player": player_state,
        "ball": ball_state,
        "left_team": left_team_state,
        "left_closest": left_closest_state,
        "right_team": right_team_state,
        "right_closest": right_closest_state,
        "avail": avail,
        # "hidden" : h_in # ([1,1,256], [1,1,256])
    }
    return state_dict_tensor


def choose_action(obs, last_action, agent_num, avail_actions, epsilon, arg_dict, model, h_in):
    inputs = obs.copy()
    avail_actions_ind = np.nonzero(avail_actions)[0]

    # transform agent_num to onehot vector
    agent_id = np.zeros(arg_dict["n_agents"])
    agent_id[agent_num] = 1.

    if arg_dict["last_action"]:
        inputs = np.hstack((inputs, last_action.reshape([inputs.shape[0], -1])))
    if arg_dict["reuse_network"]:
        inputs = np.hstack((inputs, agent_id.reshape([inputs.shape[0], -1])))

    inputs = torch.tensor(inputs, dtype=torch.float32)
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

    # get q value
    with torch.no_grad():
        q_value, h_out = model(inputs, h_in)

    # choose action from q value
    q_value[avail_actions == 0.0] = - float("inf")
    if np.random.uniform() < epsilon:
        action = np.random.choice(avail_actions_ind)  # action是一个整数
    else:
        action = torch.argmax(q_value)
    return action, h_out

def choose_action_ssg(obs, last_action, agent_num, avail_actions, epsilon, arg_dict, model, follower_model, h_in):
    inputs = obs.copy()
    avail_actions_ind_l = np.nonzero(avail_actions[0])[0]
    avail_actions_ind_f = np.nonzero(avail_actions[1])[0]

    # transform agent_num to onehot vector
    agent_id = np.zeros(arg_dict["n_agents"])
    agent_id[agent_num] = 1.

    if arg_dict["last_action"]:
        inputs_l = np.hstack((inputs[0], last_action[0].reshape([inputs[0].shape[0], -1])))
        inputs_f = np.hstack((inputs[1], last_action[1].reshape([inputs[1].shape[0], -1])))
    if arg_dict["reuse_network"]:
        inputs_l = np.hstack((inputs_l, agent_id.reshape([inputs[0].shape[0], -1])))
        inputs_f = np.hstack((inputs_f, agent_id.reshape([inputs[1].shape[0], -1])))


    inputs = torch.tensor(inputs, dtype=torch.float32)
    inputs_l = torch.tensor(inputs_l, dtype=torch.float32)
    inputs_f = torch.tensor(inputs_f, dtype=torch.float32)

    avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

    # get q value
    with torch.no_grad():
        q_value_l, h_out_l = model(inputs_l, h_in[0])
        q_value_f, h_out_f = follower_model(inputs_f, q_value_l, h_in[1])
    h_out = torch.cat([h_out_l.unsqueeze(0), h_out_f.unsqueeze(0)], dim=0)
    # choose action from q value
    q_value_l[avail_actions[:,0,:] == 0.0] = - float("inf")
    q_value_f[avail_actions[:,1,:] == 0.0] = - float("inf")
    if np.random.uniform() < epsilon:
        action_l = np.random.choice(avail_actions_ind_l)  # action是一个整数
        action_f = np.random.choice(avail_actions_ind_f)  # action是一个整数
    else:
        action_l = torch.argmax(q_value_l)
        action_f = torch.argmax(q_value_f)

    return action_l if agent_num == 0 else action_f, h_out

def add_to_inputs(obs, last_action, arg_dict):
    inputs_ = []
    inputs = obs.copy()
    # transform agent_num to onehot vector
    for i in range(arg_dict["n_agents"]):
        agent_id = np.zeros(arg_dict["n_agents"])
        agent_id[i] = 1.
        if arg_dict["last_action"]:
            input_ = np.hstack((inputs[i], np.array(last_action[i]).reshape([inputs[i].shape[0], -1])))
        if arg_dict["reuse_network"]:
            input_ = np.hstack((input_, agent_id.reshape([input_.shape[0], -1])))
        inputs_.append(input_.tolist())
    # print(inputs_)
    return np.array(inputs_)

def obs_transform(state_dict_tensor):
    '''

    :param state_dict_tensor: 7 kind of state dict with tensor for each element
    :return: flattern_obs for multi-agents [num_agent, obs_shape] (2 x 350)
    '''
    flattern_obs_0 = []
    flattern_obs_1 = []
    for k, v in enumerate(state_dict_tensor):
        if k != 'hidden': # hideen这一维度去掉
            flattern_obs_0.append(state_dict_tensor[v][0].reshape([-1]))
            flattern_obs_1.append(state_dict_tensor[v][1].reshape([-1]))

    flattern_obs_0 = torch.hstack(flattern_obs_0)
    flattern_obs_1 = torch.hstack(flattern_obs_1)
    flattern_obs = torch.stack((flattern_obs_0, flattern_obs_1), dim=0)

    return flattern_obs.unsqueeze(1).numpy()

def obs_encode(obs, h_in, fe):  # 将obs和h_out 编码成state_dict,state_dict_tensor
    # h_in = h_out
    for i in range(len(obs)):
        if obs[i]['active'] == 0:
            state_dict1 = fe.encode(obs[i])  # 长度为7的字典
            state_dict_tensor1 = state_to_tensor(state_dict1, h_in)
        else:
            state_dict2 = fe.encode(obs[i])
            state_dict_tensor2 = state_to_tensor(state_dict2, h_in)
    state_dict = [state_dict1, state_dict2]
    state_dict_tensor = {}

    for k, v in state_dict_tensor1.items():
        state_dict_tensor[k] = torch.cat((state_dict_tensor1[k], state_dict_tensor2[k]), 0)
    # state_dict_tensor['hidden'] = h_in  # ((1,1,256),(1,1,256))

    return state_dict, state_dict_tensor

def normalize_reward(rollout):
    reward = []
    for transition in rollout:
        reward.append(transition[5]) ## tuple的第五个表示reward

    r = np.array(reward)[:, 0]  # num_steps array for both agents
    r = (r - np.mean(r)) / (np.std(r) + 1e-7)
    rollout_new = []
    for i in range(r.size):
        obs_model_inputs, h_in, actions1, actions_onehot1, avail_u, reward_agency, obs_prime_inputs, h_out, avail_u_next, done = rollout[i]
        r_new = [r[i], r[i]]
        transition_new = (obs_model_inputs, h_in, actions1, actions_onehot1, avail_u, r_new, obs_prime_inputs, h_out, avail_u_next, done)
        rollout_new.append(transition_new)

    return rollout_new

def compute_win_rate(score_list):
    '''
    :param score_list: [0,0,1,1,1,0,0,1,0,1] with T timesteps
    :return: win_rate: such as [0.5] a list with one element
    '''
    if len(score_list) <= 100:
        win_rate = [sum(score_list) / len(score_list)]
    else:
        score_list = score_list[-100:]
        win_rate = [sum(score_list) / 100]
    return win_rate

def evaluator( center_model, center_follower_model, signal_queue, summary_queue, arg_dict):
    print("Evaluator process started")
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models.agents." + arg_dict["model"])
    follower_imported_model = importlib.import_module("models.agents." + arg_dict["follower_model"])

    fe = fe_module.FeatureEncoder()
    # 根据参数决定RNN的输入维度
    input_shape = arg_dict["state_shape"]
    if arg_dict["last_action"]:
        input_shape += arg_dict["n_actions"]
    if arg_dict["reuse_network"]:
        input_shape += arg_dict["n_agents"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = imported_model.RNNAgent(input_shape, arg_dict)
    follower_model = follower_imported_model.Follower_Agent(input_shape, arg_dict)
    model.load_state_dict(center_model.state_dict())
    follower_model.load_state_dict(center_follower_model.state_dict())

    env = football_env.create_environment(env_name=arg_dict["env"], number_of_left_players_agent_controls=3,
                                          representation="raw", stacked=False, logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    # print("-----------------number_of_players_agent_controls", env._players)
    n_epi = 0
    rollout = []
    score_list = []
    while True:  # episode loop
        env.reset()
        done = False
        steps, score, tot_reward, win = 0, [0, 0], [0, 0], 0
        n_epi += 1
        h_out = (torch.zeros([1, arg_dict["lstm_size"]], dtype=torch.float),
                 torch.zeros([1, arg_dict["lstm_size"]], dtype=torch.float))  ##rjq ((1,256),(1,256))

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        obs = env.observation()  # [,]

        # last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        last_action = np.zeros((arg_dict["n_agents"], arg_dict["n_actions"]))
        # epsilon
        epsilon = arg_dict["epsilon"]
        obs_prime_prime_inputs = []
        while not done and steps < arg_dict["episode_limit"]:  # step loop
            init_t = time.time()

            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
                follower_model.load_state_dict(center_follower_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            # state_dict, state_dict_tensor = obs_encode(obs, fe)  # state_dict:[dict,dict]  state_dict_tensor:dict
            state_dict, state_dict_tensor = obs_encode(obs, h_in, fe)
            obs_model = obs_transform(state_dict_tensor)
            obs_model_inputs = obs_model
            if (arg_dict["last_action"] and arg_dict["reuse_network"]):
                obs_model_inputs = add_to_inputs(obs_model, last_action, arg_dict) # 将上一动作和重用网络加入进去

            t1 = time.time()
            actions, avail_actions, actions_onehot = [], [], []
            h_out_list = []

            for agent_id in range(2):
                if arg_dict['algorithm'] == 'ssg_vdn':
                    avail_action_s = [state_dict[x]['avail'] for x in range(len(state_dict))]
                    action, h_out = choose_action_ssg(obs_model, last_action, agent_id,
                                                  avail_action_s, epsilon, arg_dict, model, follower_model, h_in)
                    h_out_list.append(h_out)
                    # generate onehot vector of th action
                    action_onehot = np.zeros(arg_dict["n_actions"])
                    action_onehot[action] = 1
                    actions.append(action)
                    actions_onehot.append([list(action_onehot)])
                    avail_actions.append(avail_action_s[agent_id])
                    last_action[agent_id] = action_onehot
                elif arg_dict['algorithm'] == 'ssg_vdn_two':
                    avail_action_s = [state_dict[x]['avail'] for x in range(len(state_dict))]
                    action, h_out = choose_action_ssg(obs_model, last_action, agent_id,
                                                  avail_action_s, epsilon, arg_dict, model, follower_model, h_in)
                    h_out_list.append(h_out)
                    # generate onehot vector of th action
                    action_onehot = np.zeros(arg_dict["n_actions"])
                    action_onehot[action] = 1
                    actions.append(action)
                    actions_onehot.append([list(action_onehot)])
                    avail_actions.append(avail_action_s[agent_id])
                    last_action[agent_id] = action_onehot
            forward_t += time.time() - t1
            h_out = (h_out[0], h_out[1])

            prev_obs = env.observation()
            p_actions_onehot = [actions_onehot[i][0] for i in range(len(actions_onehot))]
            real_action = [0, np.nonzero(p_actions_onehot[1])[0][0], np.nonzero(p_actions_onehot[0])[0][0]]
            obs, rew, done, info = env.step(real_action)
            done = done + 0

            score = rew
            fin_r0 = rewarder.calc_reward(rew[0], prev_obs[0], env.observation()[0])
            fin_r1 = rewarder.calc_reward(rew[1], prev_obs[1], env.observation()[1])
            fin_r = [fin_r0, fin_r1]
            reward = fin_r
            done = [done, done]

            reward_agency = copy.deepcopy(reward)  # for reuse and change the value next timestep
            # terminated_agency = copy.deepcopy(done)

            state_prime_dict, state_prime_dict_tensor = obs_encode(obs, h_out, fe)
            obs_prime = obs_transform(state_prime_dict_tensor)
            obs_prime_inputs = obs_prime
            if (arg_dict["last_action"] and arg_dict["reuse_network"]):
                obs_prime_inputs = add_to_inputs(obs_prime, last_action, arg_dict)  # 将上一动作和重用网络加入进去

            ### transition
            actions1 = np.reshape(actions, [arg_dict["n_agents"], 1])
            actions_onehot1 = np.array(actions_onehot).reshape([arg_dict["n_agents"], arg_dict["n_actions"]])
            avail_u = np.array(avail_actions)
            avail_u_next = []
            for agent_id in range(2):
                avail_action = list(state_prime_dict[agent_id]['avail'])
                avail_u_next.append(avail_action)
            avail_u_next= np.array(avail_u_next)




            steps += 1
            if arg_dict["epsilon_anneal_scale"] == 'step':
                epsilon = epsilon - arg_dict["anneal_epsilon"] if epsilon > arg_dict["min_epsilon"] else epsilon
            # score += rew
            # tot_reward += fin_r
            score = list(map(lambda x: x[0] + x[1], zip(score, rew)))
            tot_reward = list(map(lambda x: x[0] + x[1], zip(tot_reward, fin_r)))

            loop_t += time.time() - init_t
            done = done[0]
            # score = score[0]
            score_list.append(score[0])
            win_rate = compute_win_rate(score_list)
            if done:
                if score[0] > 0 or score[1] > 0 or score[2] > 0:
                    win = 1
                print("score", score, "total reward", tot_reward, "steps", steps)
                summary_data = (win, score, tot_reward, steps, 0, loop_t / steps, forward_t / steps, wait_t / steps)
                summary_queue.put(summary_data)
                # model.load_state_dict(center_model.state_dict())
