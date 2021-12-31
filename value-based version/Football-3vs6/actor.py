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
from models.agents.utils import pruning, is_acyclic, pruning_1
import igraph as ig
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

def choose_action_asg(obs, last_action, leader_id, step, avail_actions, epsilon, arg_dict, model, follower_model, h_in):
    inputs = obs.copy()
    # if step % arg_dict["maintain_step"] == 0:
    #     leader_id_one_hot, self.policy.election_hidden = self.policy.get_leader_id(inputs, self.policy.election_hidden)
    # leader_id = torch.argmax(leader_id_one_hot.clone().detach()).item()
    leader_id = 0
    folower_indices = [j for j in range(arg_dict["n_agents"])]
    folower_indices.remove(leader_id)
    folower_indices = torch.tensor(np.array(folower_indices), dtype=torch.int64)
    avail_actions_ind_l = np.nonzero(avail_actions[leader_id])[0]  # index of actions which can be choose
    avail_actions_ind_f = [np.nonzero(avail_actions[j])[0] for j in folower_indices]  # index of actions which can be choose

    agent_id = np.eye(arg_dict["n_agents"])

    if arg_dict["last_action"]:
        inputs = np.array([np.hstack((inputs[i].reshape([1, -1]), last_action[i].reshape([1, -1]))) for i in range(arg_dict["n_agents"])])
    if arg_dict["reuse_network"]:
        inputs = np.array([np.hstack((inputs[i].reshape([1, -1]), agent_id[i].reshape([1, -1]))) for i in range(arg_dict["n_agents"])])

    inputs_l = torch.tensor(inputs[leader_id], dtype=torch.float32)
    inputs_f = torch.tensor([inputs[j] for j in folower_indices], dtype=torch.float32)

    # get q value
    with torch.no_grad():
        q_value_l, h_out_l = model(inputs_l, h_in[0])
        f_h_in = torch.stack(h_in[1:])
        q_value_f, h_out_f = follower_model(inputs_f.squeeze(1), q_value_l, f_h_in)
    h_out = torch.cat([h_out_l.unsqueeze(0), h_out_f.permute(1,0,2)], dim=0) # 3.1.64
    # choose action from q value
    avail_actions = torch.tensor(avail_actions)
    q_value_l[avail_actions[[leader_id]] == 0.0] = -float("inf")
    q_value_f = q_value_f.unsqueeze(1)  # n_agetns-1 x act_dims
    for x in range(q_value_f.shape[0]):
        q_value_f[x][avail_actions[[[folower_indices[x]]]] == 0] = -float("inf")
    q_value_f = q_value_f.squeeze(1)

    if np.random.uniform() < epsilon:
        action_l = np.random.choice(avail_actions_ind_l)  # action是一个整数
        action_f = [np.random.choice(avail_actions_ind_f[j]) for j in range(arg_dict["n_agents"] - 1)]

    else:
        action_l = torch.argmax(q_value_l)
        action_f = torch.argmax(q_value_f, dim=-1)
    final_act_f = list(action_f)

    actions_ = [0, 0, 0]
    actions_[leader_id] = action_l
    actions_[folower_indices[0]] = final_act_f[0]
    actions_[folower_indices[1]] = final_act_f[1]

    return actions_, h_out, leader_id


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




def choose_action_dag_graph(obs, avail_actions, epsilon, model, h_in):
    inputs = obs.copy()
    avail_actions_ind = np.nonzero(avail_actions)[0] # (11,)

    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

    # get q value
    with torch.no_grad():
        q_value, h_out = model(inputs, h_in)  #(1,19)(1,256)

    # choose action from q value
    q_value[avail_actions == 0.0] = - float("inf")
    if np.random.uniform() < epsilon:
        action = np.random.choice(avail_actions_ind)  # action是一个整数
    else:
        action = torch.argmax(q_value)
    return action, h_out

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
    flattern_obs_2 = []
    for k, v in enumerate(state_dict_tensor):
        if k != 'hidden': # hideen这一维度去掉
            flattern_obs_0.append(state_dict_tensor[v][0].reshape([-1]))
            flattern_obs_1.append(state_dict_tensor[v][1].reshape([-1]))
            flattern_obs_2.append(state_dict_tensor[v][2].reshape([-1]))

    flattern_obs_0 = torch.hstack(flattern_obs_0)
    flattern_obs_1 = torch.hstack(flattern_obs_1)
    flattern_obs_2 = torch.hstack(flattern_obs_2)
    flattern_obs = torch.stack((flattern_obs_0, flattern_obs_1, flattern_obs_2), dim=0)

    return flattern_obs.unsqueeze(1).numpy()

def obs_encode(obs, h_in, fe):  # 将obs和h_out 编码成state_dict,state_dict_tensor
    # h_in = h_out
    state_dict = []
    state_dict_tensor = {}
    for i in range(len(obs)):
        state_dict1 = fe.encode(obs[i])  # 长度为7的字典
        state_dict_tensor1 = state_to_tensor(state_dict1, h_in)
        state_dict.append(state_dict1)
        if i == 0:
            state_dict_tensor = state_dict_tensor1
        else:
            for k, v in state_dict_tensor1.items():
                state_dict_tensor[k] = torch.cat((state_dict_tensor[k], state_dict_tensor1[k]), 0)
        # if obs[i]['active'] == 0:
        #     state_dict1 = fe.encode(obs[i])  # 长度为7的字典
        #     state_dict_tensor1 = state_to_tensor(state_dict1, h_in)
        # else:
        #     state_dict2 = fe.encode(obs[i])
        #     state_dict_tensor2 = state_to_tensor(state_dict2, h_in)
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

def actor(actor_num, center_model, center_actor_model, data_queue, signal_queue, summary_queue, arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    print("Actor process {} started".format(actor_num))
    fe_module = importlib.import_module("encoders." + arg_dict["encoder"])
    rewarder = importlib.import_module("rewarders." + arg_dict["rewarder"])
    imported_model = importlib.import_module("models.agents." + arg_dict["model"])
    actor_imported_model = importlib.import_module("models.agents." + arg_dict["graph_actor"])

    fe = fe_module.FeatureEncoder()
    # 根据参数决定RNN的输入维度
    input_shape = arg_dict["state_shape"]
    if arg_dict["last_action"]:
        input_shape += arg_dict["n_actions"]
    if arg_dict["reuse_network"]:
        input_shape += arg_dict["n_agents"]
    if arg_dict["father_action"]:
        input_shape += arg_dict["n_agents"] * arg_dict["n_actions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = imported_model.RNNAgent(input_shape, arg_dict)
    actor_model = actor_imported_model.Actor_graph(arg_dict)
    model.load_state_dict(center_model.state_dict())
    actor_model.load_state_dict(center_actor_model.state_dict())

    env = football_env.create_environment(env_name=arg_dict["env"], number_of_left_players_agent_controls=3,
                                          representation="raw", stacked=False, logdir='./tmp/football/ssg2', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    # print("-----------------number_of_players_agent_controls", env._players)
    n_epi = 0
    rollout = []
    score_list = []
    eps_edges = 0
    epsilon = arg_dict["epsilon"]
    while True:  # episode loop
        env.reset()
        done = False
        steps, score, tot_reward, win = 0, [0, 0, 0], [0, 0, 0], 0
        n_epi += 1
        h_out = (torch.zeros([1, arg_dict["lstm_size"]], dtype=torch.float),
                 torch.zeros([1, arg_dict["lstm_size"]], dtype=torch.float),
                 torch.zeros([1, arg_dict["lstm_size"]], dtype=torch.float))  ##rjq ((1,256),(1,256))

        loop_t, forward_t, wait_t = 0.0, 0.0, 0.0
        obs = env.observation()  # [,]

        # last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        last_action = np.zeros((arg_dict["n_agents"], arg_dict["n_actions"]))
        obs_prime_prime_inputs = []
        # step = 0
        while not done and steps < arg_dict["episode_limit"]:  # step loop
            init_t = time.time()

            is_stopped = False
            while signal_queue.qsize() > 0:
                time.sleep(0.02)
                is_stopped = True
            if is_stopped:
                model.load_state_dict(center_model.state_dict())
                actor_model.load_state_dict(center_actor_model.state_dict())
            wait_t += time.time() - init_t

            h_in = h_out
            # state_dict, state_dict_tensor = obs_encode(obs, fe)  # state_dict:[dict,dict]  state_dict_tensor:dict
            state_dict, state_dict_tensor = obs_encode(obs, h_in, fe)
            obs_model = obs_transform(state_dict_tensor)
            obs_model_inputs = obs_model  # 3.1.122
            if (arg_dict["last_action"] and arg_dict["reuse_network"]):
                obs_model_inputs = add_to_inputs(obs_model, last_action, arg_dict) # 将上一动作和重用网络加入进去  # 3.1.144

            t1 = time.time()
            actions, avail_actions, actions_onehot = [], [], []
            # h_out_list = []

            if arg_dict['algorithm'] == 'vdn_graph':
                avail_action_s = [state_dict[x]['avail'] for x in range(len(state_dict))]

                inputs_graph = torch.tensor(obs_model_inputs).float().permute(1,0,2)  ## 3.1.144 --> 1.3.144
                encoder_output, samples, mask_scores, entropy, adj_prob, \
                log_softmax_logits_for_rewards, entropy_regularization = actor_model(inputs_graph)
                graph_A = torch.stack(samples).squeeze(1).clone().numpy()
                eps_edges += np.sum(graph_A)  #### log
                ######## pruning
                G = ig.Graph.Weighted_Adjacency(graph_A.tolist())
                # print(G)
                if not is_acyclic(graph_A):
                    G, new_A = pruning_1(G, graph_A)
                    # print(graph_A)
                    # print(new_A)
                ordered_vertices = G.topological_sorting()

                actions, avail_actions, actions_onehot = [0] * arg_dict["n_agents"], [0] * arg_dict["n_agents"], [0] * arg_dict["n_agents"]
                father_action_lst = [0] * arg_dict["n_agents"]
                h_out_lst = [0] * arg_dict["n_agents"]
                for j in ordered_vertices:  # 原始采样  #   [1, 2, 0]
                    avail_action = avail_action_s[j]  # for particle env
                    father_action_0 = torch.tensor(np.zeros((arg_dict["n_agents"], arg_dict["n_actions"])))
                    parents = G.neighbors(j, mode=ig.IN)
                    if len(parents) != 0:
                        # print(steps, j, parents)
                        for i in parents:
                            # print(i)
                            father_action_0[i] = torch.tensor(actions_onehot[i][0])
                    father_action = father_action_0
                    father_action = father_action.reshape(-1)
                    father_action_lst[j] = father_action

                    obs_j = np.hstack((obs_model_inputs[j][0], father_action))
                    action, h_out = choose_action_dag_graph(obs_j, avail_action, epsilon, model, h_in[j])  # add all
                    h_out_lst[j] = h_out
                    # generate onehot vector of th action
                    action_onehot = np.zeros(arg_dict["n_actions"])
                    action_onehot[action] = 1
                    actions[j] = np.int(action)
                    actions_onehot[j] = [list(action_onehot)]
                    avail_actions[j] = avail_action
                    last_action[j] = action_onehot

            forward_t += time.time() - t1
            h_out = (h_out_lst[0], h_out_lst[1], h_out_lst[2])

            prev_obs = env.observation()
            p_actions_onehot = [actions_onehot[i][0] for i in range(len(actions_onehot))]
            # active_flag = [obs[0]['active'], obs[1]['active'], obs[2]['active']]
            # real_action = [0, 0, 0]
            # for i in range(len(active_flag)):
            #     if active_flag[i] == 0:
            #         real_action[i] = 0
            #     elif active_flag[i] == 2:
            #         real_action[i] = np.nonzero(p_actions_onehot[0])[0][0]
            #     elif active_flag[i] == 1:
            #         real_action[i] = np.nonzero(p_actions_onehot[1])[0][0]
            real_action = [np.nonzero(p_actions_onehot[0])[0][0], np.nonzero(p_actions_onehot[1])[0][0],
                           np.nonzero(p_actions_onehot[2])[0][0]]
            obs, rew, done, info = env.step(real_action)
            done = done + 0

            # score = rew
            fin_r0 = rewarder.calc_reward(rew[0], prev_obs[0], env.observation()[0])
            fin_r1 = rewarder.calc_reward(rew[1], prev_obs[1], env.observation()[1])
            fin_r2 = rewarder.calc_reward(rew[2], prev_obs[2], env.observation()[2])
            fin_r = [fin_r0, fin_r1, fin_r2]
            reward = fin_r
            done = [done, done, done]

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
            for agent_id in range(arg_dict["n_agents"]):
                avail_action = list(state_prime_dict[agent_id]['avail'])
                avail_u_next.append(avail_action)
            avail_u_next= np.array(avail_u_next)
            father_action_lst = np.array([i.tolist() for i in father_action_lst])

            transition = (obs_model_inputs, h_in, actions1, actions_onehot1, avail_u, reward_agency, reward_agency, obs_prime_inputs, h_out, avail_u_next, done, father_action_lst)

            rollout.append(transition)
            if len(rollout) == arg_dict["rollout_len"]:
                # rollout = normalize_reward(rollout)
                data_queue.put(rollout)
                rollout = []
                model.load_state_dict(center_model.state_dict())  # rjq check
                actor_model.load_state_dict(center_actor_model.state_dict())

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
            if done or steps == arg_dict["episode_limit"]:
                if score[0] > 0 or score[1] > 0 or score[2] > 0 :
                    win = 1
                print("score", score, "total reward", tot_reward, "steps", steps)
                tot_reward = list(np.array(tot_reward)/steps*200.0) # for the fairness on reward
                summary_data = (win, score, tot_reward, steps, 0, loop_t / steps, forward_t / steps, wait_t / steps, eps_edges)
                summary_queue.put(summary_data)
                # model.load_state_dict(center_model.state_dict())



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


def get_action2(a_prob, m_prob):  # a_prob(1,2,12)   m_prob(1,2,8) # 两个player各自的动作
    a_prob1 = torch.chunk(a_prob, dim=0, chunks=2)
    m_prob1 = torch.chunk(m_prob, dim=0, chunks=2)
    real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = [], [], [], [], [], [], []
    for idx, t_chunk in enumerate(a_prob1):
        real_action1, a1, m1, need_m1, prob1, prob_selected_a1, prob_selected_m1 = get_action(a_prob1[idx],
                                                                                              m_prob1[idx])
        real_action.append(real_action1)
        a.append(a1)
        m.append(m1)
        need_m.append(need_m1)
        prob.append(prob1)
        prob_selected_a.append(prob_selected_a1)
        prob_selected_m.append(prob_selected_m1)

    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m


def get_action_value(q_a, q_m):  # rjq 0114 补全  rjq 0116 debug
    # a = torch.argmax(q_a, dim=0)
    q_a_1, q_a_2 = q_a[0][0], q_a[1][0] # rjq 0116 debug
    q_m_1, q_m_2 = q_m[0][0], q_m[1][0] # rjq 0116 debug
    a1 = torch.argmax(q_a_1).item()
    a2 = torch.argmax(q_a_2).item()
    m1, need_m1 = 0, 0
    m2, need_m2 = 0, 0
    prob_selected_a1 = q_a_1[a1].item()
    prob_selected_a2 = q_a_2[a2].item()
    prob_selected_m1, prob_selected_m2 = 0, 0
    if a1 == 0:
        real_action1 = a1
        prob1 = prob_selected_a1
    elif a1 == 1:
        m1 = torch.argmax(q_m_1).item()
        need_m1 = 1
        real_action1 = m1 + 1
        prob_selected_m1 = q_m_1[m1].item()
        prob1 = prob_selected_a1 * prob_selected_m1
    else:
        real_action1 = a1 + 7
        prob1 = prob_selected_a1

    if a2 == 0:
        real_action2 = a2
        prob2 = prob_selected_a2
    elif a2 == 1:
        m2 = torch.argmax(q_m_2).item()
        need_m2 = 1
        real_action2 = m2 + 1
        prob_selected_m2 = q_m_2[m2].item()
        prob2 = prob_selected_a2 * prob_selected_m2
    else:
        real_action2 = a2 + 7
        prob2 = prob_selected_a2
    real_action = [real_action1, real_action2]
    a = [a1, a2]
    m = [m1, m2]
    need_m = [need_m1, need_m2]
    prob = [prob1, prob2]
    prob_selected_a = [prob_selected_a1, prob_selected_a2]
    prob_selected_m = [prob_selected_m1, prob_selected_m2]

    assert prob1 != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a1, m1, prob_selected_a, prob_selected_m)
    assert prob2 != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a1, m1, prob_selected_a, prob_selected_m)

    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m



def select_opponent(arg_dict):  # 50%的对手来自于最新的10个模型，其余的来自于从整个池中统一随机抽样
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
            opp_model_num = random.randint(len(model_num_lst) - arg_dict["latest_n_model"], len(model_num_lst) - 1)
        else:
            opp_model_num = len(model_num_lst) - 1
    else:
        opp_model_num = random.randint(0, len(model_num_lst) - 1)

    model_name = "/model_" + str(model_num_lst[opp_model_num]) + ".tar"
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

    env = football_env.create_environment(env_name=arg_dict["env"], number_of_left_players_agent_controls=2,
                                          representation="raw", \
                                          stacked=False, logdir='/tmp/football', write_goal_dumps=False,
                                          write_full_episode_dumps=False, \
                                          render=False)

    n_epi = 0
    rollout = []
    while True:  # episode loop
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
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))  # ([1,1,256], [1,1,256])

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
            forward_t += time.time() - t1

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

            # if arg_dict['print_mode']:
            #     print_status(steps, a, m, prob_selected_a, prob_selected_m, prev_obs, obs, fin_r, tot_reward)

            loop_t += time.time() - init_t

            if done:
                if score > 0:
                    win = 1
                print("score {}, total reward {:.2f}, opp num:{}, opp:{} ".format(score, tot_reward, opp_model_num,
                                                                                  opp_model_path))
                summary_data = (
                    win, score, tot_reward, steps, str(opp_model_num), loop_t / steps, forward_t / steps,
                    wait_t / steps)
                summary_queue.put(summary_data)
