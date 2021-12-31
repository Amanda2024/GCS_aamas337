import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import copy
from torch.autograd import Variable
from common.utils import pruning, is_acyclic, pruning_1
import igraph as ig

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        self.epsilon_ = args.epsilon_
        self.anneal_epsilon_ = args.anneal_epsilon_
        self.min_epsilon_ = args.min_epsilon_

        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded, father_actions = [], [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = [0] * self.args.n_agents # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # epsilon_
        epsilon_ = 0 if evaluate else self.epsilon_
        if self.args.epsilon_anneal_scale_ == 'episode':
            epsilon_ = epsilon_ - self.anneal_epsilon_ if epsilon_ > self.min_epsilon_ else epsilon_

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        eps_edges = 0
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = obs
            # state = self.env.get_state()
            agent_id_graph = np.eye(self.n_agents)
            inputs_graph = np.hstack((obs, last_action))
            inputs_graph = np.hstack((inputs_graph, agent_id_graph))
            # inputs_graph = Variable(torch.tensor(inputs_graph)).double()
            inputs_graph = torch.tensor(inputs_graph).unsqueeze(0).float()

            encoder_output, samples, mask_scores, entropy, adj_prob,\
            log_softmax_logits_for_rewards, entropy_regularization = self.agents.policy.actor(inputs_graph)
            # print(mask_scores)
            graph_A = torch.stack(samples).squeeze(1).clone().numpy()
            eps_edges += np.sum(graph_A)  #### log
            # graph_A[np.abs(graph_A) < self.args.graph_threshold] = 0
            ######## pruning
            G = ig.Graph.Weighted_Adjacency(graph_A.tolist())
            # print(G)
            if not is_acyclic(graph_A):
                G, new_A = pruning_1(G, graph_A)
                print(graph_A)
                print(new_A)
            ordered_vertices = G.topological_sorting()
            # ordered_vertices = G.topological_sorting()

            actions, avail_actions, actions_onehot = [0] * self.n_agents, [0] * self.n_agents, [0] * self.n_agents
            father_action_lst = [0] * self.n_agents
            for j in ordered_vertices:  # 原始采样  #   [3, 4, 1, 2, 0]
                avail_action = [1 for i in range(self.args.n_actions)]  # for particle env
                father_action_0 = torch.tensor(np.zeros((self.n_agents, self.n_actions)))
                parents = G.neighbors(j, mode=ig.IN)
                if len(parents) != 0:
                    print(step, j, parents)
                    for i in parents:
                        # print(i)
                        father_action_0[i] = torch.tensor(actions_onehot[i][0])
                father_action = father_action_0
                father_action = father_action.reshape(-1)
                father_action_lst[j] = father_action

                # obs_j = np.hstack((obs[j], father_action))
                action = self.agents.choose_action_dag_graph(obs[j],  last_action[j], j, father_action, avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions[j] = np.int(action)
                actions_onehot[j] = [list(action_onehot)]
                avail_actions[j] = avail_action
                last_action[j] = action_onehot

            if self.args.scenario == 'simple_spread.py':
                p_actions_onehot = [actions_onehot[i][0] for i in range(len(actions_onehot))]  # 3-d --> 2-d
                _, reward, terminated, info = self.env.step(p_actions_onehot)
                terminated = terminated[0] + 0
            else:
                _, reward, terminated, info = self.env.step(actions_onehot, step)

            reward_agency = copy.deepcopy(reward)  # for reuse and change the value next timestep
            terminated_agency = copy.deepcopy(terminated)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False

            if self.args.scenario == 'simple_spread.py':
                obs, state = np.array(obs), np.array(state)
                father_action_lst = np.array([i.tolist() for i in father_action_lst])
            o.append(obs.reshape([self.n_agents, self.obs_shape]))
            s.append(state.reshape([self.n_agents, self.obs_shape]))
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(np.array(actions_onehot).reshape([self.n_agents, self.n_actions]))
            avail_u.append(np.array(avail_actions))
            r.append([reward_agency])
            terminate.append([terminated_agency])
            padded.append([0.])
            father_actions.append(father_action_lst)
            for agent in range(len(reward)):
                episode_reward[agent] += reward[agent]

            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        # last obs
        if self.args.scenario == 'simple_spread.py' and len(terminate) == self.args.episode_limit:
            terminate[-1] = [1]

        o.append(obs.reshape([self.n_agents, self.obs_shape]))
        s.append(state.reshape([self.n_agents, self.obs_shape]))
        father_actions.append(father_action_lst)
        father_actions_next = father_actions[1:]
        father_actions = father_actions[:-1]
        #
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            if self.args.scenario == 'simple_spread.py':
                avail_action = [1 for i in range(self.args.n_actions)]
            else:
                avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(np.array(avail_actions))

        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros((self.n_agents, self.obs_shape)))
            r.append([[0., 0.]])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros((self.n_agents, self.state_shape)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([True])
            father_actions.append(np.zeros((self.n_agents, self.n_agents * self.n_actions)))
            father_actions_next.append(np.zeros((self.n_agents, self.n_agents * self.n_actions)))



        episode = dict(o=o.copy(),
                           s=s.copy(),
                           u=u.copy(),
                           r=r.copy(),
                           avail_u=avail_u.copy(),
                           o_next=o_next.copy(),
                           s_next=s_next.copy(),
                           avail_u_next=avail_u_next.copy(),
                           u_onehot=u_onehot.copy(),
                           padded=padded.copy(),
                           terminated=terminate.copy(),
                           father_actions=father_actions.copy(),
                           father_actions_next=father_actions_next.copy()
                           )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()


        end_step = step
        for agent in range(len(reward)):  # mean reward per episode
            episode_reward[agent] = episode_reward[agent] / end_step

        return episode, episode_reward, win_tag, end_step, eps_edges


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon

        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = self.epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step
