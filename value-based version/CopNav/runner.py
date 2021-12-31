import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
from common.utils import *

class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if args.learn and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num, writer):
        time_steps, train_steps, train_steps_dag, evaluate_steps = 0, 0, 0, -1
        end_steps_epoch = []
        step_indx = 0
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward, eps_graph_edges_mean = self.evaluate()
                writer.add_scalar('index/eps_graph_edges_mean', eps_graph_edges_mean, time_steps)
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                # self.plt(num)
                evaluate_steps += 1
            episodes = []
            end_steps = []
            episode_rewards = [0] * self.args.n_agents
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, _, steps, _ = self.rolloutWorker.generate_episode(episode_idx)
                end_steps.append(steps)
                episodes.append(episode)
                time_steps += steps

                for agent in range(len(episode_reward)):
                    episode_rewards[agent] += episode_reward[agent]

                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                # if train_steps % 5 == 0:
                #     lambda_A = 0.2
                #     c_A = 1
                #     # ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = self.agents.policy.train_dag_gnn_outer(lambda_A, c_A)
                #     graph_loss = self.agents.policy.train_dag_gnn_all(lambda_A, c_A)
                #     train_steps_dag += 1
                #     writer.add_scalar('graph/ELBO_loss', ELBO_loss, train_steps_dag)
                #     writer.add_scalar('graph/NLL_loss', NLL_loss, train_steps_dag)
                #     writer.add_scalar('graph/MSE_loss', MSE_loss, train_steps_dag)

                loss = []
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    loss_policy, loss_graph_actor, loss_graph_critic, loss_hA = self.agents.train(mini_batch, train_steps, self.rolloutWorker.epsilon_)
                    loss_policy = loss_policy.detach().numpy()
                    loss_graph_actor = loss_graph_actor.detach().numpy()
                    loss_graph_critic = loss_graph_critic.detach().numpy()
                    loss_hA = loss_hA.detach().numpy()
                    # graph_loss = self.agents.policy.train_dag_gnn_all(mini_batch, lambda_A, c_A).detach().numpy()
                    loss.append(loss_policy + loss_graph_actor + loss_graph_critic)
                train_steps += 1
                writer.add_scalar('loss/loss_all', np.mean(loss), train_steps)
                writer.add_scalar('loss/loss_policy', np.mean(loss_policy), train_steps)
                writer.add_scalar('loss/loss_graph_actor', np.mean(loss_graph_actor), train_steps)
                writer.add_scalar('loss/loss_hA', np.mean(loss_hA), train_steps)
                writer.add_scalar('loss/loss_graph_critic', np.mean(loss_graph_critic), train_steps)

            # end_steps_epoch += end_steps
            step_indx += 1
            writer.add_scalar('index/mean_end_step', np.mean(end_steps), time_steps)
            writer.add_scalar('index/episode_rewards', np.mean(episode_rewards), time_steps)  ## rjq0109 添加评估的reward  ### episode_rewards: <class 'list'>: [-6.37927245923192, -6.37927245923192]
            writer.add_scalar('index/episode_rewards_mean', np.mean(self.episode_rewards), time_steps)

        # win_rate, episode_reward = self.evaluate()
        # print('win_rate is ', win_rate)
        # self.win_rates.append(win_rate)
        # self.episode_rewards.append(episode_reward)

        return end_steps_epoch




    def evaluate(self):
        win_number = 0
        episode_rewards = [0] * self.args.n_agents  # list for two agents
        eps_graph_edges = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _, eps_graph_edge = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            for agent in range(len(episode_reward)):
                episode_rewards[agent] += episode_reward[agent]
            if win_tag:
                win_number += 1
            eps_graph_edges += eps_graph_edge
        eps_rewards = [x / self.args.evaluate_epoch for x in episode_rewards]
        eps_graph_edges_mean = eps_graph_edges / self.args.evaluate_epoch
        # return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch
        return win_number / self.args.evaluate_epoch, eps_rewards, eps_graph_edges_mean









