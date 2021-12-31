import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
from models.mixers.vdn import VDNMixer
from models.mixers.qmix_net import QMixNet
from models.agents.rnn_agent import RNNAgent
from torch.optim import Adam
import pdb


class SSG_VDN():
    def __init__(self, arg_dict, model, follower_model, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]
        self.params = list(model.parameters())
        self.last_target_update_step = 0
        self.optimization_step = 0
        self.arg_dict = arg_dict

        self.n_actions = self.arg_dict["n_actions"]
        self.n_agents = self.arg_dict["n_agents"]
        self.state_shape = self.arg_dict["state_shape"]
        self.obs_shape = self.arg_dict["obs_shape"]
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if self.arg_dict["last_action"]:
            input_shape += self.n_actions
        if self.arg_dict["reuse_network"]:
            input_shape += self.n_agents

        # 神经网络
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        # self.eval_rnn = RNNAgent(input_shape, self.arg_dict, device)  # 每个agent选动作的网络
        # self.target_rnn = RNNAgent(input_shape, self.arg_dict, device)

        self.mixer = None
        if arg_dict["mixer"] is not None:
            if arg_dict["mixer"] == "vdn":
                self.mixer = VDNMixer()
            elif arg_dict["mixer"] == "qmix":
                self.mixer = QMixNet(arg_dict)
            else:
                raise ValueError("Mixer {} not recognised".format(arg_dict["mixer"]))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = Adam(params=self.params, lr=arg_dict["learning_rate"])

        self.model = model
        self.follower_model = follower_model
        self.target_model = copy.deepcopy(self.model)
        self.target_follower_model = copy.deepcopy(self.follower_model)

    def split_agents(self, value): # 输入维度：[120,32], 其中120代表2个agent的30个transition，奇数表示agent1，偶数表示agent2
        q_x_1 = torch.Tensor([])
        q_x_2 = torch.Tensor([])
        for i in range(self.arg_dict["rollout_len"]):
            q_a_1 = value[2 * i]  # (12)
            q_a_2 = value[2 * i + 1]
            q_x_1 = torch.cat([q_x_1, q_a_1], dim=0)
            q_x_2 = torch.cat([q_x_2, q_a_2], dim=0)
        return torch.stack((q_x_1, q_x_2), dim=0)  # (2, 60*32)

    def obtain_one_state(self, state): # 输入维度：[120,32,136],其中120代表2个agent的30个transition，奇数表示agent1，偶数表示agent2
        q_x_1 = torch.Tensor([])
        q_x_2 = torch.Tensor([])
        for i in range(self.arg_dict["rollout_len"]):
            q_a_1 = state[2 * i]  # (12)
            q_a_2 = state[2 * i + 1]
            q_x_1 = torch.cat([q_x_1, q_a_1], dim=0)
            q_x_2 = torch.cat([q_x_2, q_a_2], dim=0)
        return q_x_1  # (60,32,136)

    def train(self, model, follower_model, mixer, data):

        ### rjq debug 0118
        # self.target_model.load_state_dict(model.state_dict())  # rjq 0114  传入self.model
        # if self.target_mixer is not None:
        #     self.target_mixer.load_state_dict(mixer.state_dict())  # mixer.state_dict() == self.target_mixer.state_dict()
        self.model.init_hidden()
        self.follower_model.init_hidden()

        self.target_model.init_hidden()
        self.target_follower_model.init_hidden()

        loss = []
        for mini_batch in data:
            # pdb.set_trace()
            # obs_model, h_in, actions1, avail_u, actions_onehot1, reward_agency, obs_prime, h_out, avail_u_next, done
            s, h_in, a, a1, avail_u, r, r_prime, s_prime_prime, h_out_prime, avail_u_next_next, done_mask = mini_batch
            # print("")
            ind = []
            for i in range(int(s.shape[0] / self.n_agents)):
                ind.append(2*i)
            ind = torch.from_numpy(np.array(ind))
            s = torch.cat((s[ind].unsqueeze(0), s[ind+1].unsqueeze(0)), dim=0)
            # s = s.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            h_in = torch.cat((h_in[ind].unsqueeze(0), h_in[ind+1].unsqueeze(0)), dim=0)
            # h_in = h_in.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            leader_q, _ = self.model(s[0].unsqueeze(2), h_in[0])
            leader_q = leader_q.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])#
            follower_q , _= self.follower_model(s[1], leader_q, h_in[1])
            follower_q = follower_q.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])
            # q = q.reshape(self.arg_dict["rollout_len"] *2, self.arg_dict["batch_size"], -1)

            # s_prime = torch.cat((s_prime[ind].unsqueeze(0), s_prime[ind + 1].unsqueeze(0)), dim=0)
            s_prime_prime = torch.cat((s_prime_prime[ind].unsqueeze(0), s_prime_prime[ind + 1].unsqueeze(0)), dim=0)
            # s_prime = s_prime.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            # h_out = torch.cat((h_out[ind].unsqueeze(0), h_out[ind+1].unsqueeze(0)), dim=0)
            h_out_prime = torch.cat((h_out_prime[ind].unsqueeze(0), h_out_prime[ind+1].unsqueeze(0)), dim=0)
            # h_out = h_out.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            target_q, _ = self.target_model(s_prime_prime[0], h_out_prime[0])
            target_q_ = target_q.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])
            target_follower_q, _ = self.target_follower_model(s_prime_prime[1], target_q_, h_out_prime[1])
            target_follower_q_ = target_follower_q.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])
            # target_q_ = target_q.reshape(self.arg_dict["rollout_len"] * 2, self.arg_dict["batch_size"], -1)

            a = torch.cat((a[ind].unsqueeze(0), a[ind + 1].unsqueeze(0)), dim=0)
            leader_q_ = torch.gather(leader_q, dim=2, index=a[0].long()).squeeze(2)
            follower_q_ = torch.gather(follower_q, dim=2, index=a[1].long()).squeeze(2)

            # avail_u_next = torch.cat((avail_u_next[ind].unsqueeze(0), avail_u_next[ind + 1].unsqueeze(0)), dim=0)
            avail_u_next_next = torch.cat((avail_u_next_next[ind].unsqueeze(0), avail_u_next_next[ind + 1].unsqueeze(0)), dim=0)
            target_q_[avail_u_next_next[0] == 0.0 ] = - 9999999
            target_follower_q_[avail_u_next_next[1] == 0.0 ] = - 9999999
            target_q_ = target_q_.max(dim=2)[0]
            target_follower_q_ = target_follower_q_.max(dim=2)[0]



            if self.mixer is not None:
                if self.arg_dict["mixer"] == "qmix":
                    q_ = self.split_agents(q_).permute(1, 0).reshape(self.arg_dict["rollout_len"],
                                                                     self.arg_dict["batch_size"], -1).permute(1, 0, 2)  # --> (32, 30, 2)
                    target_q_ = self.split_agents(target_q_).permute(1, 0).reshape(self.arg_dict["rollout_len"],
                                                                                   self.arg_dict["batch_size"],
                                                                                   -1).permute(1, 0, 2)  # --> (32, 30, 2)
                    s = s.reshape(self.arg_dict["rollout_len"] * 2, self.arg_dict["batch_size"], -1)
                    s_1 = self.obtain_one_state(s).reshape(self.arg_dict["rollout_len"], self.arg_dict["batch_size"],
                                                           -1).permute(1, 0, 2)  # (batch_size, rollout_len, 136)
                    s_prime = s_prime.reshape(self.arg_dict["rollout_len"] * 2, self.arg_dict["batch_size"], -1)
                    s_prime_1 = self.obtain_one_state(s_prime).reshape(self.arg_dict["rollout_len"],
                                                                       self.arg_dict["batch_size"], -1).permute(1, 0, 2)  # (batch_size, rollout_len, 136)
                    q_total_q_a = self.mixer(q_, s_1)  # ( 32,30,2) ( 32,30,136)  ### q_total_q_a ( 32,30,1)
                    q_total_target_max_q = self.target_mixer(target_q_, s_prime_1)
                    q_total_target_max_q = q_total_target_max_q.permute(1, 0, 2).reshape(
                        self.arg_dict["rollout_len"] * self.arg_dict["batch_size"], -1).permute(1, 0)
                    q_total_q_a = q_total_q_a.permute(1, 0, 2).reshape(
                        self.arg_dict["rollout_len"] * self.arg_dict["batch_size"], -1).permute(1, 0)

                elif self.arg_dict["mixer"] == "vdn":
                    # q_ = self.split_agents(q_)  # [60,5] --> (2, 30*5)
                    # target_q_ = self.split_agents(target_q_)  # (2, 30*5)
                    q_ = torch.cat([leader_q_.unsqueeze(0), follower_q_.unsqueeze(0)], dim=0)
                    target_q_ = torch.cat(([target_q_.unsqueeze(0), target_follower_q_.unsqueeze(0)]), dim=0)
                    q_total_q_a = self.mixer(q_)  # (1, 60*32)
                    q_total_target_max_q = self.target_mixer(target_q_)  # (1, 60*32)

            # s = self.split_agents((done_mask).squeeze(2))[0].unsqueeze(0)
            s = torch.cat((done_mask[ind].unsqueeze(0), done_mask[ind+1].unsqueeze(0)), dim=0)
            # r_stack = self.split_agents(r.squeeze(2))  # (120,32) --> (2, 60*32)
            r_stack = torch.cat((r[ind].unsqueeze(0), r[ind+1].unsqueeze(0)), dim=0)
            r_stack_prime = torch.cat((r_prime[ind].unsqueeze(0), r_prime[ind+1].unsqueeze(0)), dim=0)
            r_total = torch.sum(r_stack, dim=0, keepdim=True).squeeze(-1)  # (1, 60*32)
            r_total_prime = torch.sum(r_stack_prime, dim=0, keepdim=True).squeeze(-1)

            targets = r_total + self.arg_dict["gamma"] * (r_total_prime + (self.arg_dict["gamma"] * s[0].squeeze(-1) * (q_total_target_max_q)))  # (1, 60*32) tdone_mask is the same over each agent

            td_error = (q_total_q_a - targets.detach()).transpose(1,0)  #(1920, 1)
            loss_mini = torch.mean((td_error ** 2))
            loss.append(loss_mini)

        loss = torch.mean(torch.stack(loss), 0)
        # loss = torch.sum(loss)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        self.optimization_step += self.arg_dict["batch_size"] * self.arg_dict["buffer_size"] * self.arg_dict["k_epoch"]
        if (self.optimization_step - self.last_target_update_step) / self.arg_dict["target_update_interval"] >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.optimization_step
            print("self.last_target_update_step:---", self.last_target_update_step)

        return torch.mean(loss)


    def _update_targets(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_follower_model.load_state_dict(self.follower_model.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        torch.save(self.model.state_dict(), "{}agent.th".format(path))
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}mixer.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}opt.th".format(path))
        print("Model saved :", path)

    def load_models(self, path):
        self.model.load_models(path)
        self.target_model.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load("{}/mixer.th".format(path)),
                                       map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path)), map_location=lambda storage, loc: storage)
