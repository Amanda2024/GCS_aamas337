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
    def __init__(self, arg_dict, model, follower_model, mixer, coma_critic, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.td_lambda = arg_dict["td_lambda"]
        self.epsilon = arg_dict['epsilon']
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
        # device = "cpu"
        # self.eval_rnn = RNNAgent(input_shape, self.arg_dict, device)  # 每个agent选动作的网络
        # self.target_rnn = RNNAgent(input_shape, self.arg_dict, device)

        # self.mixer = None
        # if arg_dict["mixer"] is not None:
        #     if arg_dict["mixer"] == "vdn":
        #         self.mixer = VDNMixer()
        #     elif arg_dict["mixer"] == "qmix":
        #         self.mixer = QMixNet(arg_dict)
        #     else:
        #         raise ValueError("Mixer {} not recognised".format(arg_dict["mixer"]))
        #     self.params += list(self.mixer.parameters())
        #     self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimizer = Adam(params=self.params, lr=arg_dict["learning_rate"])

        self.target_model = copy.deepcopy(model)
        self.target_follower_model = copy.deepcopy(follower_model)
        self.target_mixer = copy.deepcopy(mixer)
        self.target_coma_critic = copy.deepcopy(coma_critic)
        self.eval_parameters = list(model.parameters()) + list(follower_model.parameters()) + list(mixer.parameters())
        self.coma_critic_parameters = list(coma_critic.parameters())
        if arg_dict["optimizer"] == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=arg_dict["learning_rate"])
            self.coma_critic_optimizer = torch.optim.RMSprop(self.coma_critic_parameters, lr=arg_dict["lr_critic"])


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

    # def _train_critic(self, mini_batch, max_episode_len, train_step):
    def _get_critic_inputs(self, s, ss, a1):
        # 取出所有episode上该transition_idx的经验

        obs, obs_next, s, s_next = s, ss, s, ss
        u_onehot = a1
        inputs, inputs_next = [], []
        # 添加状态
        inputs.append(s)
        inputs_next.append(s_next)
        # 添加obs
        inputs.append(obs)
        inputs_next.append(obs_next)

        u_onehot_last = torch.cat((torch.zeros(*a1.shape)[[0]], a1[1:, :, :, :]), dim=0)
        u_onehot_next = torch.cat((a1[1:, :, :, :], a1[[-1]]), dim=0)

        # 添加所有agent的上一个动作
        u_onehot_last = u_onehot_last.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], 1,
                                              -1).repeat(1, 1, self.n_agents, 1)
        u_onehot_ = u_onehot.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], 1, -1).repeat(1, 1,
                                                                                                              self.n_agents,
                                                                                                              1)
        inputs.append(u_onehot_last)
        inputs_next.append(u_onehot_)

        # 添加当前动作
        '''
        因为coma对于当前动作，输入的是其他agent的当前动作，不输入当前agent的动作，为了方便起见，每次虽然输入当前agent的
        当前动作，但是将其置为0相量，也就相当于没有输入。  当前agent指的是leader的动作
        '''
        u_onehot_leader_0 = torch.cat(
            (torch.zeros(*u_onehot[:, :, [0], :].shape), u_onehot[:, :, [1], :], u_onehot[:, :, [2], :]), dim=2)
        u_onehot_next_leader_0 = torch.cat(
            (torch.zeros(*u_onehot[:, :, [0], :].shape), u_onehot_next[:, :, [1], :], u_onehot_next[:, :, [2], :]),
            dim=2)
        u_onehot_follower_0 = torch.cat(
            (u_onehot[:, :, [0], :], torch.zeros(*u_onehot[:, :, [1], :].shape), u_onehot[:, :, [2], :]), dim=2)
        u_onehot_next_follower_0 = torch.cat(
            (u_onehot_next[:, :, [0], :], torch.zeros(*u_onehot[:, :, [1], :].shape), u_onehot[:, :, [2], :]), dim=2)
        u_onehot_follower_1 = torch.cat(
            (u_onehot[:, :, [0], :], u_onehot[:, :, [1], :], torch.zeros(*u_onehot[:, :, [2], :].shape)), dim=2)
        u_onehot_next_follower_1 = torch.cat(
            (u_onehot_next[:, :, [0], :], u_onehot_next[:, :, [1], :], torch.zeros(*u_onehot[:, :, [2], :].shape)),
            dim=2)

        u_onehot_0 = torch.cat((u_onehot_leader_0, u_onehot_follower_0, u_onehot_follower_1), dim=-1)
        u_onehot_next_0 = torch.cat((u_onehot_next_leader_0, u_onehot_next_follower_0, u_onehot_next_follower_1),
                                    dim=-1)

        inputs.append(u_onehot_0)
        inputs_next.append(u_onehot_next_0)

        # 添加agent编号对应的one-hot向量
        '''
        因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在后面添加对应的向量
        即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
        agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
        '''
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(self.arg_dict['rollout_len'],
                                                                                self.arg_dict['batch_size'], -1,
                                                                                -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).unsqueeze(0).expand(self.arg_dict['rollout_len'],
                                                                                     self.arg_dict['batch_size'],
                                                                                     -1, -1))

        # 要把inputs中的5项输入拼起来，并且要把其维度从(episode_num, n_agents, inputs)三维转换成(episode_num * n_agents, inputs)二维
        dimensions = self.arg_dict['rollout_len'] * self.arg_dict['batch_size'] * self.n_agents
        inputs = torch.cat([x.reshape(dimensions, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(dimensions, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def td_lambda_target(self, q_targets, r, mask, terminated):
        # batch.shep = (self.arg_dict['batch_size'], self.arg_dict['rollout_len'],  self.n_agents, self.n_action)
        # q_targets.shape = (self.arg_dict['batch_size'], self.arg_dict['rollout_len'], n_agents)
        episode_num = self.arg_dict['batch_size']
        max_episode_len = self.arg_dict['rollout_len']

        q_targets = q_targets.permute(2, 1, 0)  # 4.10.2
        r = r.permute(1, 0, 2).reshape(episode_num, max_episode_len, -1)
        mask = mask.permute(1, 0, 2).repeat(1, 1, self.n_agents)
        terminated = terminated.squeeze(-1).permute(2, 1, 0)

        n_step_return = torch.zeros((episode_num, max_episode_len, self.n_agents, max_episode_len))  # 4.10.2.10
        for transition_idx in range(max_episode_len - 1, -1, -1):
            # 最后计算1 step return
            n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + self.gamma * \
                                                      q_targets[:, transition_idx] * terminated[:,
                                                                                     transition_idx]) * mask[:,
                                                                                                        transition_idx]  # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
            # 同时要注意n step return对应的index为n-1
            for n in range(1, max_episode_len - transition_idx):
                # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
                # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
                n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + self.gamma * \
                                                          n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:,
                                                                                                            transition_idx]
        # --------------------------------------------------n_step_return---------------------------------------------------

        # --------------------------------------------------lambda return---------------------------------------------------
        '''
        lambda_return.shape = (episode_num, max_episode_len，n_agents)
        '''
        lambda_return = torch.zeros((episode_num, max_episode_len, self.n_agents))
        for transition_idx in range(max_episode_len):
            returns = torch.zeros((episode_num, self.n_agents))
            for n in range(1, max_episode_len - transition_idx):
                returns += pow(self.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
            lambda_return[:, transition_idx] = (1 - self.td_lambda) * returns + \
                                               pow(self.td_lambda, max_episode_len - transition_idx - 1) * \
                                               n_step_return[:, transition_idx, :,
                                               max_episode_len - transition_idx - 1]
        # --------------------------------------------------lambda return---------------------------------------------------
        return lambda_return

    def train(self, model, follower_model, mixer, coma_critic, data):

        # self.model.init_hidden()
        # self.follower_model.init_hidden()
        #
        # self.target_model.init_hidden()
        # self.target_follower_model.init_hidden()

        loss = []
        for mini_batch in data:
            # pdb.set_trace()
            # obs_model, h_in, actions1, avail_u, actions_onehot1, reward_agency, obs_prime, h_out, avail_u_next, done
            s, h_in, a, a1, avail_u, r, r_prime, s_prime_prime, h_out_prime, avail_u_next_next, done_mask = mini_batch
            # print("")
            ind = []
            for i in range(int(s.shape[0] / self.n_agents)):
                ind.append(3*i)
            ind = torch.from_numpy(np.array(ind))
            s = torch.cat((s[ind].unsqueeze(0), s[ind+1].unsqueeze(0), s[ind+2].unsqueeze(0)), dim=0) # 状态的存储是根据leader\follower存储的，每隔一个取一个
            # s = s.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            h_in = torch.cat((h_in[ind].unsqueeze(0), h_in[ind+1].unsqueeze(0), h_in[ind+2].unsqueeze(0)), dim=0)  # torch.Size([3, 10, 4, 144])
            a1 = torch.cat((a1[ind].unsqueeze(0), a1[ind + 1].unsqueeze(0), a1[ind + 2].unsqueeze(0)), dim=0)
            avail_u = torch.cat((avail_u[ind].unsqueeze(0), avail_u[ind + 1].unsqueeze(0), avail_u[ind + 2].unsqueeze(0)), dim=0)
            mask = torch.cat((done_mask[ind].unsqueeze(0), done_mask[ind+1].unsqueeze(0), done_mask[ind+2].unsqueeze(0)), dim=0)[0]  # 10.4.1 ; leader和follower的done_mask是一致的
            # h_in = h_in.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            leader_q, _ = model(s[0].unsqueeze(2), h_in[0])
            leader_q = leader_q.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])#
            q_eval_l = leader_q.clone() ### 算action_prob

            follower_q , _= follower_model(s[1:].reshape(-1, s[1:].shape[-1]), leader_q.reshape(-1, leader_q.shape[-1]),\
                                           h_in[1:].reshape(-1, h_in[1:].shape[-1]).unsqueeze(1))
            follower_q = follower_q.reshape(2, self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim']) ## 2 指的是 follower个数
            # q = q.reshape(self.arg_dict["rollout_len"] *2, self.arg_dict["batch_size"], -1)

            q_values_bk = torch.cat((leader_q.unsqueeze(0), follower_q), dim=0)  # num_agents \ rollout_len\ batch_size\ action_dim

            # s_prime = torch.cat((s_prime[ind].unsqueeze(0), s_prime[ind + 1].unsqueeze(0)), dim=0)
            s_prime_prime = torch.cat((s_prime_prime[ind].unsqueeze(0), s_prime_prime[ind + 1].unsqueeze(0), s_prime_prime[ind + 2].unsqueeze(0)), dim=0)
            # s_prime = s_prime.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            # h_out = torch.cat((h_out[ind].unsqueeze(0), h_out[ind+1].unsqueeze(0)), dim=0)
            h_out_prime = torch.cat((h_out_prime[ind].unsqueeze(0), h_out_prime[ind+1].unsqueeze(0), h_out_prime[ind+2].unsqueeze(0)), dim=0)
            # h_out = h_out.reshape(self.arg_dict["rollout_len"] * 2 * self.arg_dict["batch_size"], -1)
            target_q, _ = self.target_model(s_prime_prime[0], h_out_prime[0])
            target_q_ = target_q.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])
            target_follower_q, _ = self.target_follower_model(s_prime_prime[1:].reshape(-1, s_prime_prime[1:].shape[-1]), target_q_.reshape(-1, target_q_.shape[-1]), \
                                                              h_out_prime[1:].reshape(-1, h_out_prime[1:].shape[-1]).unsqueeze(1))
            target_follower_q_ = target_follower_q.reshape(2, self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.arg_dict['action_dim'])
            # target_q_ = target_q.reshape(self.arg_dict["rollout_len"] * 2, self.arg_dict["batch_size"], -1)
            q_targets_bk = torch.cat((target_q_.unsqueeze(0), target_follower_q_), dim=0)  # num_agents \ rollout_len\ batch_size\ action_dim

            a = torch.cat((a[ind].unsqueeze(0), a[ind + 1].unsqueeze(0), a[ind + 2].unsqueeze(0)), dim=0)
            leader_q_ = torch.gather(leader_q, dim=2, index=a[0].long()).squeeze(2)          # torch.Size([10, 4])
            follower_q_ = torch.gather(follower_q, dim=-1, index=a[1:].long()).squeeze(-1)   # torch.Size([2, 10, 4])

            # avail_u_next = torch.cat((avail_u_next[ind].unsqueeze(0), avail_u_next[ind + 1].unsqueeze(0)), dim=0)
            avail_u_next_next = torch.cat((avail_u_next_next[ind].unsqueeze(0), avail_u_next_next[ind + 1].unsqueeze(0), avail_u_next_next[ind + 2].unsqueeze(0)), dim=0)
            target_q_[avail_u_next_next[0] == 0.0 ] = - 9999999
            target_follower_q_[avail_u_next_next[1:,:,:,:] == 0.0 ] = - 9999999
            target_q_ = target_q_.max(dim=-1)[0]
            target_follower_q_ = target_follower_q_.max(dim=-1)[0]



            if mixer is not None:
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
                    q_total_q_a = mixer(q_, s_1)  # ( 32,30,2) ( 32,30,136)  ### q_total_q_a ( 32,30,1)
                    q_total_target_max_q = self.target_mixer(target_q_, s_prime_1)
                    q_total_target_max_q = q_total_target_max_q.permute(1, 0, 2).reshape(
                        self.arg_dict["rollout_len"] * self.arg_dict["batch_size"], -1).permute(1, 0)
                    q_total_q_a = q_total_q_a.permute(1, 0, 2).reshape(
                        self.arg_dict["rollout_len"] * self.arg_dict["batch_size"], -1).permute(1, 0)

                elif self.arg_dict["mixer"] == "vdn":
                    # q_ = self.split_agents(q_)  # [60,5] --> (2, 30*5)
                    # target_q_ = self.split_agents(target_q_)  # (2, 30*5)
                    q_ = torch.cat([leader_q_.unsqueeze(0), follower_q_], dim=0)  # 3.10.4
                    target_q_ = torch.cat(([target_q_.unsqueeze(0), target_follower_q_]), dim=0)
                    q_total_q_a = mixer(q_)  # (1, 10,4)
                    q_total_target_max_q = self.target_mixer(target_q_)  # (1, 10,4)

            # s = self.split_agents((done_mask).squeeze(2))[0].unsqueeze(0)
            done_mask_2 = torch.cat((done_mask[ind].unsqueeze(0), done_mask[ind+1].unsqueeze(0), done_mask[ind+2].unsqueeze(0)), dim=0) # torch.Size([3, 10, 4, 1]) ### 此处指的是done_mask
            # r_stack = self.split_agents(r.squeeze(2))  # (120,32) --> (2, 60*32)
            r_stack = torch.cat((r[ind].unsqueeze(0), r[ind+1].unsqueeze(0), r[ind+2].unsqueeze(0)), dim=0)
            r_stack_prime = torch.cat((r_prime[ind].unsqueeze(0), r_prime[ind+1].unsqueeze(0), r_prime[ind+2].unsqueeze(0)), dim=0)
            r_total = torch.sum(r_stack, dim=0, keepdim=True).squeeze(-1)  # torch.Size([1, 10, 4])
            r_total_prime = torch.sum(r_stack_prime, dim=0, keepdim=True).squeeze(-1)

            targets = r_total + self.arg_dict["gamma"] * (r_total_prime + (self.arg_dict["gamma"] * done_mask_2[[0]].squeeze(-1) * (q_total_target_max_q)))  # (1, 60*32) tdone_mask is the same over each agent

            td_error = (q_total_q_a - targets.detach())  #(1.10.4)
            # loss_mini = torch.mean((td_error ** 2))
            ########  加入加权vdn TODO:rjq add weight
            w_to_use = self.arg_dict['w_to_use']
            if not self.arg_dict['w_vdn']:
                loss_mini = torch.mean((td_error ** 2))
            else:
                # cur_max_actions = q_values_bk.max(dim=3)[0]
                cur_max_actions = q_values_bk.max(dim=3, keepdim=True)[1]  # torch.Size([2, 10, 4, 1])
                is_max_action = (a == cur_max_actions).min(dim=0)[0].squeeze(-1).unsqueeze(0)  # 10.4.1  --> 1.10.4
                target_max_agent_qvals = torch.gather(q_targets_bk, dim=3, index=cur_max_actions).squeeze(3)  # 2.10.4
                max_action_qtot = self.target_mixer(target_max_agent_qvals)  # 1.10.4
                qtot_larger = targets > max_action_qtot  # 低估 (1.10.4)
                ws = torch.ones_like(td_error) * w_to_use  # (1.10.4)
                ws = torch.where(is_max_action | qtot_larger, torch.ones_like(td_error) * 1, ws)  # Target is greater than current max --> 1
                w_to_use = ws.mean().item()
                loss_mini = (ws.detach() * (td_error ** 2)).sum() / mask.sum()
                # loss = (ws * (masked_td_error ** 2)).sum() / mask.sum()

            ############################add coma&&wvdn begin
            ##  _get_q_values
            s, ss = self._get_critic_inputs(s.permute(1,2,0,3), s_prime_prime.permute(1,2,0,3), a1.permute(1,2,0,3))## (10,4,2,157)  (10,4,2,19) --> (80,392)
            q_eval = coma_critic(s) # torch.Size([120, 19])
            q_target = self.target_coma_critic(ss)
            q_values = q_eval.clone()  # 在函数的最后返回，用来计算advantage从而更新actor
            #### _train_critic
            q_eval = q_eval.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'],self.n_agents,-1).permute(2,0,1,3)
            a = torch.tensor(a, dtype=int)
            q_evals = torch.gather(q_eval, dim=3, index=a).squeeze(3) # 3.10.4

            q_target = q_target.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.n_agents, -1).permute(2, 0, 1, 3)
            u_next = torch.cat((a[:,1:,:,:], a[:,[-1],:,:]), dim=1)
            q_next_target = torch.gather(q_target, dim=3, index=u_next).squeeze(3)

            termi = torch.ones_like(done_mask_2) - done_mask_2
            targets = self.td_lambda_target(q_next_target, r, mask, termi).permute(2,1,0)  # 2.10.4
            td_error = targets.detach() - q_evals
            masked_td_error = done_mask_2.squeeze(-1) * td_error
            loss_ = (masked_td_error ** 2).sum() / done_mask_2.squeeze(-1).sum()
            self.coma_critic_optimizer.zero_grad()
            loss_.backward()
            torch.nn.utils.clip_grad_norm_(self.coma_critic_parameters, self.grad_clip)
            self.coma_critic_optimizer.step()
            #######
            prob = torch.nn.functional.softmax(q_eval_l, dim=-1).unsqueeze(2)  # torch.Size([10, 4, 1, 19])
            leader_avail_u = avail_u[0].unsqueeze(2)
            action_num = leader_avail_u.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_u.shape[-1]) # torch.Size([10, 4, 1, 19])
            action_prob = ((1 - self.epsilon) * prob + torch.ones_like(prob) * self.epsilon / action_num) # torch.Size([10, 4, 1, 19])
            action_prob[leader_avail_u == 0] = 0.0  # 不能执行的动作概率为0
            action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
            action_prob[leader_avail_u == 0] = 0.0  # torch.Size([10, 4, 1, 19])

            q_values = q_values.reshape(self.arg_dict['rollout_len'], self.arg_dict['batch_size'], self.n_agents, -1)  # torch.Size([10, 4, 2, 19])
            q_values_leader = q_values[:,:,[0],:]  # torch.Size([10, 4, 1, 19])
            a_l = a.permute(1,2,0,3)[:,:,[0],:]  # torch.Size([10, 4, 1, 1])
            q_taken = torch.gather(q_values_leader, dim=3, index=a_l).squeeze(3)  # leader agent的选择的动作对应的Ｑ值
            pi_taken = torch.gather(action_prob, dim=3, index=a_l).squeeze(3)  # 每个agent的选择的动作对应的概率  # torch.Size([10, 4, 1])
            pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
            log_pi_taken = torch.log(pi_taken)
            # 计算advantage
            baseline = (q_values_leader * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
            advantage = (q_taken - baseline).detach()
            loss_advantage = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()

            ############################add coma&&wvdn end

            loss.append(loss_mini + loss_advantage)

        loss = torch.mean(torch.stack(loss), 0)
        # loss = torch.sum(loss)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        self.optimization_step += self.arg_dict["batch_size"] * self.arg_dict["buffer_size"] * self.arg_dict["k_epoch"]
        if (self.optimization_step - self.last_target_update_step) / self.arg_dict["target_update_interval"] >= 1.0:
            self._update_targets(model, follower_model, mixer, coma_critic)
            self.last_target_update_step = self.optimization_step
            print("self.last_target_update_step:---", self.last_target_update_step)

        return torch.mean(loss)


    def _update_targets(self, model, follower_model, mixer, coma_critic):
        self.target_model.load_state_dict(model.state_dict())
        self.target_follower_model.load_state_dict(follower_model.state_dict())
        self.target_coma_critic.load_state_dict(coma_critic.state_dict())
        if mixer is not None:
            self.target_mixer.load_state_dict(mixer.state_dict())

    # def cuda(self):
    #     self.model.cuda()
    #     self.target_model.cuda()
    #     if self.mixer is not None:
    #         self.mixer.cuda()
    #         self.target_mixer.cuda()

    def save_models(self, path, model, follower_model, mixer, coma_critic):
        torch.save(model.state_dict(), "{}agent.th".format(path))
        torch.save(follower_model.state_dict(), "{}follower.th".format(path))
        if mixer is not None:
            torch.save(mixer.state_dict(), "{}mixer.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}opt.th".format(path))
        print("Model saved :", path)

    # def load_models(self, path):
    #     self.model.load_models(path)
    #     self.target_model.load_models(path)
    #     if self.mixer is not None:
    #         self.mixer.load_state_dict(torch.load("{}/mixer.th".format(path)),
    #                                    map_location=lambda storage, loc: storage)
    #     self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path)), map_location=lambda storage, loc: storage)
