import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
from models.mixers.vdn_net import VDNMixer
from models.mixers.qmix_net import QMixNet
from models.agents.rnn_agent import RNNAgent
from torch.optim import Adam
from models.agents.utils import _h_A


class VDN_GRAPH2(): # model, actor_model, mixer, critic_model
    def __init__(self, arg_dict, model, actor_model, mixer, critic_model, device=None):
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
        self.lambda_entropy = 0.02

        self.epsilon_ = self.arg_dict["epsilon_"]
        self.anneal_epsilon_ = self.arg_dict["anneal_epsilon_"]
        self.min_epsilon_ = self.arg_dict["min_epsilon_"]
        self.epsilon_anneal_scale_ = self.arg_dict["epsilon_anneal_scale_"]

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
        self.target_mixer = copy.deepcopy(mixer)
        self.target_actor_model = copy.deepcopy(actor_model)
        self.target_critic_model = copy.deepcopy(critic_model)
        # self.eval_parameters = list(model.parameters()) + list(mixer.parameters())+ list(actor_model.parameters())+ list(critic_model.parameters())  ## 这是函数的参数，这个改变了外层也会改变
        # if arg_dict["optimizer"] == "RMS":
        #     self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=arg_dict["learning_rate"])


    def split_agents(self, value): # 输入维度：[120,32], 其中120代表2个agent的30个transition，奇数表示agent1，偶数表示agent2
        q_x_1 = torch.Tensor([])
        q_x_2 = torch.Tensor([])
        for i in range(self.arg_dict["rollout_len"]):
            q_a_1 = value[2 * i]  # (12)
            q_a_2 = value[2 * i + 1]
            q_x_1 = torch.cat([q_x_1, q_a_1], dim=0)
            q_x_2 = torch.cat([q_x_2, q_a_2], dim=0)
        return torch.stack((q_x_1, q_x_2), dim=0)  # (2, 60*32)

    def get_q_values_new(self, mini_batch, model, epsilon_):
        s, h_in, a, a1, avail_u, r, r_prime, s_prime, h_out, avail_u_next, done_mask, father, father_next_ = mini_batch  # torch.Size([30, 4, 144])  (roll_len*3, batch, 115)
        h_in = h_in.reshape(self.arg_dict["batch_size"] * self.arg_dict["rollout_len"] * 3 , -1)
        inputs_father = np.concatenate((s, father), axis=-1)
        inputs_father = inputs_father.reshape(self.arg_dict["batch_size"] * self.arg_dict["rollout_len"] * 3, -1)
        q, _ = model(torch.tensor(inputs_father).float(), h_in)  #
        q = q.reshape(self.arg_dict["batch_size"], self.arg_dict["rollout_len"], 3, -1)

        h_out = h_out.reshape(self.arg_dict["batch_size"] * self.arg_dict["rollout_len"] * 3, -1)
        inputs_prime_father = np.concatenate((s_prime, father_next_), axis=-1)
        inputs_prime_father = inputs_prime_father.reshape(self.arg_dict["batch_size"] * self.arg_dict["rollout_len"] * 3, -1)
        target_q, _ = self.target_model(torch.tensor(inputs_prime_father).float(), h_out)
        target_q_ = target_q.reshape(self.arg_dict["batch_size"], self.arg_dict["rollout_len"], 3, -1)

        ### ----------------------------rjq add prob
        q_eval_l = q.clone()
        prob = torch.nn.functional.softmax(q_eval_l, dim=-1)  # (roll_len*3, batch, 19)
        action_num = avail_u.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_u.shape[-1])  # 可以选择的动作的个数 # torch.Size([1, 25, 5, 5])
        action_prob = ((1 - epsilon_) * prob + torch.ones_like(prob) * epsilon_ / action_num)
        action_prob[avail_u == 0] = 0.0  # 不能执行的动作概率为0
        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_u == 0] = 0.0

        return q, target_q_, action_prob


    def train(self, model, actor_model, mixer, critic_model, data):

        ### rjq debug 0118
        # self.model.load_state_dict(model.state_dict())  # rjq 0114  传入self.model
        # if self.mixer is not None:
        #     self.mixer.load_state_dict(mixer.state_dict())  # mixer.state_dict() == self.target_mixer.state_dict()

        # model.init_hidden()
        # self.target_model.init_hidden()
        self.eval_parameters = list(model.parameters()) + list(mixer.parameters()) + list(actor_model.parameters()) + list(critic_model.parameters())  ## 这是函数的参数，这个改变了外层也会改变
        if self.arg_dict["optimizer"] == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.arg_dict["learning_rate"])
        # epsilon_
        epsilon_ = self.epsilon_
        if self.epsilon_anneal_scale_ == 'episode':
            self.epsilon_ = epsilon_ - self.anneal_epsilon_ if epsilon_ > self.min_epsilon_ else epsilon_

        loss_policy, loss_actors, loss_critics, loss_hAs = [],[],[],[]
        for mini_batch in data:
            # pdb.set_trace()
            # obs_model, h_in, actions1, actions_onehot1, avail_u, reward_agency, reward_agency_prime, obs_prime, h_out, avail_u_next, done_mask, father_
            s, h_in, a, a1, avail_u, r, r_prime, s_prime, h_out, avail_u_next, done_mask, father, father_next_ = mini_batch

            ###########todo get_value
            q, target_q, action_prob = self.get_q_values_new(mini_batch, model, self.epsilon_)  # 15.2.19

            q_ = torch.gather(q, dim=-1, index=a.long()).squeeze(-1)  # 15.2

            target_q[avail_u_next == 0.0 ] = - 9999999
            target_q_ = target_q.max(dim=-1)[0]

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
                    q_total_q_a = mixer(q_)  # (1, 60*32)
                    q_total_target_max_q = self.target_mixer(target_q_)  # (1, 60*32)

            # s = self.split_agents((1 - done_mask).squeeze(2))[0].unsqueeze(0)
            # s = self.split_agents((done_mask).squeeze(2))[0].unsqueeze(0)
            # r_stack = self.split_agents(r.squeeze(2))  # (60,32) --> (2, 30*32)
            # r_total = torch.sum(r_stack, dim=0, keepdim=True)  # (1, 60*32)

            r_total = torch.sum(r.squeeze(-1), dim=-1, keepdim=True) # 2.5.1
            s = done_mask.squeeze(-1)[:,:,[0]]

            targets = r_total + self.arg_dict["gamma"] * s * (q_total_target_max_q)  # (1, 60*32)

            td_error = (q_total_q_a - targets.detach())  #(1920, 1)
            loss_mini = torch.mean((td_error ** 2))
            loss_graph_actor, loss_graph_critic, loss_hA = self.train_dag_rl(mini_batch, action_prob, q, actor_model, critic_model)

            loss_policy.append(loss_mini)
            loss_actors.append(loss_graph_actor)
            loss_critics.append(loss_graph_critic)
            loss_hAs.append(loss_hA)

        loss_policy_ = torch.mean(torch.stack(loss_policy), 0)
        loss_actors_ = torch.mean(torch.stack(loss_actors), 0)
        loss_critics_ = torch.mean(torch.stack(loss_critics), 0)
        loss_hAs_ = torch.mean(torch.stack(loss_hAs), 0)

        # Optimize
        self.optimizer.zero_grad()
        (loss_policy_ + loss_actors_ + loss_critics_).backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_clip)
        self.optimizer.step()

        # self.optimization_step += self.arg_dict["batch_size"] * self.arg_dict["buffer_size"] * self.arg_dict["k_epoch"]
        # if (self.optimization_step - self.last_target_update_step) // self.arg_dict["target_update_interval"] >= 1.0:
        #     self._update_targets(model, mixer)
        #     self.last_target_update_step = self.optimization_step
        #     print("self.last_target_update_step:---", self.last_target_update_step)
        self.optimization_step += 1
        if self.optimization_step % self.arg_dict["target_update_interval"] == 0.0:
            self._update_targets(model, mixer, actor_model, critic_model)
            self.last_target_update_step = self.optimization_step
            print("self.last_target_update_step:---", self.last_target_update_step)

        return loss_policy_, loss_actors_, loss_critics_, loss_hAs_


    def train_dag_rl(self, mini_batch, action_prob, q_evals_, actor_model, critic_model):
        s, h_in, a, a1, avail_u, r, r_prime, s_prime, h_out, avail_u_next, done_mask, father, father_next_ = mini_batch

        # s = s.reshape(self.arg_dict["rollout_len"], self.arg_dict["n_agents"], self.arg_dict["batch_size"], -1)
        # s = s.permute(0,2,1,3).reshape(self.arg_dict["rollout_len"] *self.arg_dict["batch_size"], self.arg_dict["n_agents"], -1)
        s = s.reshape(self.arg_dict["batch_size"]*self.arg_dict["rollout_len"], self.arg_dict["n_agents"], -1)

        encoder_output, samples, mask_scores, entropy, adj_prob, \
        log_probs_for_rewards, entropy_regularization = actor_model(s)

        pre = critic_model.predict_rewards(encoder_output)  # 10.3.144  -->  10
        # Reward config
        reward_mean = r.mean()
        actor_model.avg_baseline = self.arg_dict["alpha"] * actor_model.avg_baseline + (1.0 - self.arg_dict["alpha"]) * reward_mean  # # moving baseline for Reinforce
        ################
        # mask = 1 - batch["padded"].float()
        a_ = torch.tensor(np.array(a).astype(np.int))
        pi_taken = torch.gather(action_prob, dim=-1, index=a_) # torch.Size([2.5.3.1])
        pi_taken[done_mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        q_taken = torch.gather(q_evals_, dim=-1, index=a_) # torch.Size([15, 2, 1])
        baseline = (q_evals_ * action_prob).sum(dim=-1, keepdim=True).detach()  # torch.Size([15, 2, 1])

        pre_1 = pre.view(self.arg_dict["batch_size"], self.arg_dict["rollout_len"]).unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.arg_dict["n_agents"], 1)
        # pre_1 = pre_1.permute(0,2,1).reshape(self.arg_dict["rollout_len"]*self.arg_dict["n_agents"], self.arg_dict["batch_size"], -1)

        # reward_baseline = r.squeeze(2) - baseline - pre_1
        reward_baseline = r - pre_1 + baseline

        advantage = (q_taken - reward_baseline).detach()
        log_adv = advantage * log_pi_taken  # 2.5.3.1

        # print(log_softmax_logits_for_rewards.shape) # 25.5.5

        # log_adv_ = log_adv.repeat(1,1,1,self.arg_dict["n_agents"])
        # log_probs_for_rewards_ = log_probs_for_rewards.view(self.arg_dict["batch_size"], self.arg_dict["rollout_len"], self.arg_dict["n_agents"], -1)
        # done_mask_ = done_mask.repeat(1,1,1,self.arg_dict["n_agents"])

        log_adv_ = torch.sum(log_adv, dim=(-1,-2)) # 2.5
        log_probs_for_rewards_ = log_probs_for_rewards.view(self.arg_dict["batch_size"], self.arg_dict["rollout_len"]) # 2.5
        done_mask_ = done_mask[:,:,0,0]

        # loss_graph_actor = log_adv_ * log_probs_for_rewards_   ### TODO: graph generation prob
        # loss_graph_actor = - ((loss_graph_actor) * done_mask_).sum() / done_mask_.sum()

        _, _, _, _, _, log_probs_for_rewards_tar, _ = self.target_actor_model(s)
        log_probs_for_rewards_tar_ = log_probs_for_rewards_tar.view(self.arg_dict["batch_size"], self.arg_dict["rollout_len"])  # 2.5


        loss_graph_actor = log_adv_ * (log_probs_for_rewards_tar_ - log_probs_for_rewards_).exp()   ### TODO:
        loss_graph_actor = - ((loss_graph_actor) * done_mask_).sum() / done_mask_.sum()
        # mask_1 = (1 - batch["padded"].float()).unsqueeze(-1).unsqueeze(0)
        entropy_regularization = entropy_regularization.view(self.arg_dict["batch_size"], self.arg_dict["rollout_len"])
        loss_graph_actor += self.lambda_entropy * ((entropy_regularization*done_mask_).sum() / done_mask_.sum())

        loss = 0
        # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
        batch_size_all = self.arg_dict["rollout_len"]* self.arg_dict["batch_size"]
        for i in range(batch_size_all):
            m_s = adj_prob[i]
            # sparse_loss = self.args.tau_A * torch.sum(torch.abs(m_s))
            h_A = _h_A(m_s, self.arg_dict["n_agents"])
            loss += h_A


        loss_hA = loss/batch_size_all
        loss_graph_actor = loss_graph_actor + loss_hA + 0.5*loss_hA*loss_hA

        # r_ = r.reshape(self.arg_dict["rollout_len"], self.arg_dict["n_agents"], self.arg_dict["batch_size"]).permute(0,2,1)[:,:,0].reshape(-1)
        r_ = r[:,:,0,0].reshape(-1)
        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss_graph_critic = 0.5 * mse_loss(pre, r_)


        return loss_graph_actor, loss_graph_critic, loss_hA





    # def train_dag_rl(self, mini_batch, action_prob, q_evals_, actor_model, critic_model):
    #     s, h_in, a, a1, avail_u, r, r_prime, s_prime, h_out, avail_u_next, done_mask, father, father_next_ = mini_batch
    #
    #     s = s.reshape(self.arg_dict["rollout_len"], self.arg_dict["n_agents"], self.arg_dict["batch_size"], -1)
    #     s = s.permute(0, 2, 1, 3).reshape(self.arg_dict["rollout_len"] * self.arg_dict["batch_size"],
    #                                       self.arg_dict["n_agents"], -1)
    #
    #     encoder_output, samples, mask_scores, entropy, adj_prob, \
    #     log_probs_for_rewards, entropy_regularization = actor_model(s)
    #
    #     pre = critic_model.predict_rewards(encoder_output)  # 10.3.144  -->  10
    #     # Reward config
    #     reward_mean = r.mean()
    #     actor_model.avg_baseline = self.arg_dict["alpha"] * actor_model.avg_baseline + (
    #                 1.0 - self.arg_dict["alpha"]) * reward_mean  # # moving baseline for Reinforce
    #     ################
    #     # mask = 1 - batch["padded"].float()
    #     a_ = torch.tensor(np.array(a).astype(np.int))
    #     pi_taken = torch.gather(action_prob, dim=-1, index=a_)  # torch.Size([15, 2, 1])
    #     pi_taken[done_mask.repeat(1, 1, pi_taken.shape[-1]) == 0] = 1.0
    #     log_pi_taken = torch.log(pi_taken)
    #     q_taken = torch.gather(q_evals_, dim=-1, index=a_)  # torch.Size([15, 2, 1])
    #     baseline = (q_evals_ * action_prob).sum(dim=-1, keepdim=True).detach()  # torch.Size([15, 2, 1])
    #
    #     pre_1 = pre.view(self.arg_dict["rollout_len"], self.arg_dict["batch_size"]).unsqueeze(-1).repeat(1, 1,
    #                                                                                                      self.arg_dict[
    #                                                                                                          "n_agents"])
    #     pre_1 = pre_1.permute(0, 2, 1).reshape(self.arg_dict["rollout_len"] * self.arg_dict["n_agents"],
    #                                            self.arg_dict["batch_size"], -1)
    #     # reward_baseline = r.squeeze(2) - baseline - pre_1
    #     reward_baseline = r - pre_1 + baseline
    #
    #     advantage = (q_taken - reward_baseline).detach()
    #     log_adv = advantage * log_pi_taken
    #
    #     # print(log_softmax_logits_for_rewards.shape) # 25.5.5
    #     log_adv_ = log_adv.reshape(self.arg_dict["rollout_len"], self.arg_dict["n_agents"], self.arg_dict["batch_size"],
    #                                -1).permute(0, 2, 1, 3)  # 5.2.3.1
    #     log_probs_for_rewards_ = log_probs_for_rewards.view(self.arg_dict["rollout_len"], self.arg_dict["batch_size"],
    #                                                         self.arg_dict["n_agents"], -1)  # 5.2.3.3
    #     done_mask_ = done_mask.view(self.arg_dict["rollout_len"], self.arg_dict["n_agents"],
    #                                 self.arg_dict["batch_size"], -1).permute(0, 2, 1, 3)  # 5.2.3.1
    #     done_mask_ = done_mask_.repeat(1, 1, 1, self.arg_dict["n_agents"])
    #
    #     loss_graph_actor = log_adv_.repeat(1, 1, 1, self.arg_dict[
    #         "n_agents"]) * log_probs_for_rewards_  ### TODO: graph generation prob
    #     loss_graph_actor = - ((loss_graph_actor) * done_mask_).sum() / done_mask_.sum()
    #     # mask_1 = (1 - batch["padded"].float()).unsqueeze(-1).unsqueeze(0)
    #     # loss_graph_actor += self.args.lambda_entropy * ((- entropy_regularization*mask_1).sum() / mask.sum())
    #
    #     loss = 0
    #     # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
    #     batch_size_all = self.arg_dict["rollout_len"] * self.arg_dict["batch_size"]
    #     for i in range(batch_size_all):
    #         m_s = adj_prob[i]
    #         # sparse_loss = self.args.tau_A * torch.sum(torch.abs(m_s))
    #         h_A = _h_A(m_s, self.arg_dict["n_agents"])
    #         loss += h_A
    #
    #     loss_hA = loss / batch_size_all
    #     loss_graph_actor = loss_graph_actor + loss_hA
    #
    #     r_ = r.reshape(self.arg_dict["rollout_len"], self.arg_dict["n_agents"], self.arg_dict["batch_size"]).permute(0,
    #                                                                                                                  2,
    #                                                                                                                  1)[
    #          :, :, 0].reshape(-1)
    #     mse_loss = torch.nn.MSELoss(reduction='mean')
    #     loss_graph_critic = 0.5 * mse_loss(pre, r_)
    #
    #     return loss_graph_actor, loss_graph_critic, loss_hA

    def _update_targets(self, model, mixer, actor_model, critic_model):
        self.target_model.load_state_dict(model.state_dict())
        self.target_actor_model.load_state_dict(actor_model.state_dict())
        self.target_critic_model.load_state_dict(critic_model.state_dict())
        if mixer is not None:
            self.target_mixer.load_state_dict(mixer.state_dict())

    # def cuda(self):
    #     self.model.cuda()
    #     self.target_model.cuda()
    #     if self.mixer is not None:
    #         self.mixer.cuda()
    #         self.target_mixer.cuda()

    def save_models(self, path, model, mixer):
        torch.save(model.state_dict(), "{}agent.th".format(path))
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
