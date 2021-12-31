from .distributions import Bernoulli, Categorical, DiagGaussian
import torch as th
import torch.nn as nn
import numpy as np
from onpolicy.utils.util import *
# import ray
import igraph as ig
from datetime import datetime, timedelta
# @ray.remote
# class InterActor(object):
#     def __init__(self, id):
#         self.id = id
#     @ray.method(num_returns=1)
#     def get_action_once(self, select_action, policy, device):
#


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
            self.action_dim = action_dim
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(Categorical(inputs_dim, action_dim, use_orthogonal, gain))
            self.action_outs = nn.ModuleList(self.action_outs)
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList(
                [DiagGaussian(inputs_dim, continous_dim, use_orthogonal, gain), Categorical(
                    inputs_dim, discrete_dim, use_orthogonal, gain)])

    def forward(self, obs, x, G_s, available_actions=None, deterministic=False):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        if self.mixed_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        elif self.multi_discrete:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)
        else:
            actions_outer, action_log_probs_outer, father_action_lst_outer = [], [], []

            cur_time = datetime.now() + timedelta(hours=0)
            # print("--------------11------start time::", cur_time)

            for i in range(len(G_s)):
                G = G_s[i]
                ordered_vertices = G.topological_sorting()
                self.n_agents = len(ordered_vertices)
                actions, action_log_probs = [0] * self.n_agents, [0] * self.n_agents
                father_action_lst = [0] * self.n_agents
                for j in ordered_vertices:
                    father_action_0 = torch.zeros(self.n_agents, self.action_dim)
                    parents = G.neighbors(j, mode=ig.IN)
                    if len(parents) != 0:
                        for k in parents:
                            father_act = torch.eye(self.action_dim)[actions[k]]
                            father_action_0[k] = torch.tensor(father_act)
                    father_action = father_action_0.reshape(-1)
                    father_action_lst[j] = father_action

                    x_ = torch.cat((x[i][j], father_action.cuda()))
                    action_logit = self.action_out(x_, available_actions)  ## 4.64、 None  --> 4.5
                    action = action_logit.mode() if deterministic else action_logit.sample()  # torch.Size([4, 1])
                    action_log_prob = action_logit.log_probs(action)   # torch.Size([4, 1])
                    actions[j], action_log_probs[j] = [action], [action_log_prob]
                actions_outer.append(actions)
                action_log_probs_outer.append(action_log_probs)
                father_action_tensor = torch.tensor([item.cpu().detach().numpy() for item in father_action_lst]).cuda()
                father_action_lst_outer.append(father_action_tensor)

        father_action_lst_outer = torch.tensor([item.cpu().detach().numpy() for item in father_action_lst_outer]).cuda()
        father_action_shape = father_action_lst_outer.shape[-1]

        cur_time = datetime.now() + timedelta(hours=0)
        # print("---------------11-----end time::", cur_time)

        return torch.tensor(actions_outer).view(-1,1), torch.tensor(action_log_probs_outer).view(-1,1), father_action_lst_outer.view(-1,father_action_shape)

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        if self.mixed_action or self.multi_discrete:
            action_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action_prob = action_logit.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, -1)
        else:
            action_logits = self.action_out(x, available_actions)
            action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(self, x, father_action, action, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.mixed_action:
            a, b = action.split((2, 1), -1)
            b = b.long()
            action = [a, b]
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    if len(action_logit.entropy().shape) == len(active_masks.shape):
                        dist_entropy.append((action_logit.entropy() * active_masks).sum() / active_masks.sum())
                    else:
                        dist_entropy.append(
                            (action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)
            dist_entropy = dist_entropy[0] / 2.0 + dist_entropy[1] / 0.98  # ! dosen't make sense

        elif self.multi_discrete:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_logit = action_out(x)
                action_log_probs.append(action_logit.log_probs(act))
                if active_masks is not None:
                    dist_entropy.append((action_logit.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum())
                else:
                    dist_entropy.append(action_logit.entropy().mean())

            action_log_probs = torch.cat(action_log_probs, -1)  # ! could be wrong
            dist_entropy = torch.tensor(dist_entropy).mean()

        else:
            father_action = torch.tensor(father_action).cuda()
            x_ = torch.cat((x, father_action), dim=1)
            action_logits = self.action_out(x_, available_actions)
            action_log_probs = action_logits.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy
