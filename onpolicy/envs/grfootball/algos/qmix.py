import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
from models.mixers.vdn import VDNMixer
from torch.optim import Adam


class QLearner():
    def __init__(self, arg_dict, model, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]

        self.model = model
        self.params = list(model.parameters())
        self.last_target_update_step = 0
        self.optimization_step = 0
        self.arg_dict = arg_dict

        self.mixer = None
        if arg_dict["mixer"] is not None:
            if arg_dict["mixer"] == "vdn":
                self.mixer = VDNMixer()
            else:
                raise ValueError("Mixer {} not recognised".format(arg_dict["mixer"]))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = Adam(params=self.params, lr=arg_dict["learning_rate"])

        self.target_model = copy.deepcopy(self.model)

    def train(self, model, data):
        tot_loss_lst = []
        pi_loss_lst = []
        entropy_lst = []
        move_entropy_lst = []
        v_loss_lst = []

        # data_with_transition = []
        model_out = []
        rewards = []
        self.model.init_hidden()
        # Calculate the Q-Values necessary for the target
        for mini_batch in data:
            s, a, m, r, s_prime, done_mask, prob, need_move = mini_batch
            rewards.append(r)  # T x 2 list
            # Calculate estimate Q-Values
            agent_outs = self.model.forward(s)
            model_out.append(agent_outs)
            # data_with_transition.append((s, m, r, s_prime, done_mask, prob, need_move, q_a, q_m))
        model_out = torch.stack(model_out, dim=1)  # concat over time
        rewards = torch.sum(torch.Tensor(r), dim=-1)

        # Pick the Q-Values for the actions taken by each agent
        # chosen_action_values = torch.sum(model_out, dim=0)  # Remove the last dim
        # chosen_action_values = torch.gather(model_out[:, :-1], dim=3, index=a).squeeze(3)  # Remove the last dim

        target_model_out = []
        self.target_model.init_hidden()
        for mini_batch in data:
            s, a, m, r, s_prime, done_mask, prob, need_move = mini_batch
            target_agent_outs = self.target_model.forward(s)
            target_model_out.append(target_agent_outs)
        target_model_out = torch.stack(target_model_out, dim=1)

        # Max over target Q-Values in each timestep
        target_max_qvals = target_model_out.max(dim=2)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_values = self.mixer(model_out)
            target_max_qvals = self.target_mixer(target_max_qvals)

        # Calculate 1-step Q-Learning targets over all timesteps
        targets = []
        for i, mini_batch in enumerate(data):
            s, a, m, r, s_prime, done_mask, prob, need_move = mini_batch
            target = r + self.arg_dict["gamma"] * (1 - done_mask) * target_max_qvals[i]
            targets.append(target)
        # targets = rewards + self.arg_dict["gamma"] * (1 - done_mask) * target_max_qvals # 1 x T

        # TD-error
        td_error = (chosen_action_values - targets)

        # Normal L2 loss, take mean over actual data
        loss = (td_error ** 2).sum()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        self.optimization_step += arg_dict["batch_size"] * arg_dict["buffer_size"] * arg_dict["k_epoch"]
        if (self.optimization_step - self.last_target_update_step) / self.arg_dict["target_update_interval"] >= 1.0:
            self._update_targets()
            self.last_target_update_step = optimization_step

        return np.mean(loss)

    def _update_targets(self):
        self.target_model.load_state(self.model)
        if self.mixer is not None:
            self.target_mixer.load_state(self.mixer)

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.model.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))
        print("Model saved :", path)

    def load_models(self, path):
        self.model.load_models(path)
        self.target_model.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load("{}/mixer.th".format(path)),
                                       map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(torch.load("{}/opt.th".format(path)), map_location=lambda storage, loc: storage)
