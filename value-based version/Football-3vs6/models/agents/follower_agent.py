# For Agent of MARL to encode the input recursively with RNN
import time
import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Value-based method to output different values depending on the current state for Q-learning
class Follower_Agent(nn.Module):
    def __init__(self, input_shape, args, device=None):
        super(Follower_Agent, self).__init__()
        self.device = None
        if device:
            self.device = device
        self.args = args

        self.fc1 = nn.Linear(input_shape, 64)
        # self.rnn = nn.GRUCell(64 + args['action_dim'], 64)
        self.rnn = nn.RNN(args['lstm_size'] + args['action_dim'], args['lstm_size'], 1)
        self.fc2 = nn.Linear(64, args["n_actions"])
        self.optimizer = optim.Adam(self.parameters(), lr=args["learning_rate"])

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, 64).zero_()

    def forward(self, inputs, f_leader, hidden_state):
        x = F.relu(self.fc1(inputs))  # 2.64  ## torch.Size([80, 64])
        f_leader = torch.repeat_interleave(f_leader, repeats=2, dim=0)  # torch.Size([2, 10, 4, 19])  # follower的个数
        x = torch.cat([x, f_leader], dim=-1)  # torch.Size([2, 10, 4, 83])
        h_in = hidden_state.reshape(1, -1, self.args["lstm_size"])
        y, h = self.rnn(x.unsqueeze(0), h_in)
        q = self.fc2(y).squeeze(0)
        return q, h

    def make_batch(self, data):
        # data = [trans1, tr2, ..., trans30] * batch_size
        state_lst_batch, state_prime_lst_batch, a_lst_batch, a1_lst_batch, avail_u_lst_batch, avail_u_next_lst_batch, r_lst_batch, done_lst_batch = [], [], [], [], [], [], [], []
        h_in_batch, h_out_batch = [], []
        for rollout in data:  # 3
            state_lst, state_prime_lst, a_lst, a1_lst, avail_u_lst, avail_u_next_lst, r_lst, done_lst = [], [], [], [], [], [], [], []
            h_in_lst, h_out_lst = [], []

            for transition in rollout:
                # s, a, m, r, s_prime, prob, done, need_move = transition
                obs_model, h_in, actions1, actions_onehot1, avail_u, reward_agency, obs_prime, h_out, avail_u_next, done = transition

                for i in range(len(obs_model)):
                    state_lst.append(obs_model[i][0].tolist())
                    state_prime_lst.append(obs_prime[i][0].tolist())
                    a_lst.append(actions1[i].tolist())
                    a1_lst.append(actions_onehot1[i].tolist())
                    r_lst.append([reward_agency[i]])
                    avail_u_lst.append(avail_u[i].tolist())
                    avail_u_next_lst.append(avail_u_next[i].tolist())
                    done_mask = 0 if done[i] else 1
                    done_lst.append([done_mask])

                h1_in, h2_in = h_in
                h_in_lst.append(h1_in.tolist())
                h_in_lst.append(h2_in.tolist())
                h1_out, h2_out = h_out
                h_out_lst.append(h1_out.tolist())
                h_out_lst.append(h2_out.tolist())

            state_lst_batch.append(state_lst)
            state_prime_lst_batch.append(state_prime_lst)
            a_lst_batch.append(a_lst)
            a1_lst_batch.append(a1_lst)
            avail_u_lst_batch.append(avail_u_lst)
            avail_u_next_lst_batch.append(avail_u_next_lst)
            r_lst_batch.append(r_lst)
            done_lst_batch.append(done_lst)

            h_in_batch.append(h_in_lst)
            # h2_in_batch.append(h2_in_lst[1::2 ])
            h_out_batch.append(h_out_lst)
            # h2_out_batch.append(h2_out_lst)

        h_in = torch.tensor(h_in_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3)  #(60,5,1,64)
        h_out = torch.tensor(h_out_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3)


        # obs_model:(roll_len*2, batch, 115)
        obs_model, actions1, actions_onehot1, avail_u, reward_agency, obs_prime, avail_u_next, done_mask = \
            torch.tensor(state_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2), \
            torch.tensor(a_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),\
            torch.tensor(a1_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),\
            torch.tensor(avail_u_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),\
            torch.tensor(r_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),\
            torch.tensor(state_prime_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2), \
            torch.tensor(avail_u_next_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),\
            torch.tensor(done_lst_batch, dtype=torch.float, device=self.device).permute(1, 0, 2)


        return obs_model, h_in, actions1, actions_onehot1, avail_u, reward_agency, obs_prime, h_out, avail_u_next, done_mask
        # obs_model obs_prime: (60,3,115)
        # h_in  h_out: 2个(1,3,64)
        # actions1 reward_agency done_mask：(60,3,1)   actions_onehot1 avail_u avail_u_next(60,3,19)



class QValueAgent(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(QValueAgent, self).__init__()
        self.device = None
        if device:
            self.device = device

        self.arg_dict = arg_dict
        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player"], 64)
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"], 64)
        self.fc_left = nn.Linear(arg_dict["feature_dims"]["left_team"], 48)
        self.fc_right = nn.Linear(arg_dict["feature_dims"]["right_team"], 48)
        self.fc_left_closest = nn.Linear(arg_dict["feature_dims"]["left_team_closest"], 48)
        self.fc_right_closest = nn.Linear(arg_dict["feature_dims"]["right_team_closest"], 48)

        self.conv1d_left = nn.Conv1d(48, 36, 1, stride=1)
        self.conv1d_right = nn.Conv1d(48, 36, 1, stride=1)
        # self.fc_left2 = nn.Linear(36 * 2, 96) # rjq0117
        # self.fc_right2 = nn.Linear(36 * 2, 96) # rjq0117
        self.fc_left2 = nn.Linear(36, 96)
        self.fc_right2 = nn.Linear(36 * 2, 96)
        self.fc_cat = nn.Linear(96 + 96 + 64 + 64 + 48 + 48, arg_dict["lstm_size"])  # 和为416

        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(48)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(48)
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])

        self.lstm = nn.LSTM(arg_dict["lstm_size"], arg_dict["lstm_size"])

        self.fc_q_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_q_a1 = nn.LayerNorm(164)
        self.fc_q_a2 = nn.Linear(164, 12)

        self.fc_q_m1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_q_m1 = nn.LayerNorm(164)
        self.fc_q_m2 = nn.Linear(164, 8)

        # self.fc_q_a1_2 = nn.Linear(arg_dict["lstm_size"], 164)
        # self.norm_q_a1_2 = nn.LayerNorm(164)  # rjq 0114
        # self.fc_q_a2_2 = nn.Linear(164, 12)
        #
        # self.fc_q_m1_2 = nn.Linear(arg_dict["lstm_size"], 164)
        # self.norm_q_m1_2 = nn.LayerNorm(164)  # rjq 0114
        # self.fc_q_m2_2 = nn.Linear(164, 8)

        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return self.fc_cat.weight.new(1, self.arg_dict["lstm_size"]).zero_()

    def forward(self, state_dict):
        player_state = state_dict["player"]
        ball_state = state_dict["ball"]
        left_team_state = state_dict["left_team"]
        left_closest_state = state_dict["left_closest"]
        right_team_state = state_dict["right_team"]
        right_closest_state = state_dict["right_closest"]
        avail = state_dict["avail"]
        # scale_m = torch.Tensor([1.,1.2,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
        # avail_scale_m = avail * scale_m
        scale_c = 1.2

        player_embed = self.norm_player(self.fc_player(player_state))    # (2,1,29) --> (2,1,64)
        ball_embed = self.norm_ball(self.fc_ball(ball_state))            # (2,1,18)  -> (2,1,64)
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim   # (2,1,1,7) --> (2,1,1,48)
        left_closest_embed = self.norm_left_closest(self.fc_left_closest(left_closest_state))       # (2,1,7) --> (2,1,48)
        right_team_embed = self.norm_right(self.fc_right(right_team_state))                         # (2,1,2,7) --> (2,1,2,48)
        right_closest_embed = self.norm_right_closest(self.fc_right_closest(right_closest_state))   # (2,1,7) --> (2,1,48)

        [horizon, batch_size, n_player, dim] = left_team_embed.size()   # [2, 1, 1, 48]
        left_team_embed = left_team_embed.view(horizon * batch_size, n_player, dim).permute(0, 2,
                                                                                            1)  # horizon * batch, dim1, n  #(2,48,2)
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(0, 2, 1)  # horizon * batch, n, dim2            #(2,1,36)
        left_team_embed = left_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size,
                                                                                 -1)  # horizon, batch, n * dim2  ##(2,1,36)
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))     ## (2, 1, 96)

        [horizon, batch_size, n_player, dim] = right_team_embed.size()  # rjq0117
        right_team_embed = right_team_embed.view(horizon * batch_size, n_player, dim).permute(0, 2,
                                                                                              1)  # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0, 2, 1)  # horizon * batch, n * dim2
        right_team_embed = right_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size, -1)
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))

        cat = torch.cat(
            [player_embed, ball_embed, left_team_embed, right_team_embed, left_closest_embed, right_closest_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden"]  # ((1,1,256),(1,1,256))
        out, h_out = self.lstm(cat, h_in)
        # Split the lstm output along the player dimension
        # out = out.permute([1, 0, 2])

        ### rjq 0116 debug
        q_a = F.relu(self.norm_q_a1(self.fc_q_a1(out)))  # (2,1,164)
        q_a = self.fc_q_a2(q_a) + (avail - 1)*1e7  # rjq 0114 加入有效动作  #(2,1,12) #rjqdebug0118
        # q_a[:,:,1] = q_a[:,:,1] * scale_c
        # q_a = q_a * avail_scale_m # rjq0118

        q_m = F.relu(self.norm_q_m1(self.fc_q_m1(out)))  # (2,1,164)
        q_m = self.fc_q_m2(q_m)                           #(2,1,8)

        # q_a_2 = F.relu(self.norm_q_a1_2(self.fc_q_a1_2(out[1])))    # (1,164)
        # q_a_2 = self.fc_q_a2(q_a_2) + (avail[1]-1)*1e-7  # rjq 0114 #(1,12)
        #
        # q_m_2 = F.relu(self.norm_q_m1_2(self.fc_q_m1_2(out[1])))  # rjq  0114 fix bug:out[0]-->out[1]  # (1,164)
        # q_m_2 = self.fc_q_m2_2(q_m_2)                                                                  # (1,8)

        # q_a = torch.cat((q_a, q_a_2), dim=0)
        # q_m = torch.cat((q_m, q_m_2), dim=0)  # n x action_space

        return q_a, q_m, h_out  # (2,1,12) # (2,1,8) # ((1,1,256),(1,1,256))

    def select_actions(self, state_dict, test_mode=False):
        avail_actions = state_dict["avail_actions"]
        agent_outs, hidden_states = self.forward(state_dict)

    def make_batch(self, data):
        # data = [trans1, tr2, ..., trans30] * batch_size
        s_player_batch, s_ball_batch, s_left_batch, s_left_closest_batch, s_right_batch, s_right_closest_batch, avail_batch = [], [], [], [], [], [], []
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_left_closest_prime_batch, \
        s_right_prime_batch, s_right_closest_prime_batch, avail_prime_batch = [], [], [], [], [], [], []
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []

        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_left_closest_lst, s_right_lst, s_right_closest_lst, avail_lst = [], [], [], [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_left_closest_prime_lst, \
            s_right_prime_lst, s_right_closest_prime_lst, avail_prime_lst = [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, m_lst, r_lst, prob_lst, done_lst, need_move_lst = [], [], [], [], [], []

            for transition in rollout:
                s, a, m, r, s_prime, prob, done, need_move = transition

                for i in range(len(s)):
                    s_player_lst.append(s[i]["player"])
                    s_ball_lst.append(s[i]["ball"])
                    s_left_lst.append(s[i]["left_team"])
                    s_left_closest_lst.append(s[i]["left_closest"])
                    s_right_lst.append(s[i]["right_team"])
                    s_right_closest_lst.append(s[i]["right_closest"])
                    avail_lst.append(s[i]["avail"])

                    # h1_in, h2_in = torch.tensor(s[i]["hidden"][0]).chunk(2, 1)
                    # h1_in = h1_in.numpy()
                    # h2_in = h2_in.numpy()
                    h1_in, h2_in = s[i]["hidden"]
                    h1_in_lst.append(h1_in)
                    h2_in_lst.append(h2_in)

                    s_player_prime_lst.append(s_prime[i]["player"])
                    s_ball_prime_lst.append(s_prime[i]["ball"])
                    s_left_prime_lst.append(s_prime[i]["left_team"])
                    s_left_closest_prime_lst.append(s_prime[i]["left_closest"])
                    s_right_prime_lst.append(s_prime[i]["right_team"])
                    s_right_closest_prime_lst.append(s_prime[i]["right_closest"])
                    avail_prime_lst.append(s_prime[i]["avail"])

                    # h1_out, h2_out = torch.tensor(s_prime[i]["hidden"][0]).chunk(2, 1)
                    # h1_out = h1_out.numpy()
                    # h2_out = h2_out.numpy()
                    h1_out, h2_out = s_prime[i]["hidden"]
                    h1_out_lst.append(h1_out)
                    h2_out_lst.append(h2_out)

                    a_lst.append([a[i]])
                    m_lst.append([m[i]])
                    r_lst.append([r[i]])
                    # prob_lst.append([prob])  # rjq 0114
                    prob_lst.append([prob[i]])
                    done_mask = 0 if done[i] else 1
                    done_lst.append([done_mask])
                    # need_move_lst.append([need_move])  # rjq 0114
                    need_move_lst.append([need_move[i]])

            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_left_closest_batch.append(s_left_closest_lst)
            s_right_batch.append(s_right_lst)
            s_right_closest_batch.append(s_right_closest_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])

            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_left_closest_prime_batch.append(s_left_closest_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
            s_right_closest_prime_batch.append(s_right_closest_prime_lst)
            avail_prime_batch.append(avail_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            a_batch.append(a_lst)
            m_batch.append(m_lst)
            r_batch.append(r_lst)
            prob_batch.append(prob_lst)
            done_batch.append(done_lst)
            need_move_batch.append(need_move_lst)

        s = {
            "player": torch.tensor(s_player_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_batch, dtype=torch.float, device=self.device).permute(1, 0,2),
            "avail": torch.tensor(avail_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_in_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2))
        }

        s_prime = {
            "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
            "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_prime_batch, dtype=torch.float, device=self.device).permute(1,0,2),
            "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=self.device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_out_batch, dtype=torch.float, device=self.device).squeeze(1).permute(1, 0, 2))
        }

        a, m, r, done_mask, prob, need_move = torch.tensor(a_batch, device=self.device).permute(1, 0, 2), \
                                              torch.tensor(m_batch, device=self.device).permute(1, 0, 2), \
                                              torch.tensor(r_batch, dtype=torch.float, device=self.device).permute(1, 0,2), \
                                              torch.tensor(done_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                              torch.tensor(prob_batch, dtype=torch.float, device=self.device).permute(1,0,2), \
                                              torch.tensor(need_move_batch, dtype=torch.float,
                                                           device=self.device).permute(1, 0, 2)

        return s, a, m, r, s_prime, done_mask, prob, need_move
