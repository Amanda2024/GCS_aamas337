import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner

import importlib


def _t2n(x):
    return x.detach().cpu().numpy()


class GRFRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(GRFRunner, self).__init__(config)

        fe_module = importlib.import_module("onpolicy.utils." + config['all_args'].encoder)
        fe = fe_module.FeatureEncoder()
        self.fe = fe
        rewarder = importlib.import_module("onpolicy.utils." + config['all_args'].rewarder)
        self.rewarder = rewarder

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads


        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            score_list = []
            prev_obs = self.envs.reset()
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, father_actions= self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions.reshape(self.n_rollout_threads, self.num_agents))

                for i in range(self.n_rollout_threads):
                    rewards[i] = np.array([self.rewarder.calc_reward(rewards[i][j], prev_obs[i][j], obs[i][j]) for j in range(self.num_agents)])
                prev_obs = obs
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions

                # insert data into buffer
                self.insert(data)

            score_list.append(rewards[0][0])
            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews
                elif self.env_name == "GRFootball":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            # if 'individual_reward' in info[agent_id].keys():
                            #     idv_rews.append(info[agent_id]['individual_reward'])
                            idv_rews.append(info['score_reward'])
                        agent_k = 'agent%i/individual_rewards' % agent_id
                        env_infos[agent_k] = idv_rews
                    # score_list.append(idv_rews[0])
                    win_rate_list = np.array(self.compute_win_rate(score_list))
                    train_infos['win_rate'] = np.array(win_rate_list)

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def compute_win_rate(self, score_list):
        '''
        :param score_list: [0,0,1,1,1,0,0,1,0,1] with T timesteps
        :return: win_rate: such as [0.5] a list with one element
        '''
        for i in range(len(score_list)):
            if score_list[i] < 0:
                score_list[i] = 0

        if len(score_list) <= 10:
            win_rate = [sum(score_list) / len(score_list)]
        else:
            score_list = score_list[-10:]
            win_rate = [sum(score_list) / 10]
        return win_rate



    def warmup(self):
        # reset env
        # obs = self.envs.reset()
        obs = self.envs.reset()

        # Tranform dicts from multi-thread into the np.array
        _, obs = self.tranf_obs(obs, self.n_rollout_threads, self.fe)

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            if self.all_args.env_name == 'GRFootball':
                share_obs = obs
            else:
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    def tranf_obs(self, obs, n_rollout_threads, fe):  # 将obs和h_out 编码成state_dict,state_dict_tensor
        # h_in = h_out
        dict_obs = []
        final_obs = []
        for i in range(n_rollout_threads):
            x = []
            for j in range(len(obs[i])):
                state_dict1 = fe.encode(obs[i][j])  # 长度为7的字典
                state_dict_tensor1 = self.state_to_tensor(state_dict1)
                x.append(state_dict1)
            state_dict_tensor = {}

            for k, v in state_dict_tensor1.items():
                # state_dict_tensor[k] = torch.cat((state_dict_tensor1[k], state_dict_tensor2[k]), 0)
                state_dict_tensor[k] = torch.Tensor([x[s][k] for s in range(len(obs[i]))])
            # state_dict_tensor['hidden'] = h_in  # ((1,1,256),(1,1,256))
            dict_obs.append(state_dict_tensor)
        for i in range(n_rollout_threads):
            final_obs.append(self.obs_transform(dict_obs[i]))

        final_obs = torch.Tensor(final_obs).numpy()  # [n_threads, state_shape]
        return x, final_obs

    def state_to_tensor(self,
                        state_dict):  # state_dict:{'player':(29,),'ball':(18,),'left_team':(10,7),'left_closest':(7,),'right_team':(11,7),'player':(7,)}
        # pdb.set_trace() #debug

        player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(
            0)  # 在第0维增加一个维度；[[   state_dict["player"]  ]] #shape(1,1,29)
        ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,18)
        left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)  # shape(1,1,10,7)
        left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(
            0)  # shape(1,1,7)
        right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(
            0)  # shape(1,1,11,7)
        right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(
            0)  # shape(1,1,7)
        avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(
            0)  # shape(1,1,12)  tensor([[[1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.]]])

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

    def obs_transform(self, state_dict_tensor):
        '''

        :param state_dict_tensor: 7 kind of state dict with tensor for each element
        :return: flattern_obs for multi-agents [num_agent, obs_shape] (3 x 115)
        '''
        flattern_obs_0 = []
        flattern_obs_1 = []
        flattern_obs = [[], [], []]
        for i in range(len(flattern_obs)):
            for k, v in enumerate(state_dict_tensor):
                if k != 'hidden':  # hideen这一维度去掉
                    flattern_obs[i].insert(0, state_dict_tensor[v][i].reshape([-1]))
                    # flattern_obs_1.append(state_dict_tensor[v][1].reshape([-1]))
            flattern_obs[i] = torch.hstack(flattern_obs[i])

        # flattern_obs_0 = torch.hstack(flattern_obs_0)
        # flattern_obs_1 = torch.hstack(flattern_obs_1)
        flattern_obs = torch.stack((flattern_obs), dim=0)

        return flattern_obs.numpy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, father_actions \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              self.buffer.actions[step] )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        father_actions = np.array(np.split(_t2n(father_actions), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space.__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space.shape[0]):  # Transform the one-hot type from the scalar
                uc_actions_env = np.eye(self.all_args.n_actions)[actions[:, i, :]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=1)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, father_actions

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)

        _, obs = self.tranf_obs(obs, self.n_rollout_threads, self.fe)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            if self.all_args.env_name == 'GRFootball':
                share_obs = obs
        else:
            share_obs = obs

        # adjust the shape
        rewards = rewards.reshape(rewards.shape[0], rewards.shape[1], 1)
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, father_actions)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array', close=False)[0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array', close=False)[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(ifi - elapsed)

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + 'render.gif', all_frames, duration=self.all_args.ifi)
