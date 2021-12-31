import numpy as np
import gym
from bisect import bisect_left
penalty = 2 # 0
# agents = 10 # 2

class GuassianSqueeze(gym.Env):  # Gaussian Squeeze
    def __init__(self, agents, episode_limit):
        self.agents = agents  ## 10
        self.episode_limit = episode_limit

        self.state = np.random.uniform(0., 2., agents*2)
        # self.state = np.repeat(np.expand_dims(self.state, axis=0), agents, axis=0)
        # self.state = np.zeros(self.agents)
        self.state = np.repeat(np.expand_dims(self.state, axis=0), self.agents, axis=0)
        # self.state_discount = np.random.uniform(0., 0.2, self.agents) ### 可以当做资源可支配单位量
        self.state_discount = np.array([0.1, 0.11, 0.1, 0.15, 0.1, 0.12, 0.08, 0.1, 0.12, 0.13]) ### 可以当做资源可支配单位量

        self.action_dim = 21
        self.state_dim = agents * 2
        print("agent number:", self.agents)

    def reset(self):
        self.state = np.random.uniform(0., 2., self.agents *2)
        # self.state = np.repeat(np.expand_dims(self.state, axis=0), self.agents, axis=0)
        # self.state = np.zeros(self.agents)
        self.state = np.repeat(np.expand_dims(self.state, axis=0), self.agents, axis=0)

        return self.state

    def f3(self, t, x):
        i = bisect_left(t, x)
        if t[i] - x > 0.5:
            i -= 1
        return i

    def step(self, action):

        info = {'n': []}
        reward = []
        done = []
        # print(action)
        action_mod = [i-10 for i in action]
        r = np.sum(np.array(action_mod) * self.state_discount)
        # r_sum_true = np.sum(np.array(action)) + 1e-100
        # r = np.sum(np.array(action))

        if penalty == 0:  ## single-domain gs
            rv = r * np.exp(-np.square(r - 8) / 0.25)
        elif penalty == 1:  ## MGS
            rv = r * np.exp(-np.square(r - 5) / 1) + r * np.exp(-np.square(r - 8) / 0.25)
        else:  ## MGS
            # rv = - r * np.exp(-np.square(r + 8) / 1.25) + r * np.exp(-np.square(r - 8) / 1.25)
            rv = r * np.exp(-np.square(r - 5) / 1.25) - r * np.exp(-np.square(r + 5) / 1.25)

        reward_1 = rv
        # if reward_1 > 8:
        #     print(action) # [5, 1, 12, tensor(0), 7, tensor(0), 4, 6, 2, 0]
        reward.append(rv)
        # reward = reward * self.agents

        if rv > 6.5:
            done.append(True)
        else:
            done.append(False)

        rv_trans = self.state_transformer(rv)
        np_one = np.concatenate((self.state[0][:10], [rv_trans] * self.agents), 0)
        next_state = np.repeat(np.expand_dims(np_one, axis=0), self.agents, axis=0)

        self.state = next_state.copy()
        return reward_1, done[0], info

    def state_transformer(self, rv): #### 0-8
        rv_trans = np.linspace(0, 10, num=400, endpoint=False)
        state_trans = np.linspace(0, 2, num=400, endpoint=False)
        index = self.f3(rv_trans, rv)
        return state_trans[index]

    def f3(self, t_array, x_num):
        i = bisect_left(t_array, x_num)
        if t_array[i] - x_num > 0.5:
            i -= 1
        return i


    def call_action_dim(self):
        return self.action_dim

    def call_state_dim(self):
        return self.state_dim

    def get_obs(self):
        return self.state
    def get_state(self):
        return self.state[0]
    def get_avail_agent_actions(self, agent_id):
        return [1 for i in range(self.action_dim)]

    def get_env_info(self):# {'state_shape': 61, 'obs_shape': 42, 'n_actions': 10, 'n_agents': 3, 'episode_limit': 200}
        info = {}
        info['state_shape'] = self.call_state_dim()
        info['obs_shape'] = self.call_state_dim()
        info['n_actions'] = self.call_action_dim()
        info['n_agents'] = self.agents
        info['episode_limit'] = self.episode_limit
        return info

    def close(self):
        pass
