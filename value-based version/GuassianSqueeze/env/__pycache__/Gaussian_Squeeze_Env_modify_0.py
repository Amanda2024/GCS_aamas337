import numpy as np
import gym

penalty = 2 # 0
# agents = 10 # 2

class GuassianSqueeze(gym.Env):  # Gaussian Squeeze
    def __init__(self, agents, episode_limit):

        self.state = np.random.uniform(0., 2., agents)
        self.state = np.repeat(np.expand_dims(self.state, axis=0), agents, axis=0)

        self.agents = agents  ## 10
        self.episode_limit = episode_limit

        self.action_dim = 21

        self.state_dim = agents
        print("agent number:", self.agents)

    def reset(self):

        self.state = np.random.uniform(0., 2., self.agents)
        self.state = np.repeat(np.expand_dims(self.state, axis=0), self.agents, axis=0)

        return self.state

    def step(self, action):

        info = {'n': []}
        reward = []
        done = []
        # print(action)
        action_mod = [i-10 for i in action]
        r = np.sum(np.array(action_mod) * self.state[0]) / 10
        # r = np.sum(np.array(action))

        if penalty == 0:  ## single-domain gs
            rv = r * np.exp(-np.square(r - 8) / 0.25)
        elif penalty == 1:  ## MGS
            rv = r * np.exp(-np.square(r - 5) / 1) + r * np.exp(-np.square(r - 8) / 0.25)
        else:  ## MGS
            rv = - r * np.exp(-np.square(r + 8) / 1.25) + r * np.exp(-np.square(r - 8) / 1.25)

            # rv = round(rv, 3)

        reward.append(rv)
        # reward = reward * self.agents

        done.append(True)

        return rv, done[0], info

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
