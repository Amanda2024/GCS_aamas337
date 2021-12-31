import numpy as np
import gym
from onpolicy.envs.mpe.scenario import BaseScenario
from onpolicy.envs.mpe.core import World

penalty = 1 # 0
# agents = 10 # 2

class Scenario(BaseScenario):  # Gaussian Squeeze  Scenario
    def make_world(self, args):

        world = World()
        world.world_length = args.episode_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = args.num_agents
        world.collaborative = True

        agents = args.n_agents
        self.state = np.random.uniform(0., 2., agents)
        self.state = np.repeat(np.expand_dims(self.state, axis=0), agents, axis=0)

        self.agents = agents
        print(self.agents)

        self.action_dim = 11

        self.state_dim = agents
        return world

    def reset_world(self, world):

        self.state = np.random.uniform(0., 2., self.agents)
        self.state = np.repeat(np.expand_dims(self.state, axis=0), self.agents, axis=0)

        return self.state

    def observation(self):
        return self.state

    def step(self, action):

        info = {'n': []}
        reward = []
        done = []

        # r = np.sum(np.array(action) * self.state[0]) / self.agents
        r = np.sum(np.array(action) * self.state[0]) / self.agents

        if penalty == 1:
            # rv = r * np.exp(-np.square(r - 5) / 1) + r * np.exp(-np.square(r - 8) / 0.25)
            # rv = r * np.exp(-np.square(r - 5) / 1.0) + r * np.exp(-np.square(r - 9) / 0.3)
            rv = r * np.exp(-np.square(r - 8) / 1.0) + r * np.exp(-np.square(r - 12) / 0.3)
        else:
            # rv = r * np.exp(-np.square(r - 8) / 0.25)
            rv = r * np.exp(-np.square(r - 8) / 1.0)  # QTRAN

        reward.append(rv)
        reward = reward * self.agents

        done.append(True)

        return self.state, reward, done, info

    def call_action_dim(self):
        return self.action_dim

    def call_state_dim(self):
        return self.state_dim


















# import numpy as np
# import gym
#
# penalty = 1 # 0
# # agents = 10 # 2
#
# class GuassianSqueeze(gym.Env):  # Gaussian Squeeze
#     def __init__(self, agents):
#
#         self.state = np.random.uniform(0., 2., agents)
#         self.state = np.repeat(np.expand_dims(self.state, axis=0), agents, axis=0)
#
#         self.agents = agents
#         print(self.agents)
#
#         self.action_dim = 11
#
#         self.state_dim = agents
#
#     def reset(self):
#
#         self.state = np.random.uniform(0., 2., self.agents)
#         self.state = np.repeat(np.expand_dims(self.state, axis=0), self.agents, axis=0)
#
#         return self.state
#
#     def step(self, action):
#
#         info = {'n': []}
#         reward = []
#         done = []
#
#         # r = np.sum(np.array(action) * self.state[0]) / self.agents
#         r = np.sum(np.array(action) * self.state[0]) / self.agents
#
#         if penalty == 1:
#             # rv = r * np.exp(-np.square(r - 5) / 1) + r * np.exp(-np.square(r - 8) / 0.25)
#             # rv = r * np.exp(-np.square(r - 5) / 1.0) + r * np.exp(-np.square(r - 9) / 0.3)
#             rv = r * np.exp(-np.square(r - 8) / 1.0) + r * np.exp(-np.square(r - 12) / 0.3)
#         else:
#             # rv = r * np.exp(-np.square(r - 8) / 0.25)
#             rv = r * np.exp(-np.square(r - 8) / 1.0)  # QTRAN
#
#         reward.append(rv)
#         reward = reward * self.agents
#
#         done.append(True)
#
#         return self.state, reward, done, info
#
#     def call_action_dim(self):
#         return self.action_dim
#
#     def call_state_dim(self):
#         return self.state_dim