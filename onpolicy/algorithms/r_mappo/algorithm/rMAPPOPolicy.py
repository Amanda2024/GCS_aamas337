import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import *
import igraph as ig
from onpolicy.algorithms.r_mappo.algorithm.graph_net_trans import Actor_graph
from gym import spaces

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.args = args

        self.tpdv = dict(dtype=torch.float32, device=device)

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space

        if args.env_name == 'GRFootball':
            self.act_space = act_space
        else:
            self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.graph_actor = Actor_graph(args, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.graph_actor_optimizer = torch.optim.Adam(self.graph_actor.parameters(),
                                                 lr=self.lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def _t2n(self, x):
        return x.detach().cpu().numpy()
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.graph_actor_optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, last_actions, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.0
        """

        ### get the sequence
        self.n_agents = last_actions.shape[-2]  ### self.n_rollout_threads, num_agents, act_shape
        self.n_rollout_threads = last_actions.shape[0]

        agent_id_graph = torch.eye(self.n_agents).unsqueeze(0).repeat(self.n_rollout_threads, 1, 1)  # self.n_rollout_threads, num_agents, num_agents
        obs_ = check(obs).to(**self.tpdv)
        if self.args.env_name == "GRFootball" or self.args.env_name == "gaussian":
            last_actions = last_actions
        else:
            last_actions = np.squeeze(np.eye(self.args.n_actions)[last_actions.astype(np.int32)], 2)
        last_actions_ = check(last_actions).to(**self.tpdv)

        obs_ = obs_.reshape(self.n_rollout_threads, self.n_agents, obs.shape[-1])
        if str(obs_.device).find('cuda') > -1:
            agent_id_graph = agent_id_graph.cuda()
        inputs_graph = torch.cat((obs_, agent_id_graph), -1).float()  # 1. 4.33
        inputs_graph = torch.cat((inputs_graph, last_actions_), -1).float()  # 1. 4.33

        encoder_output, samples, mask_scores, entropy, adj_prob, \
        log_softmax_logits_for_rewards, entropy_regularization = self.graph_actor(inputs_graph)
        graph_A = samples.clone().cpu().numpy()

        ######## pruning
        G_s = []
        for i in range(graph_A.shape[0]):
            G = ig.Graph.Weighted_Adjacency(graph_A[i].tolist())
            if not is_acyclic(graph_A[i]):
                G, new_A = pruning_1(G, graph_A[i])
            G_s.append(G)

        obs = obs.reshape(self.n_rollout_threads, self.n_agents, obs.shape[-1])
        rnn_states_actor = rnn_states_actor.reshape(self.n_rollout_threads, self.n_agents, rnn_states_actor.shape[-2],
                                                    rnn_states_actor.shape[-1])
        actions, action_log_probs, rnn_states_actor, father_actions = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks, G_s,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks) # 4.1 ，  4.1.64
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, father_actions

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, father_action, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     father_action,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
