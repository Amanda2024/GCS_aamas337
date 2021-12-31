import torch

class Config:
    def __init__(self):
        self.train = True
        self.seed = 133 # 133
        self.cuda = False
        
        # train setting 
        self.last_action = True # 使用最新动作选择动作
        self.reuse_network = True # 对所有智能体使用同一个网络
        self.n_epochs = 200000  # 20000
        self.evaluate_epoch = 20 # 20
        self.evaluate_per_epoch = 100 # 100
        self.batch_size = 32 # 32
        self.buffer_size = int(5e3)
        self.save_frequency = 5000 # 5000
        self.n_eposodes = 1 # 每个epoch有多少episodes
        self.train_steps = 1 # 每个epoch有多少train steps 
        self.gamma = 0.99
        self.grad_norm_clip = 10 # prevent gradient explosion
        self.update_target_params = 200 # 200
        self.result_dir = './results/'

        # test setting
        self.load_model = False

        # SC2 env setting
        self.map_name = 'SGS'
        self.step_mul = 8 # 多少步执行一次动作
        self.difficulty = '4'
        self.game_version = 'latest'
        self.replay_dir = './replay_buffer/'

        if not self.cuda:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda: 2" if torch.cuda.is_available() else "cpu")

        # model structure
        # drqn net
        self.drqn_hidden_dim = 64
        # qmix net
        # input: (batch_size, n_agents, qmix_hidden_dim)
        self.qmix_hidden_dim = 32
        self.two_hyper_layers = False
        self.hyper_hidden_dim = 64
        self.model_dir = './models/'
        self.optimizer = "RMS"
        self.learning_rate = 5e-4

        # epsilon greedy
        self.start_epsilon = 1
        self.end_epsilon = 0.2
        self.anneal_steps = 50000  # 50000
        self.anneal_epsilon = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        self.epsilon_anneal_scale = 'step'

        # epsilon greedy
        self.start_epsilon_ = 0.5
        self.end_epsilon_ = 0.02
        self.anneal_steps_ = 50000  # 50000
        self.anneal_epsilon_ = (self.start_epsilon_ - self.end_epsilon_) / self.anneal_steps_
        self.epsilon_anneal_scale_ = 'episode'


        ### DAG_RL
        self.n_xdims = 51
        self.nhead = 1
        self.num_layers = 4
        self.decoder_hidden_dim = 64
        self.node_num = 10
        self.critic_hidden_dim = 64
        self.lambda_entropy = 0.0001
        self.father_action = True
        self.alpha = 0.99
        self.lr_decay = 200
        self.gamma_gnn = 1.0

    def set_env_info(self, env_info):
        self.n_actions = env_info["n_actions"]
        self.state_shape = env_info["state_shape"]
        self.obs_shape = env_info["obs_shape"]
        self.n_agents = env_info["n_agents"]
        self.episode_limit = env_info["episode_limit"]

    



