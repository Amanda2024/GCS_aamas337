import os
import numpy as np
# from smac.env import StarCraft2Env
from env.Gaussian_Squeeze_Env_modify_3 import GuassianSqueeze
from agent import Agents
from utils import RolloutWorker, ReplayBuffer
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from config import Config
conf = Config()

def train():
    n_agents = 10
    episode_limit = 10
    conf.n_eposodes = 1 
    env = GuassianSqueeze(n_agents, episode_limit)
    env_info = env.get_env_info() # {'state_shape': 61, 'obs_shape': 42, 'n_actions': 10, 'n_agents': 3, 'episode_limit': 200}
    conf.set_env_info(env_info)
    agents = Agents(conf)
    rollout_worker = RolloutWorker(env, agents, conf)
    buffer = ReplayBuffer(conf)

    # save plt and pkl
    save_path = conf.result_dir + conf.map_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ##

    cur_time = datetime.now() + timedelta(hours=0)
    log_dir = save_path + "/SGS-vdn_graph-10a-env3_mod_depth1/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    conf.log_dir = log_dir
    conf.model_dir = conf.log_dir + 'models/'
    writer = SummaryWriter(logdir=log_dir)

    win_rates = []
    episode_rewards = []
    num_layers = []
    train_steps = 0
    for epoch in range(conf.n_epochs):
        # print("train epoch: %d" % epoch)
        episodes = []
        for episode_idx in range(conf.n_eposodes):
            episode, _, _, _ = rollout_worker.generate_episode(episode_idx)
            episodes.append(episode)

        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
     
        buffer.store_episode(episode_batch)
        for train_step in range(conf.train_steps):
            mini_batch = buffer.sample(min(buffer.current_size, conf.batch_size)) # obs； (64, 200, 3, 42)
            # print(mini_batch['o'].shape)
            loss_tuple = agents.train(mini_batch, train_steps, rollout_worker.start_epsilon)
            writer.add_scalar('loss/loss_policy', loss_tuple[0], epoch * episode_limit)
            writer.add_scalar('loss/loss_graph_actor', loss_tuple[1], epoch * episode_limit)  ##
            writer.add_scalar('loss/loss_graph_critic', loss_tuple[2], epoch * episode_limit)  ##
            writer.add_scalar('loss/loss_hA', loss_tuple[3], epoch * episode_limit)  ##
            train_steps += 1

        if epoch % conf.evaluate_per_epoch == 0:
            win_rate, episode_reward, num_layer = evaluate(rollout_worker)
            episode_reward = episode_reward / episode_limit
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            num_layers.append(num_layer)
            print("train epoch: {}, win rate: {}%, episode reward: {}".format(epoch, win_rate, episode_reward))
            # show_curves(win_rates, episode_rewards)
            writer.add_scalar('index/win_rate', win_rate, epoch*episode_limit)
            writer.add_scalar('index/episode', episode_reward, epoch*episode_limit)  ##
            writer.add_scalar('index/num_layers', num_layer, epoch*episode_limit)  ##

    show_curves(win_rates, episode_rewards)

def evaluate(rollout_worker):
    # print("="*15, " evaluating ", "="*15)
    win_num = 0
    episode_rewards = 0
    num_layers_sum = 0
    for epoch in range(conf.evaluate_epoch):
        _, episode_reward, win_tag, num_layers = rollout_worker.generate_episode(epoch, evaluate=True)
        num_layers_sum += np.mean(num_layers)
        episode_rewards += episode_reward
        if win_tag:
            win_num += 1

    return win_num / conf.evaluate_epoch, episode_rewards / conf.evaluate_epoch, num_layers_sum / conf.evaluate_epoch

def show_curves(win_rates, episode_rewards):
    print("="*15, " generate curves ", "="*15)
    plt.figure()
    plt.axis([0, conf.n_epochs, 0, 100])
    plt.cla()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(win_rates)), win_rates)
    plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
    plt.ylabel("win rate")

    plt.subplot(2, 1, 2)
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel('epoch*{}'.format(conf.evaluate_per_epoch))
    plt.ylabel("episode reward")

    plt.savefig(conf.log_dir + '/result_plt.png', format='png')
    np.save(conf.log_dir + '/win_rates', win_rates)
    np.save(conf.log_dir + '/episode_rewards', episode_rewards)


if __name__ == "__main__":
    if conf.train:
        train()
        

