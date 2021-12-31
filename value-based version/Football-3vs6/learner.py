import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import random

def write_summary(writer, arg_dict, summary_queue, n_game, loss_tuple, optimization_step, self_play_board, win_evaluation, score_evaluation):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []
    eps_edges_lst = []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a, b, c, d, opp_num, t1, t2, t3, eps_edges = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":  ## 这是干啥的
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]

        if 'env_evaluation' in arg_dict and opp_num == arg_dict['env_evaluation']:
            win_evaluation.append(a)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            game_len.append(d)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)
            eps_edges_lst.append(eps_edges)

    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)

    loss_policy_lst, loss_actors_lst, loss_critics_lst, loss_hAs_lst = loss_tuple
    writer.add_scalar('train/loss_policy_lst', np.mean(loss_policy_lst), n_game)
    writer.add_scalar('train/loss_actors_lst', np.mean(loss_actors_lst), n_game)
    writer.add_scalar('train/loss_critics_lst', np.mean(loss_critics_lst), n_game)
    writer.add_scalar('train/loss_hAs_lst', np.mean(loss_hAs_lst), n_game)

    writer.add_scalar('train/eps_edges', np.mean(eps_edges_lst), n_game)
    # writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    # writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)
    # writer.add_scalar('train/move_entropy', np.mean(move_entropy_lst), n_game)

    mini_window = max(1, int(arg_dict['summary_game_window'] / 3))
    if len(win_evaluation) >= mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), n_game)
        win_evaluation, score_evaluation = [], []

    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/' + opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), n_game)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation


def write_summary_simplified(writer, arg_dict, summary_queue, n_game, loss_lst, \
                             optimization_step):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('game/loss', np.mean(loss_lst), n_game)
    writer.add_scalar('train/step', float(optimization_step), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)

    mini_window = max(1, int(arg_dict['summary_game_window'] / 3))

def Random_Queue(queue, arg_dict):
    queue_list = []
    queue_size = queue.qsize()
    ranint = random.sample(range(queue_size), arg_dict["buffer_size"] * arg_dict["batch_size"])
    for i in range(queue_size):
        rollout = queue.get()
        queue_list.append(rollout)
    random_list = [queue_list[index] for index in ranint]

    for i in list(set(range(queue_size)).difference(ranint)):
        queue.put(queue_list[i])
    return random_list

def get_data(queue, arg_dict, model):
    # print("--------------------:", queue.qsize())
    data = []
    # random_list = Random_Queue(queue, arg_dict)
    for i in range(arg_dict["buffer_size"]):  # 6
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]): # 3
            # rollout = random_list[i*arg_dict["buffer_size"] + j]
            rollout = queue.get()
            mini_batch_np.append(rollout)
        mini_batch = model.make_batch(mini_batch_np)  # mini_batch_np( 32, 60*[transition, transition]) --> mini_batch含8个tuple，除s外每个的shape(120,32,1)  # 8 指的是：s, a, m, r, s_prime, done_mask, prob, need_move
        data.append(mini_batch)
    return data


def learner(center_model, center_actor_model, center_mixer, center_critic_model, queue, data_all_queue, signal_queue, summary_queue, arg_dict):
    print("Learner process started")
    imported_model = importlib.import_module("models.agents." + arg_dict["model"])
    imported_actor_model = importlib.import_module("models.agents." + arg_dict["graph_actor"])
    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # 根据参数决定RNN的输入维度
    input_shape = arg_dict["state_shape"]
    if arg_dict["last_action"]:
        input_shape += arg_dict["n_actions"]
    if arg_dict["reuse_network"]:
        input_shape += arg_dict["n_agents"]
    if arg_dict["father_action"]:
        input_shape += arg_dict["n_agents"] * arg_dict["n_actions"]

    model = imported_model.RNNAgent(input_shape, arg_dict, device)
    actor_model = imported_actor_model.Actor_graph(arg_dict)
    model.load_state_dict(center_model.state_dict())
    actor_model.load_state_dict(center_actor_model.state_dict())
    # model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    # algo = imported_algo.QLearner(arg_dict, model)
    imported_mixer = importlib.import_module("models.mixers." + arg_dict["mixer_net"])
    mixer = imported_mixer.VDNMixer()
    mixer.load_state_dict(center_mixer.state_dict())

    imported_critic = importlib.import_module("models.agents." + arg_dict["graph_critic"])
    critic_model = imported_critic.Critic_graph(arg_dict)
    critic_model.load_state_dict(center_critic_model.state_dict())

    algo = imported_algo.VDN_GRAPH2(arg_dict, model, actor_model, mixer, critic_model)

    # if torch.cuda.is_available():
    #     algo.cuda()

    writer = SummaryWriter(logdir=arg_dict["log_dir"])
    optimization_step = 0
    if "optimization_step" in arg_dict:
        optimization_step = arg_dict["optimization_step"]
    last_saved_step = optimization_step
    n_game = 0
    # loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []
    loss_policy_lst, loss_actors_lst, loss_critics_lst, loss_hAs_lst = [], [], [], []
    self_play_board = {}
    win_evaluation, score_evaluation = [], []

    while True:
        if queue.qsize() > arg_dict["batch_size"] * arg_dict["buffer_size"] * arg_dict["rich_data_scale"]:
            # if (optimization_step % arg_dict["model_save_interval"] == 0):  # rjq  save model
            #     path = arg_dict["log_dir"] + "/model_" + str(optimization_step) + ".tar"
                # algo.save_models(path, model, follower_model, mixer, coma_critic)
            signal_queue.put(1)
            data = get_data(queue, arg_dict, model)
            print("data loaded……")
            loss_policy_, loss_actors_, loss_critics_, loss_hAs_ = algo.train(model, actor_model, mixer, critic_model, data)
            # loss, pi_loss, v_loss, entropy, move_entropy = algo.train(model, data)
            # optimization_step += arg_dict["batch_size"] * arg_dict["buffer_size"] * arg_dict["k_epoch"]
            optimization_step += 1
            print("step :", optimization_step, "loss_policy_", loss_policy_, "data_q", queue.qsize(), "summary_q",
                  summary_queue.qsize())

            loss_policy_ = loss_policy_.cpu().detach().numpy()
            loss_actors_ = loss_actors_.cpu().detach().numpy()
            loss_critics_ = loss_critics_.cpu().detach().numpy()
            loss_hAs_ = loss_hAs_.cpu().detach().numpy()

            loss_policy_lst.append(loss_policy_)
            loss_actors_lst.append(loss_actors_)
            loss_critics_lst.append(loss_critics_)
            loss_hAs_lst.append(loss_hAs_)

            loss_tuple = (loss_policy_lst, loss_actors_lst, loss_critics_lst, loss_hAs_lst)

            center_model.load_state_dict(model.state_dict())  # How to load the q-agent and mixer together
            center_actor_model.load_state_dict(actor_model.state_dict())
            center_mixer.load_state_dict(mixer.state_dict())
            center_critic_model.load_state_dict(critic_model.state_dict())

            # if queue.qsize() > arg_dict["batch_size"] * arg_dict["buffer_size"]:
            #     print("warning. data remaining. queue size : ", queue.qsize())

            if summary_queue.qsize() > arg_dict["summary_game_window"]:
                # write_summary_simplified(writer, arg_dict, summary_queue, n_game, loss_lst, optimization_step)
                win_evaluation, score_evaluation = write_summary(writer, arg_dict, summary_queue, n_game, loss_tuple,
                                                                 optimization_step, self_play_board, win_evaluation, score_evaluation)

                loss_lst, pi_loss_lst, v_loss_lst, entropy_lst, move_entropy_lst = [], [], [], [], []

                data_all_queue.put(n_game)
                n_game += arg_dict["summary_game_window"]

            _ = signal_queue.get()

        else:
            time.sleep(0.1)
            # time.sleep(1000)
