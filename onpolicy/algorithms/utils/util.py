import copy
import numpy as np
import torch
import torch.nn as nn
import math
import igraph as ig

################# utils  ##################################################

def is_acyclic(adjacency):
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency, prod)
        if np.trace(prod) != 0:
            return False
    return True


EPSILON = 1e-8
def pruning(G, A):
    with torch.no_grad():
        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(A)
        for step, t in enumerate(thresholds):
            # print("Edges/thresh", G.get_adjacency()._data, t)
            to_keep = A > t + EPSILON
            new_adj = torch.tensor(G.get_adjacency()._data) * to_keep

            if is_acyclic(new_adj):
                # G.get_adjacency()._data.copy_(new_adj)
                new_G = ig.Graph.Weighted_Adjacency(new_adj.tolist())
                break
    return new_G, new_adj

def pruning_1(G, A):
    A = torch.tensor(A)
    with torch.no_grad():
        while not is_acyclic(A):
            A_nonzero = torch.nonzero(A)
            rand_int_index = np.random.randint(0, len(A_nonzero))
            A[A_nonzero[rand_int_index][0]][A_nonzero[rand_int_index][1]] = 0
        new_G = ig.Graph.Weighted_Adjacency(A.tolist())
        return new_G, A




def cal_depth(G, n_agents):
    lst = np.ones((n_agents))
    for i in G.topological_sorting():
        parents = G.neighbors(i, mode=ig.IN)
        if (len(parents) != 0):
            max_d = 1
            for j in parents:
                max_d = lst[j] if lst[j] > max_d else max_d
            lst[i] += max_d
    return max(lst)

def modify_adj(sam_1, m_s_1, depth_max):
    sam = torch.tensor(sam_1).clone()
    m_s = m_s_1.clone()
    for i in range(depth_max):
        for j in range(depth_max):
            if(i !=j and sam[i][j] == 0 and sam[j][i] == 0):
                sam[i][j] = 1
                m_s[i][j] = 1
                if not is_acyclic(sam):
                    sam[i][j] = 0
                    m_s[i][j] = 0
                    continue
                else:
                    if cal_depth(ig.Graph.Weighted_Adjacency(sam.numpy().tolist()), depth_max) == depth_max:
                        return sam, m_s

def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

def matrix_poly(matrix, d):
    x = torch.eye(d).double().cuda() + torch.div(matrix, d)
    return torch.matrix_power(x, d)

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1


def nll_gaussian(preds, target, variance, add_const=False):
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def kl_gaussian_sem(preds):
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0)))*0.5



def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output



