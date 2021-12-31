import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

import torch.optim as optim
from torch.optim import lr_scheduler


lr = 3e-3

################# utils  ##################################################
def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

def matrix_poly(matrix, d):
    x = torch.eye(d).double()+ torch.div(matrix, d)
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

###########################################################################



class MLPEncoder(nn.Module):
    """
    MLP encoder module.
    initialization:
    {
    n_xdims:
     n_hid:
      n_out:
      adj_A:
    }


    """
    def __init__(self, n_xdims, n_hid, n_out, adj_A, do_prob=0., factor=True, tol=0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)
        # print(adj_A1)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))  # 2.19 --> 2.64
        x = (self.fc2(H1))  # 2.1
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa  # 100.10.1

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa
    # myA == self.adj_A  ori_A == adj_A1


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_z, n_out, n_hid, do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)  # torch.Size([100, 10, 1])

        return mat_z, out


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

def train(data, lambda_A, c_A, optimizer):
    nll_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    scheduler.step()

    # update optimizer
    optimizer, lr = update_optimizer(optimizer, 3e-3, c_A)


    for _ in range(50):
        optimizer.zero_grad()

        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data)  # x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa
        edges = logits
        dec_x, output = decoder(edges, origin_A, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = tau_A * torch.sum(torch.abs(one_adj_A))

        h_A = _h_A(origin_A, num_nodes)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A * origin_A) + sparse_loss

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, tau_A * lr)
        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < graph_threshold] = 0

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())

    return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A



def test(data):
    enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data)  # x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa
    edges = logits
    dec_x, output = decoder(edges, origin_A, Wa)
    if torch.sum(output != output):
        print('nan error\n')

    # compute metrics
    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < graph_threshold] = 0






#### train
if __name__ == '__main__':
    x_dims = 3
    encoder_hidden = 64
    decoder_hidden = 64
    z_dims = 1
    encoder_dropout = 0.0
    factor = True
    # lr = 3e-3
    lr_decay = 200
    gamma = 1.0
    tau_A = 0.0
    lambda_A = 0.
    c_A = 1
    graph_threshold = 0.05
    k_max_iter = 10
    epochs = 20


    # add adjacency matrix A
    num_nodes = 5
    adj_A = np.zeros((num_nodes, num_nodes))

    encoder = MLPEncoder(x_dims, encoder_hidden, z_dims, adj_A, do_prob=encoder_dropout, factor=factor).double()  # n_xdims, n_hid, n_out, adj_A
    decoder = MLPDecoder(z_dims, x_dims, decoder_hidden, do_prob=encoder_dropout).double()  # n_in_z, n_out, n_hid

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)

    ### ------- train
    data = torch.rand(num_nodes, 3)
    data = Variable(data).double()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    h_A_new = torch.tensor(1.)
    h_tol = 1e-8
    k_max_iter = 2
    h_A_old = np.inf


    for step_k in range(k_max_iter):
        while c_A < 1e+20:   # 相当于notears里面的rho
            for epoch in range(epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(data, lambda_A, c_A, optimizer)

                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, num_nodes)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A *= 10
                else:
                    break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store

            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break

    graph = origin_A.data.clone().numpy()
    graph[np.abs(graph) < graph_threshold] = 0
    print(graph)

















