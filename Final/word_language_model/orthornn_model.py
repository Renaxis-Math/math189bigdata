import torch.nn as nn
import torch
from utils import exp_antisym

class uRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, bias = False, dropout = 0, bidirectional = False, ortho = True):
        super(uRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional, nonlinearity = 'relu')
        self.B = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.ortho = ortho
        self.B_grad_reduction = 0.0001

    def forward(self, t, h_0):
        return self.rnn.forward(t, h_0)

    def update_b_grad(self):
        if self.ortho:
            W = self.rnn.weight_hh_l0.data.clone()
            grad = self.rnn.weight_hh_l0.grad.data.clone()
            self.B.grad = (torch.matmul(grad, W.t()) - torch.matmul(W,grad.t())) * self.B_grad_reduction

    def update_weight_from_b(self):
        if self.ortho:
            self.rnn.weight_hh_l0.data = exp_antisym(self.B)


class uGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers = 1, bias = False, dropout = 0, bidirectional = False, ortho = True):
        super(uGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional)
        self.B = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.ortho = ortho
        self.B_grad_reduction = 0.0001

    def forward(self, t, h_0):
        return self.rnn.forward(t, h_0)

    def update_b_grad(self):
        if self.ortho:
            W = self.rnn.weight_hh_l0.data.clone().narrow(0, 2 * self.hidden_size, self.hidden_size)
            grad = self.rnn.weight_hh_l0.grad.data.clone().narrow(0, 2 * self.hidden_size, self.hidden_size)
            self.B.grad = (torch.matmul(grad, W.t()) - torch.matmul(W,grad.t())) * self.B_grad_reduction

    def update_weight_from_b(self):
        if self.ortho:
            hn_range = torch.arange(2 * self.hidden_size + 1, 3 * self.hidden_size)
            self.rnn.weight_hh_l0.data[hn_range] = exp_antisym(self.B)
