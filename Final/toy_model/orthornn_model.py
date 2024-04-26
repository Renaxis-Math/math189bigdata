import torch.nn as nn
import torch
from utils import exp_antisym

class OrthoRNN(nn.Module):

    def __init__(self, input_size, hidden_size, device, num_layers = 1, bias = False, dropout = 0, bidirectional = False, ortho = True, weightdrop = 0.0):
        super(OrthoRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional, nonlinearity = 'relu')
        self.B = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.ortho = ortho
        self.mask = None
        self.weightdrop = 1 - (1 - weightdrop)**(0.5)
        self.device = device
        # nn.init.orthogonal_(self.rnn.weight_hh_l0.data)

    def forward(self, t, h_0):
        return self.rnn.forward(t, h_0)

    def update_b_grad(self):
        if self.ortho:
            W = self.rnn.weight_hh_l0.data.clone()
            grad = self.rnn.weight_hh_l0.grad.data.clone()
            if self.mask is None:
                self.B.grad = (torch.matmul(grad, W.t()) - torch.matmul(W,grad.t()))
            else:
                self.B.grad = (torch.matmul(grad, W.t()) - torch.matmul(W,grad.t())) * self.mask

    def update_weight_from_b(self):
        if self.ortho:
            self.renew_mask(self.B.data)
            B_dropped = self.mask * self.B.data
            self.rnn.weight_hh_l0.data = exp_antisym(B_dropped, self.device)

    def renew_mask(self, x):
        x = x.clone()
        # print(x.shape)
        self.mask = x.new_empty(x.size(0), x.size(1), requires_grad=False).bernoulli_(1 - self.weightdrop)
        self.mask = self.mask * self.mask.t()
        self.mask = self.mask.div_(1 - self.weightdrop).div_(1 - self.weightdrop)
        self.mask = self.mask.expand_as(x)