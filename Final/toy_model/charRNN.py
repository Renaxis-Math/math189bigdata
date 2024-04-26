import torch
import torch.nn as nn
from orthornn_model import OrthoRNN

class CharRNN(nn.Module):
    """ Orthogonal RNN Character-based Language Model
    This model will aim to predict the next character, given a number of previous characters.
    """

    def __init__(self, n_char, embed_size, hidden_size, device, num_layers = 1, bias = False, dropout = 0.2, bidirectional = False, ortho = True, weightdrop=0.0):
        """ Initialize the parser model.

        @param n_char (int): number of characters
        @param embed_size (int): dimensionality of character embeddings
        @param hidden_size (int): dimensionality of hidden state
        @param num_layers (int): number of layers in RNN
        @param dropout_prob (float): dropout probability
        @param bidirectional (bool): if RNN is bidirectional
        """
        super(CharRNN, self).__init__()
        # initialize initial hidden state (h_0)
        h_0 = torch.zeros(1, 1, hidden_size)
        self.h_0 = nn.Parameter(h_0, requires_grad=True)
        self.orthoRNN = OrthoRNN(embed_size, hidden_size, device, num_layers=num_layers, bias=bias, dropout=dropout, bidirectional=bidirectional, ortho=ortho, weightdrop=weightdrop)
        self.embeddings = nn.Embedding(n_char, embed_size) # no padding for now.
        self.projection = nn.Linear(hidden_size, n_char)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, t):
        """
        @param t (Tensor): input tensor of tokens (batch_size, max_word_length)

        @returns distribution (Tensor): Final hidden state (batch_size, n_char)
        """
        batch_size = t.size(0)
        embedded = self.embeddings(t.permute(1,0)) # (max_word_length, batch_size, embed_size)
        embedded_d = self.dropout(embedded)
        h_0 = self.h_0.repeat(1, batch_size, 1) # (1, batch_size, hidden_size)
        output, h_n = self.orthoRNN(embedded_d, h_0) # h_n: (1, batch_size, n_char)
        distribution = self.projection(h_n).squeeze(dim=0) # (batch_size, n_char)
        
        return distribution
