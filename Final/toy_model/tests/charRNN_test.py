'''
Test file for utils.py
'''
from utils import *
from charRNN import CharRNN
import torch

def test_init():
    n_char=30
    embed_size=11
    hidden_size=12
    charRNN = CharRNN(n_char, embed_size, hidden_size)

def test_forward():
    n_char=30
    embed_size=11
    hidden_size=12

    charRNN = CharRNN(n_char, embed_size, hidden_size)

    batch_size = 13
    max_word_length= 14
    t = torch.randint(low=0, high=n_char-1, size=(batch_size,max_word_length))

    output = charRNN.forward(t)
    assert(output.shape == (batch_size, n_char))
