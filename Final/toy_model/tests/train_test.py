'''
Test file for train.py
'''
from utils import *
from charRNN import CharRNN
import torch
import torch.nn as nn
from train import *
from toy_firstchar_model import ToyFirstCharModel

def test_train():
    # prep data
    device=torch.device("cpu")
    #toy_model = ToyFirstCharModel(device=device)
    #train_set, dev_set, long_dev_set, longer_dev_set, test_set = toy_model.load_data()
    train_data= [[],[]]
    # train_data[0] = train_set[0][:100]
    # train_data[1] = train_set[1][:100]
    train_data[0] = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6 ,7],
        [4, 5, 6 ,7, 8],
    ]
    train_data[1] = [5, 6, 7, 8, 9]
    dev_data = [[[2, 3, 4, 5, 6]], [7]]
    n_char = 10
    embed_size= 10
    hidden_size= 10
    model = CharRNN(n_char, embed_size, hidden_size)
    lr = 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    batch_size = 2
    dev_data = []
    train(model, train_data, dev_data, optimizer, loss_func, device)

def test_train_big():
    # prep data
    device=torch.device("cpu")
    toy_model = ToyFirstCharModel()
    train_set, dev_set, long_dev_set, longer_dev_set, test_set = toy_model.load_data()
    train_data= [[],[]]
    dev_data = [[], []]
    train_data[0] = train_set[0][101:200]
    train_data[1] = train_set[1][101:200]
    dev_data[0] = train_set[0][201:210]
    dev_data[1] = train_set[1][201:210]
    n_char = 150
    embed_size=30
    hidden_size=30
    model = CharRNN(n_char, embed_size, hidden_size)
    lr = 100
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    batch_size = 10
    dev_data = []
    train(model, train_data, dev_data, optimizer, loss_func, device)