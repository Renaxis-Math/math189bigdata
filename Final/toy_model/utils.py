'''
utils.py
'''

import numpy as np
# import scipy.linalg.expm as expm
import torch
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os

def minibatches(data, batch_size):
    '''
    Example usage: for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
    '''

    x = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    return get_minibatches([x, y], batch_size)

def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [_minibatch(d, minibatch_indices) for d in data] if list_data \
            else _minibatch(data, minibatch_indices)


def _minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def exp_antisym(A, device):
    '''
    Performs matrix exponential of an antisymmetric matrix A by:
        1. Diagonalizing into UDU^-1
        2. Computing exp(A) = Ue^DU^-1

    @param A: pytorch Tensor to exponentiate
    @return Tensor
    '''
    B = A.cpu().numpy()
    if (len(B.shape) != 2 or B.shape[0] != B.shape[1]):
        raise ValueError("Matrix must be square.")
    if ((B.T != -1 * B).all()):
        raise ValueError("Matrix must be anti-symmetric.")
    (values, vectors) = np.linalg.eig(B)
    exp_np =   vectors @ np.diag(np.exp(values)) @ vectors.conj().T
    exp_tensor = torch.from_numpy(exp_np.real)
    exp_tensor = exp_tensor.to(device)
    return exp_tensor

'''
def exp_normal(A):
    B = A.numpy()
    exp_np = expm(B)
    exp_tensor = torch.from_numpy(exp_np.real)
    return exp_tensor
'''

def replace_grad(W, grad, lr):
    B = torch.matmul(grad, W.t()) - torch.matmul(W,grad.t())
    actual = torch.matmul(exp_antisym(- lr * B ), W)
    # actual = torch.matmul(exp_normal(- lr * B ), W)
    # actual = torch.matmul(W, grad)
    return (actual - W) / lr

def weight_step(W, grad, lr):
    B = torch.matmul(grad, W.t()) - torch.matmul(W,grad.t())
    actual = torch.matmul(exp_antisym(- lr * B ), W)
    return actual

def relu_derivative(T):
    signed = torch.sign(T)
    return (signed + torch.abs(signed)) / 2

def abs_x_derivative(T):
    return torch.abs(torch.sign(T))

def compute_rel_hidden_grad(output, recurrent_weight, first, last, act_d = relu_derivative):
    '''
    @param output: first output Tensor from rnn(), of shape (seq_len, batch, hidden_size)
    @param recurrent_weight: recurrent weight matrix
    @param first: int, denominator hidden state
    @param last: int, numerator hidden state
    @param act_d: derivative function of activation
    '''

    if (last < first or first < 0 or last >= output.size(0)):
        raise ValueError("Invalid first or last.")

    reshaped = output.view(output.size(1), output.size(0), output.size(2))
    batch = reshaped.size(0)
    hidden_size = reshaped.size(2)
    total_norm = 0
    for b in range(batch):
        array = reshaped[b]
        result = torch.eye(hidden_size)
        for t in range(first, last):
            result = torch.diag(act_d(array[t])) @ result
            result = recurrent_weight @ result
        total_norm += torch.norm(result)

    return total_norm / batch

def plot_norm(timesteps, norms, file_name):
    plt.plot(timesteps, norms, 'r--')
    plt.ylabel('Accuracy')
    plt.xlabel('Timesteps')
    full_path = os.path.join(os.getcwd(), file_name)
    plt.savefig(full_path, bbox_inches='tight')

def plot_losses(training_losses, val_losses, file_name):
    plt.plot(training_losses[0], training_losses[1], 'r--')
    plt.plot(val_losses[0], val_losses[1], 'g--')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    full_path = os.path.join(os.getcwd(), file_name)
    plt.savefig(full_path, bbox_inches='tight')

def plot_acc(acc, file_name):
    plt.plot(acc[0], acc[1], 'r--')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    full_path = os.path.join(os.getcwd(), file_name)
    plt.savefig(full_path, bbox_inches='tight')


