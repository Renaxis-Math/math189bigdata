'''
utils.py
'''

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import csv


def exp_antisym(A, device = None):
    '''
    Performs matrix exponential of an antisymmetric matrix A by:
        1. Diagonalizing into UDU^-1
        2. Computing exp(A) = Ue^DU^-1

    @param A: pytorch Tensor to exponentiate
    @return Tensor
    '''
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if (len(A.shape) != 2 or A.size(0) != A.size(1)):
        raise ValueError("Matrix must be square.")
    if (not torch.allclose(torch.t(A), -1 * A)):
        raise ValueError("Matrix must be antisymmetric.")

    if A.is_cuda:
        # Compute Hermitian B = iA
        B = gpuarray.to_gpu(1j * A.cpu().numpy().astype(complex))

        # Decompose B = UDU^-1
        linalg.init()
        vectors, values = linalg.eig(B, 'N', 'V', imag='T')

        # Then, A = U(-iD)U^-1, eigenvalues of A are -i those of B
        values = -1j * values

        # compute exp(A) = Uexp(-iD)U^-1
        exp1 = linalg.dot(linalg.transpose(vectors.conj()), linalg.diag(cumath.exp(values)))
        exp_np = linalg.dot(exp1, vectors).get()

    else:
        B = A.cpu().numpy()
        (values, vectors) = np.linalg.eig(B)
        exp_np = vectors @ np.diag(np.exp(values)) @ vectors.conj().T

    exp_tensor = torch.from_numpy(exp_np.real)
    if (device == torch.device("cuda")):
        exp_tensor.to(device)
    return exp_tensor.float()
