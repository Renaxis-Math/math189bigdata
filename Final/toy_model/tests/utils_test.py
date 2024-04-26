'''
Test file for utils.py
'''
import math
from utils import *

'''
Note exp(x * [[0, -1], [1, 0]]) = [[cos x, -sin x], [sin x, cos x]] so this is a nice check
'''
def test_exp_antisym_trigo():
    for y in range(0,11):
        x = y / 10
        true_value = torch.tensor([[math.cos(x), -math.sin(x)], [math.sin(x), math.cos(x)]], dtype = torch.float)
        A = x * torch.tensor([[0,-1],[1,0]], dtype = torch.float)
        exp_A = exp_antisym(A)
        assert(torch.allclose(true_value, exp_A, atol= 1e-20))

def test_minibatches():
    queries = [[1, 5, 6, 7], [2, 56, 7,10], [3, 6, 5, 1], [4, 5, 2, 1], [5, 6, 7, 8], [6, 7, 8, 9]]
    answers = [1, 2, 3, 4, 5, 6]
    for i, (q, a) in enumerate(get_minibatches([queries, answers], 2, shuffle=False)):
        assert(q == [queries[2 * i], queries[2 * i + 1]])
        assert(a == [answers[2 * i], answers[2 * i + 1]])

def test_compute_rel_hidden_grad():
    output = torch.tensor([[[0.1, 0.2], [0.4, 0.5]], [[0.2, 0.51], [0.514, 0.31]], [[0.531, 0.11], [0.3, 1.3]]], dtype = torch.float)
    recurrent_weight = torch.tensor([[0, -1], [-1, 0]], dtype = torch.float)
    norm = compute_rel_hidden_grad(output, recurrent_weight, 0, 2)
    assert(math.isclose(norm.item(), math.sqrt(2), abs_tol=0.00001))
