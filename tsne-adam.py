#!/usr/bin/python
from sklearn import datasets
import numpy as np

def shanon():
    """Shanon Entropy"""
    pass

def perp():
    """Perplexity"""
    pass

def high_dim_affinities(X, sigma):
    """Computes high-diemsional pairwise affinities"""
    size = X.shape[0]
    p = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            if i != j:
                p[i, j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / (2*sigma[i]**2) 

    p = p / p.sum(axis=1, keepdims=True)

    return p

def low_dim_affinities(y):
    """Computes low-diemsional pairwise affinities"""
    size = y.shape[0]
    q = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                q[i, j] = (1 + np.linalg(y[i] - y[j])**2)**(-1)

    q = q / q.sum()
    return q

def get_pijs(p):
    """Calculate p_ij for each pair"""
    size = p.shape[0]
    pij = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            pij[i,j] = (p[i, j] + p[j, i]) / (2 * size)
    return pij

def gradient(p, q, y):
    """Computes the gradient"""
    size = pij.shape[0]
    grad = np.zeros(size)
    for i in range(size):
        for j in range(size):
            grad[size] += 4*(p[i,j] - q[i,j])(y[i] - y[j])(1 + np.linalg.norm(y[i] - y[j])**2)**(-1)
    return grad

def tsne(X, ydim=2, T=100, l=1e-4, alpha=0.8):
    size = X.shape[0]

    sigma = np.array([25] * size)
    p_conditional = high_dim_affinities(X, sigma)
    p_ijs = get_pijs(p_conditional)
    # sample from N
    Y = []
    Y.append(np.array([np.random.multivariate_normal([0]*size, 1e-4*np.eye(size)) for _ in range(ydim)]).T)
    
    for t in range(T):
        q_ijs = low_dim_affinities(Y[t-1])
        grads = gradient(p_ijs, q_ijs, Y[t-1])
        # update Y
        Y[t] = Y[t-1] + l * grad + alpha * (Y[t-1] - Y[t-2])

    return Y

load = datasets.load_digits()
digits, labels = load['data'][:100], load['target']
