# -*- coding: utf-8 -*-
# @Time    : 2019/3/7 10:43
# @Author  : Ruichen Shao
# @File    : fcm.py

# fuzzy cmeans for pytorch
import torch
import numpy as np

def pairwise_distance(data1, data2=None, cuda=True):

    if data2 is None:
        data2 = data1

    if cuda is False:
        data1, data2 = data1.cpu(), data2.cpu()

    A = data1.unsqueeze(dim=1)
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1).squeeze()
    return dis

def criterion(x, v, m):
    # x: N * D
    # v: C * D
    # dis: C * N
    dis = pairwise_distance(x, v).t()
    dis = torch.where(dis>=torch.finfo(torch.float32).eps, dis, torch.tensor([torch.finfo(torch.float32).eps]))

    # if fuzzifier and distance are both small
    # d2 will be inf
    exp = -2. / (m - 1)
    d2 = dis ** exp
    # u: C * N
    u = d2 / torch.sum(d2, (0,), True)


    return u, dis

def update_clusters(x, u, m):
    # um: C * N
    um = u ** m
    # print(um.dot(x).shape)
    # print(um.sum((1,), True).shape)
    v = torch.tensor(np.dot(um.numpy(), x.numpy())) / um.sum((1,), True)
    return v

def cmeans(data, cluster_num, fuzzifier, threshold, max_iterations, v0=None):
    N, S = data.shape
    v = torch.empty((max_iterations, cluster_num, S))
    v[0] = v0
    u = torch.zeros((max_iterations, cluster_num, N))
    t = 0

    while t < max_iterations - 1:
        u[t], d = criterion(data, v[t], fuzzifier)
        v[t+1] = update_clusters(data, u[t], fuzzifier)

        if torch.norm(v[t+1]-v[t], 2) < threshold:
            print('break')
            break

        t += 1

    return v[t], v[0], u[t-1], u[0], d, t

if __name__ == '__main__':
    import numpy as np
    from numpy.linalg import cholesky
    import matplotlib.pyplot as plt
    torch.set_default_dtype(torch.float64)
    # generate gaussian distribution
    num_samples = 2000
    num_features = 2
    c = 2
    mu1 = np.array([[1, 5]])
    Sigma1 = np.array([[1, 0.5], [1.5, 3]])
    R1 = cholesky(Sigma1)
    s1 = np.dot(np.random.randn(num_samples, num_features), R1) + mu1

    mu2 = np.array([[8, 9]])
    Sigma2 = np.array([[1, 0.5], [1.5, 3]])
    R2 = cholesky(Sigma2)
    s2 = np.dot(np.random.randn(num_samples, num_features), R2) + mu2

    x = torch.cat((torch.tensor(s1), torch.tensor(s2)), dim=0)

    _, _, u, _, _, _ = cmeans(x, c, 1.2, 0.1, 100, torch.cat((x[0].reshape((1,2)), x[1].reshape(1,2))))
    idx = torch.argmax(u, dim=0)
    # u = torch.where(u>=0.5, torch.tensor([1]), torch.tensor([0]))
    # print(idx.shape)
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=idx.numpy())
    plt.show()

