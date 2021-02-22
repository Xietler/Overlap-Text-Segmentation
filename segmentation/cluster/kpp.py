# -*- coding: utf-8 -*-
# @Time    : 2019/3/6 19:01
# @Author  : Ruichen Shao
# @File    : kpp.py

from __future__ import division
import numpy
import time
import random
import numpy as np


class Point:
    def __init__(self, p, dim, id=-1):
        self.coordinates = []
        self.pointList = []
        self.id = id
        self.pointCentroid = 0
        for x in range(0, dim):
            self.coordinates.append(p[x])
        self.centroid = None


class KPP():
    def __init__(self, K, X=None, N=0):
        self.K = K
        if X is None:
            if N == 0:
                raise Exception("If no data is provided, a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X = self._init_board_gauss(N, K)
        else:
            self.X = X
            self.N = len(X)
        self.mu = None
        self.clusters = None
        self.method = None
        self.d = []
        self.D2 = []

    def _dist_from_centers(self):
        cent = self.mu
        X = self.X
        D2 = np.array([np.linalg.norm(x - self.mu[-1]) ** 2 for x in X])
        if len(self.D2) == 0:
            self.D2 = np.array(D2[:])
        else:
            for i in range(len(D2)):
                if D2[i] < self.D2[i]:
                    self.D2[i] = D2[i]

    def _choose_next_center(self):
        self.probs = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        # print(self.cumprobs.shape)
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return (self.X[ind])

    def init_centers(self):
        self.mu = random.sample(list(self.X), 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())

def makeRandomPoint(n, lower, upper):
    return numpy.random.normal(loc=upper, size=[lower, n])

if __name__ == '__main__':
    pointList = []
    numPoints = 160000
    dim = 6
    numClusters = 2
    k = 0
    for i in range(0,numClusters):
        num = int(numPoints/numClusters)
        p = makeRandomPoint(dim,num,k)
        k += 5
        pointList += p.tolist()
    kplus = KPP(numClusters, X=np.array(pointList))
    kplus.init_centers()
    cList = [Point(x, len(x)) for x in kplus.mu]

