# -*- coding: utf-8 -*-
# @Time    : 2019/3/6 19:25
# @Author  : Ruichen Shao
# @File    : kmeans.py

from __future__ import division
import numpy
import copy
import time
import threading
import math
import random
import bisect
import matplotlib.pyplot as plt
from segmentation.cluster.kpp import KPP

class Point:
    def __init__(self, p, dim, id=-1):
        self.coordinates = []
        self.pointList = []
        self.id = id
        self.pointCentroid = 0
        for x in range(0, dim):
            self.coordinates.append(p[x]);
        self.centroid = None


class Centroid:
    count = 0

    def __init__(self, point):
        self.point = point
        self.count = Centroid.count
        self.pointList = []
        self.centerPos = []
        self.predictions = []
        self.centerPos.append(self.point)
        self.centroid = None
        Centroid.count += 1

    def update(self, point):
        self.point = point
        self.centerPos.append(self.point)

    def addPoint(self, point):
        self.pointList.append(point)

    def removePoint(self, point):
        self.pointList.remove(point)


class Kmeans:
    def __init__(self, k, pointList, kmeansThreshold, centroidsToRemember, initialCentroids=None):
        self.pointList = []
        self.numPoints = len(pointList)
        self.k = k
        self.centroidsToRemember = int(k * centroidsToRemember / 100)
        print("Centroids to Remember:", self.centroidsToRemember)
        self.initPointList = []
        self.dim = len(pointList[0])
        self.kmeansThreshold = kmeansThreshold
        self.error = None
        self.errorList = []
        self.interClusterDistance = {}
        self.lowerBound = {}
        self.upperBound = {}
        self.minimumClusterDistance = {}
        self.r = {}
        self.oldCentroid = {}
        self.centroidDistance = {}
        i = 0
        temp = [0 for x in range(self.k)]
        for point in pointList:
            p = Point(point, self.dim, i)
            i += 1
            self.pointList.append(p)
            self.lowerBound[p.id] = copy.deepcopy(temp)
            self.r[p.id] = False
            self.centroidDistance[p.id] = []

        for clusters in range(k):
            self.interClusterDistance[clusters] = {}
            self.minimumClusterDistance[clusters] = -1

        if initialCentroids != None:
            self.centroidList = self.seeds(initialCentroids)
        else:
            self.centroidList = self.selectSeeds(self.k)
        self.mainFunction()

    def selectSeeds(self, k):
        seeds = random.sample(self.pointList, k)
        centroidList = []
        for seed in seeds:
            centroidList.append(Centroid(seed))
        return centroidList

    def seeds(self, initList):
        centroidList = []
        for seed in initList:
            centroidList.append(Centroid(seed))
        return centroidList

    def getDistance(self, point1, point2):
        distance = 0
        for x in range(0, self.dim):
            distance += (point1.coordinates[x] - point2.coordinates[x]) ** 2
        return (distance) ** (0.5)

    def getCentroidInit(self, point):
        minDist = -1
        pos = 0

        if point.centroid is not None:
            dist = self.getDistance(point, self.centroidList[point.centroid].point)
            minDist = dist
            closestCentroid = point.centroid
            currCentroid = point.centroid
        else:
            dist = self.getDistance(point, self.centroidList[pos].point)
            minDist = dist
            closestCentroid = pos
            currCentroid = pos
        self.lowerBound[point.id][closestCentroid] = minDist
        self.centroidDistance[point.id].append((minDist, currCentroid))
        for centroid in self.centroidList:
            if pos != currCentroid:
                if 0.5 * self.interClusterDistance[closestCentroid][pos] < minDist:
                    dist = self.getDistance(point, centroid.point)
                    self.lowerBound[point.id][pos] = dist
                    if len(self.centroidDistance[point.id]) < self.centroidsToRemember:
                        bisect.insort(self.centroidDistance[point.id], (dist, pos))
                    elif self.centroidDistance[point.id][self.centroidsToRemember - 1][0] > dist:
                        bisect.insort(self.centroidDistance[point.id], (dist, pos))
                        del self.centroidDistance[point.id][self.centroidsToRemember]
                    if minDist > dist:
                        minDist = dist
                        closestCentroid = pos
                else:
                    if len(self.centroidDistance[point.id]) < self.centroidsToRemember:
                        bisect.insort(self.centroidDistance[point.id],
                                      (self.interClusterDistance[closestCentroid][pos] - minDist, pos))
                    elif self.centroidDistance[point.id][self.centroidsToRemember - 1][0] > (
                            self.interClusterDistance[closestCentroid][pos] - minDist):
                        bisect.insort(self.centroidDistance[point.id],
                                      (self.interClusterDistance[closestCentroid][pos] - minDist, pos))
                        del self.centroidDistance[point.id][self.centroidsToRemember]
            pos += 1

        self.upperBound[point.id] = minDist
        return (closestCentroid, minDist)

    def getCentroid(self, point):
        if self.r[point.id]:
            minDist = self.getDistance(point, self.centroidList[point.centroid].point)
            self.upperBound[point.id] = minDist
            self.r[point.id] = False
        else:
            minDist = self.upperBound[point.id]
        pos = 0
        closestCentroid = point.centroid
        for x in self.initPointList[point.id]:
            centroid = self.centroidList[x]
            if x != point.centroid:
                if self.upperBound[point.id] > self.lowerBound[point.id][x]:
                    if self.upperBound[point.id] > 0.5 * self.interClusterDistance[closestCentroid][x]:
                        if minDist > self.lowerBound[point.id][x] or minDist > 0.5 * \
                                self.interClusterDistance[closestCentroid][x]:
                            dist = self.getDistance(point, centroid.point)
                            self.lowerBound[point.id][x] = dist
                            if minDist > dist:
                                minDist = dist
                                closestCentroid = x
                                self.upperBound[point.id] = minDist
            pos += 1
        return (closestCentroid, minDist)

    def getCentroidFinal(self, point):
        if self.r[point.id]:
            minDist = self.getDistance(point, self.centroidList[point.centroid].point)
            self.upperBound[point.id] = minDist
            self.r[point.id] = False
        else:
            minDist = self.upperBound[point.id]
        pos = 0
        closestCentroid = point.centroid
        for centroid in self.centroidList:
            if pos != point.centroid:
                if self.upperBound[point.id] > self.lowerBound[point.id][pos]:
                    if self.upperBound[point.id] > 0.5 * self.interClusterDistance[closestCentroid][pos]:
                        if minDist > self.lowerBound[point.id][pos] or minDist > 0.5 * \
                                self.interClusterDistance[closestCentroid][pos]:
                            dist = self.getDistance(point, centroid.point)
                            self.lowerBound[point.id][pos] = dist
                            if minDist > dist:
                                minDist = dist
                                closestCentroid = pos
                                self.upperBound[point.id] = minDist
            pos += 1
        return (closestCentroid, minDist)

    def reCalculateCentroid(self):
        pos = 0
        for centroid in self.centroidList:
            self.oldCentroid[pos] = copy.deepcopy(centroid.point)
            zeroArr = []
            for x in range(0, self.dim):
                zeroArr.append(0)
            mean = Point(zeroArr, self.dim)
            for point in centroid.pointList:
                for x in range(0, self.dim):
                    mean.coordinates[x] += point.coordinates[x]
            for x in range(0, self.dim):
                try:
                    mean.coordinates[x] = mean.coordinates[x] / len(centroid.pointList)
                except:
                    mean.coordinates[x] = 0
            centroid.update(mean)
            self.centroidList[pos] = centroid
            pos += 1

    def calcInterCluster(self):
        for i in range(0, self.k):
            for j in range(i + 1, self.k):
                temp = self.getDistance(self.centroidList[i].point, self.centroidList[j].point)
                self.interClusterDistance[i][j] = temp
                self.interClusterDistance[j][i] = temp
                if self.minimumClusterDistance[i] == -1 or self.minimumClusterDistance[i] > 0.5 * temp:
                    self.minimumClusterDistance[i] = 0.5 * temp
                if self.minimumClusterDistance[j] == -1 or self.minimumClusterDistance[j] > 0.5 * temp:
                    self.minimumClusterDistance[j] = 0.5 * temp

    def assignPointsInit(self):
        self.calcInterCluster()
        self.initPointList = {}
        for i in range(len(self.pointList) - 1, -1, -1):
            temp = self.getCentroidInit(self.pointList[i])
            self.initPointList[self.pointList[i].id] = []
            for l in range(0, self.centroidsToRemember):
                self.initPointList[self.pointList[i].id].append(self.centroidDistance[self.pointList[i].id][l][1])
            centroidPos = temp[0]
            centroidDist = temp[1]
            if self.pointList[i].centroid is None:
                self.pointList[i].centroid = centroidPos
                self.centroidList[centroidPos].pointList.append(copy.deepcopy(self.pointList[i]))

    def assignPoints(self):
        doneMap = {}
        self.calcInterCluster()
        self.distanceMap = {}
        for x in range(self.k):
            self.distanceMap[x] = self.getDistance(self.oldCentroid[x], self.centroidList[x].point)
        for i in range(len(self.centroidList) - 1, -1, -1):
            for j in range(len(self.centroidList[i].pointList) - 1, -1, -1):
                try:
                    a = doneMap[self.centroidList[i].pointList[j].id]
                except:
                    for x in self.initPointList[self.centroidList[i].pointList[j].id]:
                        self.lowerBound[self.centroidList[i].pointList[j].id][x] = max(
                            (self.lowerBound[self.centroidList[i].pointList[j].id][x] - self.distanceMap[x]), 0)
                    self.upperBound[self.centroidList[i].pointList[j].id] += self.distanceMap[
                        self.centroidList[i].pointList[j].centroid]
                    self.r[self.centroidList[i].pointList[j].id] = True
                    doneMap[self.centroidList[i].pointList[j].id] = 1
                    if self.upperBound[self.centroidList[i].pointList[j].id] > self.minimumClusterDistance[
                        self.centroidList[i].pointList[j].centroid]:
                        temp = self.getCentroid(self.centroidList[i].pointList[j])
                        centroidPos = temp[0]
                        centroidDist = temp[1]
                        if self.centroidList[i].pointList[j].centroid != centroidPos:
                            self.centroidList[i].pointList[j].centroid = centroidPos
                            self.centroidList[centroidPos].pointList.append(
                                copy.deepcopy(self.centroidList[i].pointList[j]))
                            del self.centroidList[i].pointList[j]

    def assignPointsFinal(self):
        doneMap = {}
        self.calcInterCluster()
        self.distanceMap = {}
        for x in range(self.k):
            self.distanceMap[x] = self.getDistance(self.oldCentroid[x], self.centroidList[x].point)
        for i in range(len(self.centroidList) - 1, -1, -1):
            for j in range(len(self.centroidList[i].pointList) - 1, -1, -1):
                try:
                    a = doneMap[self.centroidList[i].pointList[j].id]
                except:
                    for x in self.initPointList[self.centroidList[i].pointList[j].id]:
                        self.lowerBound[self.centroidList[i].pointList[j].id][x] = max(
                            (self.lowerBound[self.centroidList[i].pointList[j].id][x] - self.distanceMap[x]), 0)
                    self.upperBound[self.centroidList[i].pointList[j].id] += self.distanceMap[
                        self.centroidList[i].pointList[j].centroid]
                    self.r[self.centroidList[i].pointList[j].id] = True
                    doneMap[self.centroidList[i].pointList[j].id] = 1
                    if self.upperBound[self.centroidList[i].pointList[j].id] > self.minimumClusterDistance[
                        self.centroidList[i].pointList[j].centroid]:
                        temp = self.getCentroidFinal(self.centroidList[i].pointList[j])
                        centroidPos = temp[0]
                        centroidDist = temp[1]
                        if self.centroidList[i].pointList[j].centroid != centroidPos:
                            self.centroidList[i].pointList[j].centroid = centroidPos
                            self.centroidList[centroidPos].pointList.append(
                                copy.deepcopy(self.centroidList[i].pointList[j]))
                            del self.centroidList[i].pointList[j]

    def calculateError(self, config):
        error = 0
        for centroid in self.centroidList:
            for point in centroid.pointList:
                error += self.getDistance(point, centroid.point) ** 2
        return error

    def errorCount(self):
        self.t = threading.Timer(0.5, self.errorCount)
        self.t.start()
        startTime = time.time()
        timeStamp = 0
        if self.error != None:
            timeStamp = math.log(self.error)
        endTime = time.time()
        self.errorList.append(timeStamp)
        self.ti += 0.5

    def mainFunction(self):
        self.iteration = 1
        self.ti = 0.0
        self.errorCount()
        error1 = 2 * self.kmeansThreshold + 1
        error2 = 0
        iterationNo = 0
        self.currentTime = time.time()
        self.startTime = time.time()
        self.assignPointsInit()
        self.reCalculateCentroid()
        print("First Step:", time.time() - self.startTime)
        while (100 * abs(error1 - error2) / abs(error1)) > self.kmeansThreshold:
            iterationNo += 1
            self.iteration = iterationNo
            error1 = self.calculateError(self.centroidList)
            self.error = error1
            print("Iteration:", iterationNo, "Error:", error1)
            self.assignPoints()
            self.reCalculateCentroid()
            error2 = self.calculateError(self.centroidList)
            self.error = error2
        print(error1, error2)
        print(100 * abs(error1 - error2) / abs(error1))
        self.assignPointsFinal()
        self.reCalculateCentroid()
        error = self.calculateError(self.centroidList)
        self.error = error
        print("Extra Iteration Error:", error)
        time.sleep(1)
        self.t.cancel()

def makeRandomPoint(n, lower, upper):
    return numpy.random.normal(loc=upper, size=[lower, n])

if __name__ == '__main__':
    pointList = []
    x = []
    y = []
    c = []
    numPoints = 5000
    dim = 2
    numClusters = 5
    k = 0
    for i in range(0,numClusters):
        num = int(numPoints/numClusters)
        p = makeRandomPoint(dim,num,k)
        k += 5
        pointList += p.tolist()
    start = time.time()
    kplus = KPP(numClusters, X=numpy.array(pointList))
    kplus.init_centers()
    cList = [Point(x, len(x)) for x in kplus.mu]
    config1= Kmeans(numClusters, pointList, 0, 40, cList)
    print("Time taken:", time.time() - start)
    c = ['r', 'g', 'k', 'b', 'y']
    for i, centroid in enumerate(config1.centroidList):
        x = []
        y = []
        point = centroid.centerPos[0]
        for zz in centroid.pointList:
            x.append(zz.coordinates[0])
            y.append(zz.coordinates[1])
        plt.scatter(x, y, c=c[i])
    plt.show()

