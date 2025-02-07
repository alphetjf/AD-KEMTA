import random
import sys
import os
from sklearn.cluster import KMeans
import numpy as np
from PROBLEM import PROBLEM


curPath = os.path.abspath(os.path.dirname(__file__))
# print(curPath)

# '''Retrieve the higher-level directory'''
prePath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# print(prePath)
sys.path.append(prePath)

# print(os.path.abspath(os.path.dirname(os.getcwd())))
# print(os.path.abspath(os.path.join(os.getcwd(), "..")))
# """Retrieve the superior directory"""
# print(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from utils.UniformPoint import generatorPoints
from geneticOperation.selfGA import Selection, Crossover, Mutation

Nc = 1

class CKP(PROBLEM):
    def __init__(self) -> None:
        super().__init__()
        self.K = 0

    def Setting(self, D, M, K):
        self.M = M
        self.D = D
        self.K = K  # control the number of knees
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)*1
        self.upper[:M-1] = 1
        self.encoding = "real"

    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        for p in range(PopDec.shape[0]):
            sums = 0
            for i in range(1, self.D):
                sums += PopDec[p, i]
            g = 1+9*sums/(self.D-1)
            r = 5+PopDec[p, 0]*PopDec[p, 0]+np.cos(2*self.K*np.pi*PopDec[p, 0])/self.K
            PopObj[p, 0] = g*r*np.sin(np.pi*PopDec[p, 0]/2)
            PopObj[p, 1] = g*r*np.cos(np.pi*PopDec[p, 0]/2)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetPF(self, N=100):
        # print(N)
        PopDec, N = generatorPoints(N, self.D, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((self.upper - self.lower), (N, 1))*PopDec + np.tile(self.lower, (N, 1))
        R = self.GetOptimum(PopDec)
        return R

    def GetOptimum(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        for p in range(PopDec.shape[0]):
            g = 1
            r = 5+PopDec[p, 0]*PopDec[p, 0]+np.cos(2*self.K*np.pi*PopDec[p, 0])/self.K
            PopObj[p, 0] = g*r*np.sin(np.pi*PopDec[p, 0]/2)
            PopObj[p, 1] = g*r*np.cos(np.pi*PopDec[p, 0]/2)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        lower = np.ones(1)*low
        upper = np.ones(1)*up
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for p in range(PopDec.shape[0]):
            PopObj[p] = 5+PopDec[p, 0]*PopDec[p, 0]+np.cos(2*self.K*np.pi*PopDec[p, 0])/self.K

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = Crossover(PopDec, lower, upper, ProC=0.9)
            offsprings = Mutation(offsprings, lower, upper, ProM=0.1)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = 5+offsprings[i]*offsprings[i]+np.cos(2*self.K*np.pi*offsprings[i])/self.K
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = 5+PopDec[i]*PopDec[i]+np.cos(2*self.K*np.pi*PopDec[i])/self.K
        return PopDec, PopObj

    def GetKnees_in_PF(self, kn_decs, nc=Nc):
        nums_cluster = nc
        kmeans = KMeans(n_clusters=nums_cluster, random_state=2).fit(kn_decs)
        kn_dec = np.zeros((nums_cluster, 1))
        for i in range(nums_cluster):
            index = np.where(kmeans.labels_ == i)[0]
            kn_dec[i] = np.mean(kn_decs[index])
        print(kn_dec)
        N, D = np.shape(kn_dec)
        k = self.D - 1  # the number of X2
        NI = N**(self.M-1)
        Decs = np.zeros((NI, self.M-1))
        if self.M == 2:
            Decs = kn_dec
        else:
            from itertools import permutations
            group = np.asarray(list(permutations(range(N), self.M-1)))
            for i in range(NI):
                if i < N:
                    Decs[i, :] = kn_dec[i]*np.ones(self.M-1)
                else:
                    Decs[i, :] = kn_dec[group[i, :]]
        randPoints, NI = generatorPoints(NI, k, method='Latin')
        Temp = np.tile(self.upper[self.M-1:] - self.lower[self.M-1:], (NI, 1))*randPoints +\
            np.tile(self.lower[self.M-1:], (NI, 1))
        Decs = np.concatenate((Decs, Temp), axis=1)
        Objs = self.GetOptimum(Decs)
        return Objs

    def GetKneesAround_in_PF(self, kn_decs, nc=Nc):
        nums_cluster = nc
        kmeans = KMeans(n_clusters=nums_cluster, random_state=2).fit(kn_decs)
        kn_dec = np.zeros((nums_cluster, 1))
        for i in range(nums_cluster):
            index = np.where(kmeans.labels_ == i)[0]
            kn_dec[i] = np.mean(kn_decs[index])

        N, D = np.shape(kn_dec)
        print(kn_dec)
        k = self.D-1  # the number of X2
        NI = N**(self.M-1)
        Decs = np.zeros((NI, 1))
        if self.M == 2:
            Decs = kn_dec
        randPoints, NI = generatorPoints(NI, k, method='Latin')
        Temp = np.tile(self.upper[self.M-1:] - self.lower[self.M-1:], (NI, 1))*randPoints +\
            np.tile(self.lower[self.M-1:], (NI, 1))
        Decs = np.concatenate((Decs, Temp), axis=1)

        # Increase disturbance
        N = Decs.shape[0]
        for k in range(N):
            each = 20
            temp = np.ones((each, self.D))
            for i in range(each):
                temp[i, :] = Decs[k, :] + random.gauss(0, 0.01)*np.random.random((1, self.D))
            Decs = np.concatenate((Decs, temp))
        Objs = self.GetOptimum(Decs)
        return Objs


if __name__ == '__main__':
    pmop = CKP()
    pmop.Setting(30, 2, 1)
    PF = pmop.GetPF(N=100)
    # print(PF)
    import matplotlib.pyplot as plt
    kn_decs, kn_kobj = pmop.GetKnees_Of_KFunc()
    kn_decs = np.asarray([[0.482]])
    # kn_decs = np.asarray([[0.240], [0.730]])
    # kn_decs = np.asarray([[0.160], [0.5], [0.82]])
    # kn_decs = np.asarray([[0.12], [0.37], [0.615], [0.86]])
    PF_knees = pmop.GetKnees_in_PF(kn_decs, np.size(kn_decs))
    plt.figure()
    plt.scatter(PF[:, 0], PF[:, 1])
    plt.scatter(PF_knees[:, 0], PF_knees[:, 1], marker="p", c="r", linewidths=2)
    plt.show()
