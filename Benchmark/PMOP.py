import random
import sys
sys.path.append("..")   
import os
from sklearn.cluster import KMeans
import numpy as np
from PROBLEM import PROBLEM

curPath = os.path.abspath(os.path.dirname(__file__))
# print(curPath)

prePath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# print(prePath)
sys.path.append(prePath)

# print(os.path.abspath(os.path.dirname(os.getcwd())))
# print(os.path.abspath(os.path.join(os.getcwd(), "..")))
# print(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from utils.UniformPoint import generatorPoints
from geneticOperation.selfGA import Selection, OperatorGA

def k1(x, A, B, S):
    return 5 + 10*(x - 0.5)**2 + np.cos(A*np.pi*(x**B))/(A*(2**S))


def k2(x, A, B, S):
    return 1 + np.exp(np.cos(A*np.power(x, B)*np.pi + np.pi/2)) / (A*(2**S))


def k3(x, A, B, S):
    return 1 + np.exp(np.sin(A*np.power(x, B)*np.pi + np.pi/2)) / (A*(2**S))


def k4(x, A, B, S):
    return 2 + np.abs(np.sin(A*np.power(x, B)) - np.cos(A*np.power(x, B)-np.pi/4)) / (A*(2**S))


def k5(x, A, B, S, L):
    return 2 + min(np.sin(2*A*np.power(x, B)*np.pi), np.cos(2*A*np.power(x, B)*np.pi-np.pi/L)) / (A*(2**S))


def k6(x, A, B, S):
    return 2 - (np.exp(np.cos(A*np.power(x, B)*np.pi)) + 0.5*np.power((np.cos(A*np.power(x, B)*np.pi)-0.5), 4)) / (A*(2**S))


def h1(x, f, P=1):
    M = np.size(f)
    for i in range(M):
        for j in range(M-(i+1)):
            f[i] = f[i]*x[j]**P
        if i != 0:
            aux = M - (i+1)
            f[i] = f[i]*(1 - x[aux]**P)
    return f


def h2(x, f, P=1):
    M = np.size(f)
    for i in range(M):
        for j in range(M-(i+1)):
            f[i] *= np.cos(np.power(x[j], P)*np.pi/2)
        if i != 0:
            aux = M - (i+1)
            f[i] *= np.sin(np.power(x[aux], P)*np.pi/2)
    return f


def h3(x, f, P=1):
    M = np.size(f)
    for i in range(M):
        for j in range(M-(i+1)):
            f[i] *= (1 - np.cos(np.power(x[j], P)*np.pi/2))
        if i != 0:
            aux = M - (i+1)
            f[i] *= (1 - np.sin(np.power(x[aux], P)*np.pi/2))
    return f

decision_variable_linkage = True

class PMOP1(PROBLEM):
    def __init__(self) -> None:
        super().__init__()
        self.A = 2  # control the number of knees
        self.B = 1  # controls the location of the knee regions
        self.S = 2  #
        self.P = 1
        self.Linkage = False

    def Setting(self, D, M, A=2, B=1, S=2, P=1):
        self.M = M
        self.D = D
        self.A = A  # control the number of knees
        self.B = B  # controls the location of the knee regions
        self.S = S  #
        self.P = P
        k = self.D-self.M + 1  # the number of X2
        self.lower = np.zeros(self.D)
        self.upper = np.concatenate((np.ones(self.M-1), 10*np.ones(k)))
        self.encoding = "real"
        self.Linkage = decision_variable_linkage

    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g = 0
            for i in range(k):
                if Linkage:
                    tmp1 = (1 + (i+1)/k)*(PopDec[ind, self.M-1+i] - self.lower[self.M-1+i]) - \
                            PopDec[ind, 0]*(self.upper[self.M-1+i] - self.lower[self.M-1+i])
                else:
                    tmp1 = PopDec[ind, self.M-1+i]
                temp = np.abs(tmp1)
                if g < temp:
                    g = temp
            k_ = 1
            for m in range(self.M-1):
                tmp = k1(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.log(k_/(self.M-1))
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h1(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k1(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.log(k_/(self.M-1))
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h1(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetPF(self, N):
        # print(N)
        PopDec, N = generatorPoints(N, self.D, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((self.upper - self.lower), (N, 1))*PopDec + np.tile(self.lower, (N, 1))
        R = self.GetOptimum(PopDec)
        return R

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k1(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k1(offsprings[i], self.A, self.B, self.S)
            # Pool = np.concatenate((PopDec, offsprings), axis=0)
            # PoolObj = np.concatenate((PopObj, OffObj), axis=0)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k1(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj

    def GetKnees_in_PF(self, kn_decs, nc=2):
        nums_cluster = nc
        kmeans = KMeans(n_clusters=nums_cluster, random_state=2).fit(kn_decs)
        kn_dec = np.zeros((nums_cluster, 1))
        for i in range(nums_cluster):
            index = np.where(kmeans.labels_ == i)[0]
            kn_dec[i] = np.mean(kn_decs[index])
        print(kn_dec)
        N, D = np.shape(kn_dec)
        k = self.D-self.M + 1  # the number of X2
        NI = N**(self.M-1)
        Decs = np.zeros((NI, self.M-1))
        if self.M == 2:
            Decs = kn_dec
        else:
            from itertools import product

            group = np.asarray(list(product(range(N), repeat=self.M-1)))
            for i in range(NI):
                for j in range(Decs.shape[1]):
                    Decs[i, j] = kn_dec[group[i, j]]

        randPoints, NI = generatorPoints(NI, k, method='Latin')
        Temp = np.tile(self.upper[self.M-1:] - self.lower[self.M-1:], (NI, 1))*randPoints +\
            np.tile(self.lower[self.M-1:], (NI, 1))
        Decs = np.concatenate((Decs, Temp), axis=1)
        Objs = self.GetOptimum(Decs)
        return Objs

    def GetKneesAround_in_PF(self, kn_decs, nc=2):
        nums_cluster = nc
        kmeans = KMeans(n_clusters=nums_cluster, random_state=2).fit(kn_decs)
        kn_dec = np.zeros((nums_cluster, 1))
        for i in range(nums_cluster):
            index = np.where(kmeans.labels_ == i)[0]
            kn_dec[i] = np.mean(kn_decs[index])

        N, D = np.shape(kn_dec)
        print(kn_dec)
        k = self.D-self.M + 1  # the number of X2
        NI = N**(self.M-1)
        Decs = np.zeros((NI, self.M-1))
        if self.M == 2:
            Decs = kn_dec
        else:
            from itertools import product

            group = np.asarray(list(product(range(N), repeat=self.M-1)))
            for i in range(NI):
                for j in range(Decs.shape[1]):
                    Decs[i, j] = kn_dec[group[i, j]]
        randPoints, NI = generatorPoints(NI, k, method='Latin')
        Temp = np.tile(self.upper[self.M-1:] - self.lower[self.M-1:], (NI, 1))*randPoints +\
            np.tile(self.lower[self.M-1:], (NI, 1))
        Decs = np.concatenate((Decs, Temp), axis=1)

        # 增加扰动
        N = Decs.shape[0]
        for k in range(N):
            each = 20
            temp = np.ones((each, self.D))
            for i in range(each):
                temp[i, :] = Decs[k, :] + random.gauss(0, 0.01)*np.random.random((1, self.D))
            Decs = np.concatenate((Decs, temp))
        Objs = self.GetOptimum(Decs)
        return Objs


class PMOP2(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g, temp = 0, 0
            for i in range(k):
                if Linkage:
                    tmp1 = (1 + np.cos(0.5 * np.pi * (i+1)/k)) *\
                           (PopDec[ind, self.M-1+i] - self.lower[self.M-1+i]) -\
                           PopDec[ind, 0]*(self.upper[self.M-1+i] - self.lower[self.M-1+i])
                else:
                    tmp1 = PopDec[ind, self.M-1+i]
                temp += tmp1**2  # g_2 function
            g = temp
            k_ = 1
            for m in range(self.M-1):
                tmp = k2(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-1))  # k_2 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h2(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k2(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-1))
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h2(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k2(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, lower, upper)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k2(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k2(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP3(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g = 0
            for i in range(k):
                if Linkage:
                    tmp1 = (1 + (i+1)/k) *\
                           (PopDec[ind, self.M-1+i] - self.lower[self.M-1+i]) -\
                           PopDec[ind, 0]*(self.upper[self.M-1+i] - self.lower[self.M-1+i])
                else:
                    tmp1 = PopDec[ind, self.M-1+i]
                g += tmp1**2-10*np.cos(4*np.pi*tmp1)  # g_3 function
            g += 10*k + 1
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(2, k_/(self.M-1))  # k_3 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h3(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(2, k_/(self.M-1))  # k_3 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h3(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k3(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP4(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g = 0
            for i in range(k):
                if Linkage:
                    tmp1 = (1 + (i+1)/k) *\
                           (PopDec[ind, self.M-1+i] - self.lower[self.M-1+i]) -\
                           PopDec[ind, 0]*(self.upper[self.M-1+i] - self.lower[self.M-1+i])
                else:
                    tmp1 = PopDec[ind, self.M-1+i]
                g += (tmp1-0.5)**2-np.cos(20*np.pi*(tmp1-0.5))  # g_4 function
            g = 100*(k+g)
            k_ = 1
            for m in range(self.M-1):
                tmp = k4(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-1))  # k_4 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h2(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k4(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-1))  # k_4 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h2(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k4(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k4(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k4(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP5(PMOP1):
    def __init__(self) -> None:
        super().__init__()
        self.L = 12

    def Setting(self, D, M, A, B, S, P, L):
        self.M = M
        self.D = D
        self.A = A  # control the number of knees
        self.B = B  # controls the location of the knee regions
        self.S = S  #
        self.P = P
        self.L = L
        k = self.D-self.M + 1  # the number of X2
        self.lower = np.zeros(self.D)
        self.upper = np.concatenate((np.ones(self.M-1), 10*np.ones(k)))
        self.encoding = "real"
        self.Linkage = decision_variable_linkage

    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g, temp = 0, 0
            for i in range(self.D-k, self.D-1):
                if Linkage:
                    # L1 function : transform x[i] to tmp1
                    tmp1 = (1 + 0.1*(i-self.M+2)/(0.1*k)) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                    # L1 function : transform x[i+1] to tmp2
                    tmp2 = (1 + 0.1*(i-self.M+2)/(0.1*k)) *\
                           (PopDec[ind, i+1] - self.lower[i+1]) - PopDec[ind, 0]*(self.upper[i+1] - self.lower[i+1])
                else:
                    tmp1 = PopDec[ind, i]
                    tmp2 = PopDec[ind, i+1]
                temp += 100*(tmp1**2-tmp2)**2+(tmp1-1)**2  # g_5 function
            g = temp
            k_ = 1
            for m in range(self.M-1):
                tmp = k5(PopDec[ind, m], self.A, self.B, self.S, self.L)
                k_ = k_*tmp
            k_ = np.power(k_/(self.M-1), 0.4)  # k_5 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h1(PopDec[ind, :], PopObj[ind, :], self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k5(PopDec[i, m], self.A, self.B, self.S, self.L)
                k_ = k_*tmp
            k_ = np.power(k_/(self.M-1), 0.4)  # k_5 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h1(PopDec[i, :], PopObj[i, :], self.B)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k5(PopDec[i], self.A, self.B, self.S, self.L)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k5(offsprings[i], self.A, self.B, self.S, self.L)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k5(PopDec[i], self.A, self.B, self.S, self.L)
        return PopDec, PopObj


class PMOP6(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g, temp = 0, 0
            for i in range(self.D-k, self.D):
                if Linkage:
                    tmp1 = (1 + np.cos(0.5*np.pi*0.1*(i-self.M+2)/(0.1*k))) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                temp += tmp1**2-10*np.cos(2*np.pi*tmp1)+10  # g_6 function
            g = temp
            k_ = 1
            for m in range(self.M-1):
                tmp = k6(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(2, k_/(self.M-1))  # k_6 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h3(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k6(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(2, k_/(self.M-1))  # k_6 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h3(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k6(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k6(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k6(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP7(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g, temp, temp2 = 0, 0, 1.
            for i in range(self.D-k, self.D):
                if Linkage:
                    tmp1 = (1 + 0.1*(i - self.M + 2) / (0.1 * k)) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                temp += tmp1**2/4000  # g_7 function
                temp2 *= np.cos(tmp1/np.sqrt(i))    # g_7 function
            g = temp - temp2 + 1    # g_7 function
            k_ = 1
            for m in range(self.M-1):
                tmp = k2(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(3, k_/(self.M-1))  # k_2 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h1(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k2(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(3, k_/(self.M-1))  # k_2 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h1(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k2(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k2(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k2(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP8(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g, tmp2, tmp3 = 0, 0., 0.
            for i in range(0, k):
                if Linkage:
                    tmp1 = (1 + np.cos(0.5*np.pi*(i + 1)/k)) *\
                           (PopDec[ind, self.M-1+i] - self.lower[self.M-1+i]) - PopDec[ind, 0]*(self.upper[self.M-1+i] - self.lower[self.M-1+i])
                else:
                    tmp1 = PopDec[ind, self.M-1+i]
                tmp2 += tmp1**2    # g_8 function
                tmp3 += np.cos(2*np.pi*tmp1)
            g = -20*np.exp(-0.2*np.sqrt(tmp2/k)) - np.exp(tmp3/k)+20 + np.e    # g_8 function
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = k_/(self.M-1)  # k_3 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h2(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = k_/(self.M-1)  # k_3 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h2(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k3(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP9(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g = -np.infty
            for i in range(self.D-k, self.D):
                if Linkage:
                    tmp1 = (1 + 0.1 * (i - self.M + 2) / (0.1 * k)) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                temp = np.abs(tmp1)    # g_1 function
                g = max(g, temp)    # g_1 function
            k_ = 1
            for m in range(self.M-1):
                tmp = k6(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = k_/(self.M-1)  # k_6 function
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h3(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k6(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = k_/(self.M-1)  # k_6 function
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h3(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k6(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k6(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k6(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP10(PMOP5):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g3 = 0.
            tmp1, temp, temp2 = 0., 0., 1.
            for i in range(self.D-k, self.D):
                if Linkage:
                    # L1 function : transform x[i] to tmp1
                    tmp1 = (1 + 0.1*np.cos(0.5*np.pi*0.1*(i-self.M+2)/(0.1*k))) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                g3 += tmp1**2 - 10*np.cos(4*np.pi*tmp1)  # g_3
                temp += tmp1**2/4000    # g_7
                temp2 *= np.cos(tmp1/np.sqrt(i))    # g_7
            g3 += 1+10*k            # g_3
            g7 = temp - temp2 + 1   # g_7
            k_ = 1
            for m in range(self.M-1):
                tmp = k5(PopDec[ind, m], self.A, self.B, self.S, self.L)
                k_ = k_*tmp
            k_ = np.power(k_/(self.M-1), 0.2)  # k_5 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[ind, m] = (1.+g3)*k_     # odd index
                else:
                    PopObj[ind, m] = (1.+g7)*k_     # even index
            PopObj[ind, :] = h1(PopDec[ind, :], PopObj[ind, :], self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g3, g7 = 1., 0.
            k_ = 1
            for m in range(self.M-1):
                tmp = k5(PopDec[i, m], self.A, self.B, self.S, self.L)
                k_ = k_*tmp
            k_ = np.power(k_/(self.M-1), 0.2)  # k_5 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[i, m] = (1.+g3)*k_     # odd index
                else:
                    PopObj[i, m] = (1.+g7)*k_     # even index
            PopObj[i, :] = h1(PopDec[i, :], PopObj[i, :], self.B)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k5(PopDec[i], self.A, self.B, self.S, self.L)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k5(offsprings[i], self.A, self.B, self.S, self.L)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k5(PopDec[i], self.A, self.B, self.S, self.L)
        return PopDec, PopObj


class PMOP11(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            temp2, temp3 = 0, 0
            for i in range(self.D-k, self.D):
                if Linkage:
                    tmp1 = (1 + np.cos(0.5 * np.pi * 0.1 * (i-self.M+2)/(0.1*k))) * (PopDec[ind, i] - self.lower[i]) -\
                            PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                temp = np.abs(tmp1)         # g_1 function
                temp2 = max(temp2, temp)    # g_1 function
                temp3 += np.power(tmp1, 2)  # g_2 function
            g1 = temp2      # g_1 function
            g2 = temp3      # g_2 function
            k_ = 1
            for m in range(self.M-1):
                tmp = k2(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.log(1.0/(k_/(self.M-1))+1)  # k_2 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[ind, m] = (1.+g1)*k_  # odd index
                else:
                    PopObj[ind, m] = (1.+g2)*k_  # even index
            PopObj[ind, :] = h2(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g1, g2 = 0., 0.
            k_ = 1
            for m in range(self.M-1):
                tmp = k2(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.log(1.0/(k_/(self.M-1))+1)  # k_2 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[i, m] = (1.+g1)*k_  # odd index
                else:
                    PopObj[i, m] = (1.+g2)*k_  # even index
            PopObj[i, :] = h2(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k2(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, lower, upper)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k2(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k2(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP12(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            temp, tmp2, tmp3 = 0, 0., 0.
            for i in range(self.D-k, self.D):
                if Linkage:
                    tmp1 = (1 + np.cos(0.5*np.pi*0.1*(i-self.M+2)/(0.1*k))) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                temp += tmp1**2-10*np.cos(2*np.pi*tmp1)+10      # g_6 function
                tmp2 += tmp1**2                                 # g_8 function
                tmp3 += np.cos(2*np.pi*tmp1)
            g6 = temp       # g_6 function
            g8 = -20*np.exp(-0.2*np.sqrt(tmp2/k)) - np.exp(tmp3/k)+20+np.e    # g_8 function
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(k_/(self.M-1), 2)  # k_3 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[ind, m] = (1+g6)*k_      # odd index
                else:
                    PopObj[ind, m] = (1+g8)*k_      # even index
            PopObj[ind, :] = h3(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g6, g8 = 0, 0.
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.power(k_/(self.M-1), 2)  # k_3 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[i, m] = (1+g6)*k_      # odd index
                else:
                    PopObj[i, m] = (1+g8)*k_      # even index
            PopObj[i, :] = h3(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k3(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP13(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            g = -np.infty
            for i in range(k):
                if Linkage:
                    tmp1 = (1 + (i+1)/k)*(PopDec[ind, self.M-1+i] - self.lower[self.M-1+i]) - \
                            PopDec[ind, 0]*(self.upper[self.M-1+i] - self.lower[self.M-1+i])
                else:
                    tmp1 = PopDec[ind, self.M-1+i]
                temp = np.abs(tmp1)     # g_1 function
                g = max(temp, g)
            k_ = 1
            for m in range(self.M-1):
                tmp = k1(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-2))
            for m in range(self.M):
                PopObj[ind, m] = (1+g)*k_
            PopObj[ind, :] = h1(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g = 0
            k_ = 1
            for m in range(self.M-1):
                tmp = k1(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-2))
            for m in range(self.M):
                PopObj[i, m] = (1+g)*k_
            PopObj[i, :] = h1(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k1(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k1(offsprings[i], self.A, self.B, self.S)
            # Pool = np.concatenate((PopDec, offsprings), axis=0)
            # PoolObj = np.concatenate((PopObj, OffObj), axis=0)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k1(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


class PMOP14(PMOP1):
    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], self.M))
        k = self.D-self.M + 1  # the number of X2
        Linkage = self.Linkage
        for ind in range(PopDec.shape[0]):
            temp, tmp2, tmp3 = 0, 0., 0.
            for i in range(self.D-k, self.D):
                if Linkage:
                    tmp1 = (1 + np.cos(0.5*np.pi*0.1*(i-self.M+2)/(0.1*k))) *\
                           (PopDec[ind, i] - self.lower[i]) - PopDec[ind, 0]*(self.upper[i] - self.lower[i])
                else:
                    tmp1 = PopDec[ind, i]
                temp += tmp1**2-10*np.cos(2*np.pi*tmp1)+10      # g_6 function
                tmp2 += tmp1**2                                 # g_8 function
                tmp3 += np.cos(2*np.pi*tmp1)
            g6 = temp       # g_6 function
            g8 = -20*np.exp(-0.2*np.sqrt(tmp2/k)) - np.exp(tmp3/k)+20+np.e    # g_8 function
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[ind, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-2))  # k_3 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[ind, m] = (1+g6)*k_      # odd index
                else:
                    PopObj[ind, m] = (1+g8)*k_      # even index
            PopObj[ind, :] = h1(PopDec[ind, :], PopObj[ind, :], P=self.P)
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, PopDec):
        N = PopDec.shape[0]
        # print("PopDec:", PopDec)
        PopObj = np.zeros((N, self.M))
        for i in range(N):
            g6, g8 = 0, 0.
            k_ = 1
            for m in range(self.M-1):
                tmp = k3(PopDec[i, m], self.A, self.B, self.S)
                k_ = k_*tmp
            k_ = np.sqrt(k_/(self.M-2))  # k_3 function
            for m in range(self.M):
                if np.mod(m, 2) == 0:
                    PopObj[i, m] = (1+g6)*k_      # odd index
                else:
                    PopObj[i, m] = (1+g8)*k_      # even index
            PopObj[i, :] = h1(PopDec[i, :], PopObj[i, :], P=self.P)
        return PopObj

    def GetKnees_Of_KFunc(self, low=0, up=1):
        N = 100
        # l, u = 0, 1
        lower = np.zeros(1)
        upper = np.ones(1)
        # print(N)
        PopDec, N = generatorPoints(N, 1, method='Latin')
        # print(PopDec, N)
        PopDec = np.tile((up - low), (N, 1))*PopDec + np.tile(low, (N, 1))
        maxG = 100
        PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
        for i in range(PopDec.shape[0]):
            PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)

        import matplotlib.pyplot as plt
        plt.scatter(PopDec, PopObj)
        plt.title('The Knee function')
        plt.show()

        for g in range(maxG):
            offsprings = OperatorGA(PopDec, 1, 20, 1, 20, upper, lower)
            OffObj = np.zeros(offsprings.shape[0])[:, np.newaxis]
            for i in range(offsprings.shape[0]):
                OffObj[i] = k3(offsprings[i], self.A, self.B, self.S)
            PopDec = Selection(OffObj, offsprings)
            PopObj = np.zeros(PopDec.shape[0])[:, np.newaxis]
            for i in range(PopDec.shape[0]):
                PopObj[i] = k3(PopDec[i], self.A, self.B, self.S)
        return PopDec, PopObj


if __name__ == "__main__":
    pmop = PMOP8()
    M = 3
    D = M + 9
    pmop.Setting(D, M, 2, 1, -1, 1)
    # pmop.Setting(D, M, 1, 1, 2, 1, 12)
    PF = pmop.GetPF(3000)
    # print(PF)
    kn_decs, kn_kobj = pmop.GetKnees_Of_KFunc()
    # kn_decs = np.asarray([[0.3], [0.7]])      # PMOP1
    # kn_decs = np.asarray([[0.125], [0.625]])   # PMOP2
    # kn_decs = np.asarray([[0.125], [0.625]])   # PMOP3
    # kn_decs = np.asarray([[0.195], [0.725]])     # PMOP4
    # kn_decs = np.asarray([[0.55], [0.75]])     # PMOP5
    # kn_decs = np.asarray([[0.5]])     # PMOP6
    # kn_decs = np.asarray([[0.125], [0.625]])     # PMOP7
    # kn_decs = np.asarray([[0.250], [0.75]])     # PMOP8
    # kn_decs = np.asarray([[0.5]])     # PMOP9
    # kn_decs = np.asarray([[0.55], [0.75]])     # PMOP10
    # kn_decs = np.asarray([[0.125], [0.625]])   # PMOP11-concave
    # kn_decs = np.asarray([[0.375], [0.875]])   # PMOP11-convex
    # kn_decs = np.asarray([[0.250], [0.75]])     # PMOP12
    # kn_decs = np.asarray([[0.5]])      # PMOP13
    kn_decs = np.asarray([[0.5]])      # PMOP14

    PF_knees = pmop.GetKnees_in_PF(kn_decs, np.size(kn_decs))
    regions = pmop.GetKneesAround_in_PF(kn_decs, np.size(kn_decs))

    import matplotlib.pyplot as plt
    # plt.scatter(PF[:, 0], PF[:, 1], PF[:, 2])
    # # plt.scatter(PF[:, 0], PF[:, 1])
    # # plt.scatter(PF_knees[:, 0], PF_knees[:, 1], marker="p", c="r", linewidths=2)
    # plt.show()

    fig = plt.figure(figsize=(14, 10), dpi=50, facecolor='w', edgecolor='k')
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(PF[:, 0], PF[:, 1], PF[:, 2], marker='.', alpha=0.5, label='$PF$')
    ax.scatter(regions[:, 0], regions[:, 1], regions[:, 2], marker="p", c="r", linewidths=2,  label='$kregions$')
    ax.scatter(PF_knees[:, 0], PF_knees[:, 1], PF_knees[:, 2], marker="x", c="r", linewidths=2,  label='$knees$')
    ax.legend(fontsize=24, loc=0)
    ax.tick_params(labelsize=24)
    ax.set_xlabel("$f_1$", fontsize=28)
    ax.set_ylabel("$f_2$", fontsize=28)
    ax.set_zlabel("$f_2$", fontsize=28)

    plt.show()
