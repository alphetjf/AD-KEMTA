import sys
import os
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
# from geneticOperation.selfGA import Selection, OperatorGA

class DTLZ2(PROBLEM):
    def Setting(self, M=3):
        self.M = M
        self.D = self.M + 9
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)
        self.encoding = "real"

    def CalObj(self, PopDec):
        g = np.sum((PopDec[:, self.M-1:]-0.5)**2, axis=1)[:, np.newaxis]
        temp = np.append(np.ones((g.shape[0], 1)), np.cos(PopDec[:, 0:self.M-1]*np.pi/2), axis=1)
        temp2 = np.append(np.ones((g.shape[0], 1)), np.sin(PopDec[:, self.M-2::-1]*np.pi/2), axis=1)
        PopObj = np.tile(1+g, (1, self.M)) * np.fliplr(np.cumprod(temp, axis=1)) * temp2
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, *varargin):
        N = varargin[0]
        R = generatorPoints(N, self.M)[0]
        R = R / np.tile(np.sqrt(np.sum(R**2, axis=1)), (self.M, 1)).T
        return R

    def GetPF(self):
        if self.M == 2:
            R = self.GetOptimum(100)
        elif self.M == 3:
            a = np.linspace(0, np.pi/2, num=10)[:, np.newaxis]
            # R = np.asarray([np.dot(np.sin(a), np.cos(a.T)).reshape(-1, 1),
            #                 np.dot(np.sin(a), np.sin(a.T)).reshape(-1, 1),
            #                 np.dot(np.cos(a), np.cos(a.T)).reshape(-1, 1)])
            temp = np.append(np.dot(np.sin(a), np.cos(a.T)).reshape(-1, 1),
                             np.dot(np.sin(a), np.sin(a.T)).reshape(-1, 1), axis=1)
            R = np.append(temp, np.dot(np.cos(a), np.cos(a.T)).reshape(-1, 1), axis=1)
        else:
            R = []
        return R


if __name__ == "__main__":
    dtlz2 = DTLZ2()
    dtlz2.Setting()
    maxFEs = dtlz2._ParameterSet(MaxFEs=300)
    PopDec = np.ones((3, 11))
    Obj = dtlz2.CalObj(PopDec)
