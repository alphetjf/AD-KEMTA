from PROBLEM import PROBLEM
import numpy as np
from copy import deepcopy


class ZDT1(PROBLEM):
    def Setting(self):
        self.M = 2
        if self.D == 0:
            self.D = 8
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)
        self.encoding = "real"

    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], 2))
        PopObj[:, 0] = deepcopy(PopDec[:, 0])
        g = 1 + 9 * np.mean(PopDec[:, 1:], axis=1)
        h = 1 - (PopObj[:, 0] / g)**0.5
        PopObj[:, 1] = g*h
        self.FE += PopDec.shape[0]
        # print("ZDT")
        # print(PopDec[0, :])
        # print(PopObj[0, :])
        return PopObj

    def GetOptimum(self, *varargin):
        N = varargin[0]
        R = np.zeros((N, 2))
        R[:, 0] = np.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0]**0.5
        return R

    def GetPF(self, N=100):
        R = self.GetOptimum(N)
        return R


class ZDT2(PROBLEM):
    def Setting(self):
        self.M = 2
        if self.D == 0:
            self.D = 8
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)
        self.encoding = "real"

    def CalObj(self, PopDec):
        PopObj = np.zeros((PopDec.shape[0], 2))
        PopObj[:, 0] = deepcopy(PopDec[:, 0])
        g = 1 + 9 * np.mean(PopDec[:, 1:], axis=1)
        h = 1 - (PopObj[:, 0] / g)**2
        PopObj[:, 1] = g*h
        self.FE += PopDec.shape[0]
        return PopObj

    def GetOptimum(self, *varargin):
        N = varargin[0]
        R = np.zeros((N, 2))
        R[:, 0] = np.linspace(0, 1, N)
        R[:, 1] = 1 - R[:, 0]**2
        return R

    def GetPF(self):
        R = self.GetOptimum(100)
        return R
