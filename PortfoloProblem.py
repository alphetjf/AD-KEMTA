# -*- encoding: utf-8 -*-
'''
@File    :   PortfoloProblem.py
@Time    :   2024/02/26 21:16:43
@Author  :   jftang
'''
import numpy as np
import pandas as pd
from pymoo.util.remote import Remote
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling
from PROBLEM import PROBLEM
import pymoo.gradient.toolbox as anp
from pymoo.core.repair import NoRepair
import os
from pathlib import Path
import time
import math
import copy
from format_solution import SOLUTIONSET
from utils.UniformPoint import generatorPoints
from geneticOperation.selfGA import OperatorGAhalf
from utils.visualizationInManyObjective import reductionDimensionObjectives
from copy import deepcopy
from operator import attrgetter
import matplotlib.pyplot as plt
from geneticOperation.selfGA import OperatorGA, tournamentSelection
from geneticOperation.selfDE import OperatorDE_PloyMut

file = Remote.get_instance().load("examples", "portfolio_allocation.csv", to=None)
df = pd.read_csv(file, parse_dates=True, index_col="date")
returns = df.pct_change().dropna(how="all")
mu = (1 + returns).prod() ** (252 / returns.count()) - 1
cov = returns.cov() * 252
mu, cov = mu.to_numpy(), cov.to_numpy()
'''--------------------------------------------------------------------------------------------------------------------------'''
# The basic MOEA/D
class moead():
    def __init__(self) -> None:
        self.saveName = 0
        self.pro = None
        self.result = {}
        self.metric = 0
        self.preFE = 0

        self.save_frequency = 0
        self.filepath = None
        self.N0 = 100
        self.drawFig = False
        self.showTime = 1
        self.showFrequency = 0.1

    def _ParameterSet(self, **varargin):
        if len(varargin) > 0:
            Values = []
            for key, value in varargin.items():
                Values.append(value)
                if key == "save_frequency":
                    self.save_frequency = value
                elif key == "filepath":
                    self.filepath = value
                elif key == "N0":
                    self.N0 = value
                elif key == "drawFig":
                    if len(value) == 1:
                        self.drawFig = value[0]
                    elif len(value) == 2:
                        self.drawFig = value[0]
                        self.showTime = value[1]
                    else:
                        self.drawFig = value[0]
                        self.showTime = value[1]
                        self.showFrequency = value[2]
            return Values

    def NotTerminated(self, Pops):
        index = len(self.result) + 1
        self.result.update({index: [Pops.decs(), Pops.objs(), Pops.cons(), Pops.adds()]})

        nofinish = self.pro.FE < self.pro.maxFE
        if np.mod(self.pro.FE, self.showFrequency) <= 5:
            # print("process: ", self.pro.FE/self.pro.maxFE)
            print('problem evaluation numbers: ', self.pro.FE)
        if self.save_frequency > 0 and nofinish and self.filepath is not None:
            if self.preFE == 0 or (self.pro.FE - self.preFE) == self.save_frequency or \
                   (self.pro.FE - self.preFE) == self.save_frequency*self.pro.N:
                self.saveName = self.pro.FE
                self.metric['runtime'] = time.perf_counter() - self.metric['runtime']
                self.Save(self.filepath + "\\" + str(self.saveName))
                self.preFE = self.pro.FE
        if bool(1-nofinish) and self.filepath is not None:
            self.saveName = self.pro.FE
            self.metric['runtime'] = time.perf_counter() - self.metric['runtime']

            if self.save_frequency > 0:
                self.Save(self.filepath + "\\" + str(self.saveName))
            elif self.save_frequency == -1:
                pass
            else:
                self.Save(self.filepath)
        return nofinish

    def Save(self, save_path):
        my_file = Path(save_path)
        if my_file.exists():
            pass
        else:
            os.makedirs(save_path)
        import json
        jsObj = json.dumps(self.metric)
        fileObject = open(save_path+'\\metric.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()
        N = len(self.result) + 1
        Decs = self.result.get(N-1)[0]
        np.save(save_path+"\\Decs.npy", Decs)
        Objs = self.result.get(N-1)[1]
        np.save(save_path+"\\Objs.npy", Objs)
        Cons = self.result.get(N-1)[2]
        np.save(save_path+"\\Cons.npy", Cons)
        Adds = self.result.get(N-1)[3]
        np.save(save_path+"\\Adds.npy", Adds)

    def Solve(self, Pro):
        # try:
        self.pro = Pro
        self.metric = {'runtime': time.perf_counter()}
        self.pro.FE = 0
        self.main(Pro)
        return self.result, self.metric

    def main(self, Pro):
        # Generate a random population
        NI = self.N0
        PopDec, _ = generatorPoints(NI, Pro.D, method='Latin')
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        Pops = SOLUTIONSET(Pro)
        Pops.building(Decs=PopDec)
        PopObj = Pops.objs()
        Z = np.min(PopObj, axis=0)[np.newaxis,:]

        # Generate a uniform reference vector
        (W, N) = generatorPoints(NI, Pro.M)
        T = math.ceil(N/10)
        B = calculateDistMatrix(W, W)  # B saves the distance between each weight vector
        B = np.argsort(B, axis=1, kind="mergesort")  # Change B to save the nearest index of each weight vector
        B = np.asarray(B[:, :T])
        while self.NotTerminated(Pops):
            # for each solution
            for i in range(NI):
                P = B[i, np.random.permutation(B.shape[1])]
                Offsprings = OperatorGAhalf(PopDec[P[1:3],:], 1, 20, 1, 20, Pro.upper, Pro.lower)
                OffObj = self.pro.CalObj(Offsprings).reshape(1,Pro.M)
                Z = np.min(np.concatenate((Z, OffObj), axis=0), axis=0)[np.newaxis,:]
                # update the neighbors
                g_old = np.max(np.abs(PopObj[P, :]-np.tile(Z, (T, 1)))*W[P, :], axis=1)
                g_new = np.max(np.tile(np.abs(OffObj-Z), (T, 1))*W[P, :], axis=1)
                update = g_old>=g_new
                num = len(np.where(update)[0])
                Pops.updateProperties(P[update], np.tile(Offsprings, (num, 1)), key='dec')
                Pops.updateProperties(P[update], np.tile(OffObj, (num, 1)), key='obj')
            E = getExtremePoints(Pops.objs())
            Zmin = E[0, :][np.newaxis,:]    # the latest ideal point
            Zmax = E[1, :][np.newaxis,:]    # the latest nadir point
            PopObj = Pops.objs()
            PopDec = Pops.decs()
            if self.drawFig is True and np.mod(self.pro.FE, self.showFrequency) <= 1:
                self.visualResult(Pro, PopObj, Zmin, Zmax)

    def visualResult(self, Pro, PopObj, Zmin, Zmax):
        if Pro.M == 2:
            fig = plt.figure()
            plt.scatter(Pro.PF[:, 0], Pro.PF[:, 1], marker='.', c='blue')
            plt.scatter(Pro.Knees[:, 0], Pro.Knees[:, 1], marker='x', c='orange')
            plt.scatter(PopObj[:, 0], PopObj[:, 1], marker='p', c='red')
        elif Pro.M == 3:
            fig = plt.figure(figsize=(14, 10), dpi=50, facecolor='w', edgecolor='k')
            # ax = plt.axes(projection='3d')
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot(Pro.PF[:, 0], Pro.PF[:, 1], Pro.PF[:, 2], marker='.', alpha=0.5, label='$PF$')
            ax.scatter(PopObj[:, 0], PopObj[:, 1], PopObj[:, 2], marker='p', c='r')
            ax.scatter(Pro.Knees[:, 0], Pro.Knees[:, 1], Pro.Knees[:, 2], marker='s', c='black')
            ax.legend(fontsize=24, loc=0)
            ax.tick_params(labelsize=24)
            ax.set_xlabel("$f_1$", fontsize=28)
            ax.set_ylabel("$f_2$", fontsize=28)
            ax.set_zlabel("$f_2$", fontsize=28)
        elif Pro.M > 3:
            fig = plt.figure()
            dpf, ddf = reductionDimensionObjectives(Pro.PF, Zmin, Zmax)
            dpk, ddk = reductionDimensionObjectives(Pro.knees, Zmin, Zmax)
            dpdk, dddk = reductionDimensionObjectives(PopObj, Zmin, Zmax)
            # plt.scatter(dpef, ddef, marker='.', alpha=0.7, c='green', label='$ePF$')
            plt.scatter(dpf, ddf, marker='.', alpha=0.3, c='blue', label='$PF$')
            plt.scatter(dpk, ddk, marker='s', alpha=0.7, c='red', label='$knee$')
            plt.scatter(dpdk, dddk, marker='x', c='cyan', label='$dknee$')
            plt.legend(fontsize=24, loc=0)

        plt.draw()
        plt.pause(max(1, self.showTime))   # Interval in seconds
        if self.showTime > -1:
            plt.close(fig)

def calculateDistMatrix(datas, DATAS):
    dist = np.zeros((datas.shape[0], DATAS.shape[0]))  # the distance matrix
    if datas.shape[1] > 1:
        for i in range(datas.shape[0]):
            Temp = np.sum((DATAS - np.dot(np.ones((DATAS.shape[0], 1)), datas[i, :][np.newaxis, :]))**2, axis=1)
            dist[i, :] = np.sqrt(Temp)
    else:  # 1-D data
        for i in range(datas.shape[0]):
            dist[i, :] = np.abs(datas[i] - DATAS)
    return dist

def getExtremePoints(Objs, transpose=False):
    N, M = np.shape(Objs)
    E = np.zeros((2, M))
    # tmp1 -- ideal point
    # tmp2 -- nadir point
    for m in range(M):
        tmp1 = np.inf
        tmp2 = -np.inf
        for i in range(N):
            if tmp1 > Objs[i, m]:
                tmp1 = Objs[i, m]
            elif tmp2 < Objs[i, m]:
                tmp2 = Objs[i, m]
        E[0, m] = tmp1
        E[1, m] = tmp2
    if transpose:
        extremes = np.zeros((2, M))
        for i in range(M):
            extremes[i, :] = E[0, :]
            extremes[i, i] = E[1, i]
        return extremes
    return E
'''--------------------------------------------------------------------------------------------------------------------------'''
# The basic NSGA2
class nsga2():
    def __init__(self) -> None:
        self.saveName = 0
        self.pro = None
        self.result = {}
        self.metric = 0
        self.preFE = 0

        self.save_frequency = 0
        self.filepath = None
        self.N0 = 100
        self.drawFig = False
        self.showTime = 1
        self.showFrequency = 0.1

    def _ParameterSet(self, **varargin):
        if len(varargin) > 0:
            Values = []
            for key, value in varargin.items():
                Values.append(value)
                if key == "save_frequency":
                    self.save_frequency = value
                elif key == "filepath":
                    self.filepath = value
                elif key == "N0":
                    self.popsize = value
                elif key == "drawFig":
                    if len(value) == 1:
                        self.drawFig = value[0]
                    elif len(value) == 2:
                        self.drawFig = value[0]
                        self.showTime = value[1]
                    else:
                        self.drawFig = value[0]
                        self.showTime = value[1]
                        self.showFrequency = value[2]
            return Values

    def NotTerminated(self, Pops):
        index = len(self.result) + 1
        self.result.update({index: [Pops.decs(), Pops.objs(), Pops.cons(), Pops.adds()]})

        nofinish = self.pro.FE < self.pro.maxFE
        if np.mod(self.pro.FE, self.showFrequency) <= 5:
            print('problem evaluation numbers: ', self.pro.FE)
        if self.save_frequency > 0 and nofinish and self.filepath is not None:
            if self.preFE == 0 or (self.pro.FE - self.preFE) == self.save_frequency or \
                   (self.pro.FE - self.preFE) == self.save_frequency*self.pro.N:
                self.saveName = self.pro.FE
                self.metric['runtime'] = time.perf_counter() - self.metric['runtime']
                self.Save(self.filepath + "\\" + str(self.saveName))
                self.preFE = self.pro.FE
        if bool(1-nofinish) and self.filepath is not None:
            self.saveName = self.pro.FE
            self.metric['runtime'] = time.perf_counter() - self.metric['runtime']

            if self.save_frequency > 0:
                self.Save(self.filepath + "\\" + str(self.saveName))
            elif self.save_frequency == -1:
                pass
            else:
                self.Save(self.filepath)
        return nofinish

    def Save(self, save_path):
        my_file = Path(save_path)
        if my_file.exists():
            pass
        else:
            os.makedirs(save_path)
        import json
        jsObj = json.dumps(self.metric)
        fileObject = open(save_path+'\\metric.json', 'w')
        fileObject.write(jsObj)
        fileObject.close()
        N = len(self.result) + 1
        Decs = self.result.get(N-1)[0]
        np.save(save_path+"\\Decs.npy", Decs)
        Objs = self.result.get(N-1)[1]
        np.save(save_path+"\\Objs.npy", Objs)
        Cons = self.result.get(N-1)[2]
        np.save(save_path+"\\Cons.npy", Cons)
        Adds = self.result.get(N-1)[3]
        np.save(save_path+"\\Adds.npy", Adds)

    def Solve(self, Pro):
        # try:
        self.pro = Pro
        self.metric = {'runtime': time.perf_counter()}
        self.pro.FE = 0
        self.main(Pro)
        return self.result, self.metric

    def main(self, Pro):
        NI = self.popsize
        PopDec, _ = generatorPoints(NI, Pro.D, method='Latin')
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        Pops = SOLUTIONSET(Pro)
        Pops.building(Decs=PopDec)

        while self.NotTerminated(Pops):
            PopObj = Pops.objs()
            PopDec = Pops.decs()
            PopCon = Pops.cons()
            CV = violationDegree(PopCon)
            FrontNo, _ = FastAlphaNDSort(PopObj, CV)
            CrowdDis = Crowding(PopObj, FrontNo).flatten()
            winIndex = tournamentSelection(2, NI, FrontNo, -CrowdDis)
            winners = PopDec[winIndex, :]
            Offsprings = OperatorGA(winners, 1, 20, 1, 20, Pro.upper, Pro.lower)
            OffObj = self.pro.CalObj(Offsprings)
            OffCon = self.pro.CalCon(Offsprings)
            OffCV = violationDegree(OffCon)

            Pool_dec = np.append(PopDec, Offsprings, axis=0)
            Pool_obj = np.append(PopObj, OffObj, axis=0)
            Pool_con = np.append(PopCon, OffCon, axis=0)
            Pool_cv = np.append(CV, OffCV)

            FrontNo, MaxFNo = FastAlphaNDSort(Pool_obj, Pool_cv)
            CrowdDis = Crowding(Pool_obj, FrontNo).flatten()

            Next = np.lexsort((-CrowdDis, FrontNo))
            Next1 = Next[:NI]
            Pops.updateProperties(np.arange(NI), Pool_dec[Next1, :], key='dec')
            Pops.updateProperties(np.arange(NI), Pool_obj[Next1, :], key='obj')
            Pops.updateProperties(np.arange(NI), Pool_con[Next1, :], key='con')

            if self.drawFig is True and np.mod(self.pro.FE, self.showFrequency) <= 1:
                fig = plt.figure()
                # non_index = np.where(FrontNo == 1)[0]
                # plt.scatter(Pool_obj[non_index, 0], Pool_obj[non_index, 1], marker='p', c='red')
                plt.scatter(Pool_obj[Next1, 0], Pool_obj[Next1, 1], marker='p', c='red')
                plt.draw()
                plt.pause(self.showTime)
                plt.close(fig)

def Crowding(PopObj, FrontNo):
    (N, M) = np.shape(PopObj)
    # translate
    PopObj = PopObj - np.tile(np.min(PopObj, axis=0), (N, 1))
    CrowdDis = np.zeros(N)
    temp = np.unique(FrontNo)
    Fronts = np.setdiff1d(temp, np.infty)
    for f in range(len(Fronts)):
        Front = np.where(FrontNo == Fronts[f])[0]
        fmax = np.max(PopObj[Front, :], axis=0)
        fmin = np.min(PopObj[Front, :], axis=0)
        for i in range(M):
            Rank = np.argsort(PopObj[Front, i], kind='quicksort')   # Return the sequence number of the array sorted from small to large
            CrowdDis[Front[Rank[0]]] = 1e6
            CrowdDis[Front[Rank[-1]]] = 1e6
            for j in range(1, len(Front)-1):
                CrowdDis[Front[Rank[j]]] = CrowdDis[Front[Rank[j]]] + (PopObj[Front[Rank[j+1]], i] - PopObj[Front[Rank[j-1]], i]) / (fmax[i] - fmin[i]+1e-6)
    return CrowdDis

def violationDegree(PopCon):
    N, C = np.shape(PopCon)
    CV = np.zeros(N)
    for i in range(N):
        sum = 0
        for j in range(C):
            sum += -min(PopCon[i, j], 0)
        CV[i] = sum
    return CV

def AssociationWeights(PopObj0, W):
    N, M = np.shape(PopObj0)
    PopObjs = PopObj0 - np.tile(np.min(PopObj0, axis=0), (N, 1))
    Nid = np.zeros(M)
    W_size = np.shape(W)[0]

    Ri = np.zeros(N, dtype=int)
    Rc = np.zeros(W_size, dtype=int)

    for i in range(N):
        dis = np.zeros(W_size)
        for j in range(W_size):
            d, sums = 0, np.linalg.norm(W[j, :], ord=2)
            for k in range(M):
                d += np.abs((PopObjs[i, k]-Nid[k])*W[j, k]/sums)
            d2 = 0
            for k in range(M):
                d2 += (PopObjs[i, k] - (Nid[k]+d*W[j, k]))**2
            dis[j] = np.sqrt(d2)
            # sums = np.linalg.norm(W[j, :], ord=2)
            # d = np.abs(np.dot(Nid-PopObjs[i, :], W[j, :])/sums)
            # d2 = norm(PopObjs[i, :] - (Nid - d*W[j, :]))
            # dis[j] = d2
        index = np.argmin(dis)
        Ri[i] = index
        Rc[index] += 1
    return Ri, Rc

def FastAlphaNDSort(*varargin):
    nargin = len(varargin)
    # if nargin == 1:
    PopObj = varargin[0]
    alpha = None
    (N, M) = np.shape(PopObj)
    CV = np.zeros(N, dtype=int)
    if nargin == 2:
        CV = varargin[1]
    elif nargin == 3:
        CV = varargin[1]
        rvs = varargin[2]
        alpha = 0.75
    elif nargin == 4:
        rvs = varargin[2]
        alpha = varargin[3]
    FronNo = np.ones(N)*np.inf
    MaxFNo = 1
    nSort = N
    # translate
    PopObj = PopObj - np.tile(np.min(PopObj, axis=0), (N, 1))

    class AssistStruct:
        def print_self(self):
            print(self)

    class Individual(AssistStruct):
        def __init__(self, nums, liste):
            self.dominateMe = nums  # (int)
            self.iDominate = liste  # list

        def add_dominateMe(self):
            self.dominateMe = self.dominateMe + 1

        def del_dominateMe(self):
            self.dominateMe = self.dominateMe - 1

        def add_iDominate(self, serial):
            self.iDominate.append(serial)

        def del_iDominate(self, serial):
            self.iDominate.remove(serial)

        def check_if_zero(self):
            if self.dominateMe == 0:
                return True
            else:
                return False

    class Front(AssistStruct):
        def __init__(self, liste):
            self.f = liste  # list

        def add_f(self, serial):
            self.f.append(serial)

        def del_f(self, serial):
            self.f.remove(serial)

    solutions = np.array([Individual(0, []) for i in range(nSort)])
    Flist = [Front([])]
    if alpha is not None:
        Ri, _ = AssociationWeights(PopObj, rvs)
    for i in range(nSort):
        for j in range(nSort):
            if alpha is not None:
                iDominatej = pareto_dominance_operator(PopObj[i, :], PopObj[j, :], CV[i],  CV[j], alpha, Ri[i], Ri[j])
            else:
                iDominatej = pareto_dominance_operator(PopObj[i, :], PopObj[j, :], CV[i],  CV[j])
            if iDominatej == 1:
                solutions[i].add_iDominate(j)
            elif iDominatej == -1:
                solutions[i].add_dominateMe()
        if solutions[i].dominateMe == 0:
            FronNo[i] = 1
            Flist[0].add_f(i)
    front = 1
    while Flist[front-1].f:
        Q = []
        for i in Flist[front-1].f:
            if solutions[i].iDominate:
                for j in solutions[i].iDominate:
                    solutions[j].del_dominateMe()
                    if solutions[j].check_if_zero():
                        FronNo[j] = front+1
                        Q.append(j)
        front += 1
        Flist.extend([Front(Q)])
    MaxFNo = front-1  # Flist[MaxFNo-1].f非空，Flist[MaxFNo].f为空，一共MaxFNo个前沿（python下标从0开始）
    return (FronNo, MaxFNo)

def pareto_dominance_operator(si, sj, iCV, jCV, alpha=None, iIndex=None, jIndex=None):
    if iCV > 0 and jCV > 0:
        if iCV < jCV:
            iDominatej = 1      # i dominate j
        elif iCV > jCV:
            iDominatej = -1     # j dominate i
        else:
            iDominatej = 0      # non dominate each other
    elif iCV == 0 and jCV > 0:
        iDominatej = 1          # i dominate j
    elif iCV > 0 and jCV == 0:
        iDominatej = -1         # j dominate i
    elif alpha is None:
        M = np.size(si)
        xy = np.zeros(M)
        for p in range(M):
            xy[p] = si[p] - sj[p]
        dominate1 = 0
        dominate2 = 0
        for m in range(M):
            if xy[m] < 0:
                dominate1 += 1
            elif xy[m] > 0:
                dominate2 += 1
            else:
                pass
        if (dominate2 == 0) and (dominate1 > 0):  # i dominate j
            iDominatej = 1
        elif (dominate1 == 0) and (dominate2 > 0):  # j dominate i
            iDominatej = -1
        else:
            iDominatej = 0      # non dominate each other
    else:
        if iIndex != jIndex:
            iDominatej = 0
        else:
            M = np.size(si)
            xy = np.zeros(M)
            for p in range(M):
                xy[p] = si[p] - sj[p]
                for q in range(M):
                    if p != q:
                        xy[p] = xy[p] + alpha * (si[q] - sj[q])
            dominate1 = 0
            dominate2 = 0
            for m in range(M):
                if xy[m] < 0:
                    dominate1 += 1
                elif xy[m] > 0:
                    dominate2 += 1
                else:
                    pass
            if (dominate2 == 0) and (dominate1 > 0):  # i dominate j
                iDominatej = 1
            elif (dominate1 == 0) and (dominate2 > 0):  # j dominate i
                iDominatej = -1
            else:
                iDominatej = 0
    return iDominatej

'''--------------------------------------------------------------------------------------------------------------------------'''
# The portfolio allocation problem
class realProblem(PROBLEM):
    def __init__(self, n_var=-1, n_obj=1, n_ieq_constr=0, n_eq_constr=0, xl=None, xu=None,
                 vars=None, requires_kwargs=False, replace_nan_values_by=None, strict=True, **kwargs):
        """
        Parameters
        ----------
            n_var : int
                Number of Variables
            n_obj : int
                Number of Objectives
            n_ieq_constr : int
                Number of Inequality Constraints
            n_eq_constr : int
                Number of Equality Constraints
            xl : np.array, float, int
                Lower bounds for the variables. if integer all lower bounds are equal.
            xu : np.array, float, int
                Upper bounds for the variable. if integer all upper bounds are equal.
        """
        super().__init__()
        self.n_var = n_var              # number of variable
        self.n_obj = n_obj              # number of objectives
        # number of inequality constraints
        self.n_ieq_constr = n_ieq_constr if "n_constr" not in kwargs else max(n_ieq_constr, kwargs["n_constr"])
        self.n_eq_constr = n_eq_constr  # number of equality constraints
        # the lower bounds, make sure it is a numpy array with the length of n_var
        self.xl, self.xu = xl, xu
        # if the variables are provided in their explicit form
        if vars is not None:
            self.vars = vars
            self.n_var = len(vars)

            if self.xl is None:
                self.xl = {name: var.lb if hasattr(var, "lb") else None for name, var in vars.items()}
            if self.xu is None:
                self.xu = {name: var.ub if hasattr(var, "ub") else None for name, var in vars.items()}
        # whether evaluation requires kwargs (passing them can cause overhead in parallelization)
        self.requires_kwargs = requires_kwargs
        # whether the shapes are checked strictly
        self.strict = strict
        # if it is a problem with an actual number of variables - make sure xl and xu are numpy arrays
        if n_var > 0:
            if self.xl is not None:
                if not isinstance(self.xl, np.ndarray):
                    self.xl = np.ones(n_var) * xl
                self.xl = self.xl.astype(float)

            if self.xu is not None:
                if not isinstance(self.xu, np.ndarray):
                    self.xu = np.ones(n_var) * xu
                self.xu = self.xu.astype(float)
        # this defines if NaN values should be replaced or not
        self.replace_nan_values_by = replace_nan_values_by

    def has_bounds(self):
        return self.xl is not None and self.xu is not None

    def bounds(self):
        return self.xl, self.xu

class PortfolioProblem_(realProblem):
    def __init__(self, risk_free_rate=0.02, repair=None, **kwargs):
        file = Remote.get_instance().load("examples", "portfolio_allocation.csv", to=None)
        df = pd.read_csv(file, parse_dates=True, index_col="date")
        returns = df.pct_change().dropna(how="all")
        mu = (1 + returns).prod() ** (252 / returns.count()) - 1
        cov = returns.cov() * 252
        mu, cov = mu.to_numpy(), cov.to_numpy()

        super().__init__(n_var=len(df.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate
        self.df_data = df
        self.out= {}
        self.repair = repair if repair is not None else NoRepair()

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = [exp_risk, -exp_return]
        out["sharpe"] = sharpe
        return out
    
    def Setting(self):
        self.M = 2
        self.D = len(self.xl)
        self.lower = self.xl
        self.upper = self.xu
        self.encoding = "real"

    def elvaluate(self, PopDec):
        elems = [self._evaluate(x, out={}) for x in PopDec]
        # for each evaluation call
        out = {}
        for elem in elems:
            # for each key stored for this evaluation
            for k, v in elem.items():
                # if the element does not exist in out yet -> create it
                if out.get(k, None) is None:
                    out[k] = []
                out[k].append(v)
        # convert to arrays (the none check is important because otherwise an empty array is initialized)
        for k in out:
            if out[k] is not None:
                out[k] = anp.array(out[k])
        self.out = out
    
    def CalDec(self, PopDec):
        self.out = {}
        self.FE += PopDec.shape[0]
        # return super().CalDec(PopDec)
        return self.repair.do(PopDec)
    def CalObj(self, PopDec):
        if len(self.out)==0:
            self.elvaluate(PopDec)
        return self.out['F']
    def CalAdd(self, PopDec):
        if len(self.out)==0:
            self.elvaluate(PopDec)
        return self.out['sharpe']

class PortfolioSampling_():
    def __init__(self, mu=None, cov=None) -> None:
        self.mu = mu
        self.cov = cov
    
    def do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))
        if problem.has_bounds():
            xl, xu = problem.bounds()
            assert np.all(xu >= xl)
            X = xl + (xu - xl) * X

        n = len(self.mu)
        n_biased = min(n, n_samples // 2)
        order_by_ret = (-self.mu).argsort()
        order_by_cov = (self.cov.diagonal()).argsort()
        order = np.stack([order_by_ret, order_by_cov]).min(axis=0)
        X[:n_biased] = np.eye(n)[order][:n_biased]
        return X

class PortfolioRepair_():
    def __init__(self,name=None,vtype=None) -> None:
        super().__init__()
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.vtype = vtype

    def do(self, X, **kwargs):
        if self.vtype is not None:
            X = X.astype(self.vtype)
        Xp = self._do(X, **kwargs)
        return Xp

    def _do(self, X):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)

class NoRepair_(PortfolioRepair_):
    def _do(self, X, **kwargs):
        return X

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling

class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, cov, risk_free_rate=0.02, **kwargs):
        super().__init__(n_var=len(df.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = [exp_risk, -exp_return]
        out["sharpe"] = sharpe

class PortfolioSampling(FloatRandomSampling):
    def __init__(self, mu, cov) -> None:
        super().__init__()
        self.mu = mu
        self.cov = cov

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)

        n = len(mu)
        n_biased = min(n, n_samples // 2)

        order_by_ret = (-self.mu).argsort()
        order_by_cov = (self.cov.diagonal()).argsort()
        order = np.stack([order_by_ret, order_by_cov]).min(axis=0)

        X[:n_biased] = np.eye(n)[order][:n_biased]

        return X

class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)

'''--------------------------------------------------------------------------------------------------------------------------'''
# The solver of MOEA/D for the portfolio allocation problem
class solver(moead):
    def main(self, Pro):
        NI = self.N0
        sampling = PortfolioSampling(Pro.mu, Pro.cov)
        PopDec = sampling.do(pro, NI)
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        Pops = SOLUTIONSET(Pro)
        # PopDec = Pro.repair.do(PopDec)
        Pops.building(Decs=PopDec)
        Sharpe = Pops.problem.CalAdd(PopDec)
        Pops.updateProperties(np.arange(NI), Sharpe[:,np.newaxis], key='add')
        PopObj = Pops.objs()
        Z = np.min(PopObj, axis=0)[np.newaxis,:]

        (W, N) = generatorPoints(NI, Pro.M, method='Latin')
        T = math.ceil(N/10)
        B = calculateDistMatrix(W, W)
        B = np.argsort(B, axis=1, kind="mergesort")
        B = np.asarray(B[:, :T])
        while self.NotTerminated(Pops):
            # for each solution
            for i in range(NI):
                P = B[i, np.random.permutation(B.shape[1])]
                tmpDec = Pro.repair.do(PopDec[P[1:3],:])
                Offsprings = OperatorGAhalf(tmpDec, 1, 20, 1, 20, Pro.upper, Pro.lower)
                # Offsprings = Pro.repair.do(Offsprings)
                Offsprings = self.pro.CalDec(Offsprings).reshape(-1,Pro.D)
                OffObj = self.pro.CalObj(Offsprings).reshape(1, Pro.M)
                OffAdd = self.pro.CalAdd(Offsprings).reshape(1, -1)
                Z = np.min(np.concatenate((Z, OffObj), axis=0), axis=0)[np.newaxis,:]
                # update the neighbors
                g_old = np.max(np.abs(PopObj[P, :]-np.tile(Z, (T, 1)))*W[P, :], axis=1)
                g_new = np.max(np.tile(np.abs(OffObj-Z), (T, 1))*W[P, :], axis=1)
                update = g_old>=g_new
                num = len(np.where(update)[0])
                Pops.updateProperties(P[update], np.tile(Offsprings, (num, 1)), key='dec')
                Pops.updateProperties(P[update], np.tile(OffObj, (num, 1)), key='obj')
                Pops.updateProperties(P[update], np.tile(OffAdd, (num, 1)), key='add')
            E = getExtremePoints(Pops.objs())
            Zmin = E[0, :][np.newaxis,:]    # the latest ideal point
            Zmax = E[1, :][np.newaxis,:]    # the latest nadir point
            PopObj = Pops.objs()
            PopDec = Pops.decs()
            if self.drawFig is True and np.mod(self.pro.FE, self.showFrequency) <= 1:
                self.visualResult(Pro, PopObj, Zmin, Zmax)
'''--------------------------------------------------------------------------------------------------------------------------'''
# The solver of NSGA2 for the portfolio allocation problem
class solver2(nsga2):
    def main(self, Pro):
        NI = self.N0
        PopDec, _ = generatorPoints(NI, Pro.D, method='Latin')
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        NI = self.N0
        sampling = PortfolioSampling(Pro.mu, Pro.cov)
        PopDec = sampling.do(pro, NI)
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        Pops = SOLUTIONSET(Pro)
        PopDec = Pro.repair.do(PopDec)
        Pops.building(Decs=PopDec)
        Sharpe = Pops.problem.CalAdd(PopDec)
        Pops.updateProperties(np.arange(NI), Sharpe[:,np.newaxis], key='add')

        while self.NotTerminated(Pops):
            PopObj = Pops.objs()
            PopDec = Pops.decs()
            PopCon = Pops.cons()
            PopAdd = Pops.adds()
            CV = violationDegree(PopCon)
            FrontNo, _ = FastAlphaNDSort(PopObj, CV)
            CrowdDis = Crowding(PopObj, FrontNo).flatten()
            winIndex = tournamentSelection(2, NI, NI, FrontNo, -CrowdDis)
            winners = PopDec[winIndex, :]
            Offsprings = OperatorGA(winners, 1, 20, 1, 20, Pro.upper, Pro.lower)
            Offsprings = Pro.repair.do(Offsprings)
            Offsprings = self.pro.CalDec(Offsprings).reshape(-1,Pro.D)
            OffObj = self.pro.CalObj(Offsprings)
            OffCon = self.pro.CalCon(Offsprings)
            OffCV = violationDegree(OffCon)
            OffAdd = self.pro.CalAdd(Offsprings)

            Pool_dec = np.append(PopDec, Offsprings, axis=0)
            Pool_obj = np.append(PopObj, OffObj, axis=0)
            Pool_con = np.append(PopCon, OffCon, axis=0)
            Pool_add = np.append(PopAdd, OffAdd[:, np.newaxis], axis=0)
            Pool_cv = np.append(CV, OffCV)

            FrontNo, MaxFNo = FastAlphaNDSort(Pool_obj, Pool_cv)
            CrowdDis = Crowding(Pool_obj, FrontNo).flatten()

            Next = np.lexsort((-CrowdDis, FrontNo))
            Next1 = Next[:NI]
            Pops.updateProperties(np.arange(NI), Pool_dec[Next1, :], key='dec')
            Pops.updateProperties(np.arange(NI), Pool_obj[Next1, :], key='obj')
            Pops.updateProperties(np.arange(NI), Pool_con[Next1, :], key='con')
            Pops.updateProperties(np.arange(NI), Pool_add[Next1, :], key='add')
            E = getExtremePoints(Pops.objs())
            Zmin = E[0, :][np.newaxis,:]    # the latest ideal point
            Zmax = E[1, :][np.newaxis,:]    # the latest nadir point
            if self.drawFig is True and np.mod(self.pro.FE, self.showFrequency) <= 1:
                self.visualResult(Pro, PopObj, Zmin, Zmax)
'''--------------------------------------------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------------------------------------------'''
# The solver of AD-KEMTA for the portfolio allocation problem
from algorithm.AD_KEMTA import alg, get_reference_directions, ensure_design_space, getProperties, KPLS, evolutionViaSurrogateModels, ENS_SS_NDSort
from algorithm.AD_KEMTA import AssociationWeights_acuteAngle, preferredPBI, sample2Evaluation_convergenceInfilling, UpdataArchive
from algorithm.AD_KEMTA import estimation_UnitHyperplane, KneePointIdentification_LocalScalarFunc, UpdateParameters
class solver3(alg):
    def __init__(self) -> None:
        super().__init__()
        self.wmax = 10            # A prefixed number of generations using surrogate models
    def main(self, Pro):
        print('The maximum number of the budget is {}'.format(Pro.maxFE))
        NI = self.N0
        (PopDec, NI) = generatorPoints(NI, Pro.D, method='Latin')
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        PopDec = repairDecisionVariables(PopDec)
        Pops = SOLUTIONSET(Pro)
        Pops.building(Decs=PopDec, AddProper=np.zeros((NI, 1)))
        Sharpe = Pops.problem.CalAdd(PopDec)
        Pops.updateProperties(np.arange(NI), Sharpe[:,np.newaxis], key='add')

        # W, Nr = ReferenceVectorGenerator(self.H1, self.H2, Pro.M)
        W = get_reference_directions("energy", n_dim=Pro.M, n_points=self.H2)
        Nr = len(W)
        print('\tNumber of reference vectors:{}\t'.format(Nr))

        # train_index = np.arange(self.N0)
        A = copy.copy(Pops)

        wmax = self.wmax            # A prefixed number of generations using surrogate models
        theta = self.theta          # A threshold determining if diversity should be prioritized
        THETA = 5*np.ones((Pro.M, Pro.D))
        thetas = np.ones(Pro.M)*1e-2    # params for kriging
        design_space = np.concatenate((Pro.lower[:,np.newaxis], Pro.upper[:,np.newaxis]), axis=1)
        design_space = ensure_design_space(xlimits=design_space)
        d = 0.5

        preV = get_reference_directions("energy", n_dim=Pro.M, n_points=self.H2)
        nk = len(preV)
        print('\tNumber of preferred vectors:{}\t'.format(nk))
        preV = preV*d+(1-d)/Pro.M
        pAlpha = [1/2, 2/3, 1, 2, 3, 4, 5, 10, 15, 1000]
        palpha = [1]*nk
        thetas = np.ones(Pro.M)*5e-1
        # 开始优化
        while self.NotTerminated(Pops):
            Obj = Pops.objs()
            Dec = Pops.decs()
            trDec = getProperties(A.solutions, 'dec')
            trObj = getProperties(A.solutions, 'obj')
            trSharpe = getProperties(A.solutions, 'add')
            trCon = getProperties(A.solutions, 'add')
            # if np.any(trObj<0):
            #     trObj_ = (trObj - np.tile(np.min(trObj, axis=0), (len(trObj), 1))) \
            #             / np.tile(np.max(trObj, axis=0)-np.min(trObj, axis=0),(len(trObj), 1))
            # train surrogate models
            Models = []
            for i in range(Pro.M):
                model_ = KPLS(theta0=[thetas[i]], design_space=design_space,
                              print_training=False, print_prediction=False, print_problem=True, print_solver=False, 
                              poly='constant', corr='pow_exp')   
                # corr=[‘pow_exp’, ‘abs_exp’, ‘squar_exp’], poly=[‘constant’, ‘linear’, ‘quadratic’]
                model_.set_training_values(trDec, trObj[:, i].reshape(-1, 1))
                model_.train()
                Models.append(model_)
            # the search using surrogate models
            w = 1

            preV_ = preV.copy()
            palpha_ = palpha.copy()
            PopDec, PopObj, APD, MSE = evolutionViaSurrogateModels(Dec.copy(), Obj.copy(),
                                                                   preV_, palpha_, theta, Pro, w, wmax, Models)
            '''Infill-sampling after multitasking evolution'''
            # infilling of the internal convergence search
            ia1 = np.unique(PopDec, axis=0, return_index=True)[1]
            PopDec, PopObj = PopDec[ia1, :], PopObj[ia1, :]
            APD, MSE = APD[ia1, :], MSE[ia1, :]
            mu = min(int(Nr*1.), len(ia1))     # the number of solutions to be sampled, >=Nr
            # the previous number of nondominated solutions
            AllObj = np.unique(getProperties(A.solutions, 'obj'), axis=0)
            FrontNo, _ = ENS_SS_NDSort(AllObj, nSort=1)
            nonIndex0 = np.where(FrontNo == 1)[0]
            Transmission, _ = AssociationWeights_acuteAngle(PopObj, preV_)
            APD = preferredPBI(PopObj, preV_, Transmission)
            # APD = preferredAPD(PopObj, translateVector(preV_), 1.001**theta, palpha_, Transmission)
            re_index = sample2Evaluation_convergenceInfilling(PopObj, PopDec, preV_, APD, MSE, mu, W, self.kappa)
            mu = len(re_index)
            print('\tNumber of re-evaluation:{}\t'.format(mu))
            # re-evaluation
            PnewDec = PopDec[re_index, :]
            op = hasattr(self.pro,'CalDec')
            if op:
                _ = self.pro.CalDec(PnewDec)
            PnewObj = self.pro.CalObj(PnewDec)
            PnewCon = self.pro.CalCon(PnewDec)
            PnewSharpe = self.pro.CalAdd(PnewDec)[:,np.newaxis]
            '''model management'''
            Pops = UpdataArchive(Pops, {'decs':PnewDec, 'objs':PnewObj}, W, mu, NI)     # Update the initial population Pop
            '''PF estimation, updating preV, palpha'''
             # Update Training Profile A
            A.group(np.vstack((trDec, PnewDec)), np.vstack((trObj, PnewObj)),
                    np.vstack((trCon, PnewCon)), np.vstack((trSharpe, PnewSharpe)))          
            AllObj = np.unique(getProperties(A.solutions, 'obj'), axis=0)
            FrontNo, _ = ENS_SS_NDSort(AllObj, nSort=1)
            nonIndex = np.where(FrontNo == 1)[0]
            if len(nonIndex) < 2:
                continue
            print('-----Update reference vectors and alpha values-------')
            ePF, _ = estimation_UnitHyperplane(AllObj[nonIndex, :].reshape(-1, Pro.M), model='kriging', nums=20)
            Kappa = self.kappa
            kneeIndex, _ = KneePointIdentification_LocalScalarFunc(ePF, W, Kappa, Zmin=np.min(ePF, axis=0)[np.newaxis, :])
            preV, palpha = UpdateParameters(ePF[kneeIndex,:], ePF, pAlpha, W, palpha, Nr)
            nk = np.shape(preV)[0]

def repairReferenceVector(V):
    # repairs the invalid
    contain_nan = (True in np.isnan(V))
    contain_inf = (True in np.isinf(V))
    if contain_nan:
        V[np.isnan(V)] = 1e-8
        return V / V.sum(axis=1, keepdim=True)
    if contain_inf:
        V[np.isinf(V)] = 1e8
        return V / V.sum(axis=1, keepdims=True)
    else:
        return V

def evolutionViaSurrogateModels(Dec, Obj, preV_, palpha_, theta, Pro, w, wmax, Models):
    ia1 = np.unique(np.round(Dec*1e6)/1e6, axis=0, return_index=True)[1]
    ia2 = np.unique(np.round(Dec*1e6)/1e6, axis=0, return_index=True)[1]
    ia = np.intersect1d(ia1, ia2)
    PopDec = Dec[ia, :].reshape(np.size(ia), -1)
    PopObj = Obj[ia, :].reshape(np.size(ia), -1)
    MSE = np.zeros((np.shape(PopDec)[0], Pro.M))
    # When aggregating inflection point vectors, 
    #   pay attention to the translation before aggregating along the vector direction
    APD = preferredAPD(PopObj, translateVector(preV_), 1.01**theta, palpha_)
    # APD = preferredPBI(PopObj, preV_)
    FactorialCost, FactorialRank, ScalarFitness, SkillFactor = multifactorialEvaluation(APD)
    while w <= wmax:
        OffDec, Transmission = AssortativeMating_SBXPloyMut(PopDec, SkillFactor, Pro.lower, Pro.upper, 1, 20, 1, 20, 0.9)
        # OffDec, Transmission = AssortativeMating_DEPloyMut(PopDec, SkillFactor, Pro.lower, Pro.upper, 1, 0.5, 1, 20, 0.5)
        if np.isnan(OffDec).any():
            print('there exist nan in OffDec!!')
        # Evaluate the individuals in offspring-pop
        N = np.shape(OffDec)[0]
        OffObj = np.zeros((N, Pro.M))
        OffMSE = np.zeros((N, Pro.M))
        for i in range(N):
            for j in range(Pro.M):
                OffObj[i, j] = Models[j].predict_values(OffDec[i, :].reshape(1, -1))
        #         OffMSE[i, j] = np.sqrt(Models[j].predict_variances(OffDec[i, :].reshape(1, -1)))
        # OffObj = self.pro.CalObj(OffDec)
        OffAPD = preferredAPD(OffObj, translateVector(preV_), (w/wmax)**theta, palpha_, Transmission)
        # OffAPD = preferredPBI(OffObj, preV_, Transmission)

        # Concatenate offspring-pop and current-pop
        APD = np.append(APD, OffAPD, axis=0)
        PopDec = np.append(PopDec, OffDec, axis=0)
        PopObj = np.append(PopObj, OffObj, axis=0)
        MSE = np.append(MSE, OffMSE, axis=0)

        # Update the scalar fitness and skill factor of every individual in intermediate-pop.
        FactorialCost, FactorialRank, ScalarFitness, SkillFactor = multifactorialEvaluation(APD)
        # Select the fittest individuals from intermediate-pop to form the next current-pop
        rank = np.argsort(-ScalarFitness)
        index = rank[:N]
        PopDec, PopObj = PopDec[index, :], PopObj[index, :]
        APD, MSE = APD[index, :], MSE[index, :]
        SkillFactor = SkillFactor[index]
        w += 1
    if np.isnan(PopDec).any():
        print('there exist nan in PopDec!!')
    return PopDec, PopObj, APD, MSE

def translateVector(W):
    M = np.shape(W)[1]
    W = np.clip(W, 1e-6, 1e-6+1)
    return 1/W / np.tile((np.sum(1/W, axis=1))[:, np.newaxis], (M,))

def multifactorialEvaluation(APD):
    N, nv = np.shape(APD)
    # MFO properties of each of solutions
    FactorialCost = APD
    FactorialRank = np.zeros((N, nv))
    for j in range(nv): # for each mission
        tmpFC, ic = np.unique(FactorialCost[:, j], return_inverse=True)
        tmpFR = np.arange(0, len(ic)) + 1
        FactorialRank[:, j] = tmpFR[ic]
    denominator = np.min(FactorialRank, axis=1)
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    ScalarFitness = 1 / denominator
    SkillFactor = np.argmin(FactorialRank, axis=1)
    return FactorialCost, FactorialRank, ScalarFitness, SkillFactor

def AssortativeMating_SBXPloyMut(Pops, skillFactors, low, up, proC, disC, proM, disM, rmp):
    N, D = np.shape(Pops)
    if np.mod(N, 2) == 0:
        parents = np.random.randint(N, size=(2, int(N/2)))
        oddOrEven = 0
    else:
        parents = np.random.randint(N, size=(2, int(N/2)+1))
        oddOrEven = 1
    Offsprings = np.zeros((N, D))
    Transmission = np.zeros(N, dtype=int)
    for i in range(int(N/2)):
        if skillFactors[parents[0, i]] == skillFactors[parents[1, i]] or np.random.rand() < rmp:
            miu = np.random.rand(D)
            beta = np.zeros(D)
            index = miu <= 0.5
            beta[index] = (2*miu[index])**(1/(disC+1))
            beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
            beta = beta*(-1)**np.random.randint(2, size=D)
            beta[np.random.rand(D) > proC] = 1
            if oddOrEven == 1 and i == np.shape(parents)[1]-1:
                if np.random.rand() <= 0.5:
                    Offsprings[i*2, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 + beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                    Transmission[i*2] = skillFactors[parents[0, i]].copy()
                else:
                    Offsprings[i*2, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 - beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                    Transmission[i*2] = skillFactors[parents[1, i]].copy()
            else:
                Offsprings[i*2, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 + beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                Offsprings[i*2+1, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 - beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                Transmission[i*2] = skillFactors[parents[0, i]].copy()
                Transmission[i*2+1] = skillFactors[parents[1, i]].copy()
        else:
            k = np.random.rand(D)
            miu = np.random.rand(D)
            if oddOrEven == 1 and i == np.shape(parents)[1]-1:
                Offsprings[i*2, :] = Pops[parents[0, i], :]
                if np.random.rand() <= 0.5:
                    Temp = (k <= proM/D) & (miu < 0.5)  # 变异的基因
                    Offsprings[i*2, Temp] = Offsprings[i*2, Temp] + (up[Temp] - low[Temp]) * \
                        ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[i*2, Temp]-low[Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1))-1)
                    Transmission[i*2] = skillFactors[parents[0, i]].copy()
                else:
                    Temp = (k <= proM/D) & (miu >= 0.5)
                    Offsprings[i*2, Temp] = Offsprings[i*2, Temp] + (up[Temp] - low[Temp]) * \
                        (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(up[Temp]-Offsprings[i*2, Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1)))
                    Transmission[i*2] = skillFactors[parents[1, i]].copy()
            else:
                Offsprings[i*2, :] = Pops[parents[0, i], :]
                Offsprings[i*2+1, :] = Pops[parents[1, i], :]
                Temp = (k <= proM/D) & (miu < 0.5)  # 变异的基因
                Offsprings[i*2, Temp] = Offsprings[i*2, Temp] + (up[Temp] - low[Temp]) * \
                    ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[i*2, Temp]-low[Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1))-1)
                Temp = (k <= proM/D) & (miu >= 0.5)
                Offsprings[i*2+1, Temp] = Offsprings[i*2+1, Temp] + (up[Temp] - low[Temp]) * \
                    (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(up[Temp]-Offsprings[i*2+1, Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1)))
                Transmission[i*2] = skillFactors[parents[0, i]].copy()
                Transmission[i*2+1] = skillFactors[parents[1, i]].copy()

    # Normalization processing
    Offsprings = repairDecisionVariables(Offsprings)
    return Offsprings, Transmission

def repairDecisionVariables(Dec):
    _, D = np.shape(Dec)
    Dec_ = (Dec - np.tile(np.min(Dec, axis=1)[:, np.newaxis], (D,)))
    Dec_ = np.clip(Dec_, a_min=1e-6, a_max=1e6)
    hatDec = Dec_ / np.tile(np.sum(Dec_, axis=1)[:, np.newaxis], (D,))
    return hatDec

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def preferredAPD(PopObj0, V, theta, palpha, *varargin):
    N, M = np.shape(PopObj0)
    nv = np.shape(V)[0]
    nargin = len(varargin)
    V = repairReferenceVector(V)
    # translate the population
    PopObj = (PopObj0 - np.tile(np.min(PopObj0, axis=0), (N, 1))) \
        / np.clip(np.tile(np.max(PopObj0, axis=0)-np.min(PopObj0, axis=0), (N, 1)), a_min=1e-8, a_max=1e8)
    # Calculate the smallest angle value between each vector and others
    cosine_dist = cosine_similarity(V, V)
    cosine_dist = np.where(np.abs(cosine_dist-1.) <= 1e-4, 0, cosine_dist)
    gamma = np.min(np.arccos(cosine_dist), axis=1)
    # Calculate the angle value of each solution to a reference vector
    angle_dist = np.arccos(cosine_similarity(PopObj, V))
    # associates = np.argmin(angle_dist, axis=1)
    APD = np.ones((N, nv))*np.infty
    V = np.clip(V, 1e-6, 1+1e-6)
    # Calculate the APD value of each solution
    if nargin < 1:
        for j in range(N):
            for i in range(nv):
                wfunc = weightedFunc(PopObj[j, :], V[i, :], palpha[i])
                # APD[j, i] = (1+M*theta*angle_dist[j, i]/gamma[i])*wfunc
                APD[j, i] = wfunc
    else:
        skillFactor = varargin[0]
        for j in range(N):
            wfunc = weightedFunc(PopObj[j, :], V[skillFactor[j], :], palpha[skillFactor[j]])
            # APD[j, skillFactor[j]] = (1+M*theta*angle_dist[j, skillFactor[j]]/gamma[skillFactor[j]])*wfunc
            APD[j, skillFactor[j]] = wfunc
    return APD

def weightedFunc(x, w, p):
    if p < 1000:
        try:
            value = np.dot(x**p, w**p)**(1/p)
        except FloatingPointError:
            p = 2
            value = np.dot(x**p, w**p)**(1/p)
            print('FloatingPointError:',x,p,w)
    else:
        value = np.max(x*w)
    return value

'''--------------------------------------------------------------------------------------------------------------------------'''
if __name__ == "__main__":
    # pro = PortfolioProblem()
    # pro.Setting()
    # maxFEs = pro._ParameterSet(MaxFEs=300)
    # sampling = PortfolioSampling(pro.mu, pro.cov)
    # PopDec = sampling.do(pro, 3)
    # Obj = pro.CalObj(PopDec)
    # print(Obj)

    curPath = os.path.abspath(os.path.dirname(__file__))
    start_run = True
    showResults = True
    runOurAlg = True
    findKnee = False
    ourAlgName = 'AD-KEMTA'
    solver_name = ['NSGA2', 'MOEAD', 'NSGA3'][2]
    if start_run:          # Run real applications
        pro = PortfolioProblem_(repair=PortfolioRepair_())
        pro.Setting()
        maxFEs = pro._ParameterSet(MaxFEs=30000)
        if solver_name != 'NSGA3':
            if solver_name == 'NSGA2':
                dsolver = solver2()
            elif solver_name == 'MOEAD':
                dsolver = solver()
            path = os.path.join(curPath, 'experiment', 'real_problem', 'portfolio_allocation', solver_name)
            my_file = Path(path)
            if bool(1-my_file.exists()):
                os.makedirs(path)
                _ = dsolver._ParameterSet(save_frequency=0, filepath=path, drawFig=(False, 2, 500), N0=100)
                t_start = time.time()
                results, metrics = dsolver.Solve(pro)
                t_stop = time.time()
                print("The algorithm is completed and takes time %0.2f" % (t_stop-t_start))
            else:
                pass
        elif solver_name == 'NSGA3':
            from pymoo.algorithms.moo.nsga3 import NSGA3
            from pymoo.optimize import minimize

            problem = PortfolioProblem(mu, cov)
            from pymoo.util.ref_dirs import get_reference_directions
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

            algorithm = NSGA3(sampling=PortfolioSampling(mu, cov), repair=PortfolioRepair(), ref_dirs=ref_dirs, pop_size=100)
            res = minimize(problem,
                        algorithm,
                        seed=1,
                        verbose=True,
                        termination=('n_gen', 300))
            X, F, sharpe = res.opt.get("X", "F", "sharpe")
    if runOurAlg:
        path = os.path.join(curPath, 'experiment', 'real_problem', 'portfolio_allocation', ourAlgName)
        my_file = Path(path)
        if bool(1-my_file.exists()):
            os.makedirs(path)
            pro = PortfolioProblem_(repair=PortfolioRepair_())
            pro.Setting()
            maxFEs = pro._ParameterSet(MaxFEs=300)
            dsolver = solver3()
            _ = dsolver._ParameterSet(save_frequency=0, filepath=path, drawFig=(False, 2, 500), N0=100)
            t_start = time.time()
            results, metrics = dsolver.Solve(pro)
            t_stop = time.time()
            print("The algorithm is completed and takes time %0.2f" % (t_stop-t_start))
        else:
            pass
    if showResults:
        import matplotlib.pyplot as plt
        if solver_name != 'NSGA3':
            path = os.path.join(curPath, 'experiment', 'real_problem', 'portfolio_allocation', solver_name)
            X = np.load(os.path.join(path, "Decs.npy"))
            F = np.load(os.path.join(path, "Objs.npy"))
            sharpe = np.load(os.path.join(path, "Adds.npy"))
        F = F * [1, -1]
        max_sharpe = sharpe.argmax()

        # f = plt.figure(figsize=(8, 6))
        fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(F[:, 0], F[:, 1], facecolor="none", marker='o', edgecolors="blue", alpha=0.9, s=30, label="Pareto-Optimal Portfolio")
        # plt.scatter(cov.diagonal() ** 0.5, mu, facecolor="none", marker='o', edgecolors="black", s=30, label="Asset")
        plt.scatter(F[max_sharpe, 0], F[max_sharpe, 1], marker="x", s=150, color="green", label="Max Sharpe Portfolio")

        if runOurAlg:
            '''For our proposed algorithm'''
            path_ours = os.path.join(curPath, 'experiment', 'real_problem', 'portfolio_allocation', ourAlgName)
            X_ours = np.load(os.path.join(path_ours, "Decs.npy"))
            F_ours0 = np.load(os.path.join(path_ours, "Objs.npy"))
            F_ours = F_ours0 * [1, -1]
            FrontNo, _ = ENS_SS_NDSort(F_ours0, nSort=1)
            nonIndex = np.where(FrontNo == 1)[0]
            if findKnee:
                from algorithm.AD_KEMTA import KneePointIdentification_LocalScalarFunc
                W = get_reference_directions("energy", n_dim=2, n_points=8)
                Nw = len(W)
                d = 0.5
                W = W*d+(1-d)/2
                approximatePF = F_ours0[nonIndex,:] - np.tile(np.min(F_ours0[nonIndex,:], axis=0), (len(nonIndex), 1))
                kps1, _ = KneePointIdentification_LocalScalarFunc(approximatePF, vectors=W, Kappa=5, Zmin=np.min(approximatePF, axis=0))
                max_sharpe_ours = kps1
            else:
                # sharpe_ours = np.load(os.path.join(path_ours, "Adds.npy"))
                # max_sharpe_ours = sharpe_ours.argmax()
                dis = calculateDistMatrix(F[max_sharpe, :].reshape(1, -1), F_ours[nonIndex, :])
                max_sharpe_ours = np.argmin(dis)
            # max_sharpe_ours = np.argsort(dis)
            # plt.scatter(F_ours[nonIndex , 0], F_ours[nonIndex , 1], facecolor="none", marker='.', edgecolors="m", alpha=0.5, label="Found Solutions")
            plt.scatter(F_ours[nonIndex, 0], F_ours[nonIndex, 1], color='red',
                        facecolor="none", marker='o', edgecolors="red", alpha=1., label="Found Solutions")
            # plt.scatter(cov.diagonal() ** 0.5, mu, facecolor="none", edgecolors="m", s=30, label="Asset")
            plt.scatter(F_ours[nonIndex[max_sharpe_ours], 0], F_ours[nonIndex[max_sharpe_ours], 1], marker="*", s=200, color="red", label="Found Knee Point")

        labelss = plt.legend(fontsize=24, loc=0, ncol=1, frameon=False).get_texts()
        [label.set_fontname('Times New Roman') for label in labelss]
        [label.set_fontsize(20) for label in labelss]
        plt.title("The obtained Pareto front of the portfolio",fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.xlabel('Expected volatility',fontsize=24, fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.ylabel('Expected return',fontsize=24, fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.xticks(color='black', fontsize=18)
        plt.yticks(color='black', fontsize=18)
        plt.xlim(left=0.15, right=0.38)
        plt.ylim(bottom=0.15, top=0.43)
        plt.show()

        # view decision variables
        import operator

        allocation = {name: w for name, w in zip(pro.df_data.columns, X[max_sharpe])}
        allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)

        print("Allocation With Best Sharpe")
        for name, w in allocation:
            print(f"{name:<5} {w}")