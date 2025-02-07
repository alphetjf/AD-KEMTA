import os
import sys
import numpy as np
import argparse
from pathlib import Path
import csv
from tqdm import tqdm
from scipy.stats import kendalltau
import scipy.stats as ss
import matplotlib.pyplot as plt
from utils.UniformPoint import ReferenceVectorGenerator
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.optimize import minimize
import time
import copy
from abc import abstractmethod
from smt.surrogate_models import KRG, KPLS, GEKPLS
from smt.utils.design_space import ensure_design_space
from sklearn.cluster import KMeans
from scipy.linalg import norm
from format_solution import SOLUTIONSET, getProperties
from utils.Nondomination import NDSort, weightedAugmentedTchebycheff
from utils.UniformPoint import generatorPoints, ReferenceVectorGenerator
from tools.multitaskCrossoverMutation import AssortativeMating_SBXPloyMut
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from utils.FrontEstimatie import estimation_UnitHyperplane
from pymoo.util.ref_dirs import get_reference_directions
from ismember import ismember
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

# -------------------------------------------------------------------------------------------------------------------------------------
class alg():
    def __init__(self) -> None:
        self.saveName = 0
        self.pro = None
        self.result = {}
        self.metric = 0
        self.preFE = 0

        self.drawFig = False
        self.showTime = 1
        self.showFrequency = 0.1
        self.save_frequency = 0
        self.filepath = None
        self.N0 = 100
        self.wmax = 10            # A prefixed number of generations using surrogate models
        self.theta = 2            # A threshold determining if diversity should be prioritized
        self.rate = 0.0

    def _ParameterSet(self, **varargin):
        if len(varargin) > 0:
            Values = []
            for key, value in varargin.items():
                Values.append(value)
                if key == "save_frequency":
                    self.save_frequency = value
                elif key == "filepath":
                    self.filepath = value
                elif key == "wmax":
                    self.wmax = value
                elif key == "theta":
                    self.theta = value
                elif key == 'rate':
                    self.rate = value
                elif key == 'N0':
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
            print('problem evaluation numbers:\n', self.pro.FE)
        if self.save_frequency > 0 and nofinish and self.filepath is not None:
            if self.preFE == 0 or (self.pro.FE - self.preFE) >= self.save_frequency or \
                   (self.pro.FE - self.preFE) >= self.save_frequency*self.pro.maxFE:
                self.saveName = self.pro.FE
                self.metric['runtime'] = time.perf_counter() - self.metric['runtime']
                # self.metric['times'] = self.times
                self.Save(os.path.join(self.filepath, str(self.saveName)))
                self.preFE = self.pro.FE
        if bool(1-nofinish) and self.filepath is not None:
            self.saveName = self.pro.FE
            self.metric['runtime'] = time.perf_counter() - self.metric['runtime']
            # self.metric['times'] = self.times

            if self.save_frequency > 0:
                self.Save(os.path.join(self.filepath, str(self.saveName)))
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
        fileObject = open(os.path.join(save_path, 'metric.json'), 'w')
        fileObject.write(jsObj)
        fileObject.close()
        N = len(self.result) + 1
        Decs = self.result.get(N-1)[0]
        np.save(os.path.join(save_path,"Decs.npy"), Decs)
        Objs = self.result.get(N-1)[1]
        np.save(os.path.join(save_path, "Objs.npy"), Objs)
        Cons = self.result.get(N-1)[2]
        np.save(os.path.join(save_path,"Cons.npy"), Cons)
        Adds = self.result.get(N-1)[3]
        np.save(os.path.join(save_path, "Adds.npy"), Adds)

    def Solve(self, Pro):
        # try:
        self.pro = Pro
        self.metric = {'runtime': time.perf_counter()}
        self.pro.FE = 0
        self.setParameters()
        self.main(Pro)
        return self.result, self.metric

    def setParameters(self):
        self.Wn = self.pro.M + 1
        self.H1,self.H2 = 1,8
        self.kappa = 4

    @abstractmethod
    def main(self, Pro):
        print('The maximum number of the budget is {}'.format(Pro.maxFE))
        NI = self.N0
        (PopDec, NI) = generatorPoints(NI, Pro.D, method='Latin')
        PopDec = np.tile((Pro.upper - Pro.lower), (NI, 1))*PopDec + np.tile(Pro.lower, (NI, 1))
        Pops = SOLUTIONSET(Pro)
        Pops.building(Decs=PopDec, AddProper=np.zeros((NI, 1)))

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
        # preV, nk = generatorPoints(Nr, Pro.M)    # the preferred vectors
        # preV, nk = ReferenceVectorGenerator(self.H1, self.H2, Pro.M)
        preV = get_reference_directions("energy", n_dim=Pro.M, n_points=self.H2)
        nk = len(preV)
        print('\tNumber of preferred vectors:{}\t'.format(nk))
        preV = preV*d+(1-d)/Pro.M
        pAlpha = [1/2, 2/3, 1, 2, 3, 4, 5, 10, 15, 1000]
        palpha = [1]*nk
        thetas = np.ones(Pro.M)*5e-1
        # 开始优化
        while self.NotTerminated(A):
            Obj = Pops.objs()
            Dec = Pops.decs()
            trDec = getProperties(A.solutions, 'dec')
            trObj = getProperties(A.solutions, 'obj')
            # train surrogate models
            Models = []
            for i in range(Pro.M):
                model_ = KPLS(theta0=[thetas[i]], design_space=design_space,
                              print_training=False, print_prediction=False, print_problem=True, print_solver=False, 
                              poly='linear', corr='pow_exp')   
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
            # ia2 = np.unique(PopObj, axis=0, return_index=True)[1]
            # ia = np.intersect1d(ia1, ia2)
            PopDec, PopObj = PopDec[ia1, :], PopObj[ia1, :]
            APD, MSE = APD[ia1, :], MSE[ia1, :]
            mu = min(int(Nr*1.), len(ia1))     # the number of solutions to be sampled, >=Nr
            # the previous number of nondominated solutions
            # AllObj = np.unique(Pops.objs(), axis=0)
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
            PnewObj = self.pro.CalObj(PnewDec)
            '''model management'''
            Pops = UpdataArchive(Pops, {'decs':PnewDec, 'objs':PnewObj}, W, mu, NI)     # Update the initial population Pop
            '''PF estimation, updating preV, palpha'''
            A.group(np.vstack((trDec, PnewDec)), np.vstack((trObj, PnewObj)))           # Update Training Profile A
            # AllDec = Pops.decs()
            # _, ia = np.unique(AllDec, axis=0, return_index=True)
            # # AllObj = Pops.objs()
            # AllObj = np.unique(Pops.objs(), axis=0)
            AllObj = np.unique(getProperties(A.solutions, 'obj'), axis=0)
            FrontNo, _ = ENS_SS_NDSort(AllObj, nSort=1)
            nonIndex = np.where(FrontNo == 1)[0]
            # if len(nonIndex) - len(nonIndex0) > 0 and len(nonIndex) > 2:
            print('-----Update reference vectors and alpha values-------')
            ePF, _ = estimation_UnitHyperplane(AllObj[nonIndex, :].reshape(-1, Pro.M), model='kriging', nums=20)
            # Kappa = calcCurvity(ePF)
            Kappa = self.kappa
            kneeIndex, _ = KneePointIdentification_LocalScalarFunc(ePF, W, Kappa, Zmin=np.min(ePF, axis=0)[np.newaxis, :])
            preV, palpha = UpdateParameters(ePF[kneeIndex,:], ePF, pAlpha, W, palpha, Nr)
            # preV = Pro.Knees / np.tile(np.sum(Pro.Knees, axis=1)[:, np.newaxis], (Pro.M, ))
            nk = np.shape(preV)[0]

##### Functional function #####
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
        # os.system("pause")
    return PopDec, PopObj, APD, MSE

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

def sample2Evaluation_convergenceInfilling(PopObj, PopDec, preV, APD, MSE, nums, rvs, Kappa):
    # Take 1 sample for each task
    N, nv = len(APD), len(preV)
    re_index, arangeIndex = np.array([], dtype=int), np.arange(0, N, dtype=int)
    for i in range(nv):
        index = np.argmin(APD[arangeIndex, i])
        re_index = np.append(re_index, arangeIndex[index])
        arangeIndex = np.delete(arangeIndex, index)
        if len(arangeIndex) == 0:
            break
    if len(re_index) < nums:
        # Kappa = calcCurvity(PopObj[arangeIndex, :])
        kneeIndex, _ = KneePointIdentification_LocalScalarFunc(PopObj[arangeIndex, :], rvs, Kappa, Zmin=np.min(PopObj, axis=0)[np.newaxis,:])
        # kneeIndex = np.setdiff1d(kneeIndex, re_index)
        if len(np.unique(re_index)) + len(kneeIndex) <= nums:
            re_index = np.append(re_index, kneeIndex)
        else:
            while len(np.unique(re_index)) < nums and len(kneeIndex) > 0:
                # disMatrix = euclidean_distances(PopObj[kneeIndex, :], PopObj[re_index, :])
                disMatrix = euclidean_distances(PopDec[kneeIndex, :], PopDec[re_index, :])
                dis = np.min(disMatrix, axis=1)
                rank = np.argsort(-dis)
                re_index = np.append(re_index, kneeIndex[rank[:int(nums-len(np.unique(re_index)))]])
    return re_index

def AssociationWeights_acuteAngle(PObjs, W):
    N, _ = np.shape(PObjs)
    PObjs_ = PObjs - np.tile(np.min(PObjs, axis=0), (N, 1))

    wSize = np.shape(W)[0]
    Ri = np.zeros(N, dtype=int)
    Rc = np.zeros(wSize, dtype=int)

    # Associate each solution to a reference vector
    angle_dist = np.arccos(cosine_similarity(PObjs_, W))
    Ri = np.argmin(angle_dist, axis=1)

    for i in range(wSize):
        Rc[i] = len(np.where(Ri == i)[0])

    return Ri, Rc

def UpdateParameters(Kps0, pf0, pAlpha, rvs, palpha, Nr):
    # remove duplication
    pf = np.unique(pf0, axis=0)
    Kps = np.unique(Kps0, axis=0)
    Nk, M = np.shape(Kps)
    Np, _ = np.shape(pf)
    new_preV, new_palpha = np.empty([Nr, M]), [1]*Nr
    Nk = np.shape(Kps)[0]
    # translate uniform
    pf_ = (pf - np.tile(np.min(pf, axis=0), (Np, 1))) / np.tile(np.max(pf, axis=0)-np.min(pf, axis=0), (Np, 1))
    Kps_ = (Kps - np.tile(np.min(pf, axis=0), (Nk, 1))) / np.tile(np.max(pf, axis=0)-np.min(pf, axis=0), (Nk, 1))
    Ri, Rc = AssociationWeights_acuteAngle(Kps_, rvs)
    for i in np.unique(Ri):
        clusterInd = np.where(Ri==i)[0]
        cluster = Kps_[clusterInd,:]
        meanCluster = np.mean(cluster, axis=0)
        if np.sum(meanCluster) < 1e-6:
            new_preV[i, :] = meanCluster / (np.sum(meanCluster) + max(1e-6, 0))
        else:
            new_preV[i, :] = meanCluster / (np.sum(meanCluster))
    NoAss = np.where(Rc==0)[0]
    if len(NoAss) >0:
        for i in NoAss:
            new_preV[i, :] = rvs[i, :]

    # Associate each solution on ePF to a reference vector
    new_preV = repairReferenceVector(new_preV)
    angle_dist = np.arccos(cosine_similarity(pf_, new_preV))
    associates = np.argmin(angle_dist, axis=1).astype(int)
    for i in np.unique(associates):
        index = np.where(associates == i)[0]
        subpf = pf_[index, :]
        storeMetric = {}
        metrics = []
        for p in pAlpha:
            # 曲率函数值
            value = (np.sum(new_preV[i, :]**p) - 1)**2 + np.sum((np.sum(subpf**p, axis=1) - 1)**2)
            metrics.append(value)
            storeMetric[value] = p
        sort_index = np.argsort(metrics)
        if storeMetric.get(metrics[sort_index[1]]) == max(pAlpha):
            new_palpha[i] = np.infty
        else:
            new_palpha[i] = storeMetric.get(metrics[sort_index[1]])
    return new_preV, new_palpha

def UpdataArchive(A, New, V, newNum, Asize):
    AllDec = np.vstack((A.decs(), New['decs']))
    AllObj = np.vstack((A.objs(), New['objs']))
    ia1 = np.unique(AllDec, axis=0, return_index=True)[1]
    TotalDec = AllDec[ia1, :]
    TotalObj = AllObj[ia1, :]
    V = repairReferenceVector(V)
    ## Select NI solutions for updating the models 
    if len(ia1) > Asize:
        noActNum, active = NoActive(New['objs'], V)
        if noActNum > 0:
            Vi = V[np.setdiff1d(np.arange(len(V)), active), :]
        else:
            Vi = V.copy()
        # Select the undeplicated solutions without re-evaluated solutions
        Iloc, _ = ismember(TotalDec, New['decs'], 'rows')
        TotalDec = TotalDec[~Iloc, :]
        # Since the number of inactive reference vectors is smaller than
        #   Asize-mu, we cluster the solutions instead of reference vectors
        PopObj0 = TotalObj[~Iloc, :]
        # Translate the population
        PopObj = PopObj0 - np.tile(np.min(PopObj0, axis=0), (len(TotalDec), 1))
        # Associate each solution to a reference vector
        #  Calculate the angle value of each solution to a reference vector
        angle_dist = np.arccos(cosine_similarity(PopObj, Vi))
        associates = np.argmin(angle_dist, axis=1)
        Via = Vi[np.unique(associates), :]
        Next = np.zeros(Asize-newNum, dtype=np.int32)
        if len(Via) > Asize-newNum:
            # Cluster solutions based on reference vectors 
            #   when the number of active reference vectors is larger than NI-mu
            kmeans = KMeans(n_clusters=Asize-newNum, random_state=2).fit(Via)
            label_each = kmeans.labels_
        else:
            # Cluster solutions based on objective vectors 
            #   when the number of active reference vectors is smaller than NI-mu
            kmeans = KMeans(n_clusters=Asize-newNum, random_state=2).fit(PopObj)
            label_each = kmeans.labels_
        for i in np.unique(label_each):
            current = np.where(label_each==i)[0]
            if len(current) > 1:
                best = np.random.randint(len(current))
            else:
                best = 0
            Next[i] = current[best]
        A.updateProperties(np.arange(Asize), np.vstack((TotalDec[Next, :], New['decs'])), key='dec')
        A.updateProperties(np.arange(Asize), np.vstack((PopObj0[Next, :], New['objs'])), key='obj')
        # A.group(np.vstack((TotalDec[Next, :], New['decs'])),  np.vstack((PopObj[Next, :], New['objs'])))
    else:
        A.updateProperties(np.arange(len(ia1)), AllDec[ia1, :], key='dec')
        A.updateProperties(np.arange(len(ia1)), AllObj[ia1, :], key='obj')
        # A.group(AllDec[ia1, :],  AllObj[ia1, :])
    return A

def NoActive(PopObj0, V):
    # Detect the number of inactive reference vectors
    #  and return the active reference vectors
    N, _ = np.shape(PopObj0)
    NV = len(V)
    # Translate the population
    PopObj = PopObj0 - np.tile(np.min(PopObj0, axis=0), (N, 1))
    # Associate each solution to a reference vector
    # Calculate the angle value of each solution to a reference vector
    V = repairReferenceVector(V)
    contain_inf1 = (True in np.isinf(V))
    contain_inf2 = (True in np.isinf(PopObj))
    if contain_inf1 or contain_inf2:
        print('there exists inf')
    angle_dist = np.arccos(cosine_similarity(PopObj, V))
    associates = np.argmin(angle_dist, axis=1)
    active  = np.unique(associates)
    num     = NV-len(active)
    return num, active

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

def ReferenceVectorAdaptation(PopObj, V):
    tmp = np.tile(np.max(PopObj,axis=0)-np.min(PopObj,axis=0),(len(V), 1))
    tmp[tmp<1e-8] = 1e-8
    V = V * tmp
    return V / V.sum(axis=1, keepdims=True)

def ENS_SS_NDSort(PObj, nSort):
    (PopObj, ia, ic) = np.unique(PObj, axis=0, return_index=True, return_inverse=True)
    (Table, bin_edges) = np.histogram(ic, bins=np.arange(max(ic)+2))
    (N, M) = np.shape(PopObj)
    FrontNo = np.ones(N)*np.inf
    MaxFNo = 0
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(ic)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                Dominated = False
                for j in range(i-1, -1, -1):  # j<i
                    if FrontNo[j] == MaxFNo:
                        m = 1
                        while (m < M) and (PopObj[i, m] >= PopObj[j, m]):
                            m += 1
                        Dominated = m >= M
                        if Dominated:
                            break
                if bool(1-Dominated):
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[ic]
    return (FrontNo, MaxFNo)

def preferredPBI(PopObj0, V, *varargin):
    N, M = np.shape(PopObj0)
    nv = np.shape(V)[0]
    nargin = len(varargin)
    # translate the population
    # PopObj = PopObj0 - np.tile(np.min(PopObj0, axis=0), (N, 1))
    PopObj = (PopObj0 - np.tile(np.min(PopObj0, axis=0), (N, 1))) \
        / np.clip(np.tile(np.max(PopObj0, axis=0)-np.min(PopObj0, axis=0), (N, 1)), a_min=1e-8, a_max=1e8)
    Nid = np.zeros(M)
    # associates = np.argmin(angle_dist, axis=1)
    PBI = np.ones((N, nv))*np.infty
    V = np.clip(V, 1e-6, 1+1e-6)
    # Calculate the APD value of each solution
    if nargin < 1:
        for j in range(N):
            for i in range(nv):
                sums = np.linalg.norm(V[i, :], ord=2)
                d = np.abs(np.dot(Nid-PopObj[j, :], V[i, :])/sums)
                d2 = norm(PopObj[j, :] - (Nid - d*V[i, :]))
            PBI[j, i] = d + d2
    else:
        skillFactor = varargin[0]
        for j in range(N):
            for i in range(nv):
                sums = np.linalg.norm(V[i, :], ord=2)
                d = np.abs(np.dot(Nid-PopObj[j, :], V[skillFactor[j], :])/sums)
                d2 = norm(PopObj[j, :] - (Nid - d*V[skillFactor[j], :]))
            PBI[j, skillFactor[j]] = d + d2
    return PBI

# -----------KPI by the local scalarizing function 
def KneePointIdentification_LocalScalarFunc(Objs, vectors, Kappa, Zmin):
    def scalar_func_operator(si, wi, iIndex, sj, wj, jIndex, kappai, kappaj):
        if iIndex != jIndex:
            iDominatej = 0
        else:
            Ws = [wi, wj]
            xy = np.zeros(2)
            Kappa = [kappai, kappaj]
            for w in range(2):
                func_lp_i = weightedFunc(si, Ws[w], Kappa[w])
                func_lp_j = weightedFunc(sj, Ws[w], Kappa[w])
                if func_lp_i - func_lp_j < 0:
                    xy[w] = 1
                elif func_lp_i - func_lp_j > 0:
                    xy[w] = -1
                else:
                    xy[w] = 0
            if np.sum(xy) > 0:  # i dominate j
                iDominatej = 1
            elif np.sum(xy) < 0:  # j dominate i
                iDominatej = -1
            else:
                iDominatej = 0
        return iDominatej

    def local_scalar_func(Obj, rvs, Kappa):
        (Obj, ia, ic) = np.unique(Obj, axis=0, return_index=True, return_inverse=True)
        Obj = np.where(Obj < 1e-8, 1e-8, Obj)
        (N, M) = np.shape(Obj)
        nSort = N
        FrontNo = np.ones(N)*np.inf
        MaxFNo = 1
        convex_combination = Obj / np.tile(np.sum(Obj, axis=1)[:, np.newaxis], (M,))
        weights = translateVector(convex_combination)
        # Associate each solution to a reference vector
        Ri, _ = AssociationWeights_acuteAngle(Objs, rvs)
        Ri = Ri.astype(int)

        class AssistStruct:     # python 使用类创建结构体
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
        for i in range(nSort):
            for j in range(nSort):
                # iDominatej = pareto_dominance_operator(PopObj[i, :], PopObj[j, :])
                if isinstance(Kappa, int) or isinstance(Kappa, float):
                    iDominatej = scalar_func_operator(Obj[i, :], weights[i, :], Ri[i], Obj[j, :], weights[j, :], Ri[j], Kappa, Kappa)
                else:
                    iDominatej = scalar_func_operator(Obj[i, :], weights[i, :], Ri[i], Obj[j, :], weights[j, :], Ri[j], Kappa[i], Kappa[j])

                if iDominatej == 1:
                    solutions[i].add_iDominate(j)
                elif iDominatej == -1:
                    solutions[i].add_dominateMe()
            if solutions[i].dominateMe == 0:
                FrontNo[i] = 1
                Flist[0].add_f(i)
        front = 1
        while Flist[front-1].f:
            Q = []
            for i in Flist[front-1].f:
                if solutions[i].iDominate:
                    for j in solutions[i].iDominate:
                        solutions[j].del_dominateMe()
                        if solutions[j].check_if_zero():
                            FrontNo[j] = front+1
                            Q.append(j)
            front += 1
            Flist.extend([Front(Q)])
        MaxFNo = front-1
        FrontNo = FrontNo[ic]
        return (FrontNo, MaxFNo)

    N, _ = np.shape(Objs)
    Objs_ = Objs - np.tile(Zmin, (N, 1))  # translate the population
    FrontNo, _ = local_scalar_func(Objs_, vectors, Kappa)
    knSortedInd = np.where(FrontNo==1)[0]
    return knSortedInd, FrontNo

def weightedFunc(x, w, p):
    x = np.atleast_1d(x)
    if 1e-3< p < 1000:
        try:
            value = np.dot(x**p, w**p)**(1/p)
        except FloatingPointError:
            p = 2
            value = np.dot(x**p, w**p)**(1/p)
            print('FloatingPointError:',x,p,w)
    elif p<=1e-3:
        value = np.min(x*w)
    else:
        value = np.max(x*w)
    return value

def calcCurvature(uPObjs):
    uPObjs = np.where(uPObjs < 1e-8, 1e-8, uPObjs)
    N, M = np.shape(uPObjs)
    P = np.ones(N)  # Initial curvature
    lamda = 1 + np.zeros(N)
    E = np.sum(uPObjs**np.tile(P[:, np.newaxis], (M,)), axis=1) - 1
    for _ in range(3000):
        try:
            # gradient descent
            G = np.sum(uPObjs**np.tile(P[:, np.newaxis], (M,))*np.log(uPObjs), axis=1)
            newP = P - lamda*E*G
            newE = np.sum(uPObjs**np.tile(newP[:, np.newaxis], (M,)), axis=1) - 1
            # Update the value of each weight
            update = (newP > 0) & (np.sum(newE**2) < np.sum(E**2))
            P[update] = newP[update]
            E[update] = newE[update]
            lamda[update] = lamda[update]*1.1
            lamda[~update] = lamda[~update]/1.1
        except FloatingPointError as e:
            print("FloatingPointError: ", e)
            break
    return P

class CalcCurvature():
    def __init__(self, Objs) -> None:
        self.N, self.M = np.shape(Objs)
        # translate
        objs = (Objs - np.tile(np.min(Objs, axis=0), (self.N, 1))) / np.tile(np.max(Objs, axis=0)-np.min(Objs, axis=0), (self.N,1))
        self.objs = np.where(objs < 1e-8, 1e-8, objs)
        self.line = 0
        self.x_i = 0
    
    def fit(self):
        def curvature_func_loss(params):
            kappa = params[0]
            # Objs = self.line
            # P = np.array(params)
            # Iloc, _ = ismember(Objs, self.x_i, 'rows')
            # try:
            #     E = (np.sum(Objs**np.tile(P[:, np.newaxis], (self.M,)), axis=1) - 1)**2
            # except FloatingPointError as e:
            #     print("FloatingPointError: ", e)
            #     E = np.zeros(len(Objs))
            # # err = np.sum(E)
            # err = E[Iloc]
            x = self.x_i
            try:
                err = (np.sum(x**kappa)-1)**2
            except FloatingPointError as e:
                print("FloatingPointError: ", e)
                err = 0
            return err
        P = np.ones(self.N)  # Initial curvature
        # exe = getExtremePoints(self.objs, transpose=True)
        for xi, xx in enumerate(self.objs):
            # arbitraryPoint = np.atleast_2d(xx)
            # uObj = np.concatenate((exe, arbitraryPoint), axis=0)
            # self.line = np.unique(uObj, axis=0)
            # self.x_i = arbitraryPoint
            # x0_ = [P[xi]]+[1]*(len(uObj)-1)
            # res = minimize(curvature_func_loss, x0=x0_, bounds=[(1e-3, 1e3)]*len(uObj), method='L-BFGS-B')
            self.x_i = xx
            res = minimize(curvature_func_loss, x0=[P[xi]], bounds=[(1e-3, 1e3)], method='L-BFGS-B')
            # P[xi] = res.x[0]
            # P[xi] = np.clip(res.x[0] - 1/(10*self.M), 1e-4, 1e3)
            P[xi] = max(1.05, 1/res.x[0] - 1/(10*self.M))
        return P

def calcCurvity(Objs):
    N, M = np.shape(Objs)
    # translate
    Objs_ = (Objs - np.tile(np.min(Objs, axis=0), (N, 1))) / np.tile(np.max(Objs, axis=0)-np.min(Objs, axis=0), (N,1))
    exe = getExtremePoints(Objs_, transpose=True)
    Kappas = np.ones(N)
    for xi, xx in enumerate(Objs_):
        arbitraryPoint = np.atleast_2d(xx)
        uObj = np.concatenate((exe, arbitraryPoint), axis=0)
        uObj = np.unique(uObj, axis=0)
        p = calcCurvature(uObj)
        Iloc, _ = ismember(uObj, arbitraryPoint, 'rows')
        curvity = np.clip(p[Iloc], 1e-4, 1e3)
        Kappas[xi] = max(1.05, 1/curvity - 1/(10*M))
    return Kappas

def getExtremePoints(Objs, transpose=False):
    N, M = np.shape(Objs)
    if transpose:
        extremes = np.zeros((2, M))
        for i in range(M):
            ind = np.argmin(Objs[:, i])
            extremes[i, :] = Objs[ind, :]
        return extremes
    else:
        E = np.zeros((2, M))
        # tmp1 -- ideal point
        # tmp2 -- nadir point
        E[0, :] = np.min(Objs)
        E[1, :] = np.max(Objs)
        return E