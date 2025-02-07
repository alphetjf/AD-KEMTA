import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ismember import ismember
from scipy.linalg import norm
import random


def KDetection(PObj, rvs, gamma=0.5):
    N, M = np.shape(PObj)
    wSize = np.shape(rvs)[0]
    Ri, _ = AssociationWeights_acuteAngle(PObj, rvs)
    Kps = np.empty((1, np.shape(PObj)[1]))
    ifKnees = np.zeros(N, dtype=int)
    for r in range(wSize):
        index = np.where(Ri == r)[0]
        if len(index) > 0:
            sub_PObj = PObj[index, :]
            # knee_index = NBI_variant(sub_PObj, gamma)
            FrontNo, _ = FastAlphaNDSort(sub_PObj, Ri[index], np.shape(sub_PObj)[0],gamma)
            knee_index = np.where(FrontNo == 1)[0]
            kps = sub_PObj[knee_index, :]
            Kps = np.append(Kps, kps, axis=0)
            ifKnees[index[knee_index]] = 1
    return Kps[1:, :].reshape(-1, M), np.where(ifKnees == 1)[0]

def AssociationWeights_acuteAngle(PObjs, W):
    N, M = np.shape(PObjs)
    # Normalize the objective space
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

def NBI_variant(PF, gamma=0.5, Zmin=None, Zmax=None):
    N, M = np.shape(PF)

    # translate the population
    if Zmin is None:
        uPF = (PF - np.tile(np.min(PF, axis=0), (np.shape(PF)[0], 1))) / np.tile(np.max(PF, axis=0)-np.min(PF, axis=0), (np.shape(PF)[0], 1))
    else:
        uPF = (PF - np.tile(Zmin, (np.shape(PF)[0], 1))) / np.tile(Zmax-Zmin, (np.shape(PF)[0], 1))
    LowerBound = np.min(uPF, axis=0)
    UpperBound = np.max(uPF, axis=0)
    sizeBound = (UpperBound - LowerBound)*gamma/2

    Hyperplane = uPF / np.tile(np.sum(uPF, axis=1)[:, np.newaxis], (M,))
    distance = np.sqrt(np.sum((uPF - Hyperplane)**2, axis=1))

    ifKnees = np.zeros(N)
    for j in range(N):
        tempPF = np.delete(uPF, j, axis=0)
        dis = np.abs(uPF[j, :] - tempPF)
        near_index = np.where(np.cumprod(dis <= sizeBound, axis=1)[:, -1] == 1)[0]
        if len(near_index) == 0:
            ifKnees[j] = 1
            continue
        # Neighbors = tempPF[near_index, :]
        tempDis = np.delete(distance, j)
        Neighbors_dis = tempDis[near_index]
        if np.max(Neighbors_dis) < distance[j]:
            ifKnees[j] = 1
    return np.where(ifKnees == 1)[0]

def alpha_ENS_SS_NDSort(alpha, *varargin):
    PObj = varargin[0]
    (N, M) = np.shape(PObj)
    nargin = len(varargin)
    if nargin == 1:
        nSort = N
        Ri = np.ones(N, dtype=int)
    elif nargin == 2:
        nSort = varargin[1]
        Ri = np.ones(N, dtype=int)
    elif nargin == 3:
        Ri = varargin[2]
        nSort = varargin[1]
    (PopObj, ia, ic) = np.unique(PObj, axis=0, return_index=True, return_inverse=True)
    Ric = Ri[ia]
    (Table, bin_edges) = np.histogram(ic, bins=np.arange(max(ic)+2))
    (N, M) = np.shape(PopObj)
    FrontNo = np.ones(N)*np.inf
    MaxFNo = 0
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(ic)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                Dominated = False
                for j in range(i-1, -1, -1):  # j<i，
                    if FrontNo[j] == MaxFNo:
                        jDominatei = localized_alpha_operator(PopObj[j, :], PopObj[i, :], Ric[j], Ric[i], alpha)
                        if jDominatei == 1:
                            Dominated = True
                        if Dominated:
                            break 
                if bool(1-Dominated):
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[ic]
    return (FrontNo, MaxFNo)

def localized_alpha_operator(si, sj, iIndex, jIndex, alpha=0.75):
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
                if isinstance(Kappa, (int, float)):
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
        MaxFNo = front-1  # Flist[MaxFNo-1].f非空，Flist[MaxFNo].f为空，一共MaxFNo个前沿（python下标从0开始）
        FrontNo = FrontNo[ic]
        return (FrontNo, MaxFNo)

    N, _ = np.shape(Objs)
    Objs_ = Objs - np.tile(Zmin, (N, 1))  # translate the population
    FrontNo, _ = local_scalar_func(Objs_, vectors, Kappa)
    knSortedInd = np.where(FrontNo==1)[0]
    return knSortedInd, FrontNo

def translateVector(W):
    M = np.shape(W)[1]
    W = np.clip(W, 1e-6, 1e-6+1)
    return 1/W / np.tile((np.sum(1/W, axis=1))[:, np.newaxis], (M,))

def weightedFunc(x, w, p):
    if p < 1000:
        try:
            value = (np.dot(x**p, w))**(1/p)
        except FloatingPointError:
            p = 2
            value = (np.dot(x**p, w))**(1/p)
            print('FloatingPointError:',x,p,w)
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

# -----------------------local alpha-dominance relationship
def local_alpha_dominance(PObjs, vectors, alpha=0.75):
    Ri, Rc = AssociationWeights_acuteAngle(PObjs, vectors)
    Ri = Ri.astype(int)
    FrontNo, maxF = FastAlphaNDSort(PObjs, Ri, np.shape(PObjs)[0], alpha)
    ind = np.where(FrontNo == 1)[0]
    return ind

def localized_alpha_operator(si, sj, iIndex, jIndex, alpha):
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
        if (dominate2 == 0) and (dominate1 > 0):  # i dominates j
            iDominatej = 1
        elif (dominate1 == 0) and (dominate2 > 0):  # j dominates i
            iDominatej = -1
        else:
            iDominatej = 0
    return iDominatej

def FastAlphaNDSort(*varargin):
    PopObj = varargin[0]
    (N, M) = np.shape(PopObj)
    nargin = len(varargin)
    if nargin == 3:
        Ri = varargin[1]
        nSort = varargin[2]
        alpha = 0.75
    elif nargin == 4:
        Ri = varargin[1]
        nSort = varargin[2]
        alpha = varargin[3]
    FronNo = np.ones(N)*np.inf
    MaxFNo = 1

    # python: Create a structure using a class
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
    for i in range(nSort):
        for j in range(nSort):
            # iDominatej = pareto_dominance_operator(PopObj[i, :], PopObj[j, :])
            iDominatej = localized_alpha_operator(PopObj[i, :], PopObj[j, :], Ri[i], Ri[j], alpha)
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
    MaxFNo = front-1  # Flist[MaxFNo-1].f: non-empty，Flist[MaxFNo].f: empty，there are total MaxFNo numbers of front ranks
    return (FronNo, MaxFNo)

# -----------KPI by ' Bridging the Gap: Many-Objective Optimization and Informed Decision-Making'
def KneePointIdentification_ExpectedMarginalUtility(Objs0, lamdas):
    Nl = len(lamdas)
    (Objs, ia, ic) = np.unique(Objs0, axis=0, return_index=True, return_inverse=True)
    N, M = np.shape(Objs)
    # translate
    Objs_ = (Objs - np.tile(np.min(Objs, axis=0), (N, 1))) / np.tile(np.max(Objs, axis=0)-np.min(Objs, axis=0), (N, 1))
    # associate each solution to every lamda
    Ri, Rc = AssociationWeights_acuteAngle(Objs_, lamdas)
    # calculate the expected marginal utility
    EMUs = np.zeros(N)
    for wi, ww in enumerate(lamdas):
        current = np.where(Ri==wi)[0]
        Nc = len(current)
        if Nc == 0:
            continue
        matrix = Objs_[current, :].reshape(Nc, -1)
        lamda = ww.reshape(M, 1)
        utilities = np.dot(matrix, lamda).flatten()
        sortIndex = np.argsort(utilities)
        if Nc==1:
            EMUs[current[0]] = utilities
        else:
            EMUs[current[sortIndex[0]]] = utilities[sortIndex[1]] - utilities[sortIndex[0]]
    EMUs = EMUs[ic]
    ind = np.where(EMUs > 0)[0]
    return ind

#  ---------------------------------identifying Knees according to the curvature
def identificationCurvature(PObjs, vectors, K=None, numsOfNeighbor=np.infty):
    N = np.shape(PObjs)[0]
    E = getExtremePoints(PObjs)
    # normalization
    uPObjs = (PObjs - np.tile(E[0, :], (N, 1))) / np.tile(E[1, :] - E[0, :], (N, 1))
    Ri, _ = AssociationWeights_acuteAngle(uPObjs, vectors)
    Ri = Ri.astype(int)
    w_size = np.shape(vectors)[0]

    ifKnees = np.zeros(N)
    for j in range(w_size):
        wj_index = np.where(Ri == j)[0]
        if np.size(wj_index) == 0:
            continue
        curvatures = np.ones(np.size(wj_index))*np.infty
        curvatures = calcCurvature(uPObjs[wj_index, :])
        if K is None:
            soi = np.where(curvatures == np.min(curvatures))[0]  # Solution of convex region corresponding to minimum curvature
            ifKnees[wj_index[soi]] = 1
        elif K == w_size:
            soi = np.where(curvatures == np.min(curvatures))[0]  # Solution of convex region corresponding to minimum curvature
            ifKnees[wj_index[soi]] = 1
        else:
            numsOfSoi = 0
            for i in range(len(wj_index)):
                dis = calculateDistMatrix(uPObjs[wj_index[i], :], uPObjs)
                ind = np.sort(dis)
                nl = int(min(numsOfNeighbor, len(wj_index)+1, len(ind)))
                if nl == 1:
                    ifKnees[wj_index[i]] = 1
                    numsOfSoi += 1
                    continue
                neighbors = np.take(curvatures, ind[1:nl], 0)
                if np.min(neighbors) == curvatures[i]:  # If its curvature is the smallest in the field, it may be a promising solution
                    ifKnees[wj_index[i]] = 1
                    numsOfSoi += 1
                    if numsOfSoi == K:
                        break
    ind = np.where(ifKnees == 1)[0]
    return ind

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

def calcCurvature(uPObjs):
    uPObjs = np.where(uPObjs > 1e-12, uPObjs, 1e-12)
    N, M = np.shape(uPObjs)
    P = np.ones(N)  # Initial curvature
    lamda = 1 + np.zeros(N)
    E = np.sum(uPObjs**np.tile(P[:, np.newaxis], (M,)), axis=1) - 1
    for epoch in range(5000):
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
    return P

# --------Posterior Decision-Making Based on Decomposition-Driven Knee Point Identification
def TradeUtility(s1, s2, exe):
    M = np.size(s1)
    G12, D12 = 0, 0
    for m in range(M):
        G12 += min(0, (s1[m] - s2[m])/(exe[1, m] - exe[0, m]))
        D12 += max(0, (s1[m] - s2[m])/(exe[1, m] - exe[0, m]))
    U12 = G12 + D12
    if U12 < 0:
        return -1  # s1 is defined to knee_dominate s2
    elif U12 == 0:
        return 0  # s1 is defined to non_knee_dominate s2
    else:
        return 1  # s1 is defined to be knee_dominated by s2

def KneePointIdentification_TradeUtility(PObjs, vectors):
    N, M = np.shape(PObjs)
    E = getExtremePoints(PObjs)
    # Find the neighborhood of each sub-region
    Neighbors = NeighborSearch(PObjs, vectors)
    ifKnees = np.zeros(N)
    for i in range(N):
        flag = 1
        for j in range(np.size(Neighbors[i])):
            ind = Neighbors[i][j]
            if TradeUtility(PObjs[i, :], PObjs[ind, :], E) > 0:
                flag = 0
                break
        if flag == 1:
            ifKnees[i] = 1
    kneesIndex = np.where(ifKnees == 1)[0]
    kneesIndex_sorted = KneeSorted_accumulative_utility(PObjs, kneesIndex, Neighbors, E)
    return kneesIndex_sorted

def KneeSorted_accumulative_utility(PObjs, kneesIndex, Neighbors, E):
    Nk = np.size(kneesIndex)
    K = np.zeros(Nk)
    for i in range(Nk):
        numOfNeighbors = np.size(Neighbors[kneesIndex[i]])
        for j in range(numOfNeighbors):
            K[i] += TradeUtility(PObjs[kneesIndex[i], :], PObjs[Neighbors[kneesIndex[i]][j], :], E)
    sort_index = np.argsort(-K, kind='quicksort', )  # descending sort of K
    return kneesIndex[sort_index]

def AssociationWeights_acuteAngle(PObjs, W):
    N, M = np.shape(PObjs)
    # Normalize the objective space
    uPObjs = (PObjs - np.tile(np.min(PObjs, axis=0), (N, 1))) / np.tile(np.max(PObjs, axis=0)-np.min(PObjs, axis=0)+1e-8, (N, 1))

    Nid = np.zeros(M)
    W_size = np.shape(W)[0]
    Ri = np.zeros(N)
    Rc = np.zeros(W_size)
    for i in range(N):
        angles = np.zeros(W_size)
        for j in range(W_size):
            norm1 = norm(uPObjs[i, :] - Nid, 2)
            norm2 = norm(W[j, :] - Nid, 2)
            angles[j] = np.arccos(np.dot(uPObjs[i, :]-Nid, W[j, :]-Nid)/(norm1*norm2))
        index = np.where(angles == np.min(angles))[0]
        if len(index) > 0:
            Ri[i] = index[0]
            Rc[index[0]] += 1
        else:
            index = random.randint(0, W_size-1)
            Ri[i] = index
            Rc[index] += 1
    return Ri, Rc

def NeighborSearch(PObjs, W):
    W_size = np.shape(W)[0]
    Neighborhoods = []
    for i in range(W_size):
        sub_neighborhood = np.array([])
        min_angle = np.infty
        for j in range(W_size):
            if j != i:
                if AcuteAngle(W[i, :], W[j, :]) < min_angle:
                    min_angle = AcuteAngle(W[i, :], W[j, :])
                    sub_neighborhood = np.array([])
                    sub_neighborhood = np.append(sub_neighborhood, j)
                elif AcuteAngle(W[i, :], W[j, :]) == min_angle:
                    sub_neighborhood = np.append(sub_neighborhood, j)
        sub_neighborhood = np.append(sub_neighborhood, i)
        sub_neighborhood = sub_neighborhood.astype(int)
        Neighborhoods.append(sub_neighborhood)
    Ri, Rc = AssociationWeights_acuteAngle(PObjs, W)
    N = np.shape(PObjs)[0]
    Neighbors = []
    for i in range(N):
        for j in range(W_size):
            if Ri[i] == j:
                sizeOfNeigborhood = np.size(Neighborhoods[j])
                for k in range(sizeOfNeigborhood):
                    k_index = np.where(Ri == Neighborhoods[j][k])[0]
                    Neighbor_individual = np.setdiff1d(k_index, i)
                Neighbors.append(Neighbor_individual)
    return Neighbors

def AcuteAngle(v, u):
    return np.arccos(np.dot(v, u)/(norm(v, 2)*norm(u, 2)))
