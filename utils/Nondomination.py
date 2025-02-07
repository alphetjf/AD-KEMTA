import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from pymoo.indicators.hv import HV


def NDSort(*varargin):
    # Varargin is a variable parameter that returns the sorted result and the maximum index
    #    Example:
    #        [FrontNo,MaxFNo] = NDSort(PopObj,1)
    #        [FrontNo,MaxFNo] = NDSort(PopObj,PopCon,inf)
    # FrontNo=NDSort (F, s) performs non dominated sorting on population F, 
    #   where F is a matrix of population target values and s represents the 
    #   number of solutions sorted at least,
    # FrontNo[i] represents the leading edge number of the i-th solution, 
    #   and the solution number that has not been assigned a leading edge is inf
    PopObj = varargin[0]
    (N, M) = np.shape(PopObj)
    nargin = len(varargin)
    if nargin == 2:
        nSort = varargin[1]
    elif nargin == 3:
        PopCon = varargin[1]
        nSort = varargin[2]
        # The index of infeasible solutions, Np. any (axis=1) determines whether any array element in each row is non-zero
        Infeasible = np.any(PopCon > 0, axis=1)  
        PopObj[Infeasible, :] = np.tile(np.max(PopObj, axis=0), (sum(Infeasible), 1)) + np.tile(np.sum(np.maximum(0, PopCon[Infeasible, :]), axis=1), (1, M))
    if M < 3 or N > 500:
        (FronNo, MaxFNo) = ENS_SS_NDSort(PopObj, nSort)
    else:
        (FronNo, MaxFNo) = ENS_SS_NDSort(PopObj, nSort)

    return (FronNo, MaxFNo)

def ENS_SS_NDSort(PObj, nSort):
    # PopObj sorts in ascending order according to the first target value, 
    #   where nSort represents the number of individuals to be sorted in non dominated order
    (PopObj, ia, ic) = np.unique(PObj, axis=0, return_index=True, return_inverse=True)
    (Table, bin_edges) = np.histogram(ic, bins=np.arange(max(ic)+2))
    (N, M) = np.shape(PopObj)
    FrontNo = np.ones(N)*np.inf
    MaxFNo = 0
    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(ic)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                # FrontNo[i] compared to the previous individual
                Dominated = False
                for j in range(i-1, -1, -1):  # j<i
                    # Only compare with individuals in the current front
                    if FrontNo[j] == MaxFNo:
                        m = 1  # Check from the second objective onwards
                        while (m < M) and (PopObj[i, m] >= PopObj[j, m]):
                            m += 1
                        Dominated = m >= M
                        # if Dominated or M == 2:
                        if Dominated:
                            break  # Dominated==True, i is dominated by the solution j of the current front
                if bool(1-Dominated):  # Domiatetd==False, otherwise
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[ic]
    return (FrontNo, MaxFNo)

def weightedAugmentedTchebycheff(PObj):
    PObj = np.unique(PObj, axis=0)
    nSort, M = np.shape(PObj)
    # translate
    tPObj = PObj - np.tile(np.min(PObj, axis=0), (nSort, 1))
    # tPObj = PObj

    # reference vector
    rVector = tPObj / np.tile((np.sum(tPObj, axis=1)+1e-6)[:, np.newaxis], (M,))

    non_dominated_index = np.array([], dtype=int)
    for i in range(nSort):
        values = augmentedTchebycheff_func(tPObj, rVector[i, :].reshape(1, -1))
        top_index = np.argmin(values)
        if top_index == i:
            non_dominated_index = np.append(non_dominated_index, i)
    return non_dominated_index

def augmentedTchebycheff_func(x, w):
    N = np.shape(x)[0]
    w = translateVector(w)
    if N == 1:
        return np.max(x*w) + 0.00*np.sum(x*w)
    else:
        return np.max(x*np.tile(w, (N, 1)), axis=1).reshape(N,) + 0.00*np.sum(x*np.tile(w, (N, 1)), axis=1).reshape(N,)

def translateVector(W):
    M = np.shape(W)[1]
    W = np.clip(W, 1e-6, 1.1)
    if np.sum(1/W, axis=1) == 0:
        print("0000")
    return 1/W / np.tile((np.sum(1/W, axis=1))[:, np.newaxis], (M,))

def AssociationWeights_acuteAngle(PObjs, W):
    N, M = np.shape(PObjs)
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

def alpha_ENS_SS_NDSort(alpha, *varargin):
    PObj = varargin[0]
    (N, M) = np.shape(PObj)
    nargin = len(varargin)
    if nargin == 1:
        nSort = N
        Ri = np.ones(N, dtype=int)
    elif nargin == 2:
        nSort = N
        Ri = varargin[1]
    elif nargin == 3:
        Ri = varargin[1]
        nSort = varargin[2]
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
                for j in range(i-1, -1, -1):  # j<i
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
