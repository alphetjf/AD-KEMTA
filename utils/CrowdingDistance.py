import numpy as np


def Crowding(PopObj, FrontNo):
    (N, M) = np.shape(PopObj)
    CrowdDis = np.zeros((N, 1))
    temp = np.unique(FrontNo)
    Fronts = np.setdiff1d(temp, np.infty)
    for f in range(len(Fronts)):
        Front = np.where(FrontNo == Fronts[f])[0]
        fmax = np.max(PopObj[Front, :], axis=0)
        fmin = np.min(PopObj[Front, :], axis=0)
        for i in range(M):
            Rank = np.argsort(PopObj[Front, i], kind='quicksort')
            CrowdDis[Front[Rank[0]]] = 1e6
            CrowdDis[Front[Rank[-1]]] = 1e6
            for j in range(1, len(Front)-1):
                CrowdDis[Front[Rank[j]]] = CrowdDis[Front[Rank[j]]] + (PopObj[Front[Rank[j+1]], i] - PopObj[Front[Rank[j-1]], i]) / (fmax[i] - fmin[i]+1e-6)
    return CrowdDis


def DistanceWeighted(PopObj, FrontNo):
    (N, M) = np.shape(PopObj)
    DW = np.zeros((N, 1))
    temp = np.unique(FrontNo)
    Fronts = np.setdiff1d(temp, np.inf)
    for f in range(len(Fronts)):
        Front = np.where(FrontNo == Fronts[f])[0]
        Nf = np.size(Front)
        index = np.arange(Nf)
        for i in range(Nf):
            dist = np.sqrt(np.sum((PopObj[Front[i], :] - PopObj[Front[np.where(index != i)], :])**2, axis=1))
            rp = np.zeros(Nf-1)
            wp = np.zeros(Nf-1)
            for j in range(Nf-1):
                rp[j] = 1/np.abs(dist[j] - np.mean(dist, dtype=np.float64) + 1e-6)
            for j in range(Nf-1):
                wp[j] = rp[j]/np.sum(rp)
            DW[Front[i]] = np.dot(wp, dist)
    return DW
