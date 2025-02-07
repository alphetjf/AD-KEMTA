from scipy.special import comb
import numpy as np
from math import gcd
from itertools import combinations
from sklearn.cluster import KMeans
from pymoo.util.ref_dirs import get_reference_directions

def generatorPoints(N, M, method='MUD'):
    if method == 'NBI':
        (W, N) = NBI(N, M)
    elif method == 'Latin':
        (W, N) = Latin(N, M)
    elif method == 'MUD':
        (W, N) = MixtureUniformDesign(N, M)
    elif method == 'ILD':
        (W, N) = ILD(N, M)
    return (W, N)


def NBI(N, M):
    H1 = 1
    while comb(H1+M, M-1) <= N:  # comb(H1+M, M-1) Combination number (binomial coefficient) "H1+M choose M-1"
        H1 = H1 + 1
    s = range(1, H1+M)
    W = np.asarray(list(combinations(s, M-1))) - np.tile(np.arange(0, M-1), (int(comb(H1+M-1, M-1)), 1)) - 1

    # W = np.array(W) - np.tile(np.arange(0, M-1), (int(comb(H1+M-1, M-1)), 1)) - 1
    W = (np.append(W, np.zeros((W.shape[0], 1))+H1, axis=1) - np.append(np.zeros((W.shape[0], 1)), W, axis=1))/H1
    if H1 < M:
        H2 = 0
        while comb(H1+M-1, M-1) + comb(H2+M, M-1) <= N:
            H2 += 1
        if H2 > 0:
            W2 = []
            s2 = range(1, H2+M)
            W2 = np.asarray(list(combinations(s2, M-1))) - np.tile(np.arange(0, M-1), (int(comb(H2+M-1, M-1)), 1)) - 1
            # W2 = np.array(W2) - np.tile(np.arange(0, M-1), (int(comb(H2+M-1, M-1)), 1)) - 1
            W2 = (np.append(W2, np.zeros((W2.shape[0], 1))+H2, axis=1) - np.append(np.zeros((W2.shape[0], 1)), W2, axis=1))/H2
            W = np.append(W, W2/2+1/(2*M), axis=0)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] < 1e-6:
                W[i, j] = 1e-6
    N = W.shape[0]
    return (W, N)


def Latin(N, M):
    W = np.random.random((N, M))
    W = np.argsort(W, axis=0, kind='mergesort') + 1
    W = (np.random.random((N, M)) + W - 1)/N
    return (W, N)


def ILD(N, M):
    In = M * np.eye(M)
    W = np.zeros((1, M))
    edgeW = W
    while np.shape(W)[0] < N:
        edgeW = np.tile(edgeW, (M, 1)) + np.repeat(In, np.shape(edgeW)[0], axis=0)
        edgeW = np.unique(edgeW, axis=0)
        ind = np.where(np.min(edgeW, axis=0) == 0)[0]
        edgeW = np.take(edgeW, ind, axis=0)
        W = np.append(W+1, edgeW, axis=0)
    W = W / np.tile(np.sum(W, axis=1)[:, np.newaxis], (np.shape(W)[1],))
    W = np.where(W > 1e6, 1e6, W)
    N = np.shape(W)[0]
    return W, N


def MixtureUniformDesign(N, M):
    X = GoodLatticePoint(N, M-1)**(1/np.tile(np.arange(M-1, 0, -1), (N, 1)))
    X = np.clip(X, -np.infty, 1e6)
    X = np.where(X == 0, 1e-12, X)
    W = np.zeros((N, M))
    W[:, :-1] = (1-X)*np.cumprod(X, axis=1)/X
    W[:, -1] = np.prod(X, axis=1)
    return W, N


def GoodLatticePoint(N, M):
    range_nums = np.arange(1, N+1, 1)
    ind = np.asarray([], dtype=np.int64)
    for i in range(np.size(range_nums)):
        if gcd(range_nums[i], N) == 1:
            ind = np.append(ind, i)
    W1 = range_nums[ind]
    W = np.mod(np.dot(np.arange(1, N+1, 1).reshape(-1, 1), W1.reshape(1, -1)), N)
    W = np.where(W == 0, N, W)
    nCombination = int(comb(np.size(W1), M))
    if nCombination < 1e4:
        Combination = np.asarray(list(combinations(np.arange(1, np.size(W1)+1, 1), M)))
        CD2 = np.zeros((nCombination, 1))
        for i in range(nCombination):
            tmp = Combination[i, :].tolist()
            UT = np.empty((np.shape(W)[0], len(tmp)))
            for j in range(len(tmp)):
                UT[:, j] = W[:, tmp[j]-1]
            CD2[i] = CalCD2(UT)
        minIndex = np.argmin(CD2)
        tmp = Combination[minIndex, :].tolist()
        Data = np.empty((np.shape(W)[0], len(tmp)))
        for j in range(len(tmp)):
            Data[:, j] = W[:, tmp[j]-1]
    else:
        CD2 = np.zeros((N, 1))
        for i in range(N):
            UT = np.mod(np.dot(np.arange(1, N+1, 1).reshape(-1, 1), (i+1)**np.arange(0, M, 1).reshape(1, -1)), N)
            CD2[i] = CalCD2(UT)
        minIndex = np.argmin(CD2)
        Data = np.mod(np.dot(np.arange(1, N+1, 1).reshape(-1, 1), (minIndex+1)**np.arange(0, M, 1).reshape(1, -1)), N)
        Data = np.where(Data == 0, N, Data)
    Data = (Data-1)/(N-1)
    return Data


def CalCD2(UT):
    N, S = np.shape(UT)
    X = (2*UT-1)/(2*N)
    CS1 = np.sum(np.prod(2+np.abs(X-1/2)-(X-1/2)**2, axis=1))
    CS2 = np.zeros((N, 1))
    for i in range(N):
        CS2[i] = np.sum(np.prod((1+1/2*np.abs(np.tile(X[i, :], (N, 1))-1/2)
                        + 1/2*np.abs(X-1/2)
                        - 1/2*np.abs(np.tile(X[i, :], (N, 1))-X)), axis=1))
    CS2 = np.sum(CS2)
    CD2 = (13/12)**S - 2**(1-S)/N*CS1 + 1/(N**2)*CS2
    return CD2


def ReferenceVectorGenerator(divisionOuter, divisionInner, M):
    def generateWeights(divisions, M):
        # Generates the reference points (weights) for the given number of divisions
        w = np.zeros(M)
        W = []
        W = generateRecursive(W, w, M, divisions, divisions, 1)
        return np.array(W), len(W)

    def generateRecursive(W, w, M, left, total, index):
        # Generate reference points (weights) recursively.
        if index == M or index==total:
            w[index-1] = left/total
            W.append(w)
        else:
            for i in range(1, left+1):
                w[index-1] = i/total
                W = generateRecursive(W.copy(), w.copy(), M, left-i, total, index+1)
            if len(W)==0:
                W.append(w)
        return W
    
    # def unitWeights(W):
    #     # unit the weight vectors
    #     unit_W = np.zeros(np.shape(W))
    #     for i in range(W.shape[0]):
    #         length = 0
    #         for j in range(W.shape[1]):
    #             length += W[i,j]*W[i,j]
    #         length = np.sqrt(length)
    #         for j in range(W.shape[1]):
    #             unit_W[i,j] = W[i,j]/length
    #     return unit_W
    if divisionInner > 0:
        if divisionOuter >= M:
            raise ValueError('The specified number of outer divisions produces intermediate reference points, recommend setting divisionsOuter < numberOfObjectives.')
        W, N_out = generateWeights(divisionOuter, M)
        # offset the inner weights
        inner_W, N_inner = generateWeights(divisionInner, M)
        for wi, w in enumerate(inner_W):
            for j in range(M):
                w[j] = (1./M + w[j])/2
            inner_W[wi,:] = w
        if N_out > 0 and N_inner > 0:
            W, Nw = np.concatenate((W, inner_W), axis=0), N_out + N_inner
        elif N_out > 0:
            Nw = N_out
        else:
            W, Nw = inner_W, N_inner
    else:
        if divisionOuter < M:
            raise ValueError('No intermediate reference points will be generated for the specified number of divisions, recommend increasing divisions')
        W, Nw = generateWeights(divisionOuter, M)
    return W, len(W)


if __name__ == '__main__':
    # W, N = NBI(100, 8)
    W, N = Latin(100, 8)
    # (W, N) = MixtureUniformDesign(100, 8)
    # (W, N) = ILD(100,8)
    # d = 0.7
    # W = W*d + (1-d)/len(W[0, :])

    # W, N = ReferenceVectorGenerator(divisionOuter=1, divisionInner=9, M=5)
    # kmeans = KMeans(n_clusters=(1+9), random_state=2).fit(W)
    # label_each = kmeans.labels_
    # W = np.unique(W, axis=0)
    # N = len(W)
    # print(W)
    # print(N)
    print()
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("energy", n_dim=8, n_points=15)
    ref_dirs = np.unique(ref_dirs, axis=0)
    print(ref_dirs, len(ref_dirs))