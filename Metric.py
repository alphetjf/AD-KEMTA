import numpy as np
# from scipy.spatial.distance import pdist


class Descending:
    """ for np_sortrows: sort column in descending order """
    def __init__(self, column_index):
        self.column_index = column_index

    def __int__(self):  # when cast to integer
        return self.column_index


def np_sortrows(M, columns=None):
    """  sorting 2D matrix by rows
    :param M: 2D numpy array to be sorted by rows
    :param columns: None for all columns to be used,
                    iterable of indexes or Descending objects
    :return: returns sorted M
    """
    if len(M.shape) != 2:
        raise ValueError('M must be 2d numpy.array')
    if columns is None:  # no columns specified, use all in reversed order
        M_columns = tuple(M[:, c] for c in range(M.shape[1]-1, -1, -1))
    else:
        M_columns = []
        for c in columns:
            M_c = M[:, int(c)]
            if isinstance(c, Descending):
                M_columns.append(M_c[::-1])
            else:
                M_columns.append(M_c)
        M_columns.reverse()
    return M[np.lexsort(M_columns), :]


def IGD(PF, PopObj):
    N = np.shape(PF)[0]

    D = np.zeros(N)
    Index = np.zeros(N)

    for i, p in enumerate(PF):
        ed = np.sqrt(np.sum((np.tile(p, (PopObj.shape[0], 1))-PopObj)**2, axis=1))
        # print(ed)
        D[i] = ed.min()
        Index[i] = ed.argmin() + 1
        # print(D, Index)
    Score = np.mean(D)
    return Score


def IGD2(PF, PopObj):
    N = np.shape(PF)[0]

    D = np.zeros(N)
    Index = np.zeros(N)

    for i, p in enumerate(PF):
        ed = np.sqrt(np.sum((p-PopObj)**2, axis=1))
        # print(ed)
        D[i] = ed.min()
        Index[i] = ed.argmin() + 1
        # print(D, Index)
    Score = np.mean(D)
    return Score


def HyperVolume(P, R):
    Ss = np.shape(P)
    Rs = np.shape(R)
    if Ss[1] != Rs[1]:
        score = np.NaN
    else:
        fmin = np.where(np.min(P, axis=0) < np.zeros((1, Ss[1])), np.min(P, axis=0), np.zeros((1, Ss[1])))
        fmax = np.max(R, axis=0)
        # 归一化
        uPopObj = (P - np.tile(fmin, (Ss[0], 1))) / np.tile((fmax-fmin+1e-180)*1.1, (Ss[0], 1))
        uPopObj = np.delete(uPopObj, np.where(np.any(uPopObj > 1, axis=1))[0], axis=0)
        (uPopObj, ia, ic) = np.unique(uPopObj, axis=0, return_index=True, return_inverse=True)
        RefPoint = np.ones(Ss[1])
        if np.size(uPopObj) == 0:
            score = 0
        elif Ss[1] < 4:
            # Calculate the exact HV value
            pl = np_sortrows(uPopObj)
            S = [[1, pl]]
            for k in range(Ss[1]-1):
                S_ = []
                for i in range(len(S)):
                    Stemp = Slice(S[i][1], k, RefPoint)
                    for j in range(len(Stemp)):
                        temp = [Stemp[j][0]*S[i][0], Stemp[j][1]]
                        S_ = Add(temp, S_)
                S = S_.copy()
            score = 0
            for i in range(len(S)):
                p = Head(S[i][1])
                score = score + S[i][0]*abs(p[Ss[1]-1]-RefPoint[Ss[1]-1])
        else:
            # Estimate the HV value by Monte Carlo estimation
            SampleNum = 1e6
            MaxValue = RefPoint.copy()
            MinValue = np.min(uPopObj, axis=0)
            Samples = np.uniform(np.tile(MinValue, (SampleNum, 1)), np.tile(MaxValue, (SampleNum, 1)))
            for i in range(np.shape(uPopObj)[0]):
                m = 1
                domi = np.ones(np.shape(Samples)[0])*True
                while m <= Ss[1] and np.any(domi):
                    domi = domi and uPopObj[i, m] <= Samples[:, m]
                    m = m + 1
                Samples = np.delete(Samples, np.where(domi)[0], axis=0)
            score = np.prod(MaxValue-MinValue)*(1-np.shape(Samples)[0]/SampleNum)
    return score


def Slice(pl, k, RefPoint):
    p = Head(pl)
    pl = Tail(pl)
    ql = np.array([])
    S = []
    while np.size(pl) != 0:
        ql = Insert(p, k+1, ql)
        p_ = Head(pl)
        cell_ = [abs(p[k]-p_[k]), ql]
        S = Add(cell_, S)
        p = p_.copy()
        pl = Tail(pl)
    ql = Insert(p, k+1, ql)
    cell_ = [abs(p[k]-RefPoint[k]), ql]
    S = Add(cell_, S)
    return S


def Head(pl):
    if np.size(pl) == 0:
        p = np.array([])
    else:
        # if p.shape()
        p = pl[0, :]
        # p = p.reshape(1, -1)
    return p


def Tail(pl):
    if np.shape(pl)[0] < 2:
        ql = np.array([])
    else:
        ql = pl[1:, :].copy()
        # ql = ql.reshape(1, -1)
    return ql


def Insert(p, k, pl):
    flag1 = 0
    flag2 = 0
    ql = np.array([])
    hp = Head(pl)
    while np.size(pl) != 0 and hp[k] < p[k]:
        ql = np.concatenate((ql, hp), axis=0)
        pl = Tail(pl)
        hp = Head(pl)
    ql = np.concatenate((ql, p), axis=0)
    m = np.size(p)
    while np.any(pl):
        q = Head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            else:
                if p[i] > q[i]:
                    flag2 = 1
        if bool(1-(flag1 == 1 and flag2 == 0)):
            ql = np.concatenate((ql, Head(pl)), axis=0)
        pl = Tail(pl)
    return ql.reshape(-1, m)


def Add(cell_, S):
    n = len(S)
    m = 0
    for k in range(n):
        if np.all(cell_[1] == S[k][1]):
            S[k][1] = S[k][0] + cell_[0]
            m = 1
            break
    if m == 0:
        S.append(cell_)
    return S

