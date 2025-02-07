# -*- encoding: utf-8 -*-
'''
@File    :   FrontEstimatie.py
@Time    :   2022/09/20 22:17:39
@Author  :   jftang
'''

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from random import sample
from smt.surrogate_models import KRG
import math
from scipy.special import comb
from itertools import combinations

def PF_sampling(PF, sampleNums, uniformSpacing=False):
    lists = [i for i in range(np.shape(PF)[0])]
    if bool(1-uniformSpacing):
        sp = sample(lists, k=sampleNums)
    else:
        sp = np.linspace(0, len(lists), num=sampleNums, endpoint=False)
        for i in range(np.size(sp)):
            sp[i] = np.where(np.mod(sp[i], 1) == 0, sp[i], math.floor(sp[i]))
        sp = sp.astype(int)
    return sp

def NDSort(PObj, nSort):
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

def generatorPoints(N, M):
    # method='Latin'
    W = np.random.random((N, M))
    W = np.argsort(W, axis=0, kind='mergesort') + 1 # Return the element numbers sorted from small to large for each column
    W = (np.random.random((N, M)) + W - 1)/N
    return (W, N)

'''first method'''
def estimation_UnitHyperplane(samples, model="kriging", dge=2, nums=65):
    samples = np.clip(samples, 1e-6, 1e6)
    N, M = np.shape(samples)
    # L1 unit vector
    unit_samples = samples / np.tile(np.sum(samples, axis=1), (M, 1)).T
    # L1 norm
    normL1 = np.sum(samples, axis=1)
    ia1 = np.unique(np.round(unit_samples*1e6)/1e6, axis=0, return_index=True)[1]
    # ia1 = np.unique(unit_samples, axis=0, return_index=True)[1]
    # Estimation
    (Hyperplane, Nw) = generatorPoints(nums*M, M)
    Hyperplane = Hyperplane / np.tile(np.sum(Hyperplane, axis=1)[:,np.newaxis], (M,))
    Hyperplane = np.unique(Hyperplane, axis=0)
    approximatePF = np.zeros((Nw, M))
    approximateSTD = np.zeros(Nw)

    # Model Prediction
    if model == "kriging":
        from smt.surrogate_models import KRG
        print("kriging model")
        gpr = KRG(theta0=[1e-2], print_training=False, print_prediction=False, print_problem=False,
                  print_solver=False, poly='constant', corr='squar_exp')
        gpr.set_training_values(unit_samples[ia1, :M-1], normL1[ia1].reshape(-1, 1))
        gpr.train()
        for i in range(Nw):
            mu = gpr.predict_values(Hyperplane[i, :M-1].reshape(1, -1))
            std = np.sqrt(gpr.predict_variances(Hyperplane[i, :M-1].reshape(1, -1)))
            approximatePF[i, :] = mu*Hyperplane[i, :]
            approximateSTD[i] = std
    elif model == "poly":
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline  # 在Pipeline使用列表建对,可用于将多个估计器链接为一个

        print("Polynomial model")
        poly = Pipeline([('poly', PolynomialFeatures(degree=dge)), ('linear', LinearRegression(fit_intercept=False))])
        poly.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = poly.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    elif model == "mlp":
        from sklearn.neural_network import MLPRegressor

        print("Multi-layer Perceptrons model")
        regre = MLPRegressor(solver="sgd", activation="logistic", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=True, max_iter=500)
        regre.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = regre.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    else:
        print("Mo such model")
    return approximatePF, approximateSTD

def estimationUsingUnitHyperplane(samples, model="kriging", dge=2):
    samples = np.clip(samples, 1e-6, 1e6)
    N, M = np.shape(samples)
    # L1 unit vector
    unit_samples = samples / np.tile(np.sum(samples, axis=1), (M, 1)).T
    # L1 norm
    normL1 = np.sum(samples, axis=1)
    # duplicate removal
    ia1 = np.unique(np.round(unit_samples*1e6)/1e6, axis=0, return_index=True)[1]
    # ia1 = np.unique(unit_samples, axis=0, return_index=True)[1]
    # Estimation
    (Hyperplane, Nw) = generatorPoints(65*M, M)
    Hyperplane = Hyperplane / np.tile(np.sum(Hyperplane, axis=1)[:,np.newaxis], (M,))
    Hyperplane = np.unique(Hyperplane, axis=0)
    approximatePF = np.zeros((Nw, M))
    approximateSTD = np.zeros(Nw)

    # Surrogate model prediction: various models to be used
    if model == "kriging":
        from smt.surrogate_models import KRG
        print("kriging model")
        gpr = KRG(theta0=[1e-2], print_training=False, print_prediction=False, print_problem=False,
                  print_solver=False, poly='constant', corr='squar_exp')
        gpr.set_training_values(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        gpr.train()
        for i in range(Nw):
            mu = gpr.predict_values(Hyperplane[i, :M-1].reshape(1, -1))
            std = np.sqrt(gpr.predict_variances(Hyperplane[i, :M-1].reshape(1, -1)))
            approximatePF[i, :] = mu*Hyperplane[i, :]
            approximateSTD[i] = std
    elif model == "poly":
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline  # Use the list to build a pair in Pipeline, which can be used to link multiple estimators into one

        print("Polynomial model")
        poly = Pipeline([('poly', PolynomialFeatures(degree=dge)), ('linear', LinearRegression(fit_intercept=False))])
        poly.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = poly.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    elif model == "mlp":
        from sklearn.neural_network import MLPRegressor

        print("Multi-layer Perceptrons model")
        regre = MLPRegressor(solver="sgd", activation="logistic", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=True, max_iter=500)
        regre.fit(unit_samples[ia1, :M-1], normL1[ia1].reshape(np.size(ia1), 1))
        for i in range(Nw):
            mu = regre.predict(Hyperplane[i, :M-1].reshape(1, -1))
            approximatePF[i, :] = mu*Hyperplane[i, :]
    else:
        print("Mo such model")
    return approximatePF, approximateSTD

'''second method'''
def GenericFrontModeling(samples0):
    samples = np.where(samples0 > 1e-12, samples0, 1e-12)
    N, M = np.shape(samples)
    P = np.ones((1, M))
    A = np.ones((1, M))
    lamda = 1
    E = np.sum(np.tile(A, (N, 1))*samples**np.tile(P, (N, 1)), axis=1) - 1
    MSE = np.mean(E**2)
    for epoch in range(5000):
        # Calculate the Jacobian matrix
        J = np.concatenate((np.tile(A, (N, 1))*samples**np.tile(P, (N, 1))*np.log(samples), samples**np.tile(P, (N, 1))), axis=1)
        # Update the value of each weight
        while True:
            Delta = -np.linalg.inv(J.T@J + lamda*np.eye(np.size(np.diag(J.T@J))))@J.T@E
            newP = P + Delta[:M]
            newA = A + Delta[M:]
            newE = np.sum(np.tile(newA, (N, 1))*samples**np.tile(newP, (N, 1)), axis=1) - 1
            newMSE = np.mean(newE**2)
            if newMSE < MSE and np.all(newP > 1e-3) and np.all(newA > 1e-3):
                P = newP.copy()
                A = newA.copy()
                E = newE.copy()
                MSE = newMSE.copy()
                lamda = lamda / 1.1
                break
            elif lamda > 1e12:
                return P, A
            else:
                lamda *= 1.1
    return P, A

def estimization_GFM(samples):
    N, M = np.shape(samples)
    P, A = GenericFrontModeling(samples)
    # Calculate the intersections by gradient descent
    # Original sample+Gaussian noise
    # noise = np.random.normal(0, 0.05, size=(N, M))
    # X = np.concatenate((samples, samples+noise), axis=0)
    # X = samples
    # N, M = np.shape(X)
    # Generate uniformly distributed reference points again
    (X, N) = generatorPoints(10*N, M)
    A = np.tile(A, (N, 1))  # Coefficients
    P = np.tile(P, (N, 1))  # Power
    r = np.ones(N)          # Parameters to be optimized
    lamda = np.zeros(N) + 0.1   # Learning rates
    E = np.sum(A*(X*np.tile(r[:, np.newaxis], (M,)))**P, axis=1) - 1  # errors
    for i in range(2000):
        newr = r - lamda*E*np.sum(A*P*X**P*np.tile(r[:, np.newaxis], (M,))**(P-1), axis=1)
        newE = np.sum(A*(X*np.tile(newr[:, np.newaxis], (M,)))**P, axis=1) - 1
        update = (newr > 0) & (np.sum(newE**2) < np.sum(E**2))
        r[update] = newr[update]
        E[update] = newE[update]
        lamda[update] = lamda[update]*1.1
        lamda[~update] = lamda[~update]/1.1
    approximatePF = X*np.tile(r[:, np.newaxis], (M,))
    return approximatePF


'''third method'''
def estimation_LocalModel(samples, K=10):
    def adaptiveDivision(samples, K):
        '''Generate K subregions'''
        N, M = np.shape(samples)
        if K == 1:
            Centres = np.zeros((1, M))
            R = np.infty * np.ones(K)
            subNum = np.array([0, N-1])
        else:
            '''Detect the number of subregions'''
            # calculate the distance between each solution
            fmin = np.min(samples, axis=0)
            fmax = np.max(samples, axis=0)
            # normalization
            samples_ = (samples - np.tile(fmin, (N, 1))) / np.tile(fmax-fmin, (N, 1))
            Distance = euclidean_distances(samples_, samples_)
            Distance = (np.diag(np.diag(np.eye(N)*np.infty))+1) * (Distance+1e-9)
            radius = np.max(np.min(Distance, axis=0))
            # Detect subregion(s)
            Transformation = np.ones(N, dtype=int)*(-1)
            Remain = np.where(Transformation == -1)[0]
            RegionID = 0
            while len(Remain) > 0:
                seeds = np.where(Transformation == -1)[0][0]
                Transformation[seeds] = RegionID
                Remain = np.where(Transformation == -1)[0]
                while True:
                    neighbors = np.where(Distance[seeds, Remain] <= radius)[0]
                    if len(neighbors) == 0:
                        break
                    seeds = Remain[neighbors][0]
                    Transformation[seeds] = RegionID
                    Remain = np.where(Transformation == -1)[0]
                RegionID += 1
            '''Region division'''
            # Count the number of subregions of the true PF
            TrueNum = len(np.unique(Transformation))
            # Calculate the centre point of each subregion
            Centres = np.zeros((TrueNum, M))
            R = np.ones(TrueNum)
            for i in range(TrueNum):
                current = np.where(Transformation == i)[0]
                Centres[i, :] = np.mean(samples_[current, :], axis=0)
                R[i] = np.max(euclidean_distances(samples_[current, :], Centres[i, :].reshape(1, -1)))

            # Select K points
            subNum = Counter(Transformation.flatten())
            if TrueNum > K:
                # Merging small subregions
                while len([key for key, value in subNum.items() if value != np.infty]) > K:
                    Ind = np.argmin([value for _, value in subNum.items()])
                    Centres[Ind, :] = np.ones(M)*np.infty
                    subNum[Ind] = np.infty
                    R[Ind] = -np.infty
                    current = np.where(Transformation == Ind)[0]
                    T = np.argmin(euclidean_distances(samples_[current, :], Centres), axis=1)
                    Transformation[current] = T

                    # Update reference point
                    Idx = [key for key, value in subNum.items() if value != np.infty]
                    for k in Idx:
                        Centres[k, :] = np.mean(samples_[Transformation == k, :], axis=0)
                        R[k] = np.max(euclidean_distances(samples_[Transformation == k, :], Centres[k, :])) / np.sqrt(M-1)
            elif TrueNum < K:
                # Splite large subregions
                while len([key for key, value in subNum.items() if value != -np.infty]) < K:
                    Ind = np.argmax([value for _, value in subNum.items()])
                    Centres[Ind, :] = -np.ones(M)*np.infty
                    subNum[Ind] = -np.infty
                    R[Ind] = -np.infty
                    current = np.where(Transformation == Ind)[0]
                    T1 = np.argmax(euclidean_distances(samples_[current, :], samples_[current[np.random.randint(len(current))], :][np.newaxis, :]), axis=0)
                    T2 = np.argmax(euclidean_distances(samples_[current, :], samples_[current[T1], :]), axis=0)
                    T = np.argmax(euclidean_distances(samples_[current, :], samples_[current[np.append(T1, T2)], :]), axis=1)
                    ExistNum = len([key for key, _ in subNum.items()])
                    Transformation[current] = T + ExistNum

                    # Update reference point
                    newCentre1 = np.mean(samples_[Transformation == ExistNum+1-1, :], axis=0)[np.newaxis, :]
                    newCentre2 = np.mean(samples_[Transformation == ExistNum+2-1, :], axis=0)[np.newaxis, :]
                    Centres = np.append(Centres, newCentre1, axis=0)
                    Centres = np.append(Centres, newCentre2, axis=0)
                    newR1 = 0.5*euclidean_distances(Centres[ExistNum+1-1, :].reshape(1, -1), Centres[ExistNum+2-1, :].reshape(1, -1)).flatten()
                    newR2 = 0.5*euclidean_distances(Centres[ExistNum+1-1, :].reshape(1, -1), Centres[ExistNum+2-1, :].reshape(1, -1)).flatten()
                    R = np.append(R, newR1)
                    R = np.append(R, newR2)
                    subNum[ExistNum+1-1] = len(np.where(T == 0)[0])
                    subNum[ExistNum+2-1] = len(np.where(T == 1)[0])
        # Select reference point
        select = np.abs([value for _, value in subNum.items()]) != np.infty
        Centres = Centres[select, :]
        R = R[select]
        return Centres, R

    def PFEstimate_localModels(samples, Centres, R):
        '''Update intersection point solution in each subregion'''
        K = len(R)
        N, M = np.shape(samples)
        samples_ = np.where(samples > 1e-12, samples, 1e-12)

        # Normalize the population
        fmin = np.min(samples_, axis=0)
        fmax = np.max(samples_, axis=0)
        # normalization
        samples0 = (samples_ - np.tile(fmin, (N, 1))) / np.tile(fmax-fmin, (N, 1))
        # Initialize parameter P
        P = np.ones((K, M))
        A = np.ones((K, M))
        '''Calculate intersection point in each subregion'''
        if K == 1:
            P, A = GenericFrontModeling(samples_)
            interPoint = InterPoint(samples_, P, A)
        else:
            interPoint = np.empty((1, M))

            # Allocation
            transformation = Allocation(samples0, Centres, R)
            for i in range(K):
                current = np.where(transformation == i)[0]
                if len(current) > 0:
                    P[i, :], A[i, :] = GenericFrontModeling(samples_[current, :])
                    sInterPoint = InterPoint(samples_[current, :], P[i, :], A[i, :])
                    # sInterPoint = InterPoint(samples_[current, :], P[i, :], extraNum=10)
                    interPoint = np.append(sInterPoint, interPoint, axis=0)

        valid = np.arange(1, np.shape(interPoint)[0], step=1, dtype=int)
        FrontNo, _ = NDSort(interPoint[valid, :], nSort=1)
        index = np.where(FrontNo == 1)[0]
        return interPoint[valid[index], :]

    def InterPoint(samples, P, A=None, extraNum=10):
        '''Calcualte the approximation degree of each solution, and the distances
        between the intersection points of the solutions
        '''
        N, M = np.shape(samples)

        (X, N) = generatorPoints(extraNum*N, M)
        # X = samples
        if A is not None:
            A = np.tile(A, (N, 1))  # Coefficients
        P = np.tile(P, (N, 1))  # Power
        r = np.ones(N)          # Parameters to be optimized
        lamda = np.zeros(N) + 0.002   # Learning rates
        if A is not None:
            E = np.sum(A*(X*np.tile(r[:, np.newaxis], (M,)))**P, axis=1) - 1  # errors
        else:
            E = np.sum((np.tile(r[:, np.newaxis], (M,))*X)**P, axis=1) - 1  # errors
        for i in range(2000):
            if A is not None:
                newr = r - lamda*E*np.sum(A*P*X**P*np.tile(r[:, np.newaxis], (M,))**(P-1), axis=1)
                newE = np.sum(A*(X*np.tile(newr[:, np.newaxis], (M,)))**P, axis=1) - 1
            else:
                newr = r - lamda*E*np.sum(P*X**P*np.tile(r[:, np.newaxis], (M,))**(P-1), axis=1)
                newE = np.sum((X*np.tile(newr[:, np.newaxis], (M,)))**P, axis=1) - 1
            update = (newr > 0) & (np.sum(newE**2) < np.sum(E**2))
            r[update] = newr[update]
            E[update] = newE[update]
            lamda[update] = lamda[update]*1.1
            lamda[~update] = lamda[~update]/1.1
        interPoint = X*np.tile(r[:, np.newaxis], (M,))
        return interPoint

    def disMax(X):
        '''distMax pairwise distance between one set of observations'''
        N, _ = np.shape(X)
        Dis = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                Dis[i, j] = np.max(np.abs(X[i, :] - X[j, :]))
        Dis = Dis + Dis.T

    def Allocation(samples, kPoint, R):
        K = len(R)
        N, _ = np.shape(samples)
        # Allocation of each solution
        if K == 1:
            Transformation = np.zeros(N)
        else:
            Transformation = np.ones(N)*(-1)
            for i in range(K):
                T = np.where(Transformation == -1)[0]
                current = np.where(euclidean_distances(kPoint[i, :].reshape(1, -1), samples[T, :]) <= R[i])[1]
                Transformation[T[current]] = i
            Remain = np.where(Transformation == -1)[0]
            if len(Remain) != 0:
                transformation = np.argmin(euclidean_distances(samples[Remain, :], kPoint))
                Transformation[Remain] = transformation
        return Transformation

    # N, M = np.shape(samples)
    Centres, R = adaptiveDivision(samples, K)
    # (X, N) = generatorPoints(10*np.shape(samples)[0], np.shape(samples)[1])
    approximatePF = PFEstimate_localModels(samples, Centres, R)
    return approximatePF


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
            if tmp2 < Objs[i, m]:
                tmp2 = Objs[i, m]
        E[0, m] = tmp1
        E[1, m] = tmp2
    if transpose:
        extremes = np.zeros((M, M))
        for i in range(M):
            extremes[i, :] = E[0, :]
            extremes[i, i] = E[1, i]
        return extremes
    return E


if __name__ == "__main__":
    # from Benchmark.DTLZ import DTLZ2
    # pro = DTLZ2()
    # pro.Setting(M=2)
    # PF = pro.GetPF()
    paperUse = False
    from Benchmark.DEBDK import DEB2DK
    from Benchmark.DO2DK import DO2DK
    from Benchmark.CKP import CKP
    # pro = DEB2DK()
    # pro = DO2DK()
    pro = CKP()
    pro.Setting(5, 2, 2)
    PF = pro.GetPF(300)

    PF = np.unique(PF, axis=0)
    sp = PF_sampling(PF, 12, True)
    sp = np.unique(np.concatenate((sp, [0, np.shape(PF)[0]-1])))
    print("Sampling mark sp:", sp)

    approximatePF, approximateSTD = estimation_UnitHyperplane(PF[sp, :], model='kriging', nums=80)
    # approximatePF, approximateSTD = estimationUsingUnitHyperplane(PF[sp, :], model='kriging')
    # approximatePF = estimation_LocalModel(PF[sp, :], K=3)
    # approximatePF = estimization_GFM(PF[sp, :])
    print(approximatePF)
    # approximatePF, _ = localFrontEstimation(PF[sp, :], W)

    test_y = approximatePF[:, 1]
    uncertainty = approximateSTD*5

    from utils.KPI import KneePointIdentification_LocalScalarFunc, NBI_variant, local_alpha_dominance, AssociationWeights_acuteAngle
    from utils.UniformPoint import ReferenceVectorGenerator
    from sklearn.cluster import KMeans
    # W, Nw = generatorPoints(min(pro.M+2, 5), pro.M)
    W, Nw = ReferenceVectorGenerator(divisionOuter=1, divisionInner=3, M=pro.M)
    d = 0.5
    W = W*d+(1-d)/2
    # kps, _ = KneePointIdentification_LocalScalarFunc(approximatePF, vectors=W, Kappa=3, Zmin=np.min(PF, axis=0))
    # kps = NBI_variant(PF, gamma=0.7)
    kps1 = local_alpha_dominance(PF, vectors=W, alpha=0.6)
    kmeans = KMeans(n_clusters=2).fit(PF[kps1, :])
    kps2 = []
    extremes = getExtremePoints(PF, transpose=False)
    Zmin,Zmax = extremes[0,:], extremes[1, :]
    for wi in np.unique(kmeans.labels_):
        index = np.where(kmeans.labels_==wi)[0]
        kps_index = NBI_variant(PF[kps1[index], :], gamma=1)
        kps2 += kps_index.tolist()
    kps2 = np.array(kps2, dtype=int)
    kps = kps1[kps2]

    import matplotlib.pyplot as plt
    # plt.rcParams["mathtext.default"]="regular" # Set the font of the formula to be consistent with other fonts
    # font_options = {"family":"Arial","size":16}
    if pro.M == 2:
        if paperUse:
            fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            # plt.scatter(PF[:, 0], PF[:, 1], marker='.', s=150, c='b', label='$PF$')
            plt.plot(approximatePF[:, 0], approximatePF[:, 1], c='black', ls='--', lw=3, alpha=0.8, label='Estimated Pareto Front')
            plt.fill_between(approximatePF[:, 0], test_y + uncertainty, test_y - uncertainty, alpha=0.4)
            # plt.plot(PF[:, 0], PF[:, 1], c='b', ls='-', lw=2, alpha=0.6, label='True Pareto Front')
            plt.plot(PF[:, 0], PF[:, 1], c='b', ls='-', lw=2, alpha=0.6, label='  ')
            # plt.scatter(PF[sp, 0], PF[sp, 1], s=150, c='b', label='Samples')
            plt.scatter(PF[sp, 0], PF[sp, 1], s=150, c='b', label='  ')
            labelss = plt.legend(fontsize=24, loc=0, frameon=False).get_texts()
            [label.set_fontname('Times New Roman') for label in labelss]
            [label.set_fontsize(24) for label in labelss]
            # label = labelss[0]
            # label.set_fontproperties('SimSun')
            plt.show()

            fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(approximatePF[:, 0], approximatePF[:, 1], c='black', ls='--', lw=3, alpha=0.8, label='Estimated Pareto Front')
            plt.fill_between(approximatePF[:, 0], test_y + uncertainty, test_y - uncertainty, alpha=0.4)
            # plt.scatter(approximatePF[kps, 0], approximatePF[kps, 1], s=150, c='r', label='Knee Point')
            plt.scatter(approximatePF[kps, 0], approximatePF[kps, 1], s=150, c='r', label='                     ')
            labelss = plt.legend(fontsize=24, loc=0, frameon=False).get_texts()
            [label.set_fontname('Times New Roman') for label in labelss]
            [label.set_fontsize(24) for label in labelss]
            # label = labelss[0]
            # label.set_fontproperties('SimSun')
            plt.show()
        else:
            fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
            plt.plot(approximatePF[:, 0], approximatePF[:, 1], c='black', ls='--', lw=3, alpha=0.8, label='Estimation')
            plt.fill_between(approximatePF[:, 0], test_y + uncertainty, test_y - uncertainty, alpha=0.4)
            plt.plot(PF[:, 0], PF[:, 1], c='blue', ls='-', lw=2, alpha=0.6, label='Pareto Front')
            plt.scatter(PF[sp, 0], PF[sp, 1], marker='o', s=100, c='blue', alpha=0.8,label='Nondominated Solution')
            # plt.scatter(approximatePF[kps, 0], approximatePF[kps, 1],marker='s', s=180, c='r', label=r'Knee')
            # plt.scatter(PF[kps, 0], PF[kps, 1],marker='s', s=180, c='r', label=r'Knee identified by distance utility')
            # plt.scatter(PF[kps, 0], PF[kps, 1],marker='s', s=180, c='r', label=r'Knee identified by localized $\alpha$-dominance')
            # plt.scatter(PF[kps, 0], PF[kps, 1],marker='s', s=180, c='r', label=r'Knee identified by modified method')
            labelss = plt.legend(fontsize=24, loc=0, frameon=False).get_texts()
            [label.set_fontname('Times New Roman') for label in labelss]
            [label.set_fontsize(22) for label in labelss]
            # # label = labelss[0]
            # # label.set_fontproperties('SimSun')
            plt.show()

