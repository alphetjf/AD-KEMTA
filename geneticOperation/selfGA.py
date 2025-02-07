import numpy as np
from copy import deepcopy
from tools.mutation import mutPolynomialBounded
from tools.crossover import cxSimulatedBinaryBounded
import random
from tools.selection import selTournament


def OperatorGA(PDecs, proC, disC, proM, disM, x_opt_max, x_opt_min):
    N, D = np.shape(PDecs)
    Parent1 = PDecs[:np.floor(N/2).astype(int), :]
    Parent2 = PDecs[np.floor(N/2).astype(int):np.floor(N/2).astype(int)*2, :]
    N_ = Parent1.shape[0]
    # simulated binary crossover
    Offsprings = np.zeros((N_*2, D))
    miu = np.random.rand(N_, D)
    beta = np.zeros((N_, D))
    index = miu <= 0.5
    beta[index] = (2*miu[index])**(1/(disC+1))
    beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
    beta = beta*(-1)**np.random.randint(2, size=D)
    beta[np.random.rand(N_, D) < 0.5] = 1
    beta[np.tile(np.random.rand(N_,1)>proC, (1, D))] = 1
    Offsprings[:N_, :] = (Parent1 + Parent2)/2 + beta*(Parent1 - Parent2)/2
    Offsprings[N_:, :] = (Parent1 + Parent2)/2 - beta*(Parent1 - Parent2)/2
    # polynomial mutation
    N = N_*2
    if N == 1:
        MaxValue = x_opt_max.reshape(N, -1)
        MinValue = x_opt_min.reshape(N, -1)
    else:
        MaxValue = np.tile(x_opt_max, (N, 1))
        MinValue = np.tile(x_opt_min, (N, 1))

    k = np.random.rand(N, D)
    miu = np.random.rand(N, D)
    try:
        Temp = (k <= proM/D) & (miu < 0.5)  # Mutated genes
        Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
            ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[Temp]-MinValue[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1))-1)
    except FloatingPointError as e:
        print("FloatingPointError: ", e)
    try:
        Temp = (k <= proM/D) & (miu >= 0.5)
        Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
            (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(MaxValue[Temp]-Offsprings[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1)))
    except FloatingPointError as e:
        print("FloatingPointError: ", e)

    Offsprings = np.where(Offsprings <= MaxValue, Offsprings, MaxValue)
    Offsprings = np.where(Offsprings >= MinValue, Offsprings, MinValue)
    # Offsprings = np.clip(Offsprings, x_opt_min, x_opt_max)
    return Offsprings

def OperatorGAhalf(PDecs, proC, disC, proM, disM, x_opt_max, x_opt_min):
    N, D = np.shape(PDecs)
    Parent1 = PDecs[:np.floor(N/2).astype(int), :]
    Parent2 = PDecs[np.floor(N/2).astype(int):np.floor(N/2).astype(int)*2, :]
    N_ = Parent1.shape[0]
    # simulated binary crossover
    Offsprings = np.zeros((N_, D))
    miu = np.random.rand(N_, D)
    beta = np.zeros((N_, D))
    index = miu <= 0.5
    beta[index] = (2*miu[index])**(1/(disC+1))
    beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
    beta = beta*(-1)**np.random.randint(2, size=D)
    beta[np.random.rand(N_, D) < 0.5] = 1
    beta[np.tile(np.random.rand(N_,1)>proC, (1, D))] = 1
    Offsprings = (Parent1 + Parent2)/2 + beta*(Parent1 - Parent2)/2
    # polynominal mutation
    if N_ == 1:
        MaxValue = x_opt_max.reshape(N_, -1)
        MinValue = x_opt_min.reshape(N_, -1)
    else:
        MaxValue = np.tile(x_opt_max, (N_, 1))
        MinValue = np.tile(x_opt_min, (N_, 1))
    k = np.random.rand(N_, D)
    miu = np.random.rand(N_, D)
    Temp = (k <= proM/D) & (miu < 0.5)  # Mutated genes
    Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
        ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[Temp]-MinValue[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1))-1)
    Temp = (k <= proM/D) & (miu >= 0.5)
    Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
        (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(MaxValue[Temp]-Offsprings[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1)))
    Offsprings = np.where(Offsprings <= MaxValue, Offsprings, MaxValue)
    Offsprings = np.where(Offsprings >= MinValue, Offsprings, MinValue)
    # Offsprings = np.clip(Offsprings, x_opt_min, x_opt_max)
    return Offsprings


def Crossover(population, low, up, ProC=1, DisC=20):
    low = low.tolist()
    up = up.tolist()
    (N, D) = population.shape
    off = np.zeros((N, D))
    for i in range(0, N, 2):
        if (np.mod(N, 2) != 0) and (i == N-1):
            ind1 = population[i, :]
            ind2 = population[random.randint(0, max(i-1, 0)), :]
        else:
            ind1 = population[i, :]
            ind2 = population[i+1, :]
        child1, child2 = [deepcopy(ind) for ind in (ind1, ind2)]
        # print(ind1, ind2)
        child1, child2 = cxSimulatedBinaryBounded(child1, child2, DisC, ProC, low, up)
        # print(child1, child2)
        # break
        if (np.mod(N, 2) != 0) and (i == N-1):
            off[i, :] = child1.copy()
        else:
            off[i, :] = child1.copy()
            off[i+1, :] = child2.copy()
    return off

def Mutation(population, low, up, ProM=1, DisM=20):
    (N, D) = population.shape
    ProM = ProM/D
    low = low.tolist()
    up = up.tolist()
    off = np.zeros((N, D))
    for i in range(N):
        mutant = deepcopy(population[i, :])
        # print(population[i, :])
        mutant = mutPolynomialBounded(mutant, DisM, low, up, ProM)
        # print(mutant)
        # break
        mutant = np.asarray(list(mutant))
        off[i, :] = mutant.copy()
    return off

def tournamentSelection(K, N, elevation, *varargin):
    nargin = len(varargin)
    Fit = np.empty((N, nargin))
    fields = []
    for col in range(nargin):
        Fit[:, col] = varargin[col]
        fields.append(varargin[col])
    #     field.append((str(col), float))
    #     columns.append(str(col))
    # Fit2 = np.array([tuple(x) for x in Fit.tolist()], dtype=np.dtype(field))
    # ind = np.argsort(Fit2, order=tuple(columns[::-1]))
    ind = np.lexsort(tuple(fields[::-1]))
    rank = np.argsort(ind)
    parents = np.random.randint(np.shape(Fit)[0], size=(K, elevation))
    best = np.argmin(rank[parents.reshape(-1)].reshape(K, elevation), axis=0)
    index = parents.reshape(-1)[np.arange(elevation) + best * elevation]
    return index

def CreatSolutions(PCheby, PopDec):
    N = np.shape(PopDec)[0]
    population = []
    for i in range(N):
        population.append(SOLUTION(PopDec[i, :], PCheby[i]))
    return population

def Selection(fitness, chromos, Toursize=2):
    population = CreatSolutions(-fitness, chromos)
    selected = selTournament(population, len(population), Toursize, fit_attr="obj")
    selected_clone = [deepcopy(ind) for ind in selected]
    selected_clone = disintegrationSolutions(selected_clone)
    return selected_clone

def disintegrationSolutions(population):
    N = len(population)
    Temp = population[0].dec
    for i in range(1, N):
        Temp = np.append(Temp, population[i].dec, axis=0)
    PopDecs = Temp.reshape(N, -1)
    return PopDecs


class SOLUTION:
    # SOLUTION - The class of a solution.
    #   This is the class of a solution. An object of SOLUTION stores all the
    #   properties including decision variables, objective values, constraint
    #   violations, and additional properties of a solution.
    # SOLUTION properties:
    #   dec         <read-only>     decision variables of the solution
    #   obj         <read-only>     objective values of the solution
    #   con         <read-only>     constraint violations of the solution
    #   add         <read-only>     additional properties of the solution
    def __init__(self, *varvargin) -> None:
        self.dec = 0
        self.obj = 0
        self.con = 0
        self.add = 0
        if len(varvargin) > 0:
            self.dec = varvargin[0]
            if len(varvargin) > 1:
                self.obj = varvargin[1]
                if len(varvargin) > 2:
                    self.con = varvargin[2]
                    if len(varvargin) > 3:
                        self.add = varvargin[3]
