import numpy as np
from copy import deepcopy
from tools.selection import selTournament


def tournamentSelection(K, N, *varargin):
    nargin = len(varargin)
    Fit = np.empty((N, nargin))
    fields = []
    # field = []
    # columns = []
    for col in range(nargin):
        Fit[:, col] = varargin[col]
        fields.append(varargin[col])
    #     field.append((str(col), float))
    #     columns.append(str(col))
    # Fit2 = np.array([tuple(x) for x in Fit.tolist()], dtype=np.dtype(field))
    # ind = np.argsort(Fit2, order=tuple(columns[::-1]))
    ind = np.lexsort(tuple(fields[::-1]))
    rank = np.argsort(ind)
    parents = np.random.randint(np.shape(Fit)[0], size=(K, N))
    best = np.argmin(rank[parents.reshape(-1)].reshape(K, N), axis=0)
    index = parents.reshape(-1)[np.arange(N) + best * N]
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