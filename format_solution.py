# -*- encoding: utf-8 -*-
'''
@File    :   SOLUTION.py
@Time    :   2022/09/18 20:31:22
@Author  :   jftang
'''


import numpy as np
from copy import deepcopy


def err_print(msg, original_line=None):
    print('ERROR  ' * 3)
    print(msg)
    if original_line:
        print(original_line)
    print('ERROR  ' * 3)
    exit(1)


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


class SOLUTIONSET:
    # SOLUTIONSET methods:
    #   building	<public>        the constructor, which sets all the
    #                               properties of the solution
    #   decs        <public>      	get the matrix of decision variables of
    #                               multiple solutions
    #   objs        <public>        get the matrix of objective values of
    #                               multiple solutions
    #   cons        <public>        get the matrix of constraint violations of
    #                               multiple solutions
    #   adds        <public>        get the matrix of additional properties of
    #                               multiple solutions
    #   best        <public>        get the feasible and nondominated solutions
    #                               among multiple solutions

    def __init__(self, instance) -> None:
        self.problem = instance
        self.solutions = 0
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index >= len(self.solutions):
            raise StopIteration
        element = self.solutions[self.index]
        self.index += 1
        return element
    def building(self, **varvargin):
        Decs = varvargin['Decs']
        self.problem.N = Decs.shape[0]
        solutions = np.array([SOLUTION() for i in range(self.problem.N)])
        if len(varvargin.keys()) > 1:
            AddProper = varvargin['AddProper']
        Decs = self.problem.CalDec(Decs)
        Objs = self.problem.CalObj(Decs)
        Cons = self.problem.CalCon(Decs)
        for i in range(self.problem.N):
            solutions[i].dec = np.array(Decs[i, :])
            solutions[i].obj = Objs[i, :]
            solutions[i].con = Cons[i, :]
            if len(varvargin.keys()) > 1:
                solutions[i].add = AddProper[i, :]
        self.solutions = solutions
        return solutions

    def group(self, *varvargin):
        Decs = varvargin[0]
        Objs = varvargin[1]
        if len(varvargin) == 3:
            Cons = varvargin[2]
        if len(varvargin) == 4:
            Cons = varvargin[2]
            Adds = varvargin[3]
        self.problem.N = Decs.shape[0]
        solutions = np.array([SOLUTION() for i in range(self.problem.N)])
        for i in range(self.problem.N):
            solutions[i].dec = np.array(Decs[i, :])
            solutions[i].obj = Objs[i, :]
            if len(varvargin) == 3:
                solutions[i].con = Cons[i, :]
            if len(varvargin) == 4:
                solutions[i].con = Cons[i, :]
                solutions[i].add = Adds[i, :]
        self.solutions = solutions
        return solutions

    def decs(self):
        Ns = len(self.solutions)
        Decs = np.zeros((Ns, self.problem.D))
        for i in range(Ns):
            Decs[i, :] = deepcopy(self.solutions[i].dec)
        return Decs

    def objs(self):
        Ns = len(self.solutions)
        Objs = np.zeros((Ns, self.problem.M))
        for i in range(Ns):
            Objs[i, :] = deepcopy(self.solutions[i].obj)
        return Objs

    def cons(self):
        Ns = len(self.solutions)
        C = np.size(self.solutions[0].con)
        Cons = np.zeros((Ns, C))
        for i in range(Ns):
            Cons[i, :] = deepcopy(self.solutions[i].con)
        return Cons

    def adds(self):
        Ns = len(self.solutions)
        a = np.size(self.solutions[0].add)
        AddProper = np.zeros((Ns, a))
        for i in range(Ns):
            AddProper[i, :] = deepcopy(self.solutions[i].add)
        return AddProper

    def updateProperties(self, whos, Properties, key='dec'):
        try:
            if key == 'dec':
                self.updateDecs(whos, Properties)
            elif key == 'obj':
                self.updateObjs(whos, Properties)
            elif key == 'con':
                self.updateCons(whos, Properties)
            elif key == 'add':
                self.updateAdds(whos, Properties)
        except Exception as e:
            err_print('can not find such property', e)
        finally:
            pass

    def updateAdds(self, whos, AddPropers):
        for i in range(len(whos)):
            self.solutions[whos[i]].add = AddPropers[i, :].copy()

    def updateDecs(self, whos, Decs):
        for i in range(len(whos)):
            self.solutions[whos[i]].dec = Decs[i, :].copy()

    def updateObjs(self, whos, Objs):
        for i in range(len(whos)):
            self.solutions[whos[i]].obj = Objs[i, :].copy()

    def updateCons(self, whos, Cons):
        for i in range(len(whos)):
            self.solutions[whos[i]].con = Cons[i, :].copy()

    def best(self):
        # best - Get the best solutions in a population.
        #   P = obj.best returns the feasible and non-dominated solutions
        #   among multiple solutions obj. If the solutions have a single
        #   objective, the feasible solution with minimum objective value
        #   is returned.
        Feasible = np.any(self.cons() <= 0, axis=1)  # determines whether any array element in each row is non-zero
        Objs = self.objs()
        Decs, Cons, AddPropers = self.decs(), self.cons(), self.adds()
        # if np.all(Feasible, where=False):
        #     Best = []
        if self.problem.M > 1:
            temp = NDSort(Objs[Feasible, :], Objs[Feasible, :].shape[0])
            Best = np.where(temp[0] == 1)[0]
        else:
            Best = np.argsort(Objs[Feasible, :], kind="quicksort")  # Return the sequence number of the array sorted from small to large
            Best = Best[0]
        P = [Decs[Best, :], Objs[Best, :], Cons[Best, :], AddPropers[Best, :]]
        return P

    def AddOnes(self, **varvargin):
        if len(varvargin) == 1:
            Dec = varvargin['Dec']
        elif len(varvargin) == 2:
            Dec = varvargin['Dec']
            AddProper = varvargin['AddProper']
        Obj = self.problem.CalObj(Dec)
        Con = self.problem.CalCon(Dec)
        N = Dec.shape[0]
        new_solution = np.array([SOLUTION() for i in range(N)])
        for i in range(N):
            new_solution[i].dec = np.array(Dec[i, :])
            new_solution[i].obj = Obj[i, :]
            new_solution[i].con = Con[i, :]
            if len(varvargin) > 1:
                new_solution[i].add = AddProper[i, :]
        # print("new_solution.dec.shape", new_solution.dec.shape)
        self.solutions = np.append(self.solutions, new_solution)
        self.problem.N += N
        return self


def getProperties(TempSolutionArray, key='dec'):
    def getPDec(TempSolutionArray):
        PDec = np.array(TempSolutionArray[0].dec)
        for i in range(1, len(TempSolutionArray)):
            PDec = np.append(PDec, TempSolutionArray[i].dec, axis=0)
        return PDec.reshape(len(TempSolutionArray), -1)

    def getPObj(TempSolutionArray):
        PObj = np.array(TempSolutionArray[0].obj)
        for i in range(1, len(TempSolutionArray)):
            PObj = np.append(PObj, TempSolutionArray[i].obj, axis=0)
        return PObj.reshape(len(TempSolutionArray), -1)

    def getPCon(TempSolutionArray):
        PCon = np.array(TempSolutionArray[0].con)
        for i in range(1, len(TempSolutionArray)):
            PCon = np.append(PCon, TempSolutionArray[i].con, axis=0)
        return PCon.reshape(len(TempSolutionArray), -1)

    def getPAdd(TempSolutionArray):
        PAdd = np.array(TempSolutionArray[0].add)
        for i in range(1, len(TempSolutionArray)):
            PAdd = np.append(PAdd, TempSolutionArray[i].add)
        return PAdd.reshape(len(TempSolutionArray), -1)

    try:
        if key == 'dec':
            return getPDec(TempSolutionArray)
        elif key == 'obj':
            return getPObj(TempSolutionArray)
        elif key == 'con':
            return getPCon(TempSolutionArray)
        elif key == 'add':
            return getPAdd(TempSolutionArray)
    except Exception as e:
        # print(e)
        err_print('can not find such property', e)
    finally:
        pass


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
