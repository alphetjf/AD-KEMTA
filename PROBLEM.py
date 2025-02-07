import numpy as np
import format_solution as SOL


class PROBLEM:
    #    This is the superclass of problems. An object of PROBLEM stores all the
    #    settings of the problem.
    #  PROBLEM properties:
    #    N               <read-only> population size
    #    M               <read-only> number of objectives
    #    D               <read-only> number of decision variables
    #    maxFE           <read-only> maximum number of function evaluations
    #    FE              <read-only> number of consumed function evaluations
    #    encoding        <read-only> encoding scheme of decision variables
    #    lower           <read-only> lower bound of decision variables
    #    upper           <read-only> upper bound of decision variables
    #    optimum         <read-only> optimal values of the problem
    #    PF              <read-only> image of Pareto front
    #    parameter       <read-only> other parameters of the problem
    def __init__(self) -> None:
        self.N = 100
        self.FE = 0
        # protected: 单下划线  即保护类型只能允许其本身与子类进行访问，不能用于 from module import *
        self.M = 0
        self.D = 0
        self.maxFE = 500
        self.encoding = 'real'
        self.lower = 0
        self.upper = 1
        self.optimim = 0
        self.PF = 0
        self.parameter = {}  # dict

    # PROBLEM methods:
    #   PROBLEM         <protected> the constructor, which sets all the properties specified by user
    #   Setting         <public>    default settings of the problem
    #   Initialization 	<public>    generate initial solutions
    #   CalDec          <public>    repair invalid solutions
    #   CalObj          <public>    calculate the objective values of solutions
    #   CalCon          <public>    calculate the constraint violations of solutions
    #   GetOptimum      <public>    generate the optimums of the problem
    #   GetPF          	<public>    generate the image of Pareto front
    #   DrawDec         <public>    display a population in the decision space
    #   DrawObj         <public>    display a population in the objective space
    #   Current         <static>    get or set the current PROBLEM object
    #   ParameterSet	<protected>	obtain the parameters of the problem
    def _Building(self, **varargin):
        # varargin: (M==5,D==10)  'N','M','D','maxFE','parameter'
        for key, values in varargin.items():
            if key == 'M':
                self.M = values
            elif key == 'D':
                self.D = values
            elif key == 'maxFE':
                self.maxFE = values
            elif key == 'parameter':
                self.parameter = values
            elif key == 'N':
                self.N = values
        self.Setting()
        self.optimum = self.GetOptimum(500)
        self.PF = self.GetPF()

    def Setting(self):
        # Setting - Default settings of the problem.
        #   This function is expected to be implemented in each subclass of
        #   PROBLEM, which will be called automatically.
        pass

    def Initialization(self, *, N=100):
        if self.encoding == 'binary':
            pass
        elif self.encoding == 'real':
            PopDec = np.random.rand(N, self.D) * np.tile(self.upper - self.lower, (N, 1)) + self.lower  # 数组元素对应相乘相加
        # Problem = self.Current()
        # print(Problem.N)
        # print(self.N)
        Population = SOL.SOLUTIONSET(self)
        Population.building(Decs=PopDec)
        return Population

    def CalDec(self, PopDec):
        # CalDec - Repair invalid solutions.
        #    Dec = obj.CalDec(Dec) repairs the invalid (not infeasible)
        #    decision variables of Dec.
        if self.encoding == 'real':
            Temp = np.tile(self.upper, (np.size(PopDec, axis=0), 1))
            for i in range(len(PopDec)):
                PopDec[i] = np.min(np.vstack((PopDec[i], Temp[i])), axis=0)
            Temp2 = np.tile(self.lower, (np.size(PopDec, axis=0), 1))
            for i in range(len(PopDec)):
                PopDec[i] = np.max(np.vstack((PopDec[i], Temp2[i])), axis=0)
        return PopDec

    def CalObj(self, PopDec):
        # CalObj - Calculate the objective values of solutions.
        #   Obj = obj.CalObj(Dec) returns the objective values of Dec.
        PopObj = np.zeros((np.size(PopDec, axis=0), self.M))
        return PopObj

    def CalCon(self, PopDec):
        # CalCon - Calculate the constraint violations of solutions.
        #   Con = obj.CalCon(Dec) returns the constraint violations of Dec.
        PopCon = np.zeros((np.size(PopDec, axis=0), 1))
        return PopCon

    def CalAdd(self, Adds):
        pass

    def GetOptimum(self, *varargin):
        # GetOptimum - Generate the optimums of the problem.
        #   R = obj.GetOptimum(N) returns N optimums of the problem for
        #   metric calculation.
        # N = varargin[0]
        if self.M > 1:
            R = np.ones((1, self.M))
        else:
            R = 0
        return R

    def GetPF(self):
        # GetPF - Generate the image of Pareto front.
        #   R = obj.GetPF() returns the image of Pareto front for objective
        #   visualization.
        R = []
        return R

    def _ParameterSet(self, **varargin):
        # ParameterSet - Obtain the parameters of the problem.
        #   [p1,p2,...] = obj.ParameterSet(v1,v2,...) sets the values of
        #   parameters p1, p2, ..., where each parameter is set to the
        #   value given in obj.parameter if obj.parameter is specified, and
        #   set to the value given in v1, v2, ... otherwise.
        #   Example:
        #       [p1,p2,p3] = obj.ParameterSet(1,2,3)
        # if len(varargin) > 0:
        #     self._parameter.update(varargin)  # 把字典dict2的键/值对更新到dict里, 相同键值会更新，否则增加新键值
        # varargout = []
        # for key in varargin.keys():
        #     varargout.append(self._parameter.get(key))
        # return varargout
        if len(varargin) > 0:
            for key, value in varargin.items():
                if key == "MaxFEs":
                    self.maxFE = value
                    break
            return value

    def Current(self):
        # Current - Get or set the current PROBLEM object.
        #   Pro = PROBLEM.Current() returns the current PROBLEM object.
        #   PROBLEM.Current(Pro) sets the current PROBLEM object to Pro and
        #   sets Pro.evaluated to 0.
        #   Example:
        #       Problem = PROBLEM.Current()

        import weakref
        wref = weakref.ref(self)  # 弱引用: 不影响垃圾回收的计数
        return wref
