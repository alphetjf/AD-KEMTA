import numpy as np
import os
import csv
from scipy.stats import kendalltau
from Metric import IGD
import scipy.stats as ss
import matplotlib.pyplot as plt
from utils.UniformPoint import generatorPoints, MixtureUniformDesign
np.set_printoptions(suppress=True)


curPath = os.path.abspath(os.path.dirname(__file__))
BenchmarksPath = os.path.join(curPath, "Benchmark")
print(BenchmarksPath)
ExperimentalDataPath = os.path.join(curPath, "Data")
print(ExperimentalDataPath)


def err_print(msg, original_line=None):
    print('ERROR  ' * 10)
    print(msg)
    if original_line:
        print(original_line)
    print('ERROR  ' * 10)
    exit(1)

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

def realReferenceData(test, D, M, K):
    # print(test)
    if test.lower() == 'ckp':
        from Benchmark.CKP import CKP

        prob = CKP()
        prob.Setting(D, M, K)

    elif test.lower() == 'deb2dk':
        from Benchmark.DEBDK import DEB2DK, DEB3DK

        prob = DEB2DK()
        prob.Setting(D, M, K)

    elif test.lower() == 'deb3dk':
        from Benchmark.DEBDK import DEB3DK

        prob = DEB3DK()
        prob.Setting(D, M, K)

    elif test.lower() == 'do2dk':
        from Benchmark.DO2DK import DO2DK

        prob = DO2DK()
        prob.Setting(D, M, K, 1)
    
    elif test.lower() == 'pmop1':
        from Benchmark.PMOP import PMOP1
        prob = PMOP1()
        prob.Setting(D, M, 4, 1, -1, 1)
        K = 4
    elif test.lower() == 'pmop2':
        from Benchmark.PMOP import PMOP2
        prob = PMOP2()
        prob.Setting(D, M, 4, 1, 2, 1)
        K = 4
    elif test.lower() == 'pmop3':
        from Benchmark.PMOP import PMOP3
        prob = PMOP3()
        prob.Setting(D, M, 4, 1, 2, 1)
        K = 4
    elif test.lower() == 'pmop4':
        from Benchmark.PMOP import PMOP4
        prob = PMOP4()
        prob.Setting(D, M, 6, 1, -1, 1)
        K = 6
    elif test.lower() == 'pmop5':
        from Benchmark.PMOP import PMOP5
        prob = PMOP5()
        prob.Setting(D, M, 1, 1, 2, 1, 12)
        K = 1
    elif test.lower() == 'pmop6':
        from Benchmark.PMOP import PMOP6
        prob = PMOP6()
        prob.Setting(D, M, 2, 1, 2, 1)
        K = 2
    elif test.lower() == 'pmop7':
        from Benchmark.PMOP import PMOP7
        prob = PMOP7()
        prob.Setting(D, M, 4, 1, 2, 1)
        K = 4
    elif test.lower() == 'pmop8':
        from Benchmark.PMOP import PMOP8
        prob = PMOP8()
        prob.Setting(D, M, 4, 1, 2, 1)
        K = 4
    elif test.lower() == 'pmop9':
        from Benchmark.PMOP import PMOP9
        prob = PMOP9()
        prob.Setting(D, M, 2, 1, 2, 1)
        K = 2
    elif test.lower() == 'pmop10':
        from Benchmark.PMOP import PMOP10
        prob = PMOP10()
        prob.Setting(D, M, 1, 1, 2, 1, 12)
        K = 1
    elif test.lower() == 'pmop11':
        from Benchmark.PMOP import PMOP11
        prob = PMOP11()
        prob.Setting(D, M, 4, 1, 2, 1)
        K = 4
    elif test.lower() == 'pmop12':
        from Benchmark.PMOP import PMOP12
        prob = PMOP12()
        prob.Setting(D, M, 4, 1, 2, 1)
        K = 4
    elif test.lower() == 'pmop13':
        from Benchmark.PMOP import PMOP13
        prob = PMOP13()
        prob.Setting(D, M, 2, 1, -2, 1)
        K = 2
    elif test.lower() == 'pmop14':
        from Benchmark.PMOP import PMOP14
        prob = PMOP14()
        prob.Setting(D, M, 2, 1, -1, 1)
        K = 2
    PF = prob.GetPF(500)
    global BenchmarksPath
    if test.lower() == 'do2dk':
        regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\regions.npy")
        PF_knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\PF_knees.npy")
    elif test.lower() == 'deb3dk' and K > 1:
        regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")"+"\\regions.npy")
        PF_knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")"+"\\PF_knees.npy")
    elif test.lower()[:-1] == 'pmop' or test.lower()[:-2] == 'pmop':
        regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K)+"\\regions.npy")
        PF_knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K)+"\\PF_knees.npy")
    else:
        regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\regions.npy")
        PF_knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\PF_knees.npy")
    return PF, regions, PF_knees

def evasOfInstances(test, D):
    if test.lower() == 'ckp' or test.upper() == 'CKP':
        if D == 10:
            return 200
        elif D == 30:
            return 400
    if test.lower() == 'do2dk' or test.upper() == 'DO2DK':
        if D == 10:
            return 200
        elif D == 30:
            return 400
    if test.lower() == 'deb2dk' or test.upper() == 'DEB2DK':
        if D == 10:
            return 200
        elif D == 30:
            return 400
    if test.lower() == 'deb3dk' or test.upper() == 'DEB3DK':
        if D == 10:
            return 350
        elif D == 30:
            return 500
    return None

def collation(pathfile, problem_list, check_D):
    print(problem_list)
    problems = os.listdir(pathfile)
    if len(problems) == 0:
        err_print('No such result')
        return None
    else:
        print(problems)
        # IGDs = dict.fromkeys(problem_list)
        IGDs = {}
        nObjs = {}
        for p in problem_list:
            ind = problems.index(p)
            filePath = os.path.join(pathfile, problems[ind])
            dimensions = os.listdir(filePath)
            for dim in dimensions:
                x_D = int(dim[:dim.find('D')])
                x_M = int(dim[dim.find('D')+1:dim.find('M')])
                if x_D == check_D:
                    dim_path = os.path.join(filePath, dim)
                    knees = os.listdir(dim_path)
                    # topFEs = evasOfInstances(p, x_D)
                    for K in knees:
                        knee_path = os.path.join(dim_path, K)
                        PF, realRegions, _ = realReferenceData(p, x_D, x_M, int(K[0]))
                        runs = os.listdir(knee_path)
                        tmpIGDs = []
                        tmpObjs = {}
                        ii = 0
                        for r in runs:
                            runPath = os.path.join(knee_path, r)
                            filesName = os.listdir(runPath)
                            topFEs = int(filesName[-1])
                            path = os.path.join(runPath, str(topFEs))
                            # print(filePath)
                            Objs = np.load(path+"\\Objs.npy")
                            FrontNo, _ = ENS_SS_NDSort(Objs, np.shape(Objs)[0])
                            index = np.where(FrontNo == 1)[0]
                            nondominated_Objs = Objs[index, :]
                            tmpObjs[ii] = nondominated_Objs.copy()
                            ii += 1
                            tmpIGDs.append(IGD(realRegions, nondominated_Objs))
                        IGDs[p+'_'+K[:2]] = tmpIGDs
                        nObjs[p+'_'+K[:2]] = tmpObjs
        # print(IGDs)
        return IGDs, nObjs

def collationOfPlatemo(pathfile, problem_list, check_D):
    print(problem_list)
    problems = os.listdir(pathfile)
    print(problems)
    if len(problems) == 0:
        err_print('No such result')
        return None
    else:
        print(problems)
        # IGDs = dict.fromkeys(problem_list)
        IGDs = {}
        nObjs = {}
        for p in problem_list:
            p = p.upper()
            ind = problems.index(p)
            filePath = os.path.join(pathfile, problems[ind])

            dimensions = os.listdir(filePath)
            for dim in dimensions:
                x_D = int(dim[:dim.find('D')])
                x_M = int(dim[dim.find('D')+1:dim.find('M')])
                if x_D == check_D:
                    dim_path = os.path.join(filePath, dim)
                    knees = os.listdir(dim_path)

                    for K in knees:
                        knee_path = os.path.join(dim_path, K)
                        _, realRegions, _ = realReferenceData(p, x_D, x_M, int(K[0]))
                        runs = os.listdir(knee_path)
                        tmpIGDs = []
                        tmpObjs = {}
                        ii = 0
                        for r in runs:
                            runPath = os.path.join(knee_path, r)
                            Objs = np.loadtxt(runPath)
                            # print(filePath)
                            FrontNo, _ = ENS_SS_NDSort(Objs, np.shape(Objs)[0])
                            index = np.where(FrontNo == 1)[0]
                            nondominated_Objs = Objs[index, :]
                            tmpIGDs.append(IGD(realRegions, nondominated_Objs))
                            tmpObjs[ii] = nondominated_Objs.copy()
                            ii += 1
                        IGDs[p+'_'+K[:2]] = tmpIGDs
                        nObjs[p+'_'+K[:2]] = tmpObjs
        return IGDs, nObjs

def printInExcel(IGDs, algorithm):
    if len(IGDs) == 0:
        err_print('IGDs is empty')
    else:
        with open(algorithm+'.csv', 'w', encoding='utf-8-sig', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in IGDs.items():
                row = [key]
                mu = np.mean(value)
                std = np.std(value)
                row.append('{:.4e}'.format(mu) + '(' + '{:.2e}'.format(std)+')')
                writer.writerow(row)

def metricAssignedAlg(algName, problem, decisionNum, experimentalData_path):
    if algName not in ['KTA2', 'PBRVEA', 'SMSEGO']:
        IGDs, nObjs = collation(experimentalData_path, problem, decisionNum)
    else:
        IGDs, nObjs = collationOfPlatemo(experimentalData_path, problem, decisionNum)
    return IGDs, nObjs


if __name__ == '__main__':
    print('current folder:', curPath)
    print('experimental data folder:', ExperimentalDataPath)
    # ExperimentalDataPath = os.path.join(ExperimentalDataPath, 'AblationStudy')

    '''Print data in Excel format'''
    # alg1s = ['SOI', 'KPITU', 'NBI-variant', 'KPNSGA-II', 'LBD-MOEA', 'ParEGO', 'ParEGO-error']
    alg = 'Alg_MFEA_GP'
    experimentalData_path = os.path.join(ExperimentalDataPath, alg)

    problem = ['do2dk', 'ckp', 'deb2dk', 'deb3dk']
    # problem = ['pmop1','pmop2','pmop3','pmop4','pmop5','pmop6','pmop7','pmop8', 'pmop9','pmop10','pmop11','pmop12','pmop13','pmop14']
    D = 30

    IGDs, nObjs = metricAssignedAlg(alg, problem, D, experimentalData_path)
    printInExcel(IGDs, alg)

    wilcon = False
    if wilcon is True:
        alg1 = 'Alg_MFEA_GP'
        experimentalData_path1 = os.path.join(ExperimentalDataPath, alg1)
        IGD1, nObj1 = metricAssignedAlg(alg1, problem, D, experimentalData_path1)
        alg2 = 'Alg_MFEA_GP_moreReference_3'
        experimentalData_path2 = os.path.join(ExperimentalDataPath, alg2)
        IGD2, nObj2 = metricAssignedAlg(alg2, problem, D, experimentalData_path2)
        print(IGD1.keys())
        keyName = 'ckp_4K'
        for keyName, _ in IGD1.items():
            print(keyName)
            igdValues1 = IGD1[keyName]
            igdValues2 = IGD2[keyName]
            # print(igdValues1, igdValues2)
            print(np.mean(igdValues1), np.mean(igdValues2))
            # print(ss.wilcoxon(igdValues1, igdValues2, zero_method='wilcox', alternative='two-sided'), mode='approx')
            print(ss.mannwhitneyu(igdValues1, igdValues2, alternative='two-sided'))
            print(ss.ranksums(igdValues1, igdValues2, alternative='two-sided'))  

    plot = False
    if plot is True:
        print(nObjs.keys())
        print(IGDs)
        if alg not in ['KTA2']:
            keyName = problem[0].lower()+'_4A'
        else:
            keyName = problem[0].upper()+'_4A'
        igdValues = IGDs[keyName]
        rank = np.argsort(igdValues)
        PF, regions, knees = realReferenceData(problem[0], D, D-9, 3)
        Obj = nObjs[keyName][rank[int(max(rank)/2)]]
        # plt.figure()
        # plt.scatter(PF[:, 0], PF[:, 1], marker='.', c='blue')
        # plt.scatter(Obj[:, 0], Obj[:, 1], marker='p', c='r')
        # plt.show()
        import matplotlib.pyplot as plt
        PF = np.unique(PF, axis=0)
        M = D-9
        if M == 2:
            fig = plt.figure()
            # plt.fill_between(ePF[:, 0], test_y + uncertainty, test_y - uncertainty, alpha=0.4)
            # plt.scatter(ePF[:, 0], ePF[:, 1], marker=".", c='g', alpha=0.6)
            plt.scatter(PF[:, 0], PF[:, 1], marker='.', c='blue')
            plt.scatter(knees[:, 0], knees[:, 1], marker='x', c='orange')
            plt.scatter(Obj[:, 0], Obj[:, 1], marker='p', c='red')
            plt.show()
        elif M == 3:
            fig = plt.figure(figsize=(14, 10), dpi=50, facecolor='w', edgecolor='k')
            # ax = plt.axes(projection='3d')
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot(PF[:, 0], PF[:, 1], PF[:, 2], marker='.', alpha=0.5, label='$PF$')
            ax.scatter(Obj[:, 0], Obj[:, 1], Obj[:, 2], marker='p', c='r')
            ax.scatter(regions[:, 0], regions[:, 1], regions[:, 2], marker='s', c='black')
            ax.legend(fontsize=24, loc=0)
            ax.tick_params(labelsize=24)
            ax.set_xlabel("$f_1$", fontsize=28)
            ax.set_ylabel("$f_2$", fontsize=28)
            ax.set_zlabel("$f_2$", fontsize=28)
            plt.show()
        elif M >= 4:
            from utils.visualizationInManyObjective import reductionDimensionObjectives

            tmp = np.append(Obj[:, :], PF, axis=0)
            Zmin = np.min(tmp, axis=0)
            Zmax = np.max(tmp, axis=0)
            dp, dd = reductionDimensionObjectives(Obj[:, :], Zmin, Zmax)
            dpf, ddf = reductionDimensionObjectives(PF, Zmin, Zmax)
            dpk, ddk = reductionDimensionObjectives(knees, Zmin, Zmax)
            plt.scatter(dpf, ddf, marker='.', c='blue')
            plt.scatter(dpk, ddk, marker='s', alpha=0.7, c='red')
            plt.scatter(dp, dd, marker='p', c='orange')
            plt.show()