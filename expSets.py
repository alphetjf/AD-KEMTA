import os
import sys
import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
BenchmarksPath = os.path.join(curPath, "Benchmark")
print(BenchmarksPath)

def parameterSet(test):
    D = int(test[test.find('\\')+1:test.find('D')])
    M = int(test[test.find('D')+1:test.find('M')])
    K = int(test[test.find('M\\')+2:test.find('K')])
    if test[:test.find('\\')] == 'deb2dk':
        from Benchmark.DEBDK import DEB2DK
        pro = DEB2DK()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        elif K == 9:
            pro.Setting(D, M, 9)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=1000*100)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=1000*100)
    elif test[:test.find('\\')] == 'do2dk':
        from Benchmark.DO2DK import DO2DK
        pro = DO2DK()
        if K == 1:
            pro.Setting(D, M, 1, 1)
        elif K == 2:
            pro.Setting(D, M, 2, 1)
        elif K == 3:
            pro.Setting(D, M, 3, 1)
        elif K == 4:
            pro.Setting(D, M, 4, 1)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
    elif test[:test.find('\\')] == 'ckp':
        from Benchmark.CKP import CKP
        pro = CKP()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=1000*100)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=1000*100)
    elif test[:test.find('\\')] == 'deb3dk':
        from Benchmark.DEBDK import DEB3DK
        pro = DEB3DK()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=1000*100)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=1000*100)
    elif test[:test.find('\\')] == 'pmop1':
        from Benchmark.PMOP import PMOP1
        pro = PMOP1()
        pro.Setting(D, M, 4, 1, -1, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop2':
        from Benchmark.PMOP import PMOP2
        pro = PMOP2()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop3':
        from Benchmark.PMOP import PMOP3
        pro = PMOP3()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop4':
        from Benchmark.PMOP import PMOP4
        pro = PMOP4()
        pro.Setting(D, M, 6, 1, -1, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=10000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=10000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=10000*100)
    elif test[:test.find('\\')] == 'pmop5':
        from Benchmark.PMOP import PMOP5
        pro = PMOP5()
        pro.Setting(D, M, 1, 1, 2, 1, 12)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=10000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=10000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=10000*100)
    elif test[:test.find('\\')] == 'pmop6':
        from Benchmark.PMOP import PMOP6
        pro = PMOP6()
        pro.Setting(D, M, 2, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop7':
        from Benchmark.PMOP import PMOP7
        pro = PMOP7()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop8':
        from Benchmark.PMOP import PMOP8
        pro = PMOP8()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop9':
        from Benchmark.PMOP import PMOP9
        pro = PMOP9()
        pro.Setting(D, M, 2, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop10':
        from Benchmark.PMOP import PMOP10
        pro = PMOP10()
        pro.Setting(D, M, 1, 1, 2, 1, 12)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
    elif test[:test.find('\\')] == 'pmop11':
        from Benchmark.PMOP import PMOP11
        pro = PMOP11()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
    elif test[:test.find('\\')] == 'pmop12':
        from Benchmark.PMOP import PMOP12
        pro = PMOP12()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
    elif test[:test.find('\\')] == 'pmop13':
        from Benchmark.PMOP import PMOP13
        pro = PMOP13()
        pro.Setting(D, M, 2, 1, -2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=3000*100)
    elif test[:test.find('\\')] == 'pmop14':
        from Benchmark.PMOP import PMOP14
        pro = PMOP14()
        pro.Setting(D, M, 2, 1, -1, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=5000*100)
    return pro, maxFEs

def parameterSet_nontrivial(test):
    D = int(test[test.find('\\')+1:test.find('D')])
    M = int(test[test.find('D')+1:test.find('M')])
    K = int(test[test.find('M\\')+2:test.find('K')])
    if test[:test.find('\\')] == 'deb2dk':
        from Benchmark.DEBDK import DEB2DK
        pro = DEB2DK()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        elif K == 9:
            pro.Setting(D, M, 9)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=300)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'do2dk':
        from Benchmark.DO2DK import DO2DK
        pro = DO2DK()
        if K == 1:
            pro.Setting(D, M, 1, 1)
        elif K == 2:
            pro.Setting(D, M, 2, 1)
        elif K == 3:
            pro.Setting(D, M, 3, 1)
        elif K == 4:
            pro.Setting(D, M, 4, 1)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=300)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'ckp':
        from Benchmark.CKP import CKP
        pro = CKP()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=300)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'deb3dk':
        from Benchmark.DEBDK import DEB3DK
        pro = DEB3DK()
        if K == 1:
            pro.Setting(D, M, 1)
        elif K == 2:
            pro.Setting(D, M, 2)
        elif K == 3:
            pro.Setting(D, M, 3)
        elif K == 4:
            pro.Setting(D, M, 4)
        if D == 10:
            maxFEs = pro._ParameterSet(MaxFEs=300)
        elif D == 30:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop1':
        from Benchmark.PMOP import PMOP1
        pro = PMOP1()
        pro.Setting(D, M, 4, 1, -1, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop2':
        from Benchmark.PMOP import PMOP2
        pro = PMOP2()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop3':
        from Benchmark.PMOP import PMOP3
        pro = PMOP3()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop4':
        from Benchmark.PMOP import PMOP4
        pro = PMOP4()
        pro.Setting(D, M, 6, 1, -1, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop5':
        from Benchmark.PMOP import PMOP5
        pro = PMOP5()
        pro.Setting(D, M, 1, 1, 2, 1, 12)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop6':
        from Benchmark.PMOP import PMOP6
        pro = PMOP6()
        pro.Setting(D, M, 2, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop7':
        from Benchmark.PMOP import PMOP7
        pro = PMOP7()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop8':
        from Benchmark.PMOP import PMOP8
        pro = PMOP8()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop9':
        from Benchmark.PMOP import PMOP9
        pro = PMOP9()
        pro.Setting(D, M, 2, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop10':
        from Benchmark.PMOP import PMOP10
        pro = PMOP10()
        pro.Setting(D, M, 1, 1, 2, 1, 12)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop11':
        from Benchmark.PMOP import PMOP11
        pro = PMOP11()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop12':
        from Benchmark.PMOP import PMOP12
        pro = PMOP12()
        pro.Setting(D, M, 4, 1, 2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop13':
        from Benchmark.PMOP import PMOP13
        pro = PMOP13()
        pro.Setting(D, M, 2, 1, -2, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    elif test[:test.find('\\')] == 'pmop14':
        from Benchmark.PMOP import PMOP14
        pro = PMOP14()
        pro.Setting(D, M, 2, 1, -1, 1)
        if M == 3:
            maxFEs = pro._ParameterSet(MaxFEs=400)
        elif M == 5:
            maxFEs = pro._ParameterSet(MaxFEs=500)
        elif M == 8:
            maxFEs = pro._ParameterSet(MaxFEs=500)
    return pro, maxFEs

def realReferenceData(test, D, M, K, needKnee=True):
    global BenchmarksPath
    sys.path.append(curPath)
    print(test)
    if test.lower() == 'ckp':
        from Benchmark.CKP import CKP
        prob = CKP()
        prob.Setting(D, M, K)

    elif test.lower() == 'deb2dk':
        from Benchmark.DEBDK import DEB2DK
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

    elif test.lower() == 'pmop2':
        from Benchmark.PMOP import PMOP2
        prob = PMOP2()
        prob.Setting(D, M, 4, 1, 2, 1)

    elif test.lower() == 'pmop3':
        from Benchmark.PMOP import PMOP3
        prob = PMOP3()
        prob.Setting(D, M, 4, 1, 2, 1)

    elif test.lower() == 'pmop4':
        from Benchmark.PMOP import PMOP4
        prob = PMOP4()
        prob.Setting(D, M, 6, 1, -1, 1)

    elif test.lower() == 'pmop5':
        from Benchmark.PMOP import PMOP5
        prob = PMOP5()
        prob.Setting(D, M, 1, 1, 2, 1, 12)

    elif test.lower() == 'pmop6':
        from Benchmark.PMOP import PMOP6
        prob = PMOP6()
        prob.Setting(D, M, 2, 1, 2, 1)

    elif test.lower() == 'pmop7':
        from Benchmark.PMOP import PMOP7
        prob = PMOP7()
        prob.Setting(D, M, 4, 1, 2, 1)

    elif test.lower() == 'pmop8':
        from Benchmark.PMOP import PMOP8
        prob = PMOP8()
        prob.Setting(D, M, 4, 1, 2, 1)

    elif test.lower() == 'pmop9':
        from Benchmark.PMOP import PMOP9
        prob = PMOP9()
        prob.Setting(D, M, 2, 1, 2, 1)

    elif test.lower() == 'pmop10':
        from Benchmark.PMOP import PMOP10
        prob = PMOP10()
        prob.Setting(D, M, 1, 1, 2, 1, 12)

    elif test.lower() == 'pmop11':
        from Benchmark.PMOP import PMOP11
        prob = PMOP11()
        prob.Setting(D, M, 4, 1, 2, 1)

    elif test.lower() == 'pmop12':
        from Benchmark.PMOP import PMOP12
        prob = PMOP12()
        prob.Setting(D, M, 4, 1, 2, 1)

    elif test.lower() == 'pmop13':
        from Benchmark.PMOP import PMOP13
        prob = PMOP13()
        prob.Setting(D, M, 2, 1, -2, 1)

    elif test.lower() == 'pmop14':
        from Benchmark.PMOP import PMOP14
        prob = PMOP14()
        prob.Setting(D, M, 2, 1, -1, 1)
    PF = prob.GetPF(300)
    if needKnee:
        if test.lower() == 'do2dk':
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\regions.npy")
        elif test.lower() == 'deb3dk':
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")"+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")"+"\\regions.npy")
        elif 'pmop' in test.lower():
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K)+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K)+"\\regions.npy")
        else:
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\regions.npy")
        return prob, PF, regions, knees
    else:
        return prob, PF, regions, knees
