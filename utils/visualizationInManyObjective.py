import numpy as np
import os
import sys

def ASR(Objs, Zmin, Zmax):
    N, _ = np.shape(Objs)
    ASR = np.sum((np.tile(Zmax, (N, 1)) - Objs) / np.tile(Zmax-Zmin, (N, 1)), axis=1)
    return ASR

def MMD(Objs, Zmin, Zmax):
    N, _ = np.shape(Objs)
    MMD = np.sum((Objs - np.tile(Zmin, (N, 1))) / np.tile(Zmax-Zmin, (N, 1)), axis=1)
    return MMD

def reductionDimensionObjectives(Objs, Zmin=None, Zmax=None):
    from sklearn.metrics.pairwise import cosine_similarity
    N, M = np.shape(Objs)
    if Zmin is None:
        Zmin = np.min(Objs, axis=0)
    if Zmax is None:
        Zmax = np.max(Objs, axis=0)
    MMDnd = MMD(Zmax.reshape(1, -1), Zmin, Zmax)
    # print(MMDnd)
    MMDx = MMD(Objs, Zmin, Zmax)
    # index of convergence performance
    dp = (np.tile(MMDnd, (N,)) - MMDx) * np.sqrt(M) / M
    # index of diversity performance
    tObjs = (Objs - np.tile(Zmin, (N, 1))) / np.tile(Zmax-Zmin, (N, 1))
    Wc = np.ones((1, M)) * (1/M)
    angle = np.arccos(cosine_similarity((tObjs-np.tile(Zmax, (N, 1))), -Wc))
    dd = np.tan(angle.flatten()) * dp
    return dp, dd

curPath = os.path.abspath(os.path.dirname(__file__))
BenchmarksPath = os.path.join(curPath, "Benchmark")
print(BenchmarksPath)

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
    PF = prob.GetPF(150)
    if needKnee:
        if test.lower() == 'do2dk':
            print(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1")
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"S1"+"\\regions.npy")
        elif test.lower() == 'deb3dk':
            print(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")")
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")"+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"(knee"+str(K**2)+")"+"\\regions.npy")
        elif 'pmop' in test.lower():
            print(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K))
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K)+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"A"+str(K)+"\\regions.npy")
        else:
            print(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K))
            knees = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\PF_knees.npy")
            regions = np.load(BenchmarksPath+"\\"+test.upper()+"\\"+str(D)+'D'+str(M)+'M\\'+"K"+str(K)+"\\regions.npy")
        return prob, PF, regions, knees
    else:
        return prob, PF, knees, regions

benchmarks = [
    'deb2dk\\10D2M\\1K', 'deb2dk\\10D2M\\2K', 'deb2dk\\10D2M\\3K', 'deb2dk\\10D2M\\4K'
    ,'do2dk\\10D2M\\1K', 'do2dk\\10D2M\\2K', 'do2dk\\10D2M\\3K', 'do2dk\\10D2M\\4K'
    ,'ckp\\10D2M\\1K', 'ckp\\10D2M\\2K', 'ckp\\10D2M\\3K', 'ckp\\10D2M\\4K'
    ,'deb3dk\\10D3M\\1K', 'deb3dk\\10D3M\\2K(knee4)', 'deb3dk\\10D3M\\3K(knee9)'
    ]

benchmarks30 = [
    'deb2dk\\30D2M\\1K', 'deb2dk\\30D2M\\2K', 'deb2dk\\30D2M\\3K', 'deb2dk\\30D2M\\4K',
    'ckp\\30D2M\\1K', 'ckp\\30D2M\\2K', 'ckp\\30D2M\\3K', 'ckp\\30D2M\\4K',
    'do2dk\\30D2M\\1K', 'do2dk\\30D2M\\2K', 'do2dk\\30D2M\\3K', 'do2dk\\30D2M\\4K',
    'deb3dk\\30D3M\\1K', 'deb3dk\\30D3M\\2K(knee4)', 'deb3dk\\30D3M\\3K(knee9)'
]

benchmarks_Many = [
    'pmop1\\12D3M\\4A', 'pmop1\\14D5M\\4A', 'pmop1\\17D8M\\4A',
    'pmop2\\12D3M\\4A', 'pmop2\\14D5M\\4A', 'pmop2\\17D8M\\4A',
    'pmop3\\12D3M\\4A', 'pmop3\\14D5M\\4A', 'pmop3\\17D8M\\4A',
    'pmop4\\12D3M\\6A', 'pmop4\\14D5M\\6A', 'pmop4\\17D8M\\6A',
    'pmop5\\12D3M\\1A', 'pmop5\\14D5M\\1A', 'pmop5\\17D8M\\1A',
    'pmop6\\12D3M\\2A', 'pmop6\\14D5M\\2A', 'pmop6\\17D8M\\2A',
    'pmop7\\12D3M\\4A', 'pmop7\\14D5M\\4A', 'pmop7\\17D8M\\4A',
    'pmop8\\12D3M\\4A', 'pmop8\\14D5M\\4A', 'pmop8\\17D8M\\4A',
    'pmop9\\12D3M\\2A', 'pmop9\\14D5M\\2A', 'pmop9\\17D8M\\2A',
    'pmop10\\12D3M\\1A', 'pmop10\\14D5M\\1A', 'pmop10\\17D8M\\1A',
    'pmop11\\12D3M\\4A', 'pmop11\\14D5M\\4A', 'pmop11\\17D8M\\4A',
    'pmop12\\12D3M\\4A', 'pmop12\\14D5M\\4A', 'pmop12\\17D8M\\4A',
    'pmop13\\12D3M\\2A', 'pmop13\\14D5M\\2A', 'pmop13\\17D8M\\2A',
    'pmop14\\12D3M\\2A', 'pmop14\\14D5M\\2A', 'pmop14\\17D8M\\2A'
]

if __name__ == "__main__":
    num_obj = 2
    # num_variable = num_obj + 9
    num_variable = 5
    num_knee = 1
    test = 'deb2dk'
    prob, pf, knees, regions = realReferenceData(test, M=num_obj, D=num_variable, K=num_knee)
    print(knees,np.shape(knees))
    knees_cluster = np.empty((num_knee, num_obj))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_knee).fit(knees)
    for wi in np.unique(kmeans.labels_):
        index = np.where(kmeans.labels_==wi)[0]
        knees_cluster[wi, :] = np.mean(knees[index, :], axis=0)
    dp, dd = reductionDimensionObjectives(pf)
    print(np.shape(dp),np.shape(dd))
    dpk, ddk = reductionDimensionObjectives(knees_cluster, Zmin=np.min(pf, axis=0), Zmax=np.max(pf, axis=0))
    print(np.shape(dpk),np.shape(ddk))

    # from utils.KPI import KneePointIdentification_LocalScalarFunc, NBI_variant, local_alpha_dominance, AssociationWeights_acuteAngle
    # from utils.UniformPoint import ReferenceVectorGenerator
    # W, Nw = ReferenceVectorGenerator(divisionOuter=1, divisionInner=3, M=num_obj)
    # print('the number of reference vectors:{}'.format(Nw))
    # d = 0.5
    # kps = local_alpha_dominance(pf, vectors=W, alpha=0.6)
    # knees2 = pf[kps,:]
    # dpk2, ddk2 = reductionDimensionObjectives(knees2, Zmin=np.min(pf, axis=0), Zmax=np.max(pf, axis=0))
    # print(np.shape(dpk),np.shape(ddk))

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    config = {
        "font.family":'serif',
        "font.size": 24,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    rcParams.update(config)
    if num_obj ==2:
        fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(pf[:,0], pf[:,1], marker='o', s=150, c='none', edgecolors='b', alpha=1, label='帕累托前沿')
        # plt.scatter(knees[:,0], knees[:,1], marker='.', s=200, c='r', alpha=0.5, label='拐点')
        plt.scatter(knees_cluster[:,0], knees_cluster[:,1], marker='o', s=150, c='r',edgecolors='r',alpha=1, label='拐点')
        # plt.scatter(knees2[:,0], knees2[:,1], marker='.', s=200, c='g', alpha=0.5, label='拐点')
        # labelss = plt.legend(fontsize=24, loc=0, frameon=False).get_texts()
        # [label.set_fontname('serif') for label in labelss]
        # [label.set_fontsize(24) for label in labelss]
        plt.xlabel('$f_1$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.ylabel('$f_2$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.xticks(color='black', fontsize=22)
        plt.yticks(color='black', fontsize=22)
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(dp, dd, marker='o', s=150, c='none', edgecolors='b', alpha=1, label='帕累托前沿')
        plt.scatter(dpk, ddk, marker='o', s=150, c='r',edgecolors='r', alpha=1, label='拐点')
        # plt.scatter(dpk2, ddk2, marker='.', s=200, c='g', alpha=0.5, label='拐点')
        # labelss = plt.legend(fontsize=24, loc=0, frameon=False).get_texts()
        # [label.set_fontname('serif') for label in labelss]
        # [label.set_fontsize(24) for label in labelss]
        plt.xlabel('$d_p$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.ylabel('$d_d$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.xticks(color='black', fontsize=22)
        plt.yticks(color='black', fontsize=22)
        plt.legend()
        plt.show()
    elif num_obj==3:
        fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        ax1 = plt.axes(projection='3d')
        fig.patch.set_alpha(0.)
        ax1.scatter3D(pf[:,0], pf[:,1], pf[:,2], marker='.', s=150, c='black', alpha=1, label='帕累托前沿')
        ax1.scatter3D(knees_cluster[:,0], knees_cluster[:,1],  knees_cluster[:,2], marker='o', s=150, c='r',edgecolors='r',alpha=1, label='拐点')
        plt.xlabel('$f_1$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.ylabel('$f_2$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.xticks(color='black', fontsize=22)
        plt.yticks(color='black', fontsize=22)
        plt.legend()
        plt.show()
    
        fig = plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(dp, dd, marker='.', s=150, c='black', alpha=1, label='帕累托前沿')
        plt.scatter(dpk, ddk, marker='o', s=150, c='r',edgecolors='r', alpha=1, label='拐点')
        # plt.scatter(dpk2, ddk2, marker='.', s=200, c='g', alpha=0.5, label='拐点')
        # labelss = plt.legend(fontsize=24, loc=0, frameon=False).get_texts()
        # [label.set_fontname('serif') for label in labelss]
        # [label.set_fontsize(24) for label in labelss]
        plt.xlabel('$d_p$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.ylabel('$d_d$',fontsize=24,fontdict={'family' : 'Times New Roman', 'size'   : 24})
        plt.xticks(color='black', fontsize=22)
        plt.yticks(color='black', fontsize=22)
        plt.legend()
        plt.show()
    else:
        print('error')

