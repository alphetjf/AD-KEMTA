import argparse
import os
from expSets import parameterSet, parameterSet_nontrivial
from pathlib import Path
import time
import numpy as np
from multiprocessing import Pool
curPath = os.path.abspath(os.path.dirname(__file__))


def err_print(msg, original_line=None):
    print('ERROR  ' * 3)
    print(msg)
    if original_line:
        print(original_line)
    print('ERROR  ' * 3)
    exit(1)


def setParameters(pro):
    Wn = pro.M + 1
    if pro.M == 2:
        H1, H2 = 1, 3
    elif pro.M == 3 or pro.M == 8:
        H1, H2 = 1, 3
    elif pro.M == 5:
        H1, H2 = 1, 2
    return H1, H2, Wn


def collation(pathfile, test):
    problem = test[:test.find('\\')]
    D = int(test[test.find('\\')+1:test.find('D')])
    M = int(test[test.find('D')+1:test.find('M')])
    K = int(test[test.find('M\\')+2:test.find('K')])
    print('test: {}-{}D-{}M-{}K/A\n'.format(problem,D,M,K))
    problems = os.listdir(pathfile)
    typeDM = str(D)+'D'+str(M)+'M'
    if len(problems) == 0 or typeDM not in problems:
        err_print('No such result')
        return None, None
    else:
        print('experimental results: {}\n'.format(problems))
        results = {}
        ind = problems.index(typeDM)
        filePath = os.path.join(pathfile, problems[ind])
        if problem in ['ckp','deb2dk','do2dk','deb3dk']:
            knee_path = os.path.join(filePath, str(K)+'K')
        else:
            knee_path = os.path.join(filePath, str(K)+'A')
        runs = os.listdir(knee_path)
        ii = np.random.randint(runs)
        r = runs[ii]
        runPath = os.path.join(knee_path, r)
        filesName = os.listdir(runPath)
        topFEs = int(filesName[-1])
        path = os.path.join(runPath, str(topFEs))
        Decs = np.load(path+"\\Decs.npy")
        Objs = np.load(path+"\\Objs.npy")
        Cons = np.load(path+"\\Cons.npy")
        Adds = np.load(path+"\\Adds.npy")
        index = topFEs
        results.update({index: [Decs, Objs, Cons, Adds]})
        return results, index


def Save(result, save_path):
    my_file = Path(save_path)
    if bool(1-my_file.exists()):
        os.makedirs(save_path)
    import json
    N = len(result)
    Decs = result.get(N)[0]
    np.save(save_path+"\\Decs.npy", Decs)
    Objs = result.get(N)[1]
    np.save(save_path+"\\Objs.npy", Objs)
    Cons = result.get(N)[2]
    np.save(save_path+"\\Cons.npy", Cons)
    Adds = result.get(N)[3]
    np.save(save_path+"\\Adds.npy", Adds)


def process(varargin):
    test = varargin[0]
    path = varargin[1]
    algSelect = varargin[2]
    ifTrivial = varargin[3]
    print_freq = varargin[4]  # trivial-5000, nontrivial-100
    save_freq = varargin[5]
    t_start = time.time()
    print("Start execution, process number is %d" % os.getpid())
    if ifTrivial:
        pro, maxFEs = parameterSet(test)
    else:
        pro, maxFEs = parameterSet_nontrivial(test)

    print("the maximum function evaluation is ", maxFEs)
    my_file = Path(path)
    if bool(1-my_file.exists()):
        os.makedirs(path)
    dirs = len(os.listdir(path))
    path = os.path.join(path, str(dirs+1)+"_times_" + str(os.getpid()))
    my_file2 = Path(path)
    if bool(1-my_file2.exists()):
        os.makedirs(path)

    if algSelect == 'AD-KEMTA':
        from algorithm.AD_KEMTA import alg
        dsolver = alg()
        _ = dsolver._ParameterSet(save_frequency=save_freq, filepath=path, drawFig=(False, 2, print_freq))
    dsolver.Solve(pro)
    t_stop = time.time()
    print("Execution completed, time-consuming: %0.2f" % (t_stop-t_start))


parser = argparse.ArgumentParser(description='expensive MaOP')

parser.add_argument('--totalRuns', type=int, default=10,
                    help='the total number of runs needed')

parser.add_argument('--ifParallel', action='store_true',
                    help='the mode of running, parallelly or not')
parser.add_argument('--ifTrivial', action='store_true',
                    help='if the fitness evaluation is trivial? trivial|nontrivial')

parser.add_argument('--parallel_num', type=int, default=10,
                    help='if run parallelly, do N process')

parser.add_argument('--alg', type=str, default='MOEAD',
                    help='the selected algorithm to run')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--print_freq', type=int, default=5000,
                    help='print running status once every __ fes in each run, trivial-5000, nontrivial-100')
parser.add_argument('--save_freq', type=int, default=100, 
                    help='save output results once every _ fes in each run, 0 meas save the last results, -1 means no save, trivial-5000, nontrivial-100')
args, _ = parser.parse_known_args()


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

print(args)

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

if __name__=='__main__':
    print('{} start running'.format(args.alg))
    experimentType = 'trivial experiments' if args.ifTrivial else 'nontrivial experiments' 
    for test in benchmarks + benchmarks_Many:
        path = os.path.join(curPath, 'experiment', experimentType, args.alg, test)
        my_file = Path(path)
        if bool(1-my_file.exists()):
            os.makedirs(path)
        dirs = len(os.listdir(path))
        if dirs >= args.totalRuns:
            continue
        else:
            while dirs < args.totalRuns:
                nums = min(args.totalRuns - dirs, args.parallel_num)
                if args.ifParallel:
                    print(test+" nums:", nums)
                    with Pool(nums) as p:
                        p.map(process, [[test, path, args.alg, args.ifTrivial, args.print_freq, args.save_freq] for _ in range(nums)])
                else:
                    for i in range(nums):
                        process([test, path, args.alg, args.ifTrivial, args.print_freq, args.save_freq])
                dirs = len(os.listdir(path))