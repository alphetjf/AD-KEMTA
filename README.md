# AD-KEMTA

This repo contains the codebase of an surrogate-assisted evolutionary algorithm for knee solutions in many-objective optimization problems.

---

# Setup

    git clone https://github.com/alphetjf/AD-KEMTA.git
    cd AD-KEMTA
    pip install -r requirements.txt

---

# Guidelines

- **main_args.py**: The main program file running in the terminal has two running modes: parallel and non-parallel(algorithm independently executed 30 times):

(1) parallel

    python main_args.py --totalRuns 30 --ifParallel --parallel_num 10

(2) non-parallel:

    python main_args.py --totalRuns 30 --ifParallel --parallel_num 10

More arguements can be seen by:

    python main_args.py -h

- **algorithm/AD-KEMTA.py**: Our algorithm.

- **PortfoloProblem.py**: The real application problem, containing 4 algorithms (MOEA/D, NSGA2, NSGA3, AD-KEMTA).

- **expSets.py**: Experimental setup.

- **printResult.py**: Print the algorithm results to an Excel file.

- **Benchmark**: The folder contains test problems.

- **utils/KPI.py**: Related knee point identification techniques.
...

# Citation
Please cite the following if you use this code or parts of it:

    @article{tang2025knee,
        title={Knee-oriented expensive many-objective optimization via aggregation-dominance: A multi-task perspective},
        author={Tang, Junfeng and Wang, Handing and Jin, Yaochu},
        journal={Swarm and Evolutionary Computation},
        volume={92},
        pages={101813},
        year={2025},
        publisher={Elsevier}
    }