# -*- encoding: utf-8 -*-
'''
@File    :   case_study.py
@Time    :   2023/10/18 21:56:03
@Author  :   jftang
'''
# Portfolio Allocation

import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.util.remote import Remote

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

file = Remote.get_instance().load("examples", "portfolio_allocation.csv", to=None)

df = pd.read_csv(file, parse_dates=True, index_col="date")

returns = df.pct_change().dropna(how="all")
mu = (1 + returns).prod() ** (252 / returns.count()) - 1
cov = returns.cov() * 252

mu, cov = mu.to_numpy(), cov.to_numpy()

labels = df.columns

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
k = np.arange(len(mu))
ax.bar(k, mu)
ax.set_xticks(k, labels, rotation = 90)
plt.show()


f = plt.figure(figsize=(10, 10))
plt.matshow(returns.corr(), fignum=f.number)
plt.xticks(k, labels, fontsize=12, rotation=90)
plt.yticks(k, labels, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
print("DONE")


class PortfolioProblem(ElementwiseProblem):
    def __init__(self, mu, cov, risk_free_rate=0.02, **kwargs):
        super().__init__(n_var=len(df.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.mu = mu
        self.cov = cov
        self.risk_free_rate = risk_free_rate

    def _evaluate(self, x, out, *args, **kwargs):
        exp_return = x @ self.mu
        exp_risk = np.sqrt(x.T @ self.cov @ x)
        sharpe = (exp_return - self.risk_free_rate) / exp_risk

        out["F"] = [exp_risk, -exp_return]
        out["sharpe"] = sharpe


class PortfolioSampling(FloatRandomSampling):
    def __init__(self, mu, cov) -> None:
        super().__init__()
        self.mu = mu
        self.cov = cov

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)

        n = len(mu)
        n_biased = min(n, n_samples // 2)

        order_by_ret = (-self.mu).argsort()
        order_by_cov = (self.cov.diagonal()).argsort()
        order = np.stack([order_by_ret, order_by_cov]).min(axis=0)

        X[:n_biased] = np.eye(n)[order][:n_biased]

        return X


class PortfolioRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)


problem = PortfolioProblem(mu, cov)
# from pymoo.algorithms.moo.sms import SMSEMOA
# algorithm = SMSEMOA(sampling=PortfolioSampling(mu, cov), repair=PortfolioRepair())
# from pymoo.algorithms.moo.nsga2 import NSGA2
# algorithm = NSGA2(sampling=PortfolioSampling(mu, cov), repair=PortfolioRepair())
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
algorithm = NSGA3(sampling=PortfolioSampling(mu, cov), repair=PortfolioRepair(), ref_dirs=ref_dirs, pop_size=100)
res = minimize(problem,
               algorithm,
               seed=1,
               verbose=True,
               termination=('n_gen', 600))

X, F, sharpe = res.opt.get("X", "F", "sharpe")
F = F * [1, -1]
max_sharpe = sharpe.argmax()

f = plt.figure(figsize=(8, 6))
plt.scatter(F[:, 0], F[:, 1], facecolor="none", edgecolors="blue", alpha=0.5, label="Pareto-Optimal Portfolio")
plt.scatter(cov.diagonal() ** 0.5, mu, facecolor="none", edgecolors="black", s=30, label="Asset")
plt.scatter(F[max_sharpe, 0], F[max_sharpe, 1], marker="x", s=100, color="red", label="Max Sharpe Portfolio")
plt.legend()
plt.show()

# view decision variables
allocation = {name: w for name, w in zip(df.columns, X[max_sharpe])}
allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)

print("Allocation With Best Sharpe")
for name, w in allocation:
    print(f"{name:<5} {w}")