import numpy as np
import random
from copy import deepcopy


def cxSimulatedBinaryBounded(ind1, ind2, eta, ProC, low, up):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param low: A value or a :term:`python:sequence` of values that is the lower
                bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that is the upper
               bound of the search space.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    .. note::
       This implementation is similar to the one implemented in the
       original NSGA-II C code presented by Deb.
    """
    size = min(len(ind1), len(ind2))

    if len(low) < size:
        raise IndexError("low must be at least the size of the shorter individual: %d < %d" % (len(low), size))
    if len(up) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= ProC:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2
    return ind1, ind2


def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x
    return individual


def mutGaussian(individual, mu, sigma, indpb):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    if len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < indpb:
            individual[i] += random.gauss(m, s)

    return individual


def AssortativeMating_SBXPloyMut(Pops, skillFactors, low, up, proC, disC, proM, disM, rmp):
    N, D = np.shape(Pops)
    if np.mod(N, 2) == 0:
        parents = np.random.randint(N, size=(2, int(N/2)))
        oddOrEven = 0
    else:
        parents = np.random.randint(N, size=(2, int(N/2)+1))
        oddOrEven = 1
    Offsprings = np.zeros((N, D))
    Transmission = np.zeros(N, dtype=int)
    for i in range(int(N/2)):
        if skillFactors[parents[0, i]] == skillFactors[parents[1, i]] or np.random.rand() < rmp:
            miu = np.random.rand(D)
            beta = np.zeros(D)
            index = miu <= 0.5
            beta[index] = (2*miu[index])**(1/(disC+1))
            beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
            beta = beta*(-1)**np.random.randint(2, size=D)
            beta[np.random.rand(D) > proC] = 1
            if oddOrEven == 1 and i == np.shape(parents)[1]-1:
                if np.random.rand() <= 0.5:
                    Offsprings[i*2, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 + beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                    Transmission[i*2] = skillFactors[parents[0, i]].copy()
                else:
                    Offsprings[i*2, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 - beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                    Transmission[i*2] = skillFactors[parents[1, i]].copy()
            else:
                Offsprings[i*2, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 + beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                Offsprings[i*2+1, :] = (Pops[parents[0, i], :] + Pops[parents[1, i], :])/2 - beta*(Pops[parents[0, i], :] + Pops[parents[1, i], :])/2
                Transmission[i*2] = skillFactors[parents[0, i]].copy()
                Transmission[i*2+1] = skillFactors[parents[1, i]].copy()
        else:
            k = np.random.rand(D)
            miu = np.random.rand(D)
            if oddOrEven == 1 and i == np.shape(parents)[1]-1:
                Offsprings[i*2, :] = Pops[parents[0, i], :]
                if np.random.rand() <= 0.5:
                    Temp = (k <= proM/D) & (miu < 0.5)  # 变异的基因
                    Offsprings[i*2, Temp] = Offsprings[i*2, Temp] + (up[Temp] - low[Temp]) * \
                        ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[i*2, Temp]-low[Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1))-1)
                    Transmission[i*2] = skillFactors[parents[0, i]].copy()
                else:
                    Temp = (k <= proM/D) & (miu >= 0.5)
                    Offsprings[i*2, Temp] = Offsprings[i*2, Temp] + (up[Temp] - low[Temp]) * \
                        (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(up[Temp]-Offsprings[i*2, Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1)))
                    Transmission[i*2] = skillFactors[parents[1, i]].copy()
            else:
                Offsprings[i*2, :] = Pops[parents[0, i], :]
                Offsprings[i*2+1, :] = Pops[parents[1, i], :]
                Temp = (k <= proM/D) & (miu < 0.5)  # 变异的基因
                Offsprings[i*2, Temp] = Offsprings[i*2, Temp] + (up[Temp] - low[Temp]) * \
                    ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[i*2, Temp]-low[Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1))-1)
                Temp = (k <= proM/D) & (miu >= 0.5)
                Offsprings[i*2+1, Temp] = Offsprings[i*2+1, Temp] + (up[Temp] - low[Temp]) * \
                    (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(up[Temp]-Offsprings[i*2+1, Temp])/(up[Temp] - low[Temp]))**(disM+1))**(1/(disM+1)))
                Transmission[i*2] = skillFactors[parents[0, i]].copy()
                Transmission[i*2+1] = skillFactors[parents[1, i]].copy()

    # 越界处理
    if N == 1:
        MaxValue = up
        MinValue = low
    else:
        MaxValue = np.tile(up, (N, 1))
        MinValue = np.tile(low, (N, 1))
    Offsprings = np.where(Offsprings <= MaxValue, Offsprings, MaxValue)
    Offsprings = np.where(Offsprings >= MinValue, Offsprings, MinValue)
    return Offsprings, Transmission


def AssortativeMating2(Pops, skillFactors, low, up, proC, disC, proM, disM, rmp):
    N, D = np.shape(Pops)
    parents = np.random.randint(N, size=(2, int(N/2)))
    offsprings = np.zeros((N, D))
    Transmission = np.zeros(N, dtype=int)
    mu = 0.5*np.ones(D)
    sigma = 2*np.ones(D)
    for i in range(int(N/2)):
        if skillFactors[parents[0, i]] == skillFactors[parents[1, i]] or np.random.rand() < rmp:
            offsprings[i*2, :], offsprings[i*2+1, :] = cxSimulatedBinaryBounded(Pops[parents[0, i], :].copy(), Pops[parents[1, i], :].copy(), disC, proC, low, up)
            # offsprings[i*2, :] = mutPolynomialBounded(offsprings[i*2, :].copy(), disM, low, up, proM)
            # offsprings[i*2+1, :] = mutPolynomialBounded(offsprings[i*2+1, :].copy(), disM, low, up, proM)
        else:
            # offsprings[i*2, :] = mutPolynomialBounded(Pops[parents[0, i], :].copy(), disM, low, up, proM/D)
            # offsprings[i*2+1, :] = mutPolynomialBounded(Pops[parents[1, i], :].copy(), disM, low, up, proM/D)
            offsprings[i*2, :] = mutGaussian(Pops[parents[0, i], :].copy(), mu, sigma, proM/D)
            offsprings[i*2+1, :] = mutGaussian(Pops[parents[1, i], :].copy(), mu, sigma, proM/D)

        Transmission[i*2] = skillFactors[parents[0, i]].copy()
        Transmission[i*2+1] = skillFactors[parents[1, i]].copy()
    if int(N/2)*2 != N:
        # offsprings[-1, :] = mutPolynomialBounded(Pops[-1, :], disM, low, up, proM/D)
        offsprings[-1, :] = mutGaussian(Pops[-1, :].copy(), mu, sigma, proM/D)
        Transmission[-1] = skillFactors[-1].copy()

    # 越界处理
    if N == 1:
        MaxValue = up
        MinValue = low
    else:
        MaxValue = np.tile(up, (N, 1))
        MinValue = np.tile(low, (N, 1))
    offsprings = np.where(offsprings <= MaxValue, offsprings, MaxValue)
    offsprings = np.where(offsprings >= MinValue, offsprings, MinValue)
    return offsprings, Transmission


def AssortativeMating_DEPloyMut(Pops, skillFactors, low, up, CR, F, proM, disM, rmp):
    N, D = np.shape(Pops)
    parents = np.random.randint(N, size=(2, int(N)))
    Offsprings = deepcopy(Pops)
    Transmission = np.zeros(N, dtype=int)
    # transfer site
    site = np.logical_or(
        np.random.rand(N) <= rmp,
        np.logical_or(
            skillFactors[parents[0, :]] == skillFactors[parents[1, :]],
            skillFactors[parents[0, :]] == skillFactors,
            skillFactors[parents[1, :]] == skillFactors
            )
        )   # 满足迁移概率，或者三个个体之间有两个是同一任务
    # site = np.logical_or(
    #     np.random.rand(N) <= rmp,
    #     skillFactors[parents[1, :]] == skillFactors
    #     )   # 满足迁移概率，或者个体F1\F2之间是同一任务
    # 掩码
    mask = np.logical_and(np.random.random((N, D)) <= CR, np.tile(site[:, np.newaxis], (D,)))
    P2 = Pops[parents[0, :], :]
    P3 = Pops[parents[1, :], :]
    Offsprings[mask] = Offsprings[mask] + F*(P2[mask] - P3[mask])

    # Polynomial mutation
    if N == 1:
        Upper = up
        Lower = low
    else:
        Lower = np.tile(low, (N, 1))
        Upper = np.tile(up, (N, 1))
    # site2 = np.logical_and(np.tile(~site[:, np.newaxis], (D,)), np.random.random((N, D)) < proM)
    site2 = np.random.random((N, D)) < proM
    mu = np.random.random((N, D))
    temp = np.logical_and(site2, mu <= 0.5)
    temp = site2 * (mu <= 0.5)
    Offsprings[temp] = Offsprings[temp] + (Upper[temp] - Lower[temp]) * \
        ((2*mu[temp] + (1-2*mu[temp]) * (1-(Offsprings[temp]-Lower[temp])/(Upper[temp]-Lower[temp]))**(disM+1))**(1/(1+disM)) - 1)
    temp = site2 * (mu > 0.5)
    Offsprings[temp] = Offsprings[temp] + (Upper[temp] - Lower[temp]) * \
        ((1-(2-2*mu[temp]) + 2*(mu[temp]-0.5)*(1-(Upper[temp]-Offsprings[temp])/(Upper[temp]-Lower[temp]))**(disM+1))**(1/(1+disM)))

    # 越界处理
    Offsprings = np.where(Offsprings <= Upper, Offsprings, Upper)
    Offsprings = np.where(Offsprings >= Lower, Offsprings, Lower)
    return Offsprings, Transmission


def crossTansfer(cPops, aPops, low, up, proC, disC, proM, disM):
    N, D = np.shape(cPops)
    if np.mod(N, 2) == 0:
        parents = np.random.randint(N, size=(2, int(N/2)))
        oddOrEven = 0
    else:
        parents = np.random.randint(N, size=(2, int(N/2)+1))
        oddOrEven = 1

    Offsprings = np.zeros((N, D))
    for i in range(np.shape(parents)[1]):
        miu = np.random.rand(D)
        beta = np.zeros(D)
        index = miu <= 0.5
        beta[index] = (2*miu[index])**(1/(disC+1))
        beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
        beta = beta*(-1)**np.random.randint(2, size=D)
        beta[np.random.rand(D) > proC] = 1
        if oddOrEven == 1 and i == np.shape(parents)[1]-1:
            if np.random.rand() <= 0.5:
                Offsprings[i*2, :] = (cPops[parents[0, i], :] + aPops[parents[1, i], :])/2 + beta*(cPops[parents[0, i], :] + aPops[parents[1, i], :])/2
            else:
                Offsprings[i*2, :] = (cPops[parents[0, i], :] + aPops[parents[1, i], :])/2 - beta*(cPops[parents[0, i], :] + aPops[parents[1, i], :])/2
        else:
            Offsprings[i*2, :] = (cPops[parents[0, i], :] + aPops[parents[1, i], :])/2 + beta*(cPops[parents[0, i], :] + aPops[parents[1, i], :])/2
            Offsprings[i*2+1, :] = (cPops[parents[0, i], :] + aPops[parents[1, i], :])/2 - beta*(cPops[parents[0, i], :] + aPops[parents[1, i], :])/2

    # polynominal mutation
    if N == 1:
        MaxValue = up
        MinValue = low
    else:
        MaxValue = np.tile(up, (N, 1))
        MinValue = np.tile(low, (N, 1))
    k = np.random.rand(N, D)
    miu = np.random.rand(N, D)
    Temp = (k <= proM/D) & (miu < 0.5)  # 变异的基因
    Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
        ((2*miu[Temp]+(1-2*miu[Temp])*(1-(Offsprings[Temp]-MinValue[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1))-1)
    Temp = (k <= proM/D) & (miu >= 0.5)
    Offsprings[Temp] = Offsprings[Temp] + (MaxValue[Temp] - MinValue[Temp]) * \
        (1-(2*(1-miu[Temp])+2*(miu[Temp]-0.5)*(1-(MaxValue[Temp]-Offsprings[Temp])/(MaxValue[Temp] - MinValue[Temp]))**(disM+1))**(1/(disM+1)))
    # 越界处理
    Offsprings = np.where(Offsprings <= MaxValue, Offsprings, MaxValue)
    Offsprings = np.where(Offsprings >= MinValue, Offsprings, MinValue)
    return Offsprings


def crossTansfer2(cPops, aPops, low, up, proC, disC, proM, disM):
    N, D = np.shape(cPops)
    if np.mod(N, 2) == 0:
        parents = np.random.randint(N, size=(2, int(N/2)))
        oddOrEven = 0
    else:
        parents = np.random.randint(N, size=(2, int(N/2)+1))
        oddOrEven = 1

    Offsprings = np.zeros((N, D))
    for i in range(np.shape(parents)[1]):
        miu = np.random.rand(D)
        beta = np.zeros(D)
        index = miu <= 0.5
        beta[index] = (2*miu[index])**(1/(disC+1))
        beta[~index] = (2-2*miu[~index])**(-1/(disC+1))
        beta = beta*(-1)**np.random.randint(2, size=D)
        beta[np.random.rand(D) > proC] = 1
        if oddOrEven == 1 and i == np.shape(parents)[1]-1:
            if np.random.rand() <= 0.5:
                Offsprings[i*2, :], _ = cxSimulatedBinaryBounded(cPops[parents[0, i], :].copy(), aPops[parents[1, i], :].copy(),
                                                                 disC, proC, low, up)
            else:
                _, Offsprings[i*2, :] = cxSimulatedBinaryBounded(cPops[parents[0, i], :].copy(), aPops[parents[1, i], :].copy(),
                                                                 disC, proC, low, up)
            Offsprings[i*2, :] = mutPolynomialBounded(Offsprings[i*2, :].copy(), disM, low, up, proM/D)
        else:
            Offsprings[i*2, :], Offsprings[i*2+1, :] = cxSimulatedBinaryBounded(cPops[parents[0, i], :].copy(),
                                                                                aPops[parents[1, i], :].copy(),
                                                                                disC, proC, low, up)
            Offsprings[i*2, :] = mutPolynomialBounded(Offsprings[i*2, :].copy(), disM, low, up, proM/D)
            Offsprings[i*2+1, :] = mutPolynomialBounded(Offsprings[i*2+1, :].copy(), disM, low, up, proM/D)
    return Offsprings
