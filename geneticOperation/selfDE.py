import numpy as np
from copy import deepcopy


# differential operator  && polynomial mutation
def OperatorDE_PloyMut(parent1, parent2, parent3, Pro, *parameters):
    if len(parameters) == 0:
        CR, F, ProM, DisM = 1, 0.5, 1, 20
    else:
        CR, F, ProM, DisM = parameters
    N, D = np.shape(parent1)

    # Differental evolution
    site = np.random.random((N, D)) < CR
    Offspring = deepcopy(parent1)
    Offspring[site] = Offspring[site] + F*(parent2[site] - parent3[site])

    # Polynomial mutation
    Lower = np.tile(Pro.lower, (N, 1))
    Upper = np.tile(Pro.upper, (N, 1))
    site = np.random.random((N, D)) < ProM/D
    mu = np.random.random((N, D))
    temp = site * (mu <= 0.5)
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * \
        ((2*mu[temp] + (1-2*mu[temp]) * (1-(Offspring[temp]-Lower[temp])/(Upper[temp]-Lower[temp]))**(DisM+1))**(1/(1+DisM)) - 1)
    # Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * ((2*mu[temp])**(1/(1+DisM)) - 1)
    temp = site * (mu > 0.5)
    Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * \
        ((1-(2-2*mu[temp]) + 2*(mu[temp]-0.5)*(1-(Upper[temp]-Offspring[temp])/(Upper[temp]-Lower[temp]))**(DisM+1))**(1/(1+DisM)))
    # Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (1 - (2-2*mu[temp])**(1/(1+DisM)))

    Offspring = np.where(Offspring <= Upper, Offspring, Upper)
    Offspring = np.where(Offspring >= Lower, Offspring, Lower)

    OffDec = Offspring
    return OffDec
