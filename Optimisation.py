# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from numba import njit, prange
import datetime as dt
import numpy as np
import csv

from Input import *

@njit(parallel=True)
def ParallelObjectiveWrapper(xs):
    result = np.empty(xs.shape[1], dtype=np.float64)
    for i in prange(xs.shape[1]):
        result[i] = Objective(xs[:,i])
    return result
    
@njit
def Objective(x):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate()
    return S.Lcoe + S.Penalties

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    print("Length of pv_lb:", len(pv_lb)) 
    print("Length of pv_ub:", len(pv_ub)) 
    print("Length of wind_ub:", len(wind_ub)) 
    print("Length of phes_lb:", len(phes_lb)) 
    print("Length of phes_ub:", len(phes_ub)) 
    print("Length of storage_lb:", len(storage_lb)) 
    print("Length of storage_ub:", len(storage_ub)) 
    print("Length of battery_lb:", len(battery_lb)) 
    print("Length of battery_ub:", len(battery_ub)) 
    print("Length of bduration_lb:", len(bduration_lb)) 
    print("Length of bduration_ub:", len(bduration_ub)) 
    print("Length of inters_lb:", len(inters_lb)) 
    print("Length of inters_ub:", len(inters_ub)) 
    print("Length of transmission_lb:", len(transmission_lb)) 
    print("Length of transmission_ub:", len(transmission_ub))
    print(lb)
    print(ub)
    result = differential_evolution(
        func=ParallelObjectiveWrapper, 
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=args.m, 
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred', 
        workers=1,
        vectorized=True,
        )

    with open('Results/Optimisation_resultx_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    from Fill import Flexible
    Flexible(result.x)