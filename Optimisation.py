# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from scipy.optimize import differential_evolution
from argparse import ArgumentParser
import datetime as dt
import csv

parser = ArgumentParser()
parser.add_argument('-i', default=400, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=2, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-e', default=3, type=int, required=False, help='per-capita electricity: 3, 6 and 9 MWh')
parser.add_argument('-n', default='Super13', type=str, required=False, help='Super1, Super2, BN, KH, ...')
args = parser.parse_args()

percapita, node, iterations, population = (args.e, args.n, args.i, args.p)

from Input import *
from Simulation import Reliability
from Network import Transmission

def F(x):
    """This is the objective function."""

    S = Solution(x)

    Deficit = Reliability(S, flexible=np.zeros(intervals)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = min(0.5 * EHydro.sum() * pow(10, 3), Flexible) # GWh to MWh, MWh p.a.
    Fossil = Flexible - Hydro # Fossil fuels: MWh p.a.
    Hydro += GBaseload.sum() * resolution / years # Hydropower & other renewables: MWh p.a.
    PenHydro = 0

    TDC = Transmission(S) if 'Super' in node else np.zeros((intervals, len(DCloss)))  # TDC: TDC(t, k), MW

    Deficit = Reliability(S, flexible=np.ones(intervals) * CPeak.sum() * pow(10, 3)) # Sj-EDE(t, j), GW to MW
    PenDeficit = max(0, Deficit.sum() * resolution - S.allowance) # MWh

    CDC = np.amax(abs(TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    PenDC = max(0, CDC[0] - CDC0max) * pow(10, 3) # GW to MW
    PenDC += max(0, CDC[1] - CDC1max) * pow(10, 3) # GW to MW
    PenDC += max(0, CDC[7] - CDC7max) * pow(10, 3) # GW to MW
    PenDC += max(0, CDC[8] - CDC8max) * pow(10, 3) # GW to MW
    # PenDC *= pow(10, 3) # GW to MW

    cost = factor * np.array([sum(S.CPV), sum(S.CWind), sum(S.CInter), sum(S.CPHP), S.CPHS] + list(CDC) + [sum(S.CPV), sum(S.CWind), Hydro * pow(10, -6), Fossil * pow(10, -6), -1, -1]) # $b p.a.
    cost = cost.sum()
    loss = np.sum(abs(TDC), axis=0) * DCloss
    loss = loss.sum() * pow(10, -9) * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss)

    Func = LCOE + PenHydro + PenDeficit + PenDC

    return Func

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    lb = [0.]     * pzones + [0.]    * wzones + contingency      + [0.]      + [0.]    * inters
    ub = [10000.] * pzones + [300.]  * wzones + [10000.] * nodes + [100000.] + [1000.] * inters

    result = differential_evolution(func=F, bounds=list(zip(lb, ub)), tol=0,
                                    maxiter=args.i, popsize=args.p, mutation=args.m, recombination=args.r,
                                    disp=True, polish=False, updating='deferred', workers=-1)

    with open('Results/Optimisation_resultx_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    from Dispatch import Analysis
    Analysis(result.x)