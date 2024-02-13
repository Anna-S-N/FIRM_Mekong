# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Reliability(solution, flexible, start=None, end=None):
    """Deficit = Simulation.Reliability(S, hydro=...)"""

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) - solution.GBaseload.sum(axis=1))[start:end] \
              - flexible # - solution.GInter.sum(axis=1)[start:end] # Sj-ENLoad(j, t), MW

    length = len(Netload)
    solution.flexible = flexible # MW

    Pcapacity = sum(solution.CPHP) * pow(10, 3) # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * pow(10, 3) # S-CPHS(j), GWh to MWh
    efficiency, resolution = (solution.efficiency, solution.resolution)

    Discharge, Charge, Storage = map(np.zeros, [length] * 3)

    for t in range(length):

        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1] if t>0 else 0.5 * Scapacity

        Discharget = min(max(0, Netloadt), Pcapacity, Storaget_1 / resolution)
        Charget = min(-1 * min(0, Netloadt), Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution)

        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency

    Deficit = np.maximum(Netload - Discharge, 0)
    Spillage = -1 * np.minimum(Netload + Charge, 0)

    assert int(np.amin(Storage)) >= 0, 'Storage below zero'
    assert int(np.amax(Storage)) <= Scapacity, 'Storage exceeds max storage capacity'
    assert np.amin(Deficit) >= 0, 'Deficit below zero'
    assert np.amin(Spillage) >= 0, 'Spillage below zero'

    solution.Discharge, solution.Charge, solution.Storage = (Discharge, Charge, Storage)
    solution.Deficit, solution.Spillage = (Deficit, Spillage)

    return Deficit