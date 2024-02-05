# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np

def Transmission(solution, output=False):
    """TDC = Network.Transmission(S)"""

    #Nodel, PVl, Windl, Interl = (solution.Nodel, solution.PVl, solution.Windl, solution.Interl)
    Nodel, PVl, Windl = (solution.Nodel, solution.PVl, solution.Windl)
    intervals, nodes = (solution.intervals, solution.nodes)

    #MPV, MWind, MInter = map(np.zeros, [(nodes, intervals)] * 3)
    MPV, MWind = map(np.zeros, [(nodes, intervals)] * 2)
    for i, j in enumerate(Nodel):
        MPV[i, :] = solution.GPV[:, np.where(PVl==j)[0]].sum(axis=1)
        MWind[i, :] = solution.GWind[:, np.where(Windl==j)[0]].sum(axis=1)
        #if solution.node=='Super2':
        #    MInter[i, :] = solution.GInter[:, np.where(Interl==j)[0]].sum(axis=1)
    #MPV, MWind, MInter = (MPV.transpose(), MWind.transpose(), MInter.transpose()) # Sij-GPV(t, i), Sij-GWind(t, i), MW
    MPV, MWind = (MPV.transpose(), MWind.transpose()) # Sij-GPV(t, i), Sij-GWind(t, i), MW

    MBaseload = solution.GBaseload + solution.GHydro # MW
    CPeak = solution.CPeak # GW
    pkfactor = np.tile(CPeak, (intervals, 1)) / CPeak.sum()
    MPeak = np.tile(solution.flexible, (nodes, 1)).transpose() * pkfactor # MW

    MLoad = solution.MLoad # EOLoad(t, j), MW

    defactor = MLoad / MLoad.sum(axis=1)[:, None]
    MDeficit = np.tile(solution.Deficit, (nodes, 1)).transpose() * defactor # MDeficit: EDE(j, t)

    MPW = MPV + MWind
    spfactor = np.divide(MPW, MPW.sum(axis=1)[:, None], where=MPW.sum(axis=1)[:, None]!=0)
    MSpillage = np.tile(solution.Spillage, (nodes, 1)).transpose() * spfactor # MSpillage: ESP(j, t)

    CPHP = solution.CPHP
    pcfactor = np.tile(CPHP, (intervals, 1)) / sum(CPHP) if sum(CPHP) != 0 else 0
    MDischarge = np.tile(solution.Discharge, (nodes, 1)).transpose() * pcfactor # MDischarge: DPH(j, t)
    MCharge = np.tile(solution.Charge, (nodes, 1)).transpose() * pcfactor # MCharge: CHPH(j, t)

    #MImport = MLoad + MCharge + MSpillage - MPV - MWind - MInter - MBaseload - MPeak - MDischarge - MDeficit  # EIM(t, j), MW
    MImport = MLoad + MCharge + MSpillage - MPV - MWind - MBaseload - MPeak - MDischarge - MDeficit  # EIM(t, j), MW

    #AWIJ = -1 * MImport[:, np.where(Nodel == 'AW')[0][0]]
    #ANIT = -1 * MImport[:, np.where(Nodel == 'AN')[0][0]]
    #IMIP = MImport[:, np.where(Nodel == 'IP')[0][0]]
    #IJIS = MImport[:, np.where(Nodel == 'IS')[0][0]]
    #PMPV = -1 * MImport[:, np.where(Nodel == 'PM')[0][0]]
    KHVS = MImport[:, np.where(Nodel == 'VS')[0][0]]
    #CNVH = -1 * MImport[:, np.where(Nodel == 'CN')[0][0]]
    #INMM = -1 * MImport[:, np.where(Nodel == 'IN')[0][0]]

    LAVH = MImport[:, np.where(Nodel == 'VH')[0][0]]# - CNVH
    #MMTH = -1 * MImport[:, np.where(Nodel == 'MM')[0][0]] + INMM
    KHTH = -1 * MImport[:, np.where(Nodel == 'KH')[0][0]] - KHVS
    #PLPV = MImport[:, np.where(Nodel == 'PV')[0][0]] - PMPV
    #IJIT = MImport[:, np.where(Nodel == 'IT')[0][0]] - ANIT
    #IMIC = -1 * MImport[:, np.where(Nodel == 'IM')[0][0]] - IMIP

    LATH = -1 * MImport[:, np.where(Nodel == 'LA')[0][0]] - LAVH
    #BNPL = MImport[:, np.where(Nodel == 'PL')[0][0]] + PLPV
    #IKIC = MImport[:, np.where(Nodel == 'IC')[0][0]] - IMIC

    #MYTH = MImport[:, np.where(Nodel == 'TH')[0][0]] - MMTH - LATH - KHTH
    #MYSG = -1 * MImport[:, np.where(Nodel == 'MY')[0][0]] - MYTH

    #BNSG = -1 * MImport[:, np.where(Nodel == 'BN')[0][0]] - BNPL
    #IJIK = MImport[:, np.where(Nodel == 'IK')[0][0]] + IKIC

    #IJSG = MImport[:, np.where(Nodel == 'SG')[0][0]] - BNSG - MYSG
    #IJSG1 = -1 * MImport[:, np.where(Nodel == 'IJ')[0][0]] - IJIS + AWIJ - IJIT - IJIK
    #Check the final node
    #KHTH1 = MImport[:, np.where(Nodel=='TH'[0][0])] + LATH + LAVH
    #assert abs(KHTH - KHTH1).max() <= 0.1, print(abs(KHTH - KHTH1).max())#assert abs(IJSG - IJSG1).max() <= 0.1, print(abs(IJSG - IJSG1).max())

    #BNIK = 0 * BNSG

    TDC = np.array([KHTH, KHVS, LATH, LAVH,]).transpose() # TDC(t, k), MW ([AWIJ, ANIT, BNIK, BNPL, BNSG, KHTH, KHVS, CNVH, INMM, IJIK, IJIS, IJIT, IJSG, IKIC, IMIP, IMIC, LATH, LAVH, MYSG, MYTH, MMTH, PLPV, PMPV]).

    if output:
        MStorage = np.tile(solution.Storage, (nodes, 1)).transpose() * pcfactor # SPH(t, j), MWh
        #solution.MPV, solution.MWind, solution.MInter, solution.MBaseload, solution.MPeak = (MPV, MWind, MInter, MBaseload, MPeak)
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (MPV, MWind, MBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage = (MDischarge, MCharge, MStorage)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return TDC