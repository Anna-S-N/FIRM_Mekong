# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from Network import Transmission

import numpy as np
import datetime as dt

def Debug(solution):
    """Debugging"""

    Load, PV, Wind, Inter = (solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1), solution.GInter.sum(axis=1))
    Baseload, Peak = (solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage = (solution.Discharge, solution.Charge, solution.Storage)
    Deficit, Spillage = (solution.Deficit, solution.Spillage)

    PHS = solution.CPHS * pow(10, 3) # MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i]
                   - PV[i] - Wind[i] - Inter[i] - Baseload[i] - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

        # Capacity: PV, wind, Discharge, Charge and Storage
        try:
            assert np.amax(PV) <= sum(solution.CPV) * pow(10, 3), print(np.amax(PV) - sum(solution.CPV) * pow(10, 3))
            assert np.amax(Wind) <= sum(solution.CWind) * pow(10, 3), print(np.amax(Wind) - sum(solution.CWind) * pow(10, 3))
            assert np.amax(Inter) <= sum(solution.CInter) * pow(10, 3), print(np.amax(Inter) - sum(solution.CInter) * pow(10, 3))

            assert np.amax(Discharge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3))
            assert np.amax(Charge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Charge) - sum(solution.CPHP) * pow(10, 3))
            assert np.amax(Storage) <= solution.CPHS * pow(10, 3), print(np.amax(Storage) - sum(solution.CPHS) * pow(10, 3))
        except AssertionError:
            pass

    print('Debugging: everything is ok')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    Debug(solution)

    C = np.stack([solution.MLoad.sum(axis=1),
                  solution.MHydro.sum(axis=1), solution.MFossil.sum(axis=1), solution.MInter.sum(axis=1), solution.GPV.sum(axis=1), solution.GWind.sum(axis=1),
                  solution.Discharge, solution.Deficit, -1 * solution.Spillage, -1 * solution.Charge,
                  solution.Storage,
                  solution.AWIJ, solution.ANIT, solution.BNIK, solution.BNPL, solution.BNSG, solution.KHTH,
                  solution.KHVS, solution.CNVH, solution.INMM, solution.IJIK, solution.IJIS, solution.IJIT,
                  solution.IJSG, solution.IKIC, solution.IMIP, solution.IMIC, solution.LATH, solution.LAVH,
                  solution.MYSG, solution.MYTH, solution.MMTH, solution.PLPV, solution.PMPV])
    C = np.around(C.transpose())

    datentime = np.array([(dt.datetime(firstyear, 1, 1, 0, 0) + x * dt.timedelta(minutes=60 * resolution)).strftime('%a %#d %b %Y %H:%M') for x in range(intervals)])
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header_main = 'Date & time,Operational demand,' \
             'Hydropower & other renewables (MW),Fossil fuels (MW),Import (MW),Solar photovoltaics (MW),Wind (MW),Pumped hydro energy storage (MW),Energy deficit (MW),Energy spillage,PHES-Charge (MW),' \
             'PHES-Storage (MWh),' \
             'AWIJ,ANIT,BNIK,BNPL,BNSG,KHTH,KHVS,CNVH,INMM,IJIK,IJIS,IJIT,IJSG,IKIC,IMIP,IMIC,LATH,LAVH,MYSG,MYTH,MMTH,PLPV,PMPV'

    #Step to create the csv file for the main data
    np.savetxt('Results/LPGM_SEAsia_{}_{}.csv'.format(node, percapita), C, fmt='%s', delimiter=',', header=header_main, comments='')

    if 'Super' in node:
        header_node = 'Date & time,Operational demand,' \
                 'Hydropower & other renewables (MW),Fossil fuels (MW),Import (MW),Solar photovoltaics (MW),Wind (MW),Pumped hydro energy storage (MW),Energy deficit (MW),Energy spillage,' \
                 'Transmission,PHES-Charge (MW),' \
                 'PHES-Storage'

#   This iterates over the nodes and creates csv files for each one
        for j in range(nodes):
            C_node = np.stack([solution.MLoad[:, j],
                          solution.MHydro[:, j], solution.MFossil[:, j], solution.MInter[:, j], solution.MPV[:, j], solution.MWind[:, j],
                          solution.MDischarge[:, j], solution.MDeficit[:, j], -1 * solution.MSpillage[:, j], solution.Topology[j], -1 * solution.MCharge[:, j],
                          solution.MStorage[:, j]])
            C_node = np.around(C_node.transpose())

            # Inserting datetime and converting to string format
            C_node = np.insert(C_node.astype('str'), 0, datentime, axis=1)

            np.savetxt('Results/LPGM_{}_{}_{}.csv'.format(node, percapita, solution.Nodel[j]), C_node, fmt='%s', delimiter=',', header=header_node, comments='')

    print('Load profiles and generation mix have been produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and US$/MWh information"""
    
    #Importing cost factors 
    if node in ['BN', 'SG']:
        factor = np.genfromtxt('Data/factor1.csv', dtype=None, delimiter=',', encoding=None)
    else:
        factor = np.genfromtxt('Data/factor.csv', dtype=None, delimiter=',', encoding=None)
    factor = dict(factor)

    #Importing capacities [GW,GWh] from the least-cost solution
    CPV, CWind, CPHP, CPHS, CInter = (sum(solution.CPV), sum(solution.CWind), sum(solution.CPHP), solution.CPHS, sum(solution.CInter)) # GW, GWh
    CapHydro = (CHydro + CGeo + CBio + CWaste).sum() # Hydropower & other resources: GW
    CapFossil = (CCoal + CGas + COil).sum() # Fossil fuels: GW

    #Importing generation energy [GWh] from the least-cost solution
    GPV, GWind, GHydro, GFossil, GInter = map(lambda x: x * pow(10, -6) * resolution / years,
                                              (solution.GPV.sum(), solution.GWind.sum(), solution.MHydro.sum(), solution.MFossil.sum(), solution.MInter.sum())) # TWh p.a.
    CFPV, CFWind = (GPV / CPV / 8.76, GWind / CWind / 8.76)

    # Calculate the annual costs for each technology
    CostPV = factor['PV'] * CPV # US$b p.a.
    CostWind = factor['Wind'] * CWind # US$b p.a.
    CostHydro = factor['Hydro'] * GHydro # US$b p.a.
    CostFossil = factor['Fossil'] * GFossil # US$b p.a.
    CostPH = factor['PHP'] * CPHP + factor['PHS'] * CPHS - factor['LegPH'] # US$b p.a.
    CostInter = factor['Inter'] * CInter # US$b p.a.

    CostDC = np.array([factor['AWIJ'], factor['ANIT'], factor['BNIK'], factor['BNPL'], factor['BNSG'], factor['KHTH'], factor['KHVS'], factor['CNVH'], factor['INMM'], factor['IJIK'], factor['IJIS'], factor['IJIT'], factor['IJSG'], factor['IKIC'], factor['IMIP'], factor['IMIC'], factor['LATH'], factor['LAVH'], factor['MYSG'], factor['MYTH'], factor['MMTH'], factor['PLPV'], factor['PMPV']])
    CostDC = (CostDC * solution.CDC).sum() - factor['LegINTC'] # US$b p.a.
    CostAC = factor['ACPV'] * CPV + factor['ACWind'] * CWind # US$b p.a.

    # Calculate the average annual energy demand
    Energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a. (TWh?)
    Loss = np.sum(abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    # Calculate the levelised cost of electricity at a network level
    LCOE = (CostPV + CostWind + CostInter + CostHydro + CostFossil + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOEPV = CostPV / (Energy - Loss)
    LCOEWind = CostWind / (Energy - Loss)
    LCOEInter = CostInter / (Energy - Loss)# Inters is the number of external interconnections
    LCOEHydro = CostHydro / (Energy - Loss)
    LCOEFossil = CostFossil / (Energy - Loss)

    # Calculate the levelised cost of generation
    LCOG = (CostPV + CostHydro + CostWind + CostFossil) * pow(10, 3) / (GPV + GHydro + GWind + GFossil)
    LCOGP = CostPV * pow(10, 3) / GPV if GPV!=0 else 0
    LCOGW = CostWind * pow(10, 3) / GWind if GWind!=0 else 0
    LCOGH = CostHydro * pow(10, 3) / (GHydro) if (GHydro)!=0 else 0
    LCOGI = CostInter * pow(10, 3) / (GInter) if (GInter)!=0 else 0
    LCOGF = CostFossil * pow(10, 3) / (GFossil) if (GFossil)!=0 else 0

    # Calculate the levelised cost of balancing
    LCOB = LCOE - LCOG
    LCOBPH = CostPH / (Energy - Loss)
    LCOBT = (CostDC + CostAC) / (Energy - Loss)
    LCOBL = LCOB - LCOBPH - LCOBT

    print('Levelised costs of electricity:')
    print('\u2022 LCOE:', LCOE)
    print('\u2022 LCOG:', LCOG)
    print('\u2022 LCOB:', LCOB)
    print('\u2022 LCOG-PV:', LCOGP, '(%s)' % CFPV)
    print('\u2022 LCOG-Wind:', LCOGW, '(%s)' % CFWind)
    print('\u2022 LCOG-Import:', LCOGI)
    print('\u2022 LCOG-Hydro & other renewables:', LCOGH)
    print('\u2022 LCOG-Fossil fuels:', LCOGF)

    print('\u2022 LCOB-Pumped hydro:', LCOBPH)
    print('\u2022 LCOB-T:', LCOBT)
    print('\u2022 LCOB-Spillage & loss:', LCOBL)

    size = 24 + len(list(solution.CDC))
    D = np.zeros((3, size))
    header_GGTA = 'Annual demand (TWh),Annual Energy Losses (TWh),' \
             'PV Capacity (GW),PV Avg Annual Gen (GWh),Wind Capacity (GW),Wind Avg Annual Gen (GWh),Hydro Capacity (GW),' \
             'Hydro Avg Annual Gen (GWh),Fossil Capacity (GW),Fossil Generation (GWh),Inter Capacity (GW),Inter Generation (GWh),' \
             'PHES-PowerCap (GW),PHES-EnergyCap (GWh),CapDCO,CapDCS,CapAC,' \
             'LCOE,LCOG,LCOB,LCOG_PV,LCOG_Wind,LCOG_Hydro,LCOG_Inter,LCOGFossil,LCOBS_PHES,LCOBT,LCOB_LossesSpillage'

    CapDC = solution.CDC * np.array([2100, 1000, 900, 1300, 1300, 500, 200, 600, 1000, 900, 1400, 2100, 900, 600, 1000, 1000, 500, 500, 300, 1300, 700, 600, 400]) * pow(10, -3) # GW-km (1000)
    CapDCO = CapDC[[2, 5, 6, 7, 8, 10, 16, 17, 18, 19, 20]].sum() # GW-km (1000)
    CapDCS = CapDC[[0, 1, 3, 4, 9, 11, 12, 13, 14, 15, 21, 22]].sum() # GW-km (1000)
    CapAC = (10 * CPV + 200 * CWind) * pow(10, -3) # GW-km (1000)

    # D = np.zeros((1, 43))
    # D[0, :] = [Energy * pow(10, 3), Loss * pow(10, 3), CPV, GPV, CWind, GWind, CapHydro, GHydro, CInter, GInter, CPHP, CPHS] \
    #           + list(solution.CDC) \
    #           + [LCOE, LCOEPV, LCOEWind, LCOEInter, LCOEHydro, LCOEPH, LCOEDC, LCOEAC]

    D = np.zeros((1, 28))
    D[0, :] = [Energy * pow(10, 3), Loss * pow(10, 3),
               CPV, GPV, CWind, GWind, CapHydro, GHydro, CapFossil, GFossil, CInter, GInter, CPHP, CPHS,
               CapDCO, CapDCS, CapAC,
               LCOE, LCOG, LCOB, LCOGP, LCOGW, LCOGH, LCOGI, LCOGF, LCOBPH, LCOBT, LCOBL]

    np.savetxt('Results/GGTA_{}_{}.csv'.format(node, percapita), D, fmt='%s', delimiter=',',header=header_GGTA)
    print('Energy generation, storage and transmission information has been produced.')

    return True

def Information(x, flexible):
    """Dispatch: Statistics.Information(x, Hydro)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)
    Deficit = Reliability(S, flexible=flexible)

    try:
        assert Deficit.sum() * resolution - S.allowance < 0.1, 'Energy generation and demand are not balanced.'
    except AssertionError:
        pass

    if 'Super' in node:
        S.TDC = Transmission(S, output=True) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW

        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        S.MBaseload = GBaseload.copy() # MW

        S.MPV = S.GPV.copy()
        S.MWind = S.GWind.copy() if S.GWind.shape[1]>0 else np.zeros((intervals, 1))
        S.MInter = S.GInter.copy()

        S.MDischarge = np.tile(S.Discharge, (nodes, 1)).transpose()
        S.MDeficit = np.tile(S.Deficit, (nodes, 1)).transpose()
        S.MCharge = np.tile(S.Charge, (nodes, 1)).transpose()
        S.MStorage = np.tile(S.Storage, (nodes, 1)).transpose()
        S.MSpillage = np.tile(S.Spillage, (nodes, 1)).transpose()

    S.CDC = np.amax(abs(S.TDC), axis=0) * pow(10, -3) # CDC(k), MW to GW
    S.AWIJ, S.ANIT, S.BNIK, S.BNPL, S.BNSG, S.KHTH, S.KHVS, S.CNVH, S.INMM, S.IJIK, S.IJIS, S.IJIT, S.IJSG, S.IKIC, S.IMIP, S.IMIC, S.LATH, S.LAVH, S.MYSG, S.MYTH, S.MMTH, S.PLPV, S.PMPV = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))

    S.MHydro = np.tile(S.CHydro - 0.5 * S.EHydro / 8760, (intervals, 1)) * pow(10, 3) # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MFossil = S.MPeak - S.MHydro # Fossil fuels
    S.MHydro += S.MBaseload # Hydropower & other renewables

    S.MPHS = S.CPHS * np.array(S.CPHP) * pow(10, 3) / sum(S.CPHP)  # GW to MW

    # 'AW', 'AN', 'BN', 'KH', 'CN', 'IN', 'IJ', 'IK', 'IM', 'IP', 'IC', 'IS', 'IT', 'LA', 'MY', 'MM', 'PL', 'PM', 'PV', 'SG', 'TH', 'VH', 'VS'
    # S.AWIJ, S.ANIT, S.BNIK, S.BNPL, S.BNSG, S.KHTH, S.KHVS, S.CNVH, S.INMM, S.IJIK, S.IJIS, S.IJIT, S.IJSG, S.IKIC, S.IMIP, S.IMIC, S.LATH, S.LAVH, S.MYSG, S.MYTH, S.MMTH, S.PLPV, S.PMPV
    S.Topology = [-1 * S.AWIJ,
                  -1 * S.ANIT,
                  -1 * S.BNIK - S.BNPL - S.BNSG,
                  -1 * S.KHTH - S.KHVS,
                  -1 * S.CNVH,
                  -1 * S.INMM,
                  S.AWIJ - S.IJIK - S.IJIS - S.IJIT - S.IJSG,
                  S.BNIK + S.IJIK - S.IKIC,
                  -1 * S.IMIP - S.IMIC,
                  S.IMIP,
                  S.IKIC + S.IMIC,
                  S.IJIS,
                  S.ANIT + S.IJIT,
                  -1 * S.LATH - S.LAVH,
                  -1 * S.MYSG - S.MYTH,
                  S.INMM - S.MMTH,
                  S.BNPL - S.PLPV,
                  -1 * S.PMPV,
                  S.PLPV + S.PMPV,
                  S.BNSG + S.IJSG + S.MYSG,
                  S.KHTH + S.LATH + S.MYTH + S.MMTH,
                  S.CNVH + S.LAVH,
                  S.KHVS]

    LPGM(S)
    GGTA(S)

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':
    capacities = np.genfromtxt('Results/Optimisation_resultx_Super1_3.csv', delimiter=',')
    flexible = np.genfromtxt('Results/Dispatch_Flexible_Super1_3.csv', delimiter=',', skip_header=1)
    Information(capacities, flexible)