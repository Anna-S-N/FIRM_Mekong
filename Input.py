# Modelling input and assumptions
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from argparse import ArgumentParser

from numba import njit, float64, int64, boolean
from numba.experimental import jitclass

from Costs import UnitCosts, calculate_costs

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=8, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-e', default=3, type=int, required=False, help='per-capita electricity: 3, 10, 20, and 99 (PDP projections) MWh')
parser.add_argument('-n', default='TH_Iso_Grid', type=str, required=False, help='Mekong_Grid, TH_Iso_Grid, TH_Imp_Grid, Vietnam_Iso_Grid, Laos_Iso_Grid, KH, LA, VH, VS, TH ...') # TH_Iso = Isolated Thailand network, TH_imp = Thailand w imports, Mekong = Mekong Power Grid
parser.add_argument('-s', default='nuclear', type=str, required=False, help='nuclear, no_nuclear')
parser.add_argument('-f', default='flexible', type=str, required=False, help='flexible, new_modelled_baseline, modelled_newbuild')
parser.add_argument('-b', default='batteries', type=str, required=False, help='batteries, noBatteries')
args = parser.parse_args()

percapita, node, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario = (args.e, args.n, args.i, args.p, args.s, args.f, args.b)

CallBack=True

###### NODAL LISTS ######
Nodel = np.array(['KH', 'LAN', 'LAS', 'VH', 'VS', 'CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC', 'MY_I', 'MM_I', 'KH_I', 'LAS_I', 'LAN_I', 'CH_I', 'TH'])
PVl =   np.array(['KH']*1 + ['LAN']*1 + ['LAS']*1 + ['VH']*1 + ['VS']*1 + ['CACE']*1 + ['CACW']*1 + ['CACN']*1 + ['MAC']*1 + ['NAC']*1 + ['NEC']*1 + ['SAC']*1 + ['TH'])
pv_lb_np = np.array([0.] + 2*[0.] + [0.] + [0.] + [3.5] + [3.] + [2.3] + [0.2] + [11.] + [9.6] + [6.7] + [0.]) #Thailand constraints based on 2037 capacity in PDP2024 draft
pv_ub_np = np.array([1000.] + 2*[1000.] + [1000.] + [1000.] + 7*[1000.] + [0.])
phes_lb_np = np.array([2.8] + [2*0.] + [2.4] + [2.4] + 5*[0.] + [1.] + [0.] + 7*[0.] + [0.]) # Lamtakong Jolabha Vadhana in Thailand (NEC) is 1000 MW
phes_ub_np = np.array([500.] + 2*[500.] + [500.] + [500.] + 7*[500.] + 7*[0.] + [0.])
storage_lb_np = np.array(19*[0.])
storage_ub_np = np.array(5*[20000.] + 7*[3000.] + 7*[0.])

if battery_scenario == 'batteries':
    battery_lb_np = np.array([0.3] + [2*0.] + [0.] + [0.] + 5*[0.] + [0.] + [0.] + 7*[0.] + [0.]) 
    battery_ub_np = np.array(12*[500.] + 6*[0.] + [0.]) 
else:
    battery_lb_np = np.array([0.] + [2*0.] + [0.] + [0.] + 5*[0.] + [0.] + [0.] + 7*[0.] + [0.]) 
    battery_ub_np = np.array(12*[0.] + 6*[0.] + [0.]) 

Windl = np.array(['KH']*1 + ['LAN']*1 + ['LAS']*1 + ['VH']*2 + ['VS']*2 + ['CACE']*1 + ['CACW']*1 + ['CACN']*1 + ['MAC']*1 + ['NAC']*1 + ['NEC']*1 + ['SAC']*1 + ['TH'])
wind_lb_np = np.array([0.] + 2*[0.] + [0.]*2 + [0.]*2 + 7*[0.] + [0.]) 
wind_ub_np = np.array([1000.] + 2*[1000.] + [1000.]*2+ [1000.]*2 + 7*[1000.] + [0.])
CInter_mask = np.zeros(len(Nodel),dtype=np.int64)
Interl = np.array([])
inters_lb_np = np.array([])
inters_ub_np = np.array([])
    
resolution = 1

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

###### DATA Imports ######
MLoad = np.genfromtxt('Data/electricity{}.csv'.format(percapita), delimiter=',', skip_header=1) # EOLoad(t, j), MW
TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1) # TSWind(t, i), MW

if nuclear_scenario == 'nuclear' and hydro_scenario == 'flexible':
    assets = np.genfromtxt('Data/assets_nuclear.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
elif nuclear_scenario == 'no_nuclear' and hydro_scenario == 'flexible':
    assets = np.genfromtxt('Data/assets.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
else:
    assets = np.genfromtxt('Data/assets_hydro_outCatchment.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)

CHydro, CGeo, CBio, CWaste, CNuclear = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])]
#constraints = np.genfromtxt('Data/constraints.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
#print(constraints)
#EHydro, EGeo, EBio, EWaste, ENuclear = np.genfromtxt('Data/constraints.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
CBaseload = CNuclear + CGeo

if hydro_scenario == 'flexible':
    hydro_baseload = np.zeros(MLoad.shape, dtype=np.float64)
    CPeak = CHydro + CBio + CWaste
elif hydro_scenario == 'new_modelled_baseline':
    hydro_baseload = np.genfromtxt('Data/new_hydro_baseline.csv', delimiter=',', skip_header=1)
    CPeak = CBio + CWaste + CHydro
elif hydro_scenario == 'modelled_newbuild':
    hydro_baseload = np.genfromtxt('Data/hydro_newbuild.csv', delimiter=',', skip_header=1)
    CPeak = CBio + CWaste + CHydro


###### CONSTRAINTS ######
# Energy constraints
hydro_weekly_cf = np.genfromtxt('Data/hydro.csv', delimiter=',', skip_header=1)

hydromax_weeks = 1000*CHydro*7*24 * hydro_weekly_cf[::7*24, :] # Weekly energy that can be generated by the hydro, MWh/week

""" Interconnection Order: [0, 18], #KH-TH
                        [0, 4], #KH-VS
                        [1, 18], #LAN-TH
                        [2, 18], #LAS-TH
                        [0, 2], #KH-LAS
                        [4, 2], #VS-LAS
                        [1, 3], #LAN-VH
                        [3, 4], #VH-VS
                        [2, 1], #LAS-LAN
                        [3, 14], #VH-CH_I
                        [12, 18], #MY_I-TH
                        [13, 18], #MM_I-TH

                        [11, 6], #SAC-CACW 
                        [6, 8], #CACW-MAC
                        [8, 5], #MAC-CACE
                        [8, 7], #MAC-CACN
                        [7, 9], #CACN-NAC
                        [7, 10], #CACN-NEC
                        [9, 10], #NAC-NEC
                        [11, 12], #SAC-MY_I
                        [9, 13], #NAC-MM_I
                        [5, 14], #CACE-KH_I
                        [10, 15], #NEC-LAS_I
                        [9, 16], #NAC-LAN_I """

transmission_loss_factors = np.array([0.07, 0.03, 0.03, 0.07, 0.03, 0.07, 0.07, 0.07, 0.07, 0.07, 0.03, 0.03,
                   0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.03, 0.03, 0.07, 0.07, 0.03], dtype=np.float64) # HVDC from https://www.adb.org/sites/default/files/publication/846471/power-trade-greater-mekong-subregion.pdf
#transmission_loss_factors = np.array([0.07, 0.03, 0.03, 0.07, 0.03, 0.07, 0.07, 0.07, 0.07, 0.07, 0.03, 0.03,
#                  0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.03], dtype=np.float64)
hvdc_mask = (transmission_loss_factors == 0.03)
DCdistance = np.array([550, 350, 640, 760, 460, 670, 520, 1280, 590, 470, 1180, 530,
                   670, 70, 110, 90, 360, 360, 440, 450, 280, 520, 320, 370], dtype=np.float64)
#DCdistance = np.array([550, 350, 640, 760, 460, 670, 520, 1280, 590, 470, 1180, 530,
#                   670, 70, 110, 90, 360, 360, 440, 450], dtype=np.float64)
DCloss = DCdistance * transmission_loss_factors * pow(10, -3)

###### Simulation Period ######
firstyear, finalyear, timestep = (2010, 2019, 1)

###### Scenario adjustments ######
# Node Values
if 'Grid' not in node:
    if 'TH' == node: # For MRC report, treat Thailand as a copper plate node like other regions
        coverage = np.array(['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC'])

        MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]].sum(axis=1, keepdims=True)
        TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]].mean(axis=1, keepdims=True)
        TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]].mean(axis=1, keepdims=True)

        CHydro, CBaseload, CPeak, CInter_mask = [x[np.where(np.in1d(Nodel, coverage)==True)[0]].sum(keepdims=True) for x in (CHydro, CBaseload, CPeak, CInter_mask)]

        hydromax_weeks = hydromax_weeks[:, np.where(np.in1d(Nodel, coverage)==True)[0]].mean(axis=1, keepdims=True)
        hydro_baseload = hydro_baseload[:, np.where(np.in1d(Nodel, coverage)==True)[0]].sum(axis=1, keepdims=True)
        hydro_weekly_cf = hydro_weekly_cf[:, np.where(np.in1d(Nodel, coverage)==True)[0]].mean(axis=1, keepdims=True)

        pv_lb_np, pv_ub_np = [x[np.where(np.in1d(PVl, coverage)==True)[0]].sum(keepdims=True) for x in (pv_lb_np, pv_ub_np)]
        wind_lb_np, wind_ub_np = [x[np.where(np.in1d(Windl, coverage)==True)[0]].sum(keepdims=True) for x in (wind_lb_np, wind_ub_np)]
        phes_lb_np, phes_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]].sum(keepdims=True) for x in (phes_lb_np, phes_ub_np)]
        storage_lb_np, storage_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]].sum(keepdims=True) for x in (storage_lb_np, storage_ub_np)]
        battery_lb_np, battery_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]].sum(keepdims=True) for x in (battery_lb_np, battery_ub_np)]

    else:
        MLoad = MLoad[:, np.where(Nodel==node)[0]]
        TSPV = TSPV[:, np.where(PVl==node)[0]]
        TSWind = TSWind[:, np.where(Windl==node)[0]]

        CHydro = CHydro[np.where(Nodel==node)[0]]
        #EHydro = EHydro[np.where(Nodel==node)[0]] # GWh
        #EBio = EBio[np.where(Nodel==node)[0]] # GWh
        #EWaste = EWaste[np.where(Nodel==node)[0]] # GWh
        CBaseload = CBaseload[np.where(Nodel==node)[0]] # GW
        CPeak = CPeak[np.where(Nodel==node)[0]] # GW
        CInter_mask = CInter_mask[np.where(Nodel==node)[0]] # GW

        hydromax_weeks = hydromax_weeks[:, np.where(Nodel==node)[0]] 
        hydro_baseload = hydro_baseload[:, np.where(Nodel==node)[0]]
        hydro_weekly_cf = hydro_weekly_cf[:, np.where(Nodel==node)[0]]

        pv_lb_np = pv_lb_np[np.where(PVl==node)[0]]
        pv_ub_np = pv_ub_np[np.where(PVl==node)[0]]
        phes_lb_np = phes_lb_np[np.where(Nodel==node)[0]]
        phes_ub_np = phes_ub_np[np.where(Nodel==node)[0]]
        storage_lb_np = storage_lb_np[np.where(Nodel==node)[0]]
        storage_ub_np = storage_ub_np[np.where(Nodel==node)[0]]
        wind_lb_np = wind_lb_np[np.where(Windl==node)[0]]
        wind_ub_np = wind_ub_np[np.where(Windl==node)[0]]
        battery_lb_np = battery_lb_np[np.where(Nodel==node)[0]]
        battery_ub_np = battery_ub_np[np.where(Nodel==node)[0]]
        
    Nodel_int = Nodel_int[np.where(Nodel==node)[0]]
    Nodel = Nodel[np.where(Nodel==node)[0]]
    PVl_int = PVl_int[np.where(PVl==node)[0]]
    PVl = PVl[np.where(PVl==node)[0]]
    Windl_int = Windl_int[np.where(Windl==node)[0]]
    Windl = Windl[np.where(Windl==node)[0]] 

    network_mask = np.zeros(len(DCloss), np.bool_)
    network = np.empty((0,0,0,0), np.int64)
    trans_tdc_mask = np.zeros((MLoad.shape[1], len(network)), np.bool_)
    for line, row in enumerate(network):
        trans_tdc_mask[row[0], line] = True
    directconns=np.empty((0,0), np.int64)

###### Transmission Network ######
if 'Grid' in node: 
    coverage = []
    if 'Mekong' in node:
        # Define aggregate TH node for regional grid
        TH_coverage = np.array(['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC'])
        MLoad[:, np.where(Nodel=='TH')[0][0]] = MLoad[:, np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum(axis=1)
        TSPV[:, np.where(PVl=='TH')[0][0]] = TSPV[:, np.where(np.in1d(PVl, TH_coverage)==True)[0]].mean(axis=1)
        TSWind[:, np.where(Windl=='TH')[0][0]] = TSWind[:, np.where(np.in1d(Windl, TH_coverage)==True)[0]].mean(axis=1)

        CHydro[np.where(Nodel=='TH')[0][0]], CBaseload[np.where(Nodel=='TH')[0][0]], CPeak[np.where(Nodel=='TH')[0][0]], CInter_mask[np.where(Nodel=='TH')[0][0]] = [x[np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum() for x in (CHydro, CBaseload, CPeak, CInter_mask)]

        hydromax_weeks[:, np.where(Nodel=='TH')[0][0]] = hydromax_weeks[:, np.where(np.in1d(Nodel, TH_coverage)==True)[0]].mean(axis=1)
        hydro_baseload[:, np.where(Nodel=='TH')[0][0]] = hydro_baseload[:, np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum(axis=1)
        hydro_weekly_cf[:, np.where(Nodel=='TH')[0][0]] = hydro_weekly_cf[:, np.where(np.in1d(Nodel, TH_coverage)==True)[0]].mean(axis=1)

        pv_lb_np[np.where(PVl=='TH')[0][0]], pv_ub_np[np.where(PVl=='TH')[0][0]] = [x[np.where(np.in1d(PVl, TH_coverage)==True)[0]].sum() for x in (pv_lb_np, pv_ub_np)]
        wind_lb_np[np.where(Windl=='TH')[0][0]], wind_ub_np[np.where(Windl=='TH')[0][0]] = [x[np.where(np.in1d(Windl, TH_coverage)==True)[0]].sum() for x in (wind_lb_np, wind_ub_np)]
        phes_lb_np[np.where(Nodel=='TH')[0][0]], phes_ub_np[np.where(Nodel=='TH')[0][0]] = [x[np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum() for x in (phes_lb_np, phes_ub_np)]
        storage_lb_np[np.where(Nodel=='TH')[0][0]], storage_ub_np[np.where(Nodel=='TH')[0][0]] = [x[np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum() for x in (storage_lb_np, storage_ub_np)]
        battery_lb_np[np.where(Nodel=='TH')[0][0]], battery_ub_np[np.where(Nodel=='TH')[0][0]] = [x[np.where(np.in1d(Nodel, coverage)==True)[0]].sum() for x in (battery_lb_np, battery_ub_np)]

        coverage = np.array(['KH', 'LAN', 'LAS', 'VH', 'VS', 'MY_I','MM_I','CH_I','TH'])

        Interl = np.array(['MY_I']*1 + ['MM_I']*1 + ['CH_I']*1)
        inters_lb_np = np.array([0.]*3)
        inters_ub_np = np.array([1.1, 0.4, 2.]) #ASEAN Power Grid + 2 GW from China
        
    if 'TH_Iso' in node:
        coverage = np.array(['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC'])

    if 'TH_Imp' in node:
        coverage = np.array(['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC', 'MY_I', 'MM_I', 'KH_I', 'LAS_I', 'LAN_I'])

        Interl = np.array(['KH_I']*1 + ['LAS_I']*1 + ['LAN_I']*1 + ['MY_I']*1 + ['MM_I']*1)

        inters_lb_np = np.array([0., 0., 0., 0., 0.])
        inters_ub_np = np.array([2.2, 6.6, 4.1, 0.4, 1.1]) # ASEAN Power Grid plans + PDP2024

    if 'Laos_Iso' in node:
        coverage = np.array(['LAN', 'LAS'])

    if 'Vietnam_Iso' in node:
        coverage = np.array(['VH', 'VS'])
    
            
    if 'TH_LA' in node:
        coverage = np.array(['TH', 'LAN', 'LAS'])
    
    
    ####### APPLY COVERAGE TO THE INPUTS ###########
        
    MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]]
    TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]]

    CHydro, CBaseload, CPeak = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBaseload, CPeak)]

    hydro_baseload = hydro_baseload[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    hydromax_weeks = hydromax_weeks[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    hydro_weekly_cf = hydro_weekly_cf[:, np.where(np.in1d(Nodel, coverage)==True)[0]]

    pv_lb_np, pv_ub_np = [x[np.where(np.in1d(PVl, coverage)==True)[0]] for x in (pv_lb_np, pv_ub_np)]
    wind_lb_np, wind_ub_np = [x[np.where(np.in1d(Windl, coverage)==True)[0]] for x in (wind_lb_np, wind_ub_np)]
    phes_lb_np, phes_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (phes_lb_np, phes_ub_np)]
    storage_lb_np, storage_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (storage_lb_np, storage_ub_np)]
    battery_lb_np, battery_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (battery_lb_np, battery_ub_np)]

    CInter_mask = CInter_mask[np.where(np.in1d(Nodel, coverage)==True)[0]]
    CInter_mask[np.where(np.in1d(coverage, Interl)==True)[0]] = 1

    Nodel_int = Nodel_int[np.where(np.in1d(Nodel, coverage)==True)[0]]
    PVl_int = PVl_int[np.where(np.in1d(PVl, coverage)==True)[0]]
    Windl_int = Windl_int[np.where(np.in1d(Windl, coverage)==True)[0]]
    Nodel, PVl, Windl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Windl)]

    ########## BUILD TRANSMISSION NETWORK VARIABLES ###############

    network = np.array([[0, 18], #KH-TH
                        [0, 4], #KH-VS
                        [1, 18], #LAN-TH
                        [2, 18], #LAS-TH
                        [0, 2], #KH-LAS
                        [4, 2], #VS-LAS
                        [1, 3], #LAN-VH
                        [3, 4], #VH-VS
                        [2, 1], #LAS-LAN
                        [3, 17], #VH-CH_I
                        [12, 18], #MY_I-TH
                        [13, 18], #MM_I-TH

                        [11, 6], #SAC-CACW 
                        [6, 8], #CACW-MAC
                        [8, 5], #MAC-CACE
                        [8, 7], #MAC-CACN
                        [7, 9], #CACN-NAC
                        [7, 10], #CACN-NEC
                        [9, 10], #NAC-NEC
                        [11, 12], #SAC-MY_I
                        [9, 13], #NAC-MM_I
                        [5, 14], #CACE-KH_I
                        [10, 15], #NEC-LAS_I
                        [9, 16], #NAC-LAN_I
                        ], dtype=np.int64)
        
    # Find and select connections between nodes being considered
    network_mask = np.array([(network==j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0)==2
    network = network[network_mask,:]
    networkdict = {v:k for k, v in enumerate(Nodel_int)}
    #translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], dtype=np.int64).reshape(network.shape)
    
    trans_tdc_mask = np.zeros((MLoad.shape[1], len(network)), dtype=np.bool_)
    for line, row in enumerate(network):
        trans_tdc_mask[row[0], line] = True
    
    directconns = -1*np.ones((len(Nodel)+1, len(Nodel)+1), dtype=np.int64)
    for n, row in enumerate(network):
        directconns[*row] = n
        directconns[*row[::-1]] = n
    
    
    def network_neighbours(n):
        isn_mask = np.isin(network, n)
        hasn_mask = isn_mask.sum(axis=1).astype(bool)
        joins_n = network[hasn_mask][~isn_mask[hasn_mask]]
        return joins_n
    
    def nthary_network(network_1):
        """primary, secondary, tertiary, ..., nthary"""
        """supply n-1thary to generate nthary etc."""
        networkn = -1*np.ones((1,network_1.shape[1]+1), dtype=np.int64)
        for row in network_1:
            _networkn = -1*np.ones((1,network_1.shape[1]+1), dtype=np.int64)
            joins_start = network_neighbours(row[0])
            joins_end = network_neighbours(row[-1])
            for n in joins_start:
                if n not in row:
                    _networkn = np.vstack((_networkn, np.insert(row, 0, n)))
            for n in joins_end:
                if n not in row:
                    _networkn = np.vstack((_networkn, np.append(row, n)))
            _networkn=_networkn[1:,:]
            dup=[]
            # find rows which are already in network
            for i, r in enumerate(_networkn): 
                for s in networkn:
                    if np.setdiff1d(r, s).size==0:
                        dup.append(i)
            # find duplicated rows within n3
            for i, r in enumerate(_networkn):
                for j, s in enumerate(_networkn):
                    if i==j:
                        continue
                    if np.setdiff1d(r, s).size==0:
                        dup.append(i)
            _networkn = np.delete(_networkn, np.unique(np.array(dup, dtype=np.int64)), axis=0)
            if _networkn.size>0:
                networkn = np.vstack((networkn, _networkn))
        networkn = networkn[1:,:]
        return networkn
    
    #This version of FIRM maxes out at quarternary transmission
    networks = [network]
    while True:
        n = nthary_network(networks[-1])
        if n.size > 0:
            networks.append(n)
        else: 
            break

    def count_lines(network):
        unique, counts = np.unique(network[:, np.array([0,-1])], return_counts=True)
        if counts.size > 0:
            return counts.max()
        return 0
    maxconnections = max([count_lines(network) for network in networks])

    perfect = np.array([0,1,3,6,10,15,21,28,36]) #that's more than enough for now

    network = -1*np.ones((2, len(Nodel), perfect[len(networks)], maxconnections), dtype=np.int64)
    for i, net in enumerate(networks):
        conns = np.zeros(len(Nodel), int)
        for j, row in enumerate(net):
            network[0, row[0], perfect[i]:perfect[i+1], conns[row[0]]] = row[1:]
            network[0, row[-1], perfect[i]:perfect[i+1], conns[row[-1]]] = row[:-1][::-1]
            conns[row[0]]+=1
            conns[row[-1]]+=1
            
    for i in range(network.shape[1]):
        for j in range(network.shape[2]):
            for k in range(network.shape[3]):
                if j in perfect:
                    start=i
                else: 
                    start=network[0, i, j-1, k]
                network[1, i, j, k] = directconns[start, network[0, i, j, k]]

    directconns=directconns[:-1, :-1]
    
    # =============================================================================
    # network is a 4d array representing network connections 
    
    # The first index specifies whether we are talking about node-node connections 
    #   or the lines which connect the nodes
    # each element in network[0, :, :, :] represents a node-node connection
    # each element in network[1, :, :, :] indicates the line that that connection uses
    
    # The second index is the reference node and the third index relates to the 
    #   length of the connection
    # network[:, m, :, :] shows the network connections to node m 
    # network[:, m, 0, :] shows the primary (direct) connections 
    # network[:, m, 1:3, :] shows the secondary connections 
    #       (i.e. m connects to network[:, m, 2, :] via network[:, m, 1, :])
    # network[:, m, 3:6, :] shows the tertiary connections etc. 

    # the fourth  index is used to hold multiple connections 

    # -1 is used ot portray empty value
    # =============================================================================

###### DECISION VARIABLE LIST INDEXES ######
Windl_Viet_int = np.array([n_node[node] for node in ['VH', 'VS']], dtype=np.int64)[::2]
intervals, nodes = MLoad.shape # The no. of intervals equals the no. of rows in the MLoad variable. The no. of nodes equals the no. of columns in MLoad 
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1]) # Number of solar and wind sites
pidx, widx = (pzones, pzones + wzones) # Integers that define the final index of solar, wind, phes, etc. sites within the decision variable list
spidx, seidx = pzones + wzones + nodes, pzones + wzones + nodes + nodes
bpidx, bhidx = seidx + nodes, seidx + nodes + nodes
inters = len(Interl) # The number of external interconnections
iidx = bhidx + inters

energy = MLoad.sum() * resolution / years # MWh p.a.

baseload = np.tile(CBaseload, (intervals, 1)) * 1000 # GW to MW, excluding hydro

###### Network Constraints
allowance = min(0.00002*np.reshape(MLoad.sum(axis=1), (-1,8760)).sum(axis=-1)) # Allowable annual deficit of 0.002%, MWh
efficiency = 0.8 #80% round-trip efficiency PHES

###### DECISION VARIABLE UPPER AND LOWER BOUNDS ######
CDCmax = 300.*np.ones_like(DCloss[network_mask], dtype=np.float64)
hvdc_mask = hvdc_mask[network_mask]

pv_lb = list(pv_lb_np)
pv_ub = list(pv_ub_np)
wind_lb = list(wind_lb_np)
wind_ub = list(wind_ub_np)
phes_lb = list(phes_lb_np)
phes_ub = list(phes_ub_np)
storage_lb = list(storage_lb_np)
storage_ub = list(storage_ub_np)
battery_lb = list(battery_lb_np)
battery_ub = list(battery_ub_np)
if battery_scenario == 'batteries':
    bduration_lb = list([2.]*(nodes-inters)+[0.]*inters)
    bduration_ub = list([24.]*(nodes-inters)+[0.]*inters)
else:
    bduration_lb = list([0.]*nodes)
    bduration_ub = list([0.]*nodes)
inters_lb = list(inters_lb_np)
inters_ub = list(inters_ub_np)
transmission_lb = list(np.array([0.] * network_mask.sum()))
transmission_ub = list(CDCmax)

print(pv_lb, wind_lb, phes_lb, storage_lb, battery_lb, bduration_lb, inters_lb, transmission_lb)
print(pv_ub, wind_ub, phes_ub, storage_ub, battery_ub, bduration_ub, inters_ub, transmission_ub)

lb = np.array(pv_lb + wind_lb + phes_lb + storage_lb + battery_lb + bduration_lb + inters_lb + transmission_lb)
ub = np.array(pv_ub + wind_ub + phes_ub + storage_ub + battery_ub + bduration_ub + inters_ub + transmission_ub)

from Simulation import Reliability

# Reshape is not compatible with JIT, so need below function for hydromax constraints
@njit()
def weekly_sum(original_array):
    num_rows, num_cols = original_array.shape
    new_num_rows = num_rows // (7*24) + 1
    new_array = np.zeros((new_num_rows, num_cols), dtype=np.float64)
    
    for i in range(new_num_rows):
        for j in range(7*24):
            if i * 7*24 + j >= num_rows:
                break
            for k in range(num_cols):
                new_array[i, k] += original_array[i * 7*24 + j, k]    
    return new_array

@njit()
def F(S):
    # Simulation with baseload
    #print(pidx,widx,spidx,seidx,bpidx,bhidx,iidx,len(S.x))
    Deficit1 = Reliability(S, flexible=np.zeros((intervals, nodes), dtype=np.float64), agg_storage=True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64)) # Sj-EDE(t, j), MW
    hydrobio1 = np.maximum(np.minimum(Deficit1,CPeak*1000),0)

    # Simulation with baseload + flexible hydro and bio
    if 'Grid' in node:
        Deficit2 = Reliability(S, flexible=hydrobio1, agg_storage=True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64)) # Sj-EDE(t, j), MW 
    else:
        Deficit2 = np.zeros((intervals,nodes), dtype=np.float64)

    """ print("1: ",Deficit1)
    print("2: ",Deficit2) """
    
    # Clip all of the profiles based on capacity at each node
    Flexible = np.maximum(Deficit1-Deficit2,0)
    imports = np.minimum(Deficit2,S.CInter*1000)
    hydrobio = np.minimum(Flexible,CPeak*1000)
    hydro = np.minimum(hydrobio,CHydro*1000)

    """ print("CInter: ",S.CInter)
    print("imports: ",imports)
    print("hydrobio: ",hydrobio) """

    # Constrain weekly generation from hydro
    Hydro_Weekly = weekly_sum(hydro)
    PenHydro = np.sum(np.maximum(Hydro_Weekly - hydromax_weeks, 0)) *1000000 # Create positive, large penalty number

    GHydroBio = (hydrobio.sum() / efficiency + hydro_baseload.sum()) * resolution / years  # MWh p.a.  
    GImports = imports.sum() * resolution / years / efficiency # MWh p.a.  
    GBaseload = baseload.sum() * resolution / years

    # Deficit calculation
    Deficit = Reliability(S, flexible=hydrobio+imports, agg_storage=True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64)) # Sj-EDE(t, j), GW to MW
    
    PenDeficit = np.maximum(0., Deficit.sum() * resolution - allowance) * 1000000 # MWh

    TDC = S.TDC
    S.TDCabs = np.abs(TDC) # Absolute value of energy flows for VOM calculation and losses

    cost, _ = calculate_costs(S, S.Discharge*(S.CPHS.sum()/(S.CPHS.sum()+S.CBS.sum())) , GHydroBio, GImports, GBaseload)
    #cost = calculate_costs(S, S.Discharge*(S.CPHS.sum()/(S.CPHS.sum()+S.CBS.sum())), GHydroBio, GImports, GBaseload)
    
    loss = np.zeros(len(network_mask), dtype=np.float64)
    loss[network_mask] = S.TDCabs.sum(axis=0) * DCloss[network_mask]
    loss = loss.sum() * resolution / years # MWh p.a.
    LCOE = cost / np.abs(energy - loss)

    """ print("Costs", cost/ np.abs(energy - loss))
    print("Tech: ", _/ np.abs(energy - loss))
    print("Energies: ", energy, loss)
    
    netload = S.MLoad.sum(axis=1)- S.GPV.sum(axis=1)- S.GWind.sum(axis=1)- S.baseload.sum(axis=1)- S.hydro_baseload.sum(axis=1)- hydrobio.sum(axis=1)- imports.sum(axis=1)- S.Discharge.sum(axis=1) + S.Charge.sum(axis=1)+ S.Spillage.sum(axis=1) - S.Deficit.sum(axis=1)
    balance = np.stack([S.MLoad.sum(axis=1), S.GPV.sum(axis=1), S.GWind.sum(axis=1), S.baseload.sum(axis=1), S.hydro_baseload.sum(axis=1), hydrobio.sum(axis=1), imports.sum(axis=1), S.Discharge.sum(axis=1), -1*S.Charge.sum(axis=1), S.Spillage.sum(axis=1), S.Storage.sum(axis=1), netload], axis=1)
    np.savetxt('Results/Balance.csv', balance, fmt='%f', delimiter=',', newline='\n')    
    np.savetxt('Results/TDC.csv', TDC, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/hydrobio1.csv', hydrobio1, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/hydrobio.csv', hydrobio, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/imports.csv', imports, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/Deficit1.csv', Deficit1, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/Deficit2.csv', Deficit2, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/hydro_weekly.csv', Hydro_Weekly, fmt='%f', delimiter=',', newline='\n')
    np.savetxt('Results/hydromax_weeks.csv', hydromax_weeks, fmt='%f', delimiter=',', newline='\n') """

    return LCOE, (PenHydro+PenDeficit)

solution_spec = [
    ('x', float64[:]),  # x is 1d array
    ('DCdistance', float64[:]),
    ('DCloss', float64[:]),
    ('UnitCosts', float64[:]),
    ('TDCabs', float64[:, :]),
    ('Storage_cost', float64),
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int64), # plain integer
    ('nodes', int64),
    ('years', int64),
    ('resolution',float64), # plain float
    ('allowance',float64),
    ('CPV', float64[:]), 
    ('CWind', float64[:]), 
    ('CInter', float64[:]),
    ('CBaseload', float64[:]),
    ('CPHP', float64[:]),
    ('CPHS', float64[:]),
    ('CBP', float64[:]),
    ('CBH', float64[:]),
    ('CBS', float64[:]),
    ('CHVDC', float64[:]),
    ('GPV', float64[:, :]), 
    ('GWind', float64[:, :]), 
    ('GWind_sites', float64[:, :]), 
    ('efficiency', float64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('Windl_int', int64[:]),
    ('Windl_Viet_int', int64[:]),
    ('baseload', float64[:, :]),
    ('hydro_baseload', float64[:, :]),
    ('GHydro', float64[:, :]), 
    ('CPeak', float64[:]), 
    ('CHydro', float64[:]),
    ('EHydro', float64[:]),
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:]),
    ('Netload' ,float64[:,:]),
    ('Import' ,float64[:,:]),
    ('Export' ,float64[:,:]),
    ('Penalties', float64),
    ('Lcoe', float64),
    ('evaluated', boolean),
    # ('MPV', float64[:, :]),
    # ('MWind', float64[:, :]),
    ('MBaseload', float64[:, :]),
    #('MPeak', float64[:, :]),
    #('MFossil', float64[:, :]),
    # ('MDischarge', float64[:, :]),
    # ('MCharge', float64[:, :]),
    # ('MStorage', float64[:, :]),
    # ('MDeficit', float64[:, :]),
    # ('MSpillage', float64[:, :]),
    ('MHydro', float64[:, :]),
    ('CDP', float64[:]),
    ('CDS', float64[:]),
    ('TDC', float64[:, :]),
    ('CDC', float64[:]),
    ('Topology', float64[:, :]),
    ('network', int64[:, :, :, :]),
    ('directconns', int64[:,:]),
    ('trans_tdc_mask', boolean[:,:]),
    ('hvdc_mask', boolean[:]),
    ('battery_charge', float64[:,:]),
    ('battery_discharge', float64[:,:]),
    ('BStorage', float64[:,:]),
]


@jitclass(solution_spec)
class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        
        self.intervals = intervals
        self.nodes = nodes
        self.years = years
        self.resolution = resolution
        self.network, self.directconns = network, directconns
        self.trans_tdc_mask = trans_tdc_mask
        self.hvdc_mask = hvdc_mask
        self.Windl_Viet_int = Windl_Viet_int
        
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW

        _CInter = x[bhidx:iidx]
        CInter = np.zeros(len(CInter_mask), dtype=np.float64)
        counter = 0
        for i in range(len(CInter)):
            if CInter_mask[i] == 1:
                CInter[i] = _CInter[counter]
                counter+=1
        self.CInter = CInter
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)), dtype=np.float64)
        CWind_tiled = np.zeros((intervals, len(self.CWind)), dtype=np.float64)
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]

        GPV = TSPV * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        GWind = TSWind * CWind_tiled * 1000.  # GWind(i, t), GW to MW
        self.GWind_sites = GWind

        self.GPV, self.GWind = np.empty((intervals, nodes), np.float64), np.empty((intervals, nodes), np.float64)
        for i, j in enumerate(Nodel_int):
            self.GPV[:,i] = GPV[:, PVl_int==j].sum(axis=1)
            self.GWind[:,i] = GWind[:, Windl_int==j].sum(axis=1) 
        
        self.CPHP = x[widx: spidx]  # CPHP(j), GW
        self.CPHS = x[spidx: seidx]  # S-CPHS(j), GWh
        self.CBP = x[seidx: bpidx] # GW
        self.CBH = x[bpidx: bhidx] # hours
        self.CBS = self.CBP * self.CBH # GWh

        self.DCloss = DCloss
        self.CHVDC = x[iidx:] # GW
        self.DCdistance = DCdistance
        
        self.efficiency = efficiency

        self.baseload = baseload # MWh
        self.hydro_baseload = hydro_baseload
        self.CPeak = CPeak # GW
        self.CHydro = CHydro # GW
        self.CBaseload = CBaseload # GW
        
        self.UnitCosts = UnitCosts
        
        self.allowance = allowance
        
        self.evaluated=False
        
    def _evaluate(self):
        self.Lcoe, self.Penalties = F(self)
        self.evaluated=True

if __name__ == '__main__':
    x = np.genfromtxt('Results/Optimisation_resultx_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',')
    def test():
        #x = np.random.rand(len(lb))*(ub-lb)+ub
        S = Solution(x)
        S._evaluate()
        print(S.Lcoe, S.Penalties)
    test()