# Modelling input and assumptions
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
import pyomo.environ as pyo
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-y', default=1, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-e', default=3, type=int, required=False, help='per-capita electricity: 3, 6 and 9 MWh')
parser.add_argument('-n', default='Mekong_Grid', type=str, required=False, help='Mekong_Grid, TH_Iso_Grid, TH_Imp_Grid, Vietnam_Iso_Grid, Laos_Iso_Grid, KH, LA, VH, VS, TH ...') # TH_Iso = Isolated Thailand network, TH_imp = Thailand w imports, Mekong = Mekong Power Grid
parser.add_argument('-s', default='nuclear', type=str, required=False, help='nuclear, no_nuclear')
parser.add_argument('-f', default='modelled_newbuild', type=str, required=False, help='flexible, new_modelled_baseline, modelled_newbuild')
parser.add_argument('-b', default='batteries', type=str, required=False, help='batteries, noBatteries')

args = parser.parse_args()

percapita, node, years = args.e, args.n, args.y
nuclear_scenario, hydro_scenario, battery_scenario = args.s, args.f, args.b



###### NODAL LISTS ######
Nodel = np.array(['KH', 'LAN', 'LAS', 'VH', 'VS', 'CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC', 'MY_I', 'MM_I', 'KH_I', 'LAS_I', 'LAN_I', 'CH_I', 'TH'])
PVl =   np.array(['KH', 'LAN', 'LAS', 'VH', 'VS', 'CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC', 'TH'])
pv_lb_np = np.array([0.] + 2*[0.] + [0.] + [0.] + [3.5] + [3.] + [2.3] + [0.2] + [11.] + [9.6] + [6.7] + [0.]) #Thailand constraints based on 2037 capacity in PDP2024 draft
pv_ub_np = np.array([1000.] + 2*[1000.] + [1000.] + [1000.] + 7*[1000.] + [0.])
phes_lb_np = np.array([2.8] + [2*0.] + [2.4] + [2.4] + 5*[0.] + [1.] + [0.] + 7*[0.] + [0.]) # Lamtakong Jolabha Vadhana in Thailand (NEC) is 1000 MW
phes_ub_np = np.array([500.] + 2*[500.] + [500.] + [500.] + 7*[500.] + 7*[0.] + [0.])
storage_lb_np = np.array(19*[0.])
storage_ub_np = np.array(5*[20000.] + 7*[3000.] + 7*[0.])

if battery_scenario == 'batteries':
    battery_lb_np = np.array([0.3] + 18*[0.]) 
    battery_ub_np = np.array(12*[500.] + 6*[0.] + [0.]) 
else:
    battery_lb_np = np.array(19*[0.]) 
    battery_ub_np = np.array(19*[0.]) 

Windl = np.array(['KH', 'LAN', 'LAS'] + ['VH']*2 + ['VS']*2 + ['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC', 'TH'])
wind_lb_np = np.array(15*[0.]) 
wind_ub_np = np.array(14*[1000.] + [0.])
CInter_mask = np.zeros(len(Nodel),dtype=np.bool_)
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


CHydro, CGeo, CBio, CWaste, CNuclear = [assets[:, x] for x in range(assets.shape[1])]
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
elif hydro_scenario == 'newbuild_noSSS':
    hydro_baseload = np.genfromtxt('Data/hydro_newbuild_noSSS.csv', delimiter=',', skip_header=1)
    CPeak = CBio + CWaste + CHydro

###### CONSTRAINTS ######
# Energy constraints
hydro_weekly_cf = np.genfromtxt('Data/hydro.csv', delimiter=',', skip_header=1)

hydromax_weeks = 1000*CHydro*7*24 * hydro_weekly_cf[::7*24, :]

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

###### Scenario adjustments ######
# Node Values
if 'Grid' not in node:
    if 'TH' == node: # For MRC report, treat Thailand as a copper plate node like other regions
        coverage = np.array(['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC'])

        MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]].sum(axis=1, keepdims=True)
        TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]].mean(axis=1, keepdims=True)
        TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]].mean(axis=1, keepdims=True)

        CHydro, CBaseload, CPeak, CInter_mask, CBio, CWaste = [x[np.where(np.in1d(Nodel, coverage)==True)[0]].sum(keepdims=True) for x in (CHydro, CBaseload, CPeak, CInter_mask, CBio, CWaste)]

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

        CHydro, CBaseload, CPeak, CInter_mask, CBio, CWaste = (x[np.where(Nodel==node)[0]] for x in (CHydro, CBaseload, CPeak, CInter_mask, CBio, CWaste))
        #EHydro = EHydro[np.where(Nodel==node)[0]] # GWh
        #EBio = EBio[np.where(Nodel==node)[0]] # GWh
        #EWaste = EWaste[np.where(Nodel==node)[0]] # GWh

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
    

###### Transmission Network ######
if 'Grid' in node: 
    coverage = []
    if 'Mekong' in node:
        # Define aggregate TH node for regional grid
        TH_coverage = np.array(['CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC'])
        MLoad[:, np.where(Nodel=='TH')[0][0]] = MLoad[:, np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum(axis=1)
        TSPV[:, np.where(PVl=='TH')[0][0]] = TSPV[:, np.where(np.in1d(PVl, TH_coverage)==True)[0]].mean(axis=1)
        TSWind[:, np.where(Windl=='TH')[0][0]] = TSWind[:, np.where(np.in1d(Windl, TH_coverage)==True)[0]].mean(axis=1)

        (CHydro[np.where(Nodel=='TH')[0][0]], CBaseload[np.where(Nodel=='TH')[0][0]], 
         CPeak[np.where(Nodel=='TH')[0][0]], CInter_mask[np.where(Nodel=='TH')[0][0]],
         CBio[np.where(Nodel=='TH')[0][0]], CWaste[np.where(Nodel=='TH')[0][0]]) = [x[np.where(np.in1d(Nodel, TH_coverage)==True)[0]].sum() for x in (CHydro, CBaseload, CPeak, CInter_mask, CBio, CWaste)]

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
    coverage_int = np.array([n_node[node] for node in coverage])
    
    MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]]
    TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]]

    CHydro, CBaseload, CPeak, CBio, CWaste = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBaseload, CPeak, CBio, CWaste)]

    hydro_baseload = hydro_baseload[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    hydromax_weeks = hydromax_weeks[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    hydro_weekly_cf = hydro_weekly_cf[:, np.where(np.in1d(Nodel, coverage)==True)[0]]

    pv_lb_np, pv_ub_np = [x[np.where(np.in1d(PVl, coverage)==True)[0]] for x in (pv_lb_np, pv_ub_np)]
    wind_lb_np, wind_ub_np = [x[np.where(np.in1d(Windl, coverage)==True)[0]] for x in (wind_lb_np, wind_ub_np)]
    phes_lb_np, phes_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (phes_lb_np, phes_ub_np)]
    storage_lb_np, storage_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (storage_lb_np, storage_ub_np)]
    battery_lb_np, battery_ub_np = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (battery_lb_np, battery_ub_np)]

    CInter_mask = CInter_mask[np.where(np.in1d(Nodel, coverage)==True)[0]]
    CInter_mask[np.where(np.in1d(coverage, Interl)==True)[0]] = True

    Nodel, PVl, Windl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Windl)]
    Nodel_int, PVl_int, Windl_int = [x[np.where(np.in1d(x, coverage_int)==True)[0]] for x in (Nodel_int, PVl_int, Windl_int)]

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
   
###### Simulation Period ######
firstyear = 2010
finalyear = firstyear+years-1 
intervals = int(years*365*24/resolution)
###### DECISION VARIABLE LIST INDEXES ######
Windl_Viet_int = np.array([n_node[node] for node in ['VH', 'VS']], dtype=np.int64)[::2]
nodes = MLoad.shape[1] # The no. of intervals equals the no. of rows in the MLoad variable. The no. of nodes equals the no. of columns in MLoad 
pzones, wzones = (TSPV.shape[1], TSWind.shape[1]) # Number of solar and wind sites
pidx, widx = (pzones, pzones + wzones) # Integers that define the final index of solar, wind, phes, etc. sites within the decision variable list
spidx, seidx = pzones + wzones + nodes, pzones + wzones + nodes + nodes
bpidx, bhidx = seidx + nodes, seidx + nodes + nodes
inters = len(Interl) # The number of external interconnections
iidx = bhidx + inters

MLoad, hydro_baseload = (x[:intervals, :] for x in (MLoad, hydro_baseload))

energy = MLoad.sum() * resolution / years # MWh p.a.

baseload = np.tile(CBaseload, (intervals, 1)) #* 1000 # GW to MW, excluding hydro

Netload = (MLoad - hydro_baseload - baseload) 

Netload, MLoad, hydro_baseload, baseload = [x/1000 for x in (Netload, MLoad, hydro_baseload, baseload )]
CHydro, CBio, CWaste = [x/1000 for x in (CHydro, CBio, CWaste)]

###### Network Constraints
allowance = min(0.00002*np.reshape(MLoad.sum(axis=1), (-1,8760)).sum(axis=-1)) # Allowable annual deficit of 0.002%, MWh
efficiency = 0.8 #80% round-trip efficiency PHES

StartCharge = 0.5 #starting storage level 

###### DECISION VARIABLE UPPER AND LOWER BOUNDS ######
DCloss = DCloss[network_mask]
DCdistance = DCdistance[network_mask]
hvdc_mask = hvdc_mask[network_mask]
HVDCloss, HVACloss = DCloss[hvdc_mask], DCloss[~hvdc_mask]

offshore_mask = np.isin(Windl_int, Windl_Viet_int)

TSWind_on, TSWind_off = TSWind[:, ~offshore_mask], TSWind[:, offshore_mask]

CDCmax = 50.*np.ones_like(DCloss, dtype=np.float64)

pv_lb = list(pv_lb_np)
pv_ub = list(pv_ub_np/100)
wind_on_lb = list(wind_lb_np[~offshore_mask])
wind_on_ub = list(wind_ub_np[~offshore_mask]/100)
wind_off_lb = list(wind_lb_np[offshore_mask])
wind_off_ub = list(wind_ub_np[offshore_mask]/100)
phes_lb = list(phes_lb_np)
phes_ub = list(phes_ub_np/20)
storage_lb = list(storage_lb_np)
storage_ub = list(storage_ub_np/100)
battery_lb = list(battery_lb_np)
battery_ub = list(battery_ub_np/50)
if battery_scenario == 'batteries':
    bduration_lb_np = np.array([2.]*(nodes-inters)+[0.]*inters)
    bduration_ub_np = np.array([24.]*(nodes-inters)+[0.]*inters)
    bduration_lb = list(bduration_lb_np*battery_lb_np)
    bduration_ub = list(bduration_ub_np*battery_ub_np/100)
else:
    bduration_lb = list([0.]*nodes)
    bduration_ub = list([0.]*nodes)
inters_lb = list(inters_lb_np)
inters_ub = list(inters_ub_np)
hvdc_lb = list(np.array([0.] * hvdc_mask.sum()))
hvdc_ub = list(CDCmax[hvdc_mask])
hvac_lb = list(np.array([0.] * (~hvdc_mask).sum()))
hvac_ub = list(CDCmax[~hvdc_mask])

print(f"""
Bounds: 
\u2022pv:  {pv_lb}
 [GW] {pv_ub}
\u2022onshore wind: {wind_on_lb}
 [GW]          {wind_on_ub}
\u2022offshore wind: {wind_off_lb}
 [GW]           {wind_off_ub}
\u2022ph-power: {phes_lb}
 [GW]      {phes_ub}
\u2022ph-energy: {storage_lb}
 [GWh]      {storage_ub}
\u2022battery-power: {battery_lb}
 [GW]           {battery_ub}
\u2022battery-energy: {bduration_lb}
 [GWh]           {bduration_ub}
\u2022imports: {inters_lb}
 [GW]     {inters_ub}
\u2022hvdc: {hvdc_lb}
 [GW]  {hvdc_ub}
\u2022hvac: {hvac_lb}
 [GW]  {hvac_ub}
""")

lb = np.array(pv_lb + wind_on_lb + wind_off_lb + phes_lb + storage_lb + battery_lb + bduration_lb + inters_lb + hvdc_lb + hvac_lb)
ub = np.array(pv_ub + wind_on_ub + wind_off_ub + phes_ub + storage_ub + battery_ub + bduration_ub + inters_ub + hvdc_ub + hvac_ub)

ntrans = len(network)
nhvdc = hvdc_mask.sum()
nhvac = (~hvdc_mask).sum()
ninters = CInter_mask.sum()


#%%
class Solution:
    def __init__(self, model, years=years):
        self.node, self.nodes = node, nodes
        self.Nodel, self.PVl, self.Windl = Nodel, PVl, Windl
        
        self.network, self.network_mask = network, network_mask
        self.HVDCloss, self.HVACloss = HVDCloss, HVACloss

        self.pos_hvdc_lines = [np.where(network[hvdc_mask,0]==n)[0] for n in range(nodes)] 
        self.neg_hvdc_lines = [np.where(network[hvdc_mask,1]==n)[0] for n in range(nodes)]
        
        self.pos_hvac_lines = [np.where(network[~hvdc_mask,0]==n)[0] for n in range(nodes)] 
        self.neg_hvac_lines = [np.where(network[~hvdc_mask,1]==n)[0] for n in range(nodes)] 

        self.firstyear, self.years = firstyear, years
        self.finalyear = self.firstyear+self.years-1
        self.resolution = resolution
        self.intervals = int((years*365)*24/resolution)

        self.efficiency, self.StartCharge = efficiency, StartCharge
        
        self.cpv      = np.array([model.cpv[i].value        for i in model.cpv])#/1000
        self.conshore = np.array([model.conshore[i].value   for i in model.conshore])#/1000
        self.coffshore= np.array([model.coffshore[i].value  for i in model.coffshore])#/1000
        self.cphp     = np.array([model.cphp[i].value       for i in model.cphp])#/1000
        self.cphe     = np.array([model.cphe[i].value       for i in model.cphe])#/1000
        self.cbp      = np.array([model.cbp[i].value        for i in model.cbp])#/1000
        self.cbe      = np.array([model.cbe[i].value        for i in model.cbe])#/1000
        self.cinter   = np.array([model.cinter[i].value     for i in model.cinter])#/1000
        self.chvdc    = np.array([model.chvdc[i].value      for i in model.chvdc])#/1000
        self.chvac    = np.array([model.chvac[i].value      for i in model.chvac])#/1000
        self.chydro   = CHydro 
        self.cbio     = CBio 
        self.cwaste   = CWaste 
        self.cwind = np.zeros(len(Windl))
        self.cwind[~offshore_mask], self.cwind[offshore_mask] = self.conshore, self.coffshore
        self.ctrans = np.zeros(ntrans)
        self.ctrans[hvdc_mask], self.ctrans[~hvdc_mask] = self.chvdc, self.chvac
        
        # operations in MW and MWh
        self.Dischargeph = np.array([model.dischargeph[i].value for i in model.dischargeph]).reshape(-1, nodes) * 1000
        self.Chargeph    = np.array([model.chargeph[i].value    for i in model.chargeph   ]).reshape(-1, nodes) * 1000
        self.Storageph   = np.array([model.storageph[i].value   for i in model.storageph  ]).reshape(-1, nodes) * 1000

        self.Dischargeb  = np.array([model.dischargeb[i].value for i in model.dischargeb]).reshape(-1, nodes) * 1000
        self.Chargeb     = np.array([model.chargeb[i].value    for i in model.chargeb   ]).reshape(-1, nodes) * 1000
        self.Storageb    = np.array([model.storageb[i].value   for i in model.storageb  ]).reshape(-1, nodes) * 1000
        
        self.Hydro   = np.array([model.hydro[i].value     for i in model.hydro     ]).reshape(-1, nodes) * 1000
        self.Bio     = np.array([model.bio[i].value       for i in model.bio       ]).reshape(-1, nodes) * 1000
        self.Waste   = np.array([model.waste[i].value     for i in model.waste     ]).reshape(-1, nodes) * 1000
        self.Imports = np.zeros((intervals, nodes))
        self.Imports[:, CInter_mask] = np.array([model.imports[i].value for i in model.imports]).reshape(-1, ninters) * 1000
        # self.Deficit = np.array([model.deficit[i].value   for i in model.deficit   ]).reshape(-1, nodes) 
        
        hvdc_pos =    np.array([model.hvdc_pos[i].value for i in model.hvdc_pos ]).reshape(-1, nhvdc) * 1000
        hvdc_neg =    np.array([model.hvdc_neg[i].value for i in model.hvdc_neg ]).reshape(-1, nhvdc) * 1000
        self.hvdc = hvdc_pos - hvdc_neg

        hvac_pos =    np.array([model.hvac_pos[i].value for i in model.hvac_pos ]).reshape(-1, nhvac) * 1000
        hvac_neg =    np.array([model.hvac_neg[i].value for i in model.hvac_neg ]).reshape(-1, nhvac) * 1000
        self.hvac = hvac_pos - hvac_neg

        self.Transmission = np.empty_like(self.Chargeph)
        for t in range(self.Chargeph.shape[0]):
            for n in range(self.Chargeph.shape[1]):
                self.Transmission[t, n]= (
                    + sum((hvdc_pos[t, l] - hvdc_neg[t, l]*(1-self.HVDCloss[l]) for l in self.pos_hvdc_lines[n]))
                    + sum((hvdc_neg[t, l] - hvdc_pos[t, l]*(1-self.HVDCloss[l]) for l in self.neg_hvdc_lines[n]))
                    + sum((hvac_pos[t, l] - hvac_neg[t, l]*(1-self.HVACloss[l]) for l in self.pos_hvac_lines[n]))
                    + sum((hvac_neg[t, l] - hvac_pos[t, l]*(1-self.HVACloss[l]) for l in self.neg_hvac_lines[n]))
                )
                
                
        self.Wind = np.zeros((intervals, len(Windl)))
        self.Wind[:, ~offshore_mask] = self.conshore*TSWind[:self.intervals, ~offshore_mask]*1000  
        self.Wind[:, offshore_mask] = self.coffshore*TSWind[:self.intervals, offshore_mask]*1000  
        
        self.PV   = self.cpv*TSPV[:self.intervals, :]*1000
        self.PV   = np.stack([self.PV[  :, np.where(self.PVl  ==node)[0]].sum(axis=1) for node in self.Nodel]).T
        self.Wind = np.stack([self.Wind[:, np.where(self.Windl==node)[0]].sum(axis=1) for node in self.Nodel]).T
        self.Load = MLoad * 1000
        self.baseload = baseload * 1000
        self.hydro_baseload = hydro_baseload * 1000
        
        self.Spillage = -np.minimum(0, 
            self.Load 
            + self.Chargeph
            + self.Chargeb
            + self.Transmission
            # - self.Deficit
            - self.hydro_baseload
            - self.baseload
            - self.Dischargeph
            - self.Dischargeb
            - self.Hydro
            - self.Bio 
            - self.Waste 
            - self.PV 
            - self.Wind 
            - self.Imports
            )
        
        self.OBJ = pyo.value(model.OBJ)



