# Modelling input and assumptions
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Optimisation import percapita, node, iterations, population

###### NODAL LISTS ######
Nodel = np.array(['KH', 'LA', 'TH', 'VH', 'VS']) #(['AW', 'AN', 'BN', 'KH', 'CN', 'IN', 'IJ', 'IK', 'IM', 'IP', 'IC', 'IS', 'IT', 'LA', 'MY', 'MM', 'PL', 'PM', 'PV', 'SG', 'TH', 'VH', 'VS'])
PVl =   np.array(['KH']*1 + ['LA']*1 + ['TH']*1 + ['VH']*1 + ['VS']*1)
#pv_lb_np = np.array([1000.] + [1000.] + [1000.] + [5000.] + [5000.])
#pv_ub_np = np.array([10000.] + [10000.] + [10000.] + [10000.] + [10000.])
#phes_lb_np = np.array([0.] + [0.] + [1500.] + [600.] + [600.])
#phes_ub_np = np.array([100000.] + [100000.] + [100000.] + [100000.] + [10000.])
Windl = np.array(['KH']*1 + ['LA']*1 + ['TH']*1 + ['VH']*1 + ['VS']*1)
#wind_ub_np = np.array([100000.] + [13000.] + [239000.] + [155000.]+ [155000.])
#Interl = np.array(['AW']*1 + ['AN']*1 + ['CN']*1 + ['IN']*1) if node=='Super2' else np.array([])
resolution = 1

###### DATA Imports ######
MLoad = np.genfromtxt('Data/electricity{}.csv'.format(percapita), delimiter=',', skip_header=1) # EOLoad(t, j), MW
TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1) # TSWind(t, i), MW

#Hydrol = np.array(['KH']*1 + ['LA']*1 + ['TH']*1 + ['VH']*1 + ['VS']*1) #do I need a node list? Cheng didn't have one

assets = np.genfromtxt('Data/assets.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW. There are the existing capacities for each tech
constraints = np.genfromtxt('Data/constraints.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
ECoal, EGas, EOil, EHydro, EGeo, EBio, EWaste = [constraints[:, x] for x in range(constraints.shape[1])] # GWh, constraints on generation from existing capacity are imported from constraints.csv
CBaseload = (EGeo + EBio + EWaste) / 8760 # 0.5 * EHydro +  24/7, GW, baseload capacity for a single time interval, defined according to fraction of each existing capacity technology assigned to baseload generation. Based on annual generation constraints (GWh) divided by number of intervals in each year (this accounts for the annual capacity factor of that tech)

hydroProfiles = np.genfromtxt('Data/hydro.csv', delimiter = ',', skip_header = 1, encoding = None).astype(float)

baseload = np.ones((MLoad.shape[0], len(CHydro))) #This makes the array of RoR values

for i in range(0,MLoad.shape[0]):
    for j in range(0,len(CHydro)):
        baseload[i,j] = hydroProfiles[i,j]

baseload += CBaseload

TotalBaseload = baseload

#CPeak = CCoal + CGas + COil + CHydro - 0.5 * EHydro / 8760 # GW
CPeak = CCoal + CGas + COil / 8760


#for i in range(0,len(hydroProfiles[0])): Not sure what this does and whether it is needed
    #hydroProfiles[i,1] = 0

###### CONSTRAINTS ######
# Energy constraints
#Hydromax = EHydro.sum() * pow(10,3) # GWh to MWh per year Will need this when I split the hydro into flex and not

inter = 0.05 if node=='Super2' else 0
#CDC0max, CDC1max, CDC7max, CDC8max = 4 * [inter * MLoad.sum() / MLoad.shape[0] / 1000] # 5%: AWIJ, ANIT, CHVH, INMM, MW to GW
DCloss = np.array([500, 200, 500, 500]) * 0.03 * pow(10, -3)#([2100, 1000, 900, 1300, 1300, 500, 200, 600, 1000, 900, 1400, 2100, 900, 600, 1000, 1000, 500, 500, 300, 1300, 700, 600, 400])

if node in ['BN', 'SG']:
    efficiency = 0.9
    factor = np.genfromtxt('Data/factor1.csv', delimiter=',', usecols=1)
else:
    efficiency = 0.8
    factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

###### Simulation Period ######
firstyear, finalyear, timestep = (2010, 2019, 1)

###### Scenario adjustments ######
# Node Values
if 'Super' not in node:
    MLoad = MLoad[:, np.where(Nodel==node)[0]]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    TSWind = TSWind[:, np.where(Windl==node)[0]]

    #CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste = [x[np.where(Nodel == node)[0]] for x in (CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste)]#
    CCoal, CGas, COil, CGeo, CBio, CWaste = [x[np.where(Nodel == node)[0]] for x in (CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste)]# if I take CHydro out of this does the order get stuffed up
    #EHydro = EHydro[np.where(Nodel==node)[0]] # GWh
    #CBaseload = CBaseload[np.where(Nodel==node)[0]] # GW
    TotalBaseload = TotalBaseload[np.where(Nodel==node)[0]] # GW Replacing the above to combine all the types of baseload
    CPeak = CPeak[np.where(Nodel==node)[0]] # GW


###### DECISION VARIABLE LIST INDEXES ######
intervals, nodes = MLoad.shape # The no. of intervals equals the no. of rows in the MLoad variable. The no. of nodes equals the no. of columns in MLoad 
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1]) # Number of solar and wind sites
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes) # Integers that define the final index of solar, wind, phes, etc. sites within the decision variable list
#inters = len(Interl) # The number of external interconnections
#iidx = sidx + 1 + inters

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = TotalBaseload * pow(10, 3) # GW to MW

###### Network Constraints
#manage = 0 # weeks
#allowance = MLoad.sum(axis=1).max() * 0.05 * manage * 168 * efficiency # MWh
allowance = min(0.00002*np.reshape(MLoad.sum(axis=1), (-1,8760)).sum(axis=-1)) # Allowable annual deficit of 0.002%, MWh

###### DECISION VARIABLE UPPER AND LOWER BOUNDS ######
#pv_lb = list(pv_lb_np)
#pv_ub = list(pv_ub_np)
#wind_ub = list(wind_ub_np)
#phes_ub = list(phes_ub_np)
#phes_lb = list(phes_lb_np)


class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals, self.nodes = (intervals, nodes)
        self.resolution = resolution

        self.CPV = list(x[: pidx]) # CPV(i), GW
        self.CWind = list(x[pidx: widx]) # CWind(i), GW
        self.GPV = TSPV * np.tile(self.CPV, (intervals, 1)) * pow(10, 3) # GPV(i, t), GW to MW
        self.GWind = TSWind * np.tile(self.CWind, (intervals, 1)) * pow(10, 3) # GWind(i, t), GW to MW

        self.CPHP = list(x[widx: sidx]) # CPHP(j), GW
        self.CPHS = x[sidx] # S-CPHS(j), GWh
        self.efficiency = efficiency

        #self.CInter = x[sidx+1: iidx] if node=='Super2' else [0] # CInter(j), GW
        #self.GInter = np.tile(self.CInter, (intervals, 1)) * pow(10, 3) # GInter(j, t), GW to MW

        self.Nodel, self.PVl, self.Windl = (Nodel, PVl, Windl)
        #self.Interl = Interl
        self.node = node

        self.GBaseload, self.CPeak = (GBaseload, CPeak) #Will need to make a change here? 
        self.CHydro, self.EHydro = (CHydro, EHydro) # GW, GWh 

        self.allowance = allowance

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)