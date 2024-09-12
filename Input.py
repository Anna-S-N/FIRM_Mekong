# Modelling input and assumptions
# Copyright (c) 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from argparse import ArgumentParser

from numba import njit, float64, int64, boolean
from numba.experimental import jitclass

parser = ArgumentParser()
parser.add_argument('-i', default=150, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=8, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-e', default=3, type=int, required=False, help='per-capita electricity: 3, 6 and 9 MWh')
parser.add_argument('-n', default='TH', type=str, required=False, help='Mekong, TH_Iso, TH_imp KH, LA, VH, VS, ...') # TH_Iso = Isolated Thailand network, TH_imp = Thailand w imports, Mekong = Mekong Power Grid
args = parser.parse_args()

percapita, node, iterations, population = (args.e, args.n, args.i, args.p)

###### NODAL LISTS ######
Nodel = np.array(['KH', 'LA', 'VH', 'VS', 'CACE', 'CACW', 'CACN', 'MAC', 'NAC', 'NEC', 'SAC'])
PVl =   np.array(['KH']*1 + ['LA']*1 + ['VH']*1 + ['VS']*1 + ['CACE']*1 + ['CACW']*1 + ['CACN']*1 + ['MAC']*1 + ['NAC']*1 + ['NEC']*1 + ['SAC']*1)
pv_lb_np = np.array([0.] + [0.] + [0.] + [0.] + [3.5] + [3.] + [2.3] + [0.2] + [11.] + [9.6] + [6.7]) #Thailand constraints based on 2037 capacity in PDP2024 draft
pv_ub_np = np.array([100000.] + [100000.] + [100000.] + [100000.] + 7*[100000.])
phes_lb_np = np.array([0.] + [0.] + [0.] + [0.] + 5*[0.] + [1.] + [0.]) # Lamtakong Jolabha Vadhana, Thailand (NEC), is 1000 MW
phes_ub_np = np.array([100000.] + [100000.] + [100000.] + [100000.] + 7*[10000.])
#Windl = np.array(['KH']*1 + ['LA']*1 + ['VH']*1 + ['VS']*1 + ['CACE']*1 + ['CACW']*1 + ['CACN']*1 + ['MAC']*1 + ['NAC']*1 + ['NEC']*1 + ['SAC']*1)
#wind_lb_np = np.array([0.] + [0.] + [0.] + [0.] + 7*[0.]) 
#wind_ub_np = np.array([100000.] + [239000.] + [155000.]+ [155000.] + 7*[13000.])
Interl = np.array(['AW']*1 + ['AN']*1 + ['CN']*1 + ['IN']*1) if node=='TH_imp' else np.array([]) ######## DEFINE THE THAI IMPORT NODES
resolution = 1

n_node = dict((name, i) for i, name in enumerate(Nodel))
#Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))
Nodel_int, PVl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl))


###### DATA Imports ######
MLoad = np.genfromtxt('Data/electricity{}.csv'.format(percapita), delimiter=',', skip_header=1) # EOLoad(t, j), MW
TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1) # TSPV(t, i), MW
#TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1) # TSWind(t, i), MW

assets = np.genfromtxt('Data/assets.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
CHydro = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])]
#CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW. There are the existing capacities for each tech
constraints = np.genfromtxt('Data/constraints.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
EHydro = np.genfromtxt('Data/constraints.csv', dtype=None, delimiter=',', encoding=None)[1:, 3:].astype(float)
Hydro_monthly_CF = [0.238, 0.262, 0.253, 0.24, 0.236, 0.217, 0.174, 0.164, 0.105, 0.078, 0.111, 0.191] # January - December
#ECoal, EGas, EOil, EHydro, EGeo, EBio, EWaste = [constraints[:, x] for x in range(constraints.shape[1])] # GWh, constraints on generation from existing capacity are imported from constraints.csv
#CBaseload = (EGeo + EBio + EWaste) / 8760 # 0.5 * EHydro +  24/7, GW, baseload capacity for a single time interval, defined according to fraction of each existing capacity technology assigned to baseload generation. Based on annual generation constraints (GWh) divided by number of intervals in each year (this accounts for the annual capacity factor of that tech)
#hydroProfiles = np.genfromtxt('Data/hydro.csv', delimiter = ',', skip_header = 1, encoding = None).astype(float) #This makes the array of RoR values
#CPeak = CCoal + CGas + COil + CHydro - 0.5 * EHydro / 8760 # GW
#CPeak = CCoal + CGas + COil / 8760


###### CONSTRAINTS ######
# Energy constraints
#Hydromax = EHydro.sum() * pow(10,3) # GWh to MWh per year Not need in this version

#inter = 0.05 if node=='Super2' else 0
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
""" if 'Super' not in node:
    MLoad = MLoad[:, np.where(Nodel==node)[0]]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    #TSWind = TSWind[:, np.where(Windl==node)[0]]

    #CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste = [x[np.where(Nodel == node)[0]] for x in (CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste)]#
    #CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste = [x[np.where(Nodel == node)[0]] for x in (CCoal, CGas, COil, CHydro, CGeo, CBio, CWaste)]
    #EHydro = EHydro[np.where(Nodel==node)[0]] # GWh
    #CBaseload = CBaseload[np.where(Nodel==node)[0]] # GW
    #hydroProfiles = hydroProfiles[:, np.where(Nodel==node)[0]] # GW 
    #CBaseload = CBaseload[np.where(Nodel==node)[0]] # GW
    #CPeak = CPeak[np.where(Nodel==node)[0]] # GW

    network_mask = np.zeros(4, np.bool_)
    network = np.empty((0,0,0,0), np.int64)
    directconns=np.empty((0,0), np.int64)
    
    
    Nodel_int = Nodel_int[np.where(Nodel==node)[0]]
    Nodel = Nodel[np.where(Nodel==node)[0]]
    PVl_int = PVl_int[np.where(PVl==node)[0]]
    pv_lb_np = pv_lb_np[np.where(PVl==node)[0]]
    pv_ub_np = pv_ub_np[np.where(PVl==node)[0]]
    PVl = PVl[np.where(PVl==node)[0]]
    Windl_int = Windl_int[np.where(Windl==node)[0]]
    Windl = Windl[np.where(Windl==node)[0]] """
    

###### Transmission Network ######
if 'Super' in node: 
    # # coverage to be fixed
    # coverage = np.where(Nodel==node)[0]
    # Nodel = Nodel[coverage]
    # Nodel_int = Nodel_int[coverage]
    
    #Full network of all node connections
    network = np.array([[0, 3], #KH-TH
                        [0, 4], #KH-VS
                        [1, 2], #LA-TH
                        [1, 3], #LA-VH
                        ], dtype=np.int64)
    
    # Find and select connections between nodes being considered
    network_mask = np.array([(network==j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0)==2
    network = network[network_mask,:]
    networkdict = {v:k for k, v in enumerate(Nodel_int)}
    #translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)
    
    trans_tdc_mask = np.zeros((MLoad.shape[1], len(network)), np.bool_)
    for line, row in enumerate(network):
        trans_tdc_mask[row[0], line] = True
    
    directconns = -1*np.ones((len(Nodel)+1, len(Nodel)+1), np.int64)
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

    perfect = np.array([0,1,3,6,10,15,21]) #that's more than enough for now

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
intervals, nodes = MLoad.shape # The no. of intervals equals the no. of rows in the MLoad variable. The no. of nodes equals the no. of columns in MLoad 
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1]) # Number of solar and wind sites
pidx, widx = (pzones, pzones + wzones) # Integers that define the final index of solar, wind, phes, etc. sites within the decision variable list
spidx, seidx = pzones + wzones + nodes, pzones + wzones + nodes + nodes


#inters = len(Interl) # The number of external interconnections
#iidx = sidx + 1 + inters

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * 0.001) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * 1000 # GW to MW, excluding hydro
GHydro = hydroProfiles

###### Network Constraints
#manage = 0 # weeks
#allowance = MLoad.sum(axis=1).max() * 0.05 * manage * 168 * efficiency # MWh
allowance = min(0.000_02*np.reshape(MLoad.sum(axis=1), (-1,8760)).sum(axis=-1)) # Allowable annual deficit of 0.002%, MWh

###### DECISION VARIABLE UPPER AND LOWER BOUNDS ######
pv_lb = list(pv_lb_np)
pv_ub = list(pv_ub_np)
#wind_ub = list(wind_ub_np)
#phes_ub = list(phes_ub_np)
#phes_lb = list(phes_lb_np)


CDCmax = 100.*np.ones_like(DCloss)

#lb = pv_lb + [0.]    * wzones + phes_lb + contingency + [0.] # 
#lb = [0.]     * pzones + [0.]    * wzones + contingency      + [0.]      #+ [0.]    * inters (previous lb)
lb = np.array(pv_lb + [0.]    * wzones + contingency + [0.] * nodes     + [0.] * network_mask.sum())
#ub = pv_ub + wind_ub + phes_ub + [10000.] * nodes + [100000.] #
#ub = [10000.] * pzones + [300.]  * wzones + [10000.] * nodes + [100000.] #+ [1000.] * inters
ub = np.array(pv_ub + [300.]  * wzones + [10000.] * nodes    + [50000.] * nodes + list(CDCmax[network_mask]))

#%%
from Simulation import Reliability

@njit()
def F(S):
    Deficit = Reliability(S, flexible=np.zeros((intervals, nodes) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / (0.5 * (1 + efficiency)) # MWh p.a.
    Hydro = resolution * GHydro.sum() / years #min(0.5 * EHydro.sum() * pow(10, 3), Flexible) # GWh to MWh, MWh p.a.
    Fossil = Flexible# - Hydro # Fossil fuels: MWh p.a.
    Hydro += resolution * GBaseload.sum() / years # Hydropower & other renewables: MWh p.a.
    PenHydro = 0

    Deficit = Reliability(S, flexible=np.ones((intervals, nodes), dtype=np.float64)*CPeak*1000) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, Deficit.sum() * resolution) # MWh

    CHVDC = np.zeros(len(network_mask), dtype=np.float64)
    CHVDC[network_mask] = S.CHVDC

    _c = -1.0 if 'super' not in node else -1.0 # unclear if rules here have changed since previous versions of FIRM
    cost = (factor * np.array([S.CPV.sum(), S.CWind.sum(), 0, S.CPHP.sum(), S.CPHS.sum()] + list(CHVDC) +
                               [S.CPV.sum(), S.CWind.sum(), Hydro * 0.000_001, Fossil*0.000_001, _c, _c])
            ).sum()

    loss = np.zeros(len(network_mask), dtype=np.float64)
    loss[network_mask] = S.TDC.sum(axis=0) * DCloss[network_mask]
    loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / np.abs(energy - loss)
    
    return LCOE, (PenHydro+PenDeficit)

solution_spec = [
    ('x', float64[:]),  # x is 1d array
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int64), # plain integer
    ('nodes', int64),
    ('resolution',float64), # plain float
    ('allowance',float64),
    ('CPV', float64[:]), 
    ('CWind', float64[:]), 
    ('CPHP', float64[:]),
    ('CPHS', float64[:]),
    ('CHVDC', float64[:]),
    ('GPV', float64[:, :]), 
    ('GWind', float64[:, :]),  
    ('CHVDC', float64[:]),
    ('efficiency', float64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('Windl_int', int64[:]),
    ('GBaseload', float64[:, :]),
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
    ('MPeak', float64[:, :]),
    ('MFossil', float64[:, :]),
    # ('MDischarge', float64[:, :]),
    # ('MCharge', float64[:, :]),
    # ('MStorage', float64[:, :]),
    # ('MDeficit', float64[:, :]),
    # ('MSpillage', float64[:, :]),
    ('MHydro', float64[:, :]),
    ('MBio', float64[:, :]),
    ('CDP', float64[:]),
    ('CDS', float64[:]),
    ('TDC', float64[:, :]),
    ('CDC', float64[:]),
    # ('KH', float64[:]),
    # ('LA', float64[:]),
    # ('TH', float64[:]),
    # ('VH', float64[:]),
    # ('VS', float64[:]),
    ('Topology', float64[:, :]),
    ('network', int64[:, :, :, :]),
    ('directconns', int64[:,:]),
    ('trans_tdc_mask', boolean[:,:]),
]


@jitclass(solution_spec)
class Solution:
    """A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)"""

    def __init__(self, x):
        self.x = x
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
        self.network, self.directconns = network, directconns
        self.trans_tdc_mask = trans_tdc_mask
        
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)))
        CWind_tiled = np.zeros((intervals, len(self.CWind)))
        # CInter_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]

        GPV = TSPV * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        GWind = TSWind * CWind_tiled * 1000.  # GWind(i, t), GW to MW

        self.GPV, self.GWind = np.empty((intervals, nodes), np.float64), np.empty((intervals, nodes), np.float64)
        for i, j in enumerate(Nodel_int):
            self.GPV[:,i] = GPV[:, PVl_int==j].sum(axis=1)
            self.GWind[:,i] = GWind[:, Windl_int==j].sum(axis=1) 
        
        self.CPHP = x[widx: spidx]  # CPHP(j), GW
        self.CPHS = x[spidx: seidx]  # S-CPHS(j), GWh
        self.CHVDC = x[seidx:]
        
        self.efficiency = efficiency

        # self.Nodel_int, self.PVl_int, self.Windl_int = Nodel_int, PVl_int, Windl_int
        
        # self.node = node

        self.GHydro, self.GBaseload, self.CPeak = (GHydro, GBaseload, CPeak) #Will need to make a change here? 
        self.CHydro, self.EHydro = (CHydro, EHydro) # GW, GWh 
        
        self.allowance = allowance
        
        self.evaluated=False
        
    def _evaluate(self):
        self.Lcoe, self.Penalties = F(self)
        self.evaluated=True

    #Incompatible with jitclass
    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)
    
#%%
if __name__ == '__main__':
    x = np.genfromtxt("Results/Optimisation_resultx_Super13_3_10_7.csv", delimiter=',')
    def test():
        x = np.random.rand(len(lb))*(ub-lb)+ub
        S = Solution(x)
        S._evaluate()
        print(S.Lcoe, S.Penalties)
    test()