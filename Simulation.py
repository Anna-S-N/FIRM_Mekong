# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit

perfect = np.array([0,1,3,6,10,15,21,28,36])


@njit
def Reliability(solution, flexible, start=None, end=None):
    """ 
    flexible = np.ones((intervals, nodes))*CPeak*1000; end=None; start=None 
    """
    network = solution.network
    trans_tdc_mask = solution.trans_tdc_mask
    networksteps = np.where(perfect == network.shape[2])[0][0]
    
    Netload = (solution.MLoad - solution.GPV - solution.GWind - solution.GBaseload)[start:end]
    Netload -= flexible
    
    shape2d = intervals, nodes = len(Netload), solution.nodes

    Pcapacity = solution.CPHP * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    Hcapacity = solution.CHVDC * 1000 # GW to MW
    nhvdc = len(solution.CHVDC)
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d, dtype=np.float64)
    Charge = np.zeros(shape2d, dtype=np.float64)
    Storage = np.zeros(shape2d, dtype=np.float64)
    Deficit = np.zeros(shape2d, dtype=np.float64)
    Transmission = np.zeros((intervals, nhvdc, nodes), dtype = np.float64)

    Storaget_1 = 0.5*Scapacity

    for t in range(intervals):
        Netloadt = Netload[t]

        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Deficitt = np.maximum(Netloadt - Discharget ,0)

        Transmissiont=np.zeros((nhvdc, nodes), dtype=np.float64)
        
        # # The following provides a relatively minor benefit to cost for increased computation time
        # if Deficitt.sum() > 1e-6:
        #     # Fill deficits with transmission without drawing down from battery reserves
        #     Fillt = np.maximum(Netloadt - Discharget, 0)
        #     Surplust = -1 * np.minimum(0, Netloadt + Charget) 
            
        #     Transmissiont = hvdc(Fillt, Surplust, Transmissiont, Hcapacity, network, networksteps)
            
        #     Netloadt = Netload[t] - Transmissiont.sum(axis=0)
        #     Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        #     Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        #     Deficitt = np.maximum(Netloadt - Discharget, 0)
    
        if Deficitt.sum() > 1e-6:
            # raise KeyboardInterrupt
            # Fill deficits with transmission allowing drawing down from neighbours battery reserves
            Surplust = -1 * np.minimum(0, Netloadt) + (np.minimum(Pcapacity, Storaget_1 / resolution) - Discharget)

            Transmissiont = hvdc(Deficitt, Surplust, Hcapacity, network, networksteps, 
                                 np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        # =============================================================================
        # TODO: If deficit Go back in time and discharge batteries 
        # This will be extemely computationally intensive
        # =============================================================================
        
        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Surplust = -1 * np.minimum(0, Netloadt + Charget)# charge itself first, then distribute
        if Surplust.sum() > 1e-6:
            # raise KeyboardInterrupt
            # Distribute surplus energy with transmission to areas with spare charging capacity
            Fillt = (Discharget # load not met by gen and transmission
                     + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #full charging capacity
                     - Charget) #charge capacity already in use

            Transmissiont = hvdc(Fillt, Surplust, Hcapacity, network, networksteps,
                                 np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
            #print(Netload[t])
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        Storaget_1 = Storaget.copy()
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget
        Transmission[t] = Transmissiont
        
    ImpExp = Transmission.sum(axis=1)
    
    Deficit = np.maximum(0, Netload - ImpExp - Discharge)
    Spillage = -1 * np.minimum(0, Netload - ImpExp + Charge)

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit
    solution.Import = np.maximum(0, ImpExp)
    solution.Export = -1 * np.minimum(0, ImpExp)
    solution.TDC = (np.atleast_3d(trans_tdc_mask).T*Transmission).sum(axis=2)
    
    return Deficit

@njit
def hvdc(Fillt, Surplust, Hcapacity, network, networksteps, Importt, Exportt):
    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    if network.size == 0:
        return Importt+Exportt

    for n in np.where(Fillt>0)[0]:
        pdonors = network[:, n, 0, :]
        valid_mask = pdonors[0] != -1
        pdonors, pdonor_lines = pdonors[0, valid_mask], pdonors[1, valid_mask]
  
        if Surplust[pdonors].sum() == 0:
            continue
  
        _transmission = np.zeros_like(Fillt)
        _transmission[pdonors] = Surplust[pdonors]
        _transmission[pdonors] = np.minimum(_transmission[pdonors], Hcapacity[pdonor_lines]-Importt[pdonor_lines,:].sum(axis=1))
        
        _transmission /= max(1, _transmission.sum()/Fillt[n])
        
        for d, l in zip(pdonors, pdonor_lines):#  print(d,l)
            Importt[l, n] += _transmission[d]
            Exportt[l, d] -= _transmission[d]
            
        Fillt[n] -= _transmission.sum()
        Surplust -= _transmission                

    # Continue with nthary transmission 
    # Note: This code block works for primary transmission too, but is slower
    if Surplust.sum() > 0 and Fillt.sum() > 0:
        for leg in range(1, networksteps):
            for n in np.where(Fillt>0)[0]:
                donors = network[:, n, perfect[leg]:perfect[leg+1], :]
                donors, donor_lines = donors[0, :, :], donors[1, :, :]
      
                valid_mask = donors[-1] != -1
                if np.prod(~valid_mask):
                    break
                donor_lines = donor_lines[:, valid_mask]
                donors = donors[:, valid_mask]
                if Surplust[donors[-1]].sum() == 0:
                    continue
      
                ndonors = valid_mask.sum()
                donors = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
                
                _import = np.zeros_like(Importt)
                for d, dl in zip(donors[-1], donor_lines.T): #print(d,dl)
                    _import[dl, d] = Surplust[d]
                
                hostingcapacity = (Hcapacity-Importt.sum(axis=1))
                zmask = hostingcapacity > 0
                _import[zmask] /= np.atleast_2d(np.maximum(1, _import.sum(axis=1)/hostingcapacity)).T[zmask]
                _import[~zmask]*=-1
                _transmission = _import.sum(axis=0)
                for _row in _import:
                    zmask = _row!=0
                    _transmission[zmask] = np.minimum(_row, _transmission)[zmask]
                _transmission=np.maximum(0, _transmission)
                _transmission /= max(1, _transmission.sum()/Fillt[n])
                
                for nd, d, dl in zip(range(ndonors), donors[-1], donor_lines.T):
                    for step, l in enumerate(dl): 
                        Importt[l, donors[step, nd]] += _transmission[d]
                        Exportt[l, donors[step+1, nd]] -= _transmission[d]
                Fillt[n] -= _transmission.sum()
                Surplust -= _transmission                
                
                if Surplust.sum() == 0 or Fillt.sum() == 0:
                    break
                
            if Surplust.sum() == 0 or Fillt.sum() == 0:
                break
        
    return Importt+Exportt