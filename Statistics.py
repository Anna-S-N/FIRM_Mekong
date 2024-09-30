from Input import *
from Simulation import Reliability
from Costs import calculate_costs

import numpy as np
import datetime as dt

def update_battery_level(battery_discharge,battery_charge,battery_efficiency,battery_energy,old_battery_level,start_t=0):
    
    ''' generate time series data for battery level based on input battery charging and discharging profiles 
    assuming all energy loss occurs during the charging process '''

    if len(old_battery_level[:,0]) > 0:
        battery_level = old_battery_level.copy()
    else:
        battery_level = np.zeros((intervals,nodes), dtype=np.float64)
    
    for n in range(nodes):      
        for t in range(start_t, len(battery_discharge[:,n])):
            
            if t == 0:
                battery_level[t,n] = 0.5 * battery_energy[n]
            else:
                battery_level[t,n] = min(battery_level[t-1,n] + battery_charge[t,n] * battery_efficiency - battery_discharge[t,n], battery_energy[n])
                
    return battery_level

def Debug(solution):
    """Debugging"""
    
    MLoad = solution.MLoad.sum(axis=1)

    PV, Wind, TDC = solution.GPV.sum(axis=1), solution.GWind.sum(axis=1), solution.TDCabs.sum(axis=1)
    
    Flexible = solution.hydro_baseload.sum(axis=1) + solution.flexible.sum(axis=1)

    Baseload = solution.baseload.sum(axis=1)
        
    Discharge, Charge, Storage, Spillage, Deficit = solution.Discharge.sum(axis=1), solution.Charge.sum(axis=1), solution.Storage.sum(axis=1), solution.Spillage.sum(axis=1), solution.Deficit.sum(axis=1)
    
    BDischarge, BCharge, BStorage = solution.battery_discharge.sum(axis=1), solution.battery_charge.sum(axis=1), solution.BStorage.sum(axis=1)

    efficiency = solution.efficiency

    for i in range(solution.intervals):
        # Energy supply-demand balance
        if abs(MLoad[i] + Charge[i] + BCharge [i] + Spillage[i] - PV[i] - Wind[i] - Flexible[i] - Baseload[i] - Discharge[i] - BDischarge[i] - Deficit[i]) > 10:
            print(i,MLoad[i],Charge[i],BCharge [i],Spillage[i],PV[i],Wind[i],Hydro[i],Discharge[i],BDischarge[i],Deficit[i])
        assert abs(MLoad[i] + Charge[i] + BCharge [i] + Spillage[i] - PV[i] - Wind[i] - Flexible[i] - Baseload[i] - Discharge[i] - BDischarge[i] - Deficit[i]) <= 10

        # Discharge, Charge and Storage
        if i != 0:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] - Charge[i] * efficiency) <= 10
            assert abs(BStorage[i] - BStorage[i - 1] + BDischarge[i] - BCharge[i] * 0.9) <= 10

    # Capacity: PV, wind, Discharge, Charge and Storage
    assert np.amax(PV) - sum(solution.CPV) * pow(10, 3) < 1
    assert np.amax(Wind) - sum(solution.CWind) * pow(10, 3) < 1
    assert np.amax(Flexible) - (max(solution.hydro_baseload.sum(axis=1)) + (sum(solution.CPeak) + sum(solution.CInter)) * pow(10, 3)) < 1

    assert np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3) < 1
    assert np.amax(Charge) - sum(solution.CPHP) * pow(10, 3) < 1
    assert np.amax(Storage) - sum(solution.CPHS) * pow(10, 3) < 1
    
    assert np.amax(BDischarge) - sum(solution.CBP) * pow(10, 3) < 1
    assert np.amax(BCharge) - sum(solution.CBP) * pow(10, 3) < 1
    assert np.amax(BStorage) - sum(solution.CBS) * pow(10, 3) < 1
    
    if np.amax(TDC) - sum(solution.CHVDC) * 1000. > 1:
        print(i, np.amax(TDC), sum(solution.CHVDC))
    assert np.amax(TDC) - sum(solution.CHVDC) * 1000. < 1

    print('Debugging: everything is ok')

    return True

def Information(x, hydrobio, imports, charge, discharge):

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)    
    
    if nodes > 1:
        S.flexible = hydrobio + imports 
        S.battery_charge = charge
        S.battery_discharge = discharge
    else: 
        S.flexible = hydrobio[:, np.newaxis] + imports[:, np.newaxis]
        S.battery_charge = charge[:, np.newaxis] 
        S.battery_discharge = discharge[:, np.newaxis]  
        hydrobio = hydrobio[:, np.newaxis]
        imports = imports[:, np.newaxis]
    
    Deficit = Reliability(S, flexible=S.flexible, agg_storage = False, battery_charge = S.battery_charge, battery_discharge = S.battery_discharge)
    
    battery_charge_input = np.zeros((intervals,nodes), dtype=np.float64)
    S.BStorage = update_battery_level(S.battery_discharge,S.battery_charge,0.9,S.CBS*1000,battery_charge_input)
    S.TDCabs = np.abs(S.TDC)

    Debug(S)  

    C = np.stack([S.MLoad.sum(axis=1), S.Charge.sum(axis=1), S.Spillage.sum(axis=1), S.GPV.sum(axis=1), S.GWind.sum(axis=1), S.Discharge.sum(axis=1), S.Storage.sum(axis=1), 
                  S.Deficit.sum(axis=1), S.baseload.sum(axis=1), S.flexible.sum(axis=1), S.battery_charge.sum(axis=1), S.battery_discharge.sum(axis=1), S.BStorage.sum(axis=1)]) 
    C = np.hstack([C.transpose(), S.TDC])    
    C = np.around(C)

    header = 'Demand(MW),Storage charge(MW),Spillage(MW),Solar photovoltaics(MW),Wind(MW),Storage discharge(MW),Storage level(MWh),Deficit(MW),'\
             'Hydro baseload(MW),Hydro peak(MW),Battery charge (MW),Battery discharge(MW),Battery level(MWh),Transmission flows (MWh)'
    np.savetxt('Results/TimeSeries_{}_{}_{}_{}_{}_{}_{}_NETWORK.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), C, fmt='%f', delimiter=',', header=header, comments='')

    
    header = 'Operational demand,' \
                 'Solar photovoltaics,Wind,Nuclear,Hydro baseload,Hydro/bio peak,External imports,Discharge,Energy deficit,Energy spillage,' \
                 'Transmission imports,Transmission exports,Charge,Battery charge,Battery discharge,Battery level,' \
                 'Storage level'

    for j in range(nodes):
        C = np.stack([S.MLoad[:, j],
                      S.GPV[:, j], S.GWind[:, j], S.baseload[:, j], S.hydro_baseload[:, j], hydrobio[:, j], imports[:, j],
                      S.Discharge[:, j], S.Deficit[:, j], S.Spillage[:, j], S.Import[:,j], S.Export[:,j], S.Charge[:, j],
                      S.battery_charge[:, j], S.battery_discharge[:, j], S.BStorage[:, j],
                      S.Storage[:, j]])
        C = np.around(C.transpose())

        np.savetxt('Results/TimeSeries_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario,Nodel[j]), C, fmt='%f', delimiter=',', header=header, comments='')

    
    print('Load profiles and generation mix is produced.')

    CPV, CWind, CPHP, CPHS, CBP, CBS, CBaseload, CPeak, CDC = sum(S.CPV), sum(S.CWind), sum(S.CPHP), sum(S.CPHS), sum(S.CBP), sum(S.CBS), sum(S.CBaseload), sum(S.CPeak), sum(S.CHVDC) # GW/GWh
    GPV, GWind, GBaseload, GPeak, GPHES, Deficit, Spillage, GImports = map(lambda x: x / years, (S.GPV.sum(), S.GWind.sum(), S.baseload.sum(), hydrobio.sum(), S.Discharge.sum(), S.Deficit.sum(), S.Spillage.sum(), imports.sum())) # MWh
    CFPV, CFWind = (GPV / CPV / 8.76, GWind / CWind / 8.76)

    GHydroBioTotal = GPeak + S.hydro_baseload.sum() / years
    CPeakTotal = CPeak + S.hydro_baseload.max(axis=0).sum()/1000
    
    costs, tech_costs = calculate_costs(S,S.Discharge,GPeak,GImports,GBaseload)
    
    if nodes>1:
        Loss = np.sum(S.TDCabs, axis=0) * DCloss[network_mask]
    else:
        Loss = np.zeros(intervals, dtype=np.float64)
    energy = ((S.MLoad.sum() - Loss.sum()) / S.years)
        
    #PV_costs, wind_costs, transmission_costs, phes_costs, battery_costs, hydro_costs, import_costs, baseload_costs
    LCOE = costs / energy
    LCOEPV = tech_costs[0] / energy
    LCOEWind = tech_costs[1] / energy
    LCOETransmission = tech_costs[2] / energy
    LCOEStorage = tech_costs[3] / energy  
    LCOEBattery = tech_costs[4] / energy  
    LCOEHydro = tech_costs[5] / energy    
    LCOEImports = tech_costs[6] / energy
    LCOENuclear = tech_costs[7] / energy

    print('Levelized cost of electricity:')
    print('LCOE:', LCOE)
    print('LCOEPV:', LCOEPV)
    print('LCOEWind:', LCOEWind)
    print('LCOEPHES:', LCOEStorage)
    print('LCOETransmission:', LCOETransmission)
    print('LCOEHydro:', LCOEHydro)
    print('LCOEBattery:', LCOEBattery)
    print('LCOEImports:', LCOEImports)
    print('LCOENuclear:', LCOENuclear)

    D = np.zeros((1, 28))
    battery_duration = CBS/CBP if CBP > 0 else 0
    header = 'Annual generation (TWh),Loss (TWh),PV (GW),PV (TWh),Wind (GW),Wind (TWh),Nuclear (GW), Nuclear (TWh), Hydrobio peak(GW),Hydrobio (TWh),Battery Power Capacity(GW),Battery Energy Capacity(GWh),Battery average duration (hour),\
    PHES Power Capacity (GW),PHES Storage Capacity (GWh),PHES average duration (h),Transmission (GW),Deficit (TWh),Spillage(TWh),LCOE,LCOE PV,LCOE Wind,LCOE Storage,LCOE Transmission,LCOE Hydro,LCOE Battery,LCOE Imports,LCOE Nuclear'
    D[0, :] = [energy * pow(10,-6), Loss.sum()/years*pow(10,-6),CPV, GPV * pow(10,-6), CWind, GWind * pow(10,-6),CBaseload, GBaseload * pow(10,-6), CPeakTotal, GHydroBioTotal * pow(10,-6), CBP, CBS, battery_duration, CPHP, CPHS, CPHS/CPHP, CDC, 
               Deficit * pow(10,-6), Spillage * pow(10,-6), LCOE, LCOEPV, LCOEWind, LCOEStorage, LCOETransmission,LCOEHydro,  LCOEBattery, LCOEImports, LCOENuclear]

    np.savetxt('Results/Summary_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), D, fmt='%f', delimiter=',', header = header)
    print('Energy generation, storage and cost information is produced.')

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':

    # assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
    # CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
    # CBaseload = np.array([0, 0, 0, 0, 1.0]) # 24/7, GW
    # CPeak = CHydro + CBio - CBaseload # GW
    
    capacities = np.genfromtxt('Results/Optimisation_resultx_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',')
    hydrobio = np.genfromtxt('Results/Dispatch_Hydro_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',', skip_header=1)
    imports = np.genfromtxt('Results/Dispatch_Imports_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',', skip_header=1)
    battery_charge = np.genfromtxt('Results/Dispatch_BatteryCharge_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',', skip_header=1)
    battery_discharge = np.genfromtxt('Results/Dispatch_BatteryDischarge_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',', skip_header=1)
    
    Information(capacities, hydrobio, imports, battery_charge, battery_discharge)