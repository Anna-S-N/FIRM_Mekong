from Input import *
from Simulation import Reliability
from Costs import calculate_costs

import numpy as np
import datetime as dt

def update_battery_level(battery_discharge,battery_charge,battery_efficiency,battery_energy,old_battery_level=[],start_t=0):
    
    ''' generate time series data for battery level based on input battery charging and discharging profiles 
    assuming all energy loss occurs during the charging process '''
    
    if len(old_battery_level) > 0:
        battery_level = old_battery_level.copy()
    else:
        battery_level = np.zeros(len(battery_discharge), dtype=np.float64)
    
    for t in range(start_t, len(battery_discharge)):
        
        if t == 0:
            battery_level[t] = 0.5 * battery_energy
        else:
            battery_level[t] = min(battery_level[t-1] + battery_charge[t] * battery_efficiency - battery_discharge[t], battery_energy)
            
    return battery_level

def Debug(solution):
    """Debugging"""
    
    MLoad = solution.MLoad.sum(axis=1)

    PV, Wind, TDC = solution.GPV.sum(axis=1), solution.GWind.sum(axis=1), solution.TDCabs.sum(axis=1)
    
    Hydro = solution.GBaseload.sum(axis=1) + solution.flexible
        
    Discharge, Charge, Storage, Spillage, Deficit = solution.Discharge, solution.Charge, solution.Storage, solution.Spillage, solution.Deficit
    
    BDischarge, BCharge, BStorage = solution.battery_discharge, solution.battery_charge, solution.BStorage

    efficiency = solution.efficiency

    for i in range(solution.intervals):
        # Energy supply-demand balance
        if abs(MLoad[i] + Charge[i] + BCharge [i] + Spillage[i] - PV[i] - Wind[i] - Hydro[i] - Discharge[i] - BDischarge[i] - Deficit[i]) > 10:
            print(i,MLoad[i],Charge[i],BCharge [i],Spillage[i],PV[i],Wind[i],Hydro[i],Discharge[i],BDischarge[i],Deficit[i])
        assert abs(MLoad[i] + Charge[i] + BCharge [i] + Spillage[i] - PV[i] - Wind[i] - Hydro[i] - Discharge[i] - BDischarge[i] - Deficit[i]) <= 10

        # Discharge, Charge and Storage
        if i != 0:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] - Charge[i] * efficiency) <= 10
            assert abs(BStorage[i] - BStorage[i - 1] + BDischarge[i] - BCharge[i] * 0.9) <= 10

    # Capacity: PV, wind, Discharge, Charge and Storage
    assert np.amax(PV) - sum(solution.CPV) * pow(10, 3) < 1
    assert np.amax(Wind) - sum(solution.CWind) * pow(10, 3) < 1
    assert np.amax(Hydro) - (sum(solution.CBaseload) + sum(solution.CPeak)) * pow(10, 3) < 1

    assert np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3) < 1
    assert np.amax(Charge) - sum(solution.CPHP) * pow(10, 3) < 1
    assert np.amax(Storage) - sum(solution.CPHS) * pow(10, 3) < 1
    
    assert np.amax(BDischarge) - sum(solution.CBP) * pow(10, 3) < 1
    assert np.amax(BCharge) - sum(solution.CBP) * pow(10, 3) < 1
    assert np.amax(BStorage) - sum(solution.CBS) * pow(10, 3) < 1
    
    if np.amax(TDC) - sum(solution.CDC) * 1000. > 1:
        print(i, np.amax(TDC), sum(solution.CDC))
    assert np.amax(TDC) - sum(solution.CDC) * 1000. < 1

    print('Debugging: everything is ok')

    return True

def Information(x, hydrobio, imports, charge, discharge):

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)

    S.flexible = hydrobio.sum(axis=1)+imports.sum(axis=1)
    
    S.battery_charge = charge
    S.battery_discharge = discharge
    
    Deficit, Discharge = Reliability(S, flexible=S.flexible, agg_storage = False, battery_charge = charge, battery_discharge = discharge)
    
    S.BStorage = update_battery_level(discharge,charge,0.9,S.CBS.sum()*1000)
    S.TDCabs = np.abs(S.TDC)

    Debug(S)    
        
    """ S.TDC = Transmission(S, output = True, agg_storage = False)
    S.TDCabs = np.abs(S.TDC)   
    S.QLD_NSW,S.NSW_VIC,S.VIC_SA,S.VIC_TAS = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))    

    # S.MPHS = S.CPHS * np.array(S.CPHP) * pow(10, 3) / sum(S.CPHP) # GW to MW

    S.Topology = np.array([S.QLD_NSW - S.NSW_VIC,
                  -1 * S.QLD_NSW,
                  -1 * S.VIC_TAS - S.VIC_SA + S.NSW_VIC,
                  S.VIC_SA,
                  S.VIC_TAS])

    """

    C = np.stack([S.MLoad.sum(axis=1), S.Charge.sum(axis=1), S.Spillage.sum(axis=1), S.GPV.sum(axis=1), S.GWind.sum(axis=1), S.Discharge.sum(axis=1), S.Storage.sum(axis=1), 
                  S.Deficit.sum(axis=1), S.GBaseload.sum(axis=1), S.flexible, S.battery_charge, S.battery_discharge, S.BStorage,
                  S.TDC]) 
    
    C = np.around(C.transpose())

    '''NEED HEADER TO INCLUDE TRANSMISSION LINES'''

    header = 'Demand(MW),Storage charge(MW),Spillage(MW),Solar photovoltaics(MW),Wind(MW),Storage discharge(MW),Storage level(MWh),Deficit(MW),'\
             'Hydro baseload(MW),Hydro peak(MW),Battery charge (MW),Battery discharge(MW),Battery level(MWh)'
    np.savetxt('Results/TimeSeries_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), C, fmt='%f', delimiter=',', header=header, comments='')

    
    header = 'Operational demand,' \
                 'Solar photovoltaics,Wind,Hydro baseload,Hydro peak,Discharge,Energy deficit,Energy spillage,' \
                 'Transmission,Charge,Battery charge,Battery discharge,Battery level,' \
                 'Storage level'

    for j in range(nodes):
        C = np.stack([S.MLoad[:, j],
                      S.MPV[:, j], S.MWind[:, j], S.MBaseload[:, j], S.MPeak[:, j],
                      S.MDischarge[:, j], S.MDeficit[:, j], S.MSpillage[:, j], S.Topology[j], S.MCharge[:, j],
                      S.MBCharge[:, j], S.MBDischarge[:, j], S.MBStorage[:, j],
                      S.MStorage[:, j]])
        C = np.around(C.transpose())

        np.savetxt('Results/TimeSeries_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario,str(S.Nodel_int[j])), C, fmt='%f', delimiter=',', header=header, comments='')

    
    print('Load profiles and generation mix is produced.')

    CPV, CWind, CPHP, CPHS, CBP, CBS, CBaseload, CPeak, CDC = sum(S.CPV), sum(S.CWind), sum(S.CPHP), sum(S.CPHS), sum(S.CBP), sum(S.CBS), sum(S.CBaseload), sum(S.CPeak), sum(S.CDC) # GW/GWh
    GPV, GWind, GBaseload, GPeak, GPHES, Deficit, Spillage = map(lambda x: x * pow(10, -6) / years, (S.GPV.sum(), S.GWind.sum(), S.GBaseload.sum(), S.flexible.sum(), S.Discharge.sum(), S.Deficit.sum(), S.Spillage.sum())) # TWh
    CFPV, CFWind = (GPV / CPV / 8.76, GWind / CWind / 8.76)

    costs, tech_costs = calculate_costs(S,Discharge,GPeak,GImports)
    Loss = np.sum(S.TDCabs, axis=0) * DCloss
    energy = ((S.MLoad.sum() - Loss.sum()) / S.years)
        
    LCOE = costs / energy
    LCOEPV = tech_costs[0] / energy
    LCOEWind = tech_costs[1] / energy
    LCOEStorage = tech_costs[2] / energy
    LCOETransmission = tech_costs[3] / energy
    LCOEHydro = tech_costs[4] / energy
    LCOEBattery = tech_costs[5] / energy
    LCOEImports = tech_costs[6] / energy

    print('Levelized cost of electricity:')
    print('LCOE:', LCOE)
    print('LCOEPV:', LCOEPV)
    print('LCOEWind:', LCOEWind)
    print('LCOEStorage:', LCOEStorage)
    print('LCOETransmission:', LCOETransmission)
    print('LCOEHydro:', LCOEHydro)
    print('LCOEBattery:', LCOEBattery)
    print('LCOEImports:', LCOEImports)

    D = np.zeros((1, 28))
    header = 'Annual generation (TWh),Loss (TWh),PV (GW),PV (TWh),PV CF,Wind (GW),Wind (TWh),Wind CF,Hydro baseload (GW),Hydro peak(GW),Hydro baseload (TWh),Hydro peak (TWh),Battery (GW),Battery (GWh),Battery (hour),\
    Storage (GW),Storage (GWh),Storage (h),Transmission (GW),Deficit (TWh),Spillage(TWh),LCOE,LCOE PV,LCOE Wind,LCOE Hydro,LCOE Storage,LCOE Transmission,LCOE Battery,LCOE Imports'
    D[0, :] = [energy * pow(10,-6), Loss.sum()/years*pow(10,-6),CPV, GPV, CFPV, CWind, GWind, CFWind, CBaseload, CPeak, GBaseload, GPeak, CBP, CBS, CBS/CBP, CPHP, CPHS, CPHS/CPHP, CDC, 
               Deficit, Spillage, LCOE, LCOEPV, LCOEWind, LCOEHydro, LCOEStorage, LCOETransmission, LCOEBattery, LCOEImports]

    np.savetxt('Results/Summary_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), D, fmt='%f', delimiter=',', header = header)
    print('Energy generation, storage and cost information is produced.')

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

if __name__ == '__main__':

    # assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
    # CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
    # CBaseload = np.array([0, 0, 0, 0, 1.0]) # 24/7, GW
    # CPeak = CHydro + CBio - CBaseload # GW
    
    capacities = np.genfromtxt('Results/Optimisation_resultx_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), delimiter=',')
    hydrobio = np.genfromtxt('Results/Dispatch_Hydro_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), delimiter=',', skip_header=1)
    imports = np.genfromtxt('Results/Dispatch_Imports_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), delimiter=',', skip_header=1)
    battery_charge = np.genfromtxt('Results/Dispatch_BatteryCharge_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), delimiter=',', skip_header=1)
    battery_discharge = np.genfromtxt('Results/Dispatch_BatteryDischarge_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), delimiter=',', skip_header=1)
    
    Information(capacities, hydrobio, imports, battery_charge, battery_discharge)