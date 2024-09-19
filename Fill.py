from Input import *
from Simulation import Reliability
import numpy as np

'''Need 2d arrays for hydro and imports'''
'''Need weekly hydro constraints'''

def fill_deficit(deficit,hydro,imports,storage,hydro_limit,import_limit,storage_cap,efficiency,step,charge,charge_cap,hflag,iflag):
        
    idx = np.where(deficit > 0)[0]
    storage_full = np.where(storage == storage_cap)[0]
    charging_full = np.where(charge == charge_cap)[0]
    
    for idd, i in np.ndenumerate(idx):
        
        # for each deficit, look backwards
        
        d = deficit[i]
        t = i
        count = 0
        # print("Filling deficit", idd[0], "of", len(idx))
        
        try:
            nearest_full = storage_full[np.where(storage_full < t)[0][-1]] 
            # only makes sense to look at the period between current t and the most recent time when the storage is full
            # as the energy used to charge storage before "nearest full" will be curtailed at or before "nearest full" anyways
        except IndexError: 
            nearest_full = 0
        
        while d > 0 and t > nearest_full and count < step:
                            
            if t == i - 1: # if unable to meet deficit from real-time hydro, then the trickle-charging storage is required
                d = d / efficiency

            if hflag: 
                hydro_c = min(hydro[t] + d, hydro_limit)
                d = d - (hydro_c - hydro[t])
                hydro[t] = hydro_c
                
                available_hydro_idx = np.where(hydro < hydro_limit)[0]
                # move to the most recent time when hydro and PHES are not operating at full power (spare power capacity available)
                try:
                    t = sorted([i for i in available_hydro_idx if i < t and i > nearest_full and i not in charging_full])[-1]
                except IndexError:
                    break
            if iflag:
                if d > 0:
                    imports_c = min(imports[t] + d, import_limit)
                    d = d - (imports_c - imports[t])
                    imports[t] = imports_c
                    
                    available_imports_idx = np.where(imports < import_limit)[0]
                    # move to the most recent time when hydro and PHES are not operating at full power (spare power capacity available)
                    try:
                        t = sorted([i for i in available_imports_idx if i < t and i > nearest_full and i not in charging_full])[-1]
                    except IndexError:
                        break
            count += 1
        
    return hydro, imports

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

def check_battery_level(t, discharget, battery_discharge, battery_charge, battery_efficiency, battery_energy, old_battery_level):
    
    ''' check whether updated discharge at t will cause trouble for future discharges '''
    
    battery_discharge[t] = discharget
    battery_level = update_battery_level(battery_discharge.copy(),battery_charge.copy(),battery_efficiency,battery_energy,old_battery_level,t)
    
    return battery_level.min()

def fill_battery_discharge(deficit,battery_discharge,storage,CBP,storage_cap,efficiency,step,battery_charge,CBS,battery_efficiency,charge,charge_cap,window=8760):
    
    ''' fill deficit from battery discharging, generally the same logic as filling defict with hydro with a few small changes '''
        
    idx = np.where(deficit > 0)[0]
    storage_full = np.where(storage == storage_cap)[0]
    
    for idd, i in np.ndenumerate(idx):
        
        d = deficit[i]
        ini_t = i
        count = 0
        print("Filling deficit with battery", idd[0], "of", len(idx), "t =", ini_t)
        
        try:
            nearest_full = storage_full[np.where(storage_full < ini_t)[0][-1]]
        except IndexError:
            nearest_full = 0
        
        # model battery level based on current battery charging and discharging profiles, useful to determine whether there is enough enegy left in the battery for discharging
        battery_level = update_battery_level(battery_discharge.copy(),battery_charge.copy(),battery_efficiency,CBS)
        
        # print(d, battery_level[ini_t-1], battery_discharge[ini_t], battery_level[ini_t])
        
        # try to meet deficit using real-time battery discharge first
        discharget = min(battery_discharge[ini_t] + d, battery_level[ini_t-1], CBP)
        # do a battery level check every time new discharge is introduced to make sure future discharges are still valid
        level_check = check_battery_level(ini_t, discharget, battery_discharge.copy(),battery_charge.copy(),battery_efficiency,CBS,battery_level.copy())
        if level_check < 0:
            discharget = discharget  + level_check
        
        d = d - (discharget - battery_discharge[ini_t])
        battery_discharge[ini_t] = discharget
        
        battery_level = update_battery_level(battery_discharge.copy(),battery_charge.copy(),battery_efficiency,CBS,battery_level.copy(),ini_t)
        assert battery_level.min() >= 0
                        
        if d > 0:
            # need to meet deficit through trickle-charging PHES
            d = d / efficiency
            
            # available timesteps: 1) there is energy left in the battery AND 2) battery has spare power capacity 3) PHES have spare charging capacity
            available_energy_idx = np.where(battery_level > 0)[0] + 1
            available_power_idx = np.where(battery_discharge < CBP)[0]
            spare_PHES_charging = np.where(charge != charge_cap)[0]
            available_idx = np.intersect1d(spare_PHES_charging, np.intersect1d(available_energy_idx, available_power_idx))
            
            try:
                t = sorted([i for i in available_idx if i < ini_t and i > max(nearest_full, ini_t - window)])[-1]
            except IndexError:
                continue
            
            while d > 0 and count < step:
                
                # exhausted = False
                
                # discharge is constrainted by 1) available energy left in the battery and 2) battery power capacity
                discharget = min(battery_discharge[t] + d, battery_level[t-1], CBP)
                # print("t, battery level, previous discharge, new discharge:", t, battery_level[t-1], battery_discharge[t], discharget)
                # do a battery level check every time new discharge is introduced to make sure future discharges are still valid
                level_check = check_battery_level(t, discharget, battery_discharge.copy(),battery_charge.copy(),battery_efficiency,CBS,battery_level.copy())
                if level_check < 0:
                    discharget = discharget  + level_check
                
                d = d - (discharget - battery_discharge[t])
                battery_discharge[t] = discharget
                
                battery_level = update_battery_level(battery_discharge.copy(),battery_charge.copy(),battery_efficiency,CBS,battery_level.copy(),t)
                assert battery_level.min() >= 0
                
                # move to next t
                available_energy_idx = np.where(battery_level > 0)[0] + 1
                available_power_idx = np.where(battery_discharge < CBP)[0]
                available_idx = np.intersect1d(spare_PHES_charging, np.intersect1d(available_energy_idx, available_power_idx))
                
                # if exhausted: # then need to move to the next timestep when battery is charged
                #     available_battery_charge_idx = np.where(battery_charge > 0)[0]
                #     available_idx = np.intersect1d(available_idx, available_battery_charge_idx)

                try:
                    t = sorted([i for i in available_idx if i < ini_t and i > t])[-1]
                except IndexError:
                    break
                
                count += 1
            
    return battery_discharge

def save(hydro,imports, discharge,charge):
    np.savetxt('Results/Dispatch_Hydro_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), hydro, fmt='%f', delimiter=',', newline='\n', header='Flexible hydro')
    np.savetxt('Results/Dispatch_Imports_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), imports, fmt='%f', delimiter=',', newline='\n', header='Flexible imports')
    np.savetxt('Results/Dispatch_BatteryDischarge_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), discharge, fmt='%f', delimiter=',', newline='\n', header='Battery discharge')
    np.savetxt('Results/Dispatch_BatteryCharge_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), charge, fmt='%f', delimiter=',', newline='\n', header='Battery charge')
    
def Flexible(capacities):    
    S = Solution(capacities)

    # fill hydro with agg_storage = True

    # Calculate initial deficit
    Deficit1, _ = Reliability(S, flexible=np.zeros((intervals,nodes), dtype=np.float64), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
    Deficit2, _ = Reliability(S, flexible=np.ones((intervals, nodes), dtype=np.float64) * CPeak * pow(10,3), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
    
    GImports = Deficit2.sum() / years / efficiency
    GHydroBio = Deficit1.sum() / years / efficiency - GImports

    print("Initial deficit:", Deficit.sum()/1e6)
    
    Storage = S.Storage
    Charge = S.Charge
    Storage_cap = (S.CPHS.sum() + S.CBS.sum()) * pow(10, 3)
    Charge_cap = (S.CPHP.sum() + S.CBP.sum()) * pow(10, 3)
    
    if (GImports == 0):
        print("HYDRO ONLY")
        print("------------------------------")
        # initial hydro (peak) = zero
        hydro = np.zeros((intervals,nodes), dtype=np.float64)
        imports = np.zeros((intervals,nodes), dtype=np.float64)
        
        Deficit, _ = Reliability(S, flexible=hydro+imports, agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
        # start filling
        h,i = fill_deficit(Deficit.copy(),hydro.copy(),imports.copy(), Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,True,False)
        
        # check deficit again
        Deficit, _ = Reliability(S, flexible=h.copy()+i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
        Storage = S.Storage
        Charge = S.Charge
        
        print("Initial hydro generation:", h.sum()/1e6/years)
        print("Remaining deficit:", Deficit.sum()/1e6)
        step = 1
        
        # repeat filling for 50 times max
        while Deficit.sum() > allowance*years and step < 50:
            print("Total deficit:", Deficit.sum(), ", No. of deficit:", len(np.where(Deficit > 0)[0]), ", Step =", step)
            h,i = fill_deficit(Deficit.copy(),h.copy(),i.copy(), Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,True,False)
            Deficit, _ = Reliability(S, flexible=h.copy() + i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
            Storage = S.Storage
            Charge = S.Charge
            step += 1
        
        print("Hydro done")
    else:
        print("HYDRO + IMPORTS")
        print("------------------------------")
        # initial hydro (peak) = zero
        hydro = np.ones((intervals,nodes), dtype=np.float64) * S.CPeak * 1000
        imports = np.zeros((intervals,nodes), dtype=np.float64)
        
        Deficit, _ = Reliability(S, flexible=hydro+imports, agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
        # start filling
        h,i = fill_deficit(Deficit.copy(),hydro.copy(),imports.copy(), Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,False,True)
        
        # check deficit again
        Deficit, _ = Reliability(S, flexible=h.copy()+i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
        Storage = S.Storage
        Charge = S.Charge
        
        print("Initial imported generation:", i.sum()/1e6/years)
        print("Remaining deficit:", Deficit.sum()/1e6)
        step = 1
        
        # repeat filling for 50 times max
        while Deficit.sum() > allowance*years and step < 50:
            print("Total deficit:", Deficit.sum(), ", No. of deficit:", len(np.where(Deficit > 0)[0]), ", Step =", step)
            h,i = fill_deficit(Deficit.copy(),h.copy(),i.copy(), Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,False,True)
            Deficit, _ = Reliability(S, flexible=h.copy() + i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
            Storage = S.Storage
            Charge = S.Charge
            step += 1
        
        print("Imports done")

        if Deficit.sum() < allowance*years:
            hydro = np.zeros((intervals,nodes), dtype=np.float64)
            Deficit, _ = Reliability(S, flexible=hydro+i, agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
            h,i = fill_deficit(Deficit.copy(),hydro.copy(),imports.copy(), Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,True,False)
        
            # check deficit again
            Deficit, _ = Reliability(S, flexible=h.copy()+i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
            Storage = S.Storage
            Charge = S.Charge
            
            print("Initial hydro generation:", i.sum()/1e6/years)
            print("Remaining deficit:", Deficit.sum()/1e6)
            step = 1
            
            # repeat filling for 50 times max
            while Deficit.sum() > allowance*years and step < 50:
                print("Total deficit:", Deficit.sum(), ", No. of deficit:", len(np.where(Deficit > 0)[0]), ", Step =", step)
                h,i = fill_deficit(Deficit.copy(),h.copy(),i.copy(), Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,True,False)
                Deficit, _ = Reliability(S, flexible=h.copy() + i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
                Storage = S.Storage
                Charge = S.Charge
                step += 1

            print("Hydro done")

    # remove unnecessary hydro (peak) if spillage presents
    '''SUM ALONG NODAL AXIS'''
    h_remove_spillage = np.array([max(0,h[i]-S.Spillage[i]) for i in range(len(h))])
    
    # check deficit again
    Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = True, battery_charge=np.zeros((intervals, nodes), dtype=np.float64),battery_discharge=np.zeros((intervals, nodes), dtype=np.float64))
        
    print("Final hydro generation:", h_remove_spillage.sum()/1e6/years)
    print("Remaining deficit:", Deficit.sum()/1e6)
    
    # fill battery charge whenever possible
    
    # set everything to zero first
    BatteryCharge = np.zeros(intervals, dtype=np.float64)
    BatteryDischarge = np.zeros(intervals, dtype=np.float64)
    BatteryStorage = np.zeros(intervals, dtype=np.float64)
    
    CBS = S.CBS.sum() * 1000
    CBP = S.CBP.sum() * 1000
    battery_efficiency = 0.9
    
    if CBP != 0:
    
        Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
        
        Spillage = S.Spillage
        # Initially, use all spillage for battery charge, as long as battery power is sufficient. This allows battery storage level to be monitored when filling deficit using battery discharge
        BatteryCharge = np.clip(Spillage, 0, CBP)
        
        # fill deficit from battery discharge
        
        Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
        Storage = S.Storage
        Charge = S.Charge
        Storage_cap = S.CPHS.sum() * pow(10, 3)
        Charge_cap = S.CPHP.sum() * pow(10, 3)
        
        print("Initial deficit for battery:", Deficit.sum()/1e6)
        
        BatteryDischarge = fill_battery_discharge(Deficit.copy(),BatteryDischarge.copy(),Storage.copy(),CBP,Storage_cap,S.efficiency,9999,BatteryCharge.copy(),CBS,battery_efficiency,Charge.copy(),Charge_cap)
        
        Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
        Storage = S.Storage
        Charge = S.Charge
        
        print("Initial battery discharge:", BatteryDischarge.sum()/1e6)
        print("Remaining deficit:", Deficit.sum()/1e6)
        step = 1
        
        while Deficit.sum() > allowance*years and step < 50:
            print("Total deficit:", Deficit.sum(), ", No. of deficit:", len(np.where(Deficit > 0)[0]), ", Step =", step)
            BatteryDischarge = fill_battery_discharge(Deficit.copy(),BatteryDischarge.copy(),Storage.copy(),CBP,Storage_cap,S.efficiency,9999,BatteryCharge.copy(),CBS,battery_efficiency,Charge.copy(),Charge_cap)
            Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
            Storage = S.Storage
            Charge = S.Charge
            step += 1
            
        BatteryDischarge = np.array([max(0,BatteryDischarge[i]-max(0,S.Spillage[i])) for i in range(len(BatteryDischarge))])
        Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
        
        print("Final battery discharge:", BatteryDischarge.sum()/1e6)
        print("Remaining deficit:", Deficit.sum()/1e6)
        
        battery_level = update_battery_level(BatteryDischarge.copy(),BatteryCharge.copy(),battery_efficiency,CBS)
        
        # revise battery charge by modelling battery storage level, if there is excess charging after battery is full, then revise battery charge accordingly
        
        battery_level = np.zeros(intervals, dtype=np.float64)
        
        for t in range(intervals):
            
            if t == 0:
                previous_level = 0.5 * CBS
            else:
                previous_level = battery_level[t-1]
            netcharget = BatteryCharge[t] * battery_efficiency - BatteryDischarge[t]
            battery_levelt = min(previous_level + netcharget, CBS)
            
            if netcharget > 0 and battery_levelt == CBS:
                BatteryCharget = BatteryCharge[t] - max(0,battery_level[t-1] + netcharget - battery_levelt) / battery_efficiency
                BatteryCharge[t] = BatteryCharget
                
            battery_level[t] = battery_levelt
                    
        Deficit, _ = Reliability(S, flexible=h_remove_spillage.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
        
        print("Battery Charge:", BatteryCharge.sum()/1e6)
        print("Final deficit:", Deficit.sum()/1e6)
        
        # sometimes, there will be a tiny deficit when battery charge is revised (for no reason). This deficit is filled by flexible hydro (which is easier)
        if Deficit.sum() > allowance*years:
            
            Storage = S.Storage
            Charge = S.Charge
            h_revise, i = fill_deficit(Deficit.copy(),h_remove_spillage.copy(),i.copy(),Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,True,False)
            Deficit, _ = Reliability(S, flexible=h_revise.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
            
            if Deficit.sum() < allowance*years:
                save(h_revise, BatteryDischarge, BatteryCharge)
                
            else:
                step = 1
                while Deficit.sum() > allowance*years and step < 50:
                    print("Total deficit:", Deficit.sum(), ", No. of deficit:", len(np.where(Deficit > 0)[0]), ", Step =", step)
                    Storage = S.Storage
                    Charge = S.Charge
                    h_revise, i = fill_deficit(Deficit.copy(),h_remove_spillage.copy(),i.copy(),Storage.copy(),S.CPeak*1000,S.CInter*1000,Storage_cap,S.efficiency,9999,Charge.copy(),Charge_cap,True,False)
                    Deficit, _ = Reliability(S, flexible=h_revise.copy()+i.copy(), agg_storage = False, battery_charge = BatteryCharge.copy(), battery_discharge = BatteryDischarge.copy())
                    step += 1
                if Deficit.sum() < allowance*years:
                    save(h_revise, i, BatteryDischarge, BatteryCharge)
                    
        else:
            save(h_remove_spillage, i, BatteryDischarge, BatteryCharge)
    else:
        
        save(h_remove_spillage, i, BatteryDischarge, BatteryCharge)

if __name__=='__main__':    
    capacities = np.genfromtxt('Results/Optimisation_resultx_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario), delimiter=',')

    Flexible(capacities)    
    
    