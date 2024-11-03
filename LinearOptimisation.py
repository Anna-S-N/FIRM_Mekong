# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from datetime import datetime as dt
import csv
import numpy as np
import pyomo.environ as pyo

from LinearInput import * 

masked_DCloss = DCloss[network_mask]

pv_zs_in_n =   [np.where(PVl  ==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing
wind_zs_in_n = [np.where(Windl==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing

pos_export_lines = [np.where(network[:,0]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing
neg_export_lines = [np.where(network[:,1]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing

nhvdc = len(network)

ndays = 365*years 
intervals = int(ndays*24/resolution)

adj_energy = (MLoad[:intervals, :].sum() * pow(10, -9) * resolution/years)

#%%
from Costs import UnitCosts, annulization, annulization_transmission

def sum_model_c(c_array):
    return sum((c_array[i].value for i in c_array))

def model_GPV(c_array):
    return np.array([c_array[i].value for i in c_array]) * TSPV * 1000.

def model_GWind(c_array):
    return np.array([c_array[i].value for i in c_array]) * TSWind * 1000.

def calculate_costs_linear(m):
    PV_costs = annulization(UnitCosts[0], UnitCosts[1], UnitCosts[2], UnitCosts[3],UnitCosts[-1], sum_model_c(m.cpv), model_GPV(m.cpv).sum()/ years)
    PV_Wind_transmission_cost = annulization_transmission(UnitCosts[8],UnitCosts[34],UnitCosts[9],UnitCosts[10],UnitCosts[11],UnitCosts[-1],sum_model_c(m.cpv),0,20)
    wind_costs = 0
    GWind_sites = model_GWind(m.cwind).sum(axis=0)
    for i in range(len(GWind_sites)):
        if i not in Windl_Viet_int: # Onshore wind 
            wind_costs += annulization(UnitCosts[4],UnitCosts[5],UnitCosts[6],UnitCosts[7],UnitCosts[-1],model.cwind[i+1].value,GWind_sites[i]/years)
            PV_Wind_transmission_cost += annulization_transmission(UnitCosts[8],UnitCosts[34],UnitCosts[9],UnitCosts[10],UnitCosts[11],UnitCosts[-1],model.cwind[i+1].value,0,20)
        else: # Offshore wind for Vietnam
            wind_costs += annulization(UnitCosts[35],UnitCosts[36],UnitCosts[37],UnitCosts[38],UnitCosts[-1],model.cwind[i+1].value,GWind_sites[i]/years)
            PV_Wind_transmission_cost += annulization_transmission(4000*0.7,0,40*0.7,0,30,UnitCosts[-1],model.cwind[i+1].value,0,100) # Submarine cable offshore
    transmission_costs = PV_Wind_transmission_cost
    for i in range(nhvdc):
        if hvdc_mask[i]: # HVDC line costs
            transmission_costs += annulization_transmission(UnitCosts[24],0,UnitCosts[25],UnitCosts[26],UnitCosts[27],UnitCosts[-1],m.chvdc[i+1].value,
                                                            (sum((m.hvdc_neg[t, i]) for t in m.time) + sum((m.hvdc_pos[t, i]) for t in m.time))/years,DCdistance[i])
        else: # HVAC line + transformer costs
            transmission_costs += annulization_transmission(UnitCosts[8],UnitCosts[34],UnitCosts[9],UnitCosts[10],UnitCosts[11],UnitCosts[-1],m.chvdc[i+1].value,
                                                            (sum((m.hvdc_neg[t, i]) for t in m.time) + sum((m.hvdc_pos[t, i]) for t in m.time))/years,DCdistance[i])
    # Converter and substation costs, a pair of stations per line
    for i in range(nhvdc):
        if hvdc_mask[i]:
            converter_costs = 2 * annulization(UnitCosts[28],UnitCosts[29],UnitCosts[30],UnitCosts[31],UnitCosts[-1],sum_model_c(m.chvdc),0)
            transmission_costs += converter_costs
        """ else:
            substation_costs = 2 * annulization(UnitCosts[35],UnitCosts[29],UnitCosts[30],UnitCosts[31],UnitCosts[-1],sum_model_c(m.chvdc),0)
            transmission_costs += substation_costs """
            
    pv_phes = (1-(1+UnitCosts[-1])**(-1*UnitCosts[18]))/UnitCosts[-1]
    phes_costs = (UnitCosts[12] * sum_model_c(m.cphp) * pow(10,6) + UnitCosts[13] * sum_model_c(m.cphe) * pow(10,6)) / pv_phes \
                    + UnitCosts[14] * sum_model_c(m.cphp) * pow(10,6) + UnitCosts[15] * (sum_model_c(m.phdischarge) + sum_model_c(m.bdischarge)) / years \
                    + UnitCosts[16] * ((1+UnitCosts[-1])**(-1*UnitCosts[17]) + (1+UnitCosts[-1])**(-1*UnitCosts[17]*2)) / pv_phes
    
    pv_battery = (1-(1+UnitCosts[-1])**(-1*UnitCosts[22]))/UnitCosts[-1] # 19, 20, 21, 22
    battery_costs = (UnitCosts[19] * sum_model_c(m.cbp) * pow(10,6) + UnitCosts[20] * sum_model_c(m.cbh) * pow(10,6)) / pv_battery \
                    + UnitCosts[21] * sum_model_c(m.cbh) * pow(10,6)
    
    hydro_costs = UnitCosts[23] * sum_model_c(m.hydro)
    # import_costs = UnitCosts[32] * GImports
    baseload_costs = UnitCosts[33] * baseload[:intervals, :].sum()
    
    costs = PV_costs + wind_costs + transmission_costs + phes_costs + battery_costs + hydro_costs  + baseload_costs #+ import_costs
    tech_costs = np.array([PV_costs, wind_costs, transmission_costs, phes_costs, battery_costs, hydro_costs,
                           # import_costs,
                           baseload_costs], dtype=np.float64)
    
    return costs, tech_costs
     




#%%

print("Instantiating optimiser:", dt.now())
model = pyo.ConcreteModel()

# indexers
model.lines = pyo.RangeSet(nhvdc)
model.nodes = pyo.RangeSet(nodes)
model.time  = pyo.RangeSet(intervals)

# cost parameter
# model.hvdcCost = pyo.Param(model.lines, domain=pyo.Reals, initialize = dict(zip(range(1, nhvdc+1), factor[5:9][network_mask])))

#capacity variables
model.cpv = pyo.Var(
    model.nodes,   
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(pv_lb, pv_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 for l, u in zip(pv_lb, pv_ub)))),
    )
model.cwind = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(wind_lb, wind_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 for l, u in zip(wind_lb, wind_ub)))),
    )
model.cphp = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(phes_lb, phes_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 for l, u in zip(phes_lb, phes_ub)))),
    )
model.cphe = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(storage_lb, storage_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 for l, u in zip(storage_lb, storage_ub)))),
    )
model.cbp = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(battery_lb, battery_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 for l, u in zip(battery_lb, battery_ub)))),
    )
model.cbh = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(bduration_lb, bduration_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 for l, u in zip(bduration_lb, bduration_ub)))),
    )
model.chvdc = pyo.Var(
    model.lines, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nhvdc+1), zip(transmission_lb, transmission_ub))),
    initialize=dict(zip(range(1, nhvdc+1), ((u-l)/2 for l, u in zip(transmission_lb, transmission_ub)))),
    )

## Operational variables
# Storage
model.chargeph   = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.dischargeph= pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.storageph  = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

model.chargeb    = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.dischargeb = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.storageb   = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

# Transmission
model.hvdc_pos = pyo.Var(model.time, model.lines, initialize=0, domain=pyo.NonNegativeReals)
model.hvdc_neg = pyo.Var(model.time, model.lines, initialize=0, domain=pyo.NonNegativeReals)

model.deficit  = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

# Flexible generation
model.hydro    = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.bio      = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.waste    = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

## Operational constraints
# Dis/Charging power and storage energy limits
model.constr_charge_power_upper_ph    = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.chargeph[t, n]    <= m.cphp[n])
model.constr_discharge_power_upper_ph = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.dischargeph[t, n] <= m.cphp[n])
model.constr_storage_energy_upper_ph  = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.storageph[t, n]   <= m.cphe[n])

model.constr_charge_power_upper_b     = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.chargeb[t, n]    <= m.cbp[n])
model.constr_discharge_power_upper_b  = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.dischargeb[t, n] <= m.cbp[n])
model.constr_storage_energy_upper_b   = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.storageb[t, n]   <= m.cbh[n])

# State of charge 
def constr_state_of_chargeph(m, t, n):
    if t==1: return m.storageph[t, n] == 0.5 * m.cphe[n]
    else: return m.storageph[t, n] == m.storageph[t-1, n] - m.dischargeph[t-1, n] * resolution + m.chargeph[t-1, n] * resolution * efficiency
model.constr_storage_state_of_chargeph = pyo.Constraint(model.time, model.nodes, rule=constr_state_of_chargeph)

# State of charge 
def constr_state_of_chargeb(m, t, n):
    if t==1: 
        return m.storageb[t, n] == 0.5 * m.cbh[n]
    ## Check this logic - units?
    else: 
        return m.storageb[t, n] == m.storageb[t-1, n] + (- m.dischargeb[t-1, n] * resolution + m.chargeb[t-1, n] * resolution * efficiency)/ m.cbp[n]
model.constr_storage_state_of_chargeb = pyo.Constraint(model.time, model.nodes, rule=constr_state_of_chargeb)

# Deficit allowance 
model.constr_deficit = pyo.Constraint(rule=lambda m: pyo.summation(m.deficit)* resolution/years <= allowance*adj_energy)

# Flexible power limits
model.constr_hydro_power   = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.hydro[t, n] <= CHydro[n-1])
model.constr_bio_power   = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.bio[t, n]   <= CBio[n-1])
model.constr_waste_power = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.waste[t, n] <= CWaste[n-1])

# Flexible energy limits (annual basis)
def constr_hydro_weekly(m, t, n):
    #check format and constr source
    sum((m.hydro[j, n] for j in range(t, min(intervals+2, t + 7 * 24 / resolution))))
    
    return energy <= hydro_max_weeks[(t-1)//(24*7), n-1]
##use constraint list?
# model.constr_hydro_weeky = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: constr_hydro_weekly) 
# model.constr_hydro_energy = pyo.Constraint(rule=lambda m: pyo.summation(m.hydro)*resolution/years <= 100000.0) 
# model.constr_bio_energy   = pyo.Constraint(rule=lambda m: pyo.summation(m.bio)  *resolution/years <= 100.0)
# model.constr_waste_energy = pyo.Constraint(rule=lambda m: pyo.summation(m.waste)*resolution/years <= 100.0)


# HVDC line capacity
model.constr_hvdc_power_import = pyo.Constraint(model.time, model.lines, rule=lambda m, t, l: m.hvdc_pos[t, l] <= m.chvdc[l])
model.constr_hvdc_power_export = pyo.Constraint(model.time, model.lines, rule=lambda m, t, l: m.hvdc_neg[t, l] <= m.chvdc[l])


model.constr_power_balance = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: (
    Netload[t-1, n-1] 
    - m.deficit[t,n]
    + m.chargeph[t,n] # charging
    + m.chargeb[t,n] 
    - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) # pv node-by-node
    - sum((m.cwind[z]*TSWind[t-1, z-1] for z in wind_zs_in_n[n-1])) # wind node-by-node
    - m.hydro[t,n] 
    # - m.geo[t,n] 
    - m.bio[t,n] 
    - m.waste[t,n] 
    - m.dischargeph[t,n] 
    - m.dischargeb[t,n] 
    + sum((m.hvdc_pos[t, l] - m.hvdc_neg[t, l]*(1-masked_DCloss[l-1]) for l in pos_export_lines[n-1]))
    + sum((m.hvdc_neg[t, l] - m.hvdc_pos[t, l]*(1-masked_DCloss[l-1]) for l in neg_export_lines[n-1]))
    ) <= 0.0
    )


model.OBJ = pyo.Objective(rule=calculate_costs_linear)

optimiser = pyo.SolverFactory('gurobi')

start=dt.now()
print("Optimisation starts:", start)
optimiser.solve(model)
end=dt.now()
print("Optimisation took:", end-start)

model.OBJ.display()

#%%

S = Solution(model)

print('pv:', S.cpv)
print('wind:', S.cwind)
print('php:', S.cphp)
print('phe:', S.cphe)
print('chvdc:', S.chvdc)

try:
    with open(f'Results/Optimisation_resultx{node}.csv', 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([S.OBJ] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.chvdc))
except FileNotFoundError:
    import os 
    os.mkdir('Results')
    with open(f'Results/Optimisation_resultx{node}.csv', 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([S.OBJ] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.chvdc))


from LinearStatistics import Information
Information(S)


