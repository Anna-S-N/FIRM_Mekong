# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from datetime import datetime as dt
import csv
import numpy as np
import pyomo.environ as pyo

from LinearInput import * 


pv_zs_in_n =   [np.where(PVl  ==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing
wind_on_zs_in_n = [np.where(Windl[~offshore_mask]==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing
wind_off_zs_in_n = [np.where(Windl[offshore_mask]==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing

pos_hvdc_lines = [np.where(network[hvdc_mask,0]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing
neg_hvdc_lines = [np.where(network[hvdc_mask,1]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing

pos_hvac_lines = [np.where(network[~hvdc_mask,0]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing
neg_hvac_lines = [np.where(network[~hvdc_mask,1]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing

import_connecs = [np.where(Interl==node)[0]+1 for node in Nodel]


ndays = 365*years 
intervals = int(ndays*24/resolution)

adj_energy = (MLoad.sum() * resolution/years) * pow(10,3)

#%%
from Costs import UnitCosts

def annualisation_constants(capex, fom, vom, life, dr):
    pv = (1-(1+dr)**(-1*life))/dr
    return (capex*pow(10,6)/pv + fom*pow(10,6), vom*pow(10,3)) # p*,  e*

def annualisation_constants_transmission(capex, transformer_capex, fom, vom, life, dr, d):
    pv = (1-(1+dr)**(-1*life))/dr
    return (pow(10,3) * ((d * capex + transformer_capex) / pv + d * fom), pow(10, 3) * vom) # p*, e*

def cost_constant_factors():
    pv_costs = annualisation_constants(UnitCosts[0], UnitCosts[1],UnitCosts[2],UnitCosts[3],UnitCosts[-1])
    pv_transmission_costs = annualisation_constants_transmission(UnitCosts[8],UnitCosts[34],UnitCosts[9],UnitCosts[10],UnitCosts[11],UnitCosts[-1], 20)
    onshore_wind_costs = annualisation_constants(UnitCosts[4],UnitCosts[5],UnitCosts[6],UnitCosts[7],UnitCosts[-1])
    onshore_wind_transmission_costs = annualisation_constants_transmission(UnitCosts[8],UnitCosts[34],UnitCosts[9],UnitCosts[10],UnitCosts[11],UnitCosts[-1], 20)
    offshore_wind_costs = annualisation_constants(UnitCosts[35],UnitCosts[36],UnitCosts[37],UnitCosts[38],UnitCosts[-1])
    offshore_wind_transmission_costs = annualisation_constants_transmission(4000*0.7,0,40*0.7,0,30,UnitCosts[-1],100)

    hvdc_transmission_costs = annualisation_constants_transmission(UnitCosts[24],0,UnitCosts[25],UnitCosts[26],UnitCosts[27],UnitCosts[-1],DCdistance[hvdc_mask])
    hvac_transmission_costs = annualisation_constants_transmission(UnitCosts[8],UnitCosts[34],UnitCosts[9],UnitCosts[10],UnitCosts[11],UnitCosts[-1],DCdistance[~hvdc_mask])
    
    converter_substation_costs = annualisation_constants(UnitCosts[28],UnitCosts[29],UnitCosts[30],UnitCosts[31],UnitCosts[-1])
    converter_substation_costs = tuple(2*i for i in converter_substation_costs)
    
    pv_phes = (1-(1+UnitCosts[-1])**(-1*UnitCosts[18]))/UnitCosts[-1]
    phes_costs = (UnitCosts[12] * pow(10,6) / pv_phes + UnitCosts[14] * pow(10,6), 
                  UnitCosts[13] * pow(10,6) / pv_phes,
                  UnitCosts[15] * pow(10,3) / years, 
                  UnitCosts[16] * ((1+UnitCosts[-1])**(-1*UnitCosts[17]) + (1+UnitCosts[-1])**(-1*UnitCosts[17]*2)) / pv_phes
    ) #pc*, ec*, dis*, 1*
    # phes_costs = tuple(x/1000 for x in phes_costs)
    
    pv_battery = (1-(1+UnitCosts[-1])**(-1*UnitCosts[22]))/UnitCosts[-1] # 19, 20, 21, 22
    battery_costs = (UnitCosts[19] * pow(10,6) / pv_battery, UnitCosts[20] * pow(10,6) / pv_battery + UnitCosts[21] * pow(10,6))# pc*, ec*
    # battery_costs = tuple(x/1000 for x in battery_costs)
    
    hydro_costs = UnitCosts[23]
    import_costs = UnitCosts[32] 
    baseload_costs = UnitCosts[33]
    
    return (pv_costs[0] + pv_transmission_costs[0], onshore_wind_costs[0] + onshore_wind_transmission_costs[0],
            offshore_wind_costs[0] + offshore_wind_transmission_costs[0], hvdc_transmission_costs[0],
            hvac_transmission_costs[0], converter_substation_costs[0], phes_costs, battery_costs,
            hydro_costs, import_costs, baseload_costs)

(pv_costs, onshore_wind_costs, offshore_wind_costs, hvdc_transmission_costs,
        hvac_transmission_costs, converter_substation_costs, phes_costs, battery_costs,
        hydro_costs, import_costs, baseload_costs) = cost_constant_factors()

#%%

print("Instantiating optimiser:", dt.now())
model = pyo.ConcreteModel()

# indexers
model.hvdc = pyo.RangeSet(nhvdc)
model.hvac = pyo.RangeSet(nhvac)
model.nodes = pyo.RangeSet(nodes)
model.pvl = pyo.RangeSet(len(PVl))
model.onshorel = pyo.RangeSet((~offshore_mask).sum())
model.offshorel = pyo.RangeSet(offshore_mask.sum())
model.time  = pyo.RangeSet(intervals)
model.imps  = pyo.RangeSet(ninters)

# static params
model.hvdcCost = pyo.Param(model.hvdc, domain=pyo.Reals, initialize = dict(zip(range(1, nhvdc+1), hvdc_transmission_costs)))
model.hvacCost = pyo.Param(model.hvac, domain=pyo.Reals, initialize = dict(zip(range(1, nhvac+1), hvac_transmission_costs)))

#capacity variables
model.cpv = pyo.Var(
    model.pvl,   
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(pv_lb, pv_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(pv_lb, pv_ub)))),
    )
model.conshore = pyo.Var(
    model.onshorel, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(wind_on_lb, wind_on_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(wind_on_lb, wind_on_ub)))),
    )
model.coffshore = pyo.Var(
    model.offshorel, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(wind_off_lb, wind_off_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(wind_off_lb, wind_off_ub)))),
    )
model.cphp = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(phes_lb, phes_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(phes_lb, phes_ub)))),
    )
model.cphe = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(storage_lb, storage_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(storage_lb, storage_ub)))),
    )
model.cbp = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(battery_lb, battery_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(battery_lb, battery_ub)))),
    )
model.cbe = pyo.Var(
    model.nodes, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nodes+1), zip(bduration_lb, bduration_ub))),
    initialize=dict(zip(range(1, nodes+1), ((u-l)/2 + l for l, u in zip(bduration_lb, bduration_ub)))),
    )
model.cinter= pyo.Var(
    model.imps, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, ninters+1), zip(inters_lb, inters_ub))),
    initialize=dict(zip(range(1, ninters+1), ((u-l)/2 + l for l, u in zip(inters_lb, inters_ub)))),
    )
model.chvdc = pyo.Var(
    model.hvdc, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nhvdc+1), zip(hvdc_lb, hvdc_ub))),
    initialize=dict(zip(range(1, nhvdc+1), ((u-l)/2 + l for l, u in zip(hvdc_lb, hvdc_ub)))),
    )
model.chvac = pyo.Var(
    model.hvac, 
    domain=pyo.NonNegativeReals, 
    bounds=dict(zip(range(1, nhvac+1), zip(hvac_lb, hvac_ub))),
    initialize=dict(zip(range(1, nhvac+1), ((u-l)/2 + l for l, u in zip(hvac_lb, hvac_ub)))),
    )

# for i in model.cphe:
#     model.cphe[i].fix((storage_ub[i-1]-storage_lb[i-1])/2)

## Operational variables
# Storage
model.chargeph   = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.dischargeph= pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.storageph  = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

model.chargeb    = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.dischargeb = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.storageb   = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

# Transmission
model.hvdc_pos = pyo.Var(model.time, model.hvdc, initialize=0, domain=pyo.NonNegativeReals)
model.hvdc_neg = pyo.Var(model.time, model.hvdc, initialize=0, domain=pyo.NonNegativeReals)
model.hvac_pos = pyo.Var(model.time, model.hvac, initialize=0, domain=pyo.NonNegativeReals)
model.hvac_neg = pyo.Var(model.time, model.hvac, initialize=0, domain=pyo.NonNegativeReals)

# model.deficit  = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)

# Flexible generation
model.hydro    = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.bio      = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.waste    = pyo.Var(model.time, model.nodes, initialize=0, domain=pyo.NonNegativeReals)
model.imports  = pyo.Var(model.time, model.imps,  initialize=0, domain=pyo.NonNegativeReals)

## Operational constraints
# Dis/Charging power and storage energy limits
model.constr_charge_power_upper_ph    = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.chargeph[t, n]    <= m.cphp[n])
model.constr_discharge_power_upper_ph = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.dischargeph[t, n] <= m.cphp[n])
model.constr_storage_energy_upper_ph  = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.storageph[t, n]   <= m.cphe[n])

model.constr_charge_power_upper_b     = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.chargeb[t, n]    <= m.cbp[n])
model.constr_discharge_power_upper_b  = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.dischargeb[t, n] <= m.cbp[n])
model.constr_storage_energy_upper_b   = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.storageb[t, n]   <= m.cbe[n])

# State of charge 
def constr_state_of_chargeph(m, t, n):
    if t==1: 
        return m.storageph[t, n] == StartCharge * m.cphe[n]
    else: 
        return m.storageph[t, n] == m.storageph[t-1, n] - m.dischargeph[t-1, n] * resolution + m.chargeph[t-1, n] * resolution * efficiency
model.constr_storage_state_of_chargeph = pyo.Constraint(model.time, model.nodes, rule=constr_state_of_chargeph)

# State of charge 
def constr_state_of_chargeb(m, t, n):
    if t==1: 
        return m.storageb[t, n] == StartCharge * m.cbe[n]
    ## Check this logic - units?
    else: 
        return m.storageb[t, n] == m.storageb[t-1, n] - m.dischargeb[t-1, n] * resolution + m.chargeb[t-1, n] * resolution * efficiency
model.constr_storage_state_of_chargeb = pyo.Constraint(model.time, model.nodes, rule=constr_state_of_chargeb)

# Deficit allowance 
# model.constr_deficit = pyo.Constraint(rule=lambda m: pyo.summation(m.deficit)* resolution/years <= allowance)

# Flexible power limits
model.constr_hydro_power  = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.hydro[t, n]   <= CHydro[n-1])
model.constr_bio_power    = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.bio[t, n]     <= CBio[n-1])
model.constr_waste_power  = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: m.waste[t, n]   <= CWaste[n-1])
model.constr_imports_power= pyo.Constraint(model.time, model.imps,  rule=lambda m, t, l: m.imports[t, l] <= m.cinter[l])

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
model.constr_hvdc_power_import = pyo.Constraint(model.time, model.hvdc, rule=lambda m, t, l: m.hvdc_pos[t, l] <= m.chvdc[l])
model.constr_hvdc_power_export = pyo.Constraint(model.time, model.hvdc, rule=lambda m, t, l: m.hvdc_neg[t, l] <= m.chvdc[l])

model.constr_hvac_power_import = pyo.Constraint(model.time, model.hvac, rule=lambda m, t, l: m.hvac_pos[t, l] <= m.chvac[l])
model.constr_hvac_power_export = pyo.Constraint(model.time, model.hvac, rule=lambda m, t, l: m.hvac_neg[t, l] <= m.chvac[l])

#supply demand
model.constr_power_balance = pyo.Constraint(model.time, model.nodes, rule=lambda m, t, n: (
    Netload[t-1, n-1] 
    + m.chargeph[t,n] 
    + m.chargeb[t,n] 
    + sum((m.hvdc_pos[t, l] - m.hvdc_neg[t, l]*(1-HVDCloss[l-1]) for l in pos_hvdc_lines[n-1]))
    + sum((m.hvdc_neg[t, l] - m.hvdc_pos[t, l]*(1-HVDCloss[l-1]) for l in neg_hvdc_lines[n-1]))
    + sum((m.hvac_pos[t, l] - m.hvac_neg[t, l]*(1-HVACloss[l-1]) for l in pos_hvac_lines[n-1]))
    + sum((m.hvac_neg[t, l] - m.hvac_pos[t, l]*(1-HVACloss[l-1]) for l in neg_hvac_lines[n-1]))
    # - m.deficit[t,n]
    - m.dischargeph[t,n] 
    - m.dischargeb[t,n] 
    - m.hydro[t,n] 
    - m.bio[t,n] 
    - m.waste[t,n] 
    # - m.cpv[n]*TSPV[t-1, n-1]
    # - m.cwind[n]*TSWind[t-1, n-1]
    - sum((m.cpv[z]*TSPV[t-1, z-1] for z in pv_zs_in_n[n-1])) # pv node-by-node
    - sum((m.conshore[z] *TSWind_on[t-1, z-1]  for z in wind_on_zs_in_n[n-1])) # wind node-by-node
    - sum((m.coffshore[z]*TSWind_off[t-1, z-1] for z in wind_off_zs_in_n[n-1])) # wind node-by-node
    - sum((m.imports[t, l] for l in import_connecs[n-1]))
    ) <= 0.0
    )

##### Cost components
model.pv_cost               = pyo.Expression(rule=lambda m: pyo.summation(m.cpv) * pv_costs)
model.onshore_wind_cost     = pyo.Expression(rule=lambda m: pyo.summation(m.conshore) * onshore_wind_costs)
model.offshore_wind_cost    = pyo.Expression(rule=lambda m: pyo.summation(m.coffshore) * offshore_wind_costs)
model.hvdc_cost             = pyo.Expression(rule=lambda m: pyo.summation(m.hvdcCost, m.chvdc))
model.hvac_cost             = pyo.Expression(rule=lambda m: pyo.summation(m.hvacCost, m.chvac))
model.converter_substation_cost = pyo.Expression(rule=lambda m: (pyo.summation(m.chvdc) + pyo.summation(m.chvac))*converter_substation_costs)
model.phes_cost             = pyo.Expression(rule=lambda m: pyo.summation(m.cphp) * phes_costs[0] +
                                                            pyo.summation(m.cphe) * phes_costs[1] + 
                                                            pyo.summation(m.dischargeph) * resolution / years * phes_costs[2] + 
                                                            phes_costs[3])
model.battery_cost          = pyo.Expression(rule=lambda m: pyo.summation(m.cbp) * battery_costs[0] +
                                                            pyo.summation(m.cbe) * battery_costs[1])

model.hydro_cost    = pyo.Expression(rule=lambda m: (pyo.summation(m.hydro) + hydro_baseload.sum()) * resolution / years * hydro_costs)
model.import_cost   = pyo.Expression(rule=lambda m: pyo.summation(m.imports) * resolution /years / efficiency * import_costs)
model.baseload_cost = pyo.Expression(rule=lambda m: baseload.sum() * resolution / years * baseload_costs)


model.OBJ = pyo.Objective(rule=lambda m:  (m.pv_cost + m.onshore_wind_cost + 
    m.offshore_wind_cost + m.hvdc_cost + m.hvac_cost + m.converter_substation_cost + 
    m.phes_cost + m.battery_cost + m.baseload_cost + m.hydro_cost) / adj_energy)

model.sensible_utilisation = pyo.Objective(rule=lambda m: (pyo.summation(m.chargeph)+
    pyo.summation(m.dischargeph) + pyo.summation(m.chargeb) + pyo.summation(m.dischargeb) +
    pyo.summation(m.hvdc_pos) + pyo.summation(m.hvdc_neg) + pyo.summation(m.hvac_pos) + pyo.summation(m.hvac_neg)
    ))

model.sensible_utilisation.deactivate()

optimiser = pyo.SolverFactory('gurobi')

start=dt.now()
print("Optimisation starts:", start)
optimiser.solve(model)

model.LCOE.display()
midway=dt.now()
print("Finished sizing. Sizing took:", midway-start)
print("Tuning operations. Starts:", midway)

model.cpv.fix()
model.conshore.fix()
model.coffshore.fix()
model.cphp.fix()
model.cphe.fix()
model.cbp.fix()
model.cbe.fix()
model.chvdc.fix()
model.chvac.fix()
model.cinter.fix()
model.hydro.fix()
model.bio.fix()
model.waste.fix()
model.imports.fix()

model.OBJ.deactivate()
model.sensible_utilisation.activate()
optimiser.solve(model)

end=dt.now()
print("Finished tuning operations. Took:", end-midway)
print("Optimisation took:", end-start)

#%%

S = Solution(model)
solution=S
print('pv (GW):', S.cpv.round(2))
print('onshore wind (GW):', S.conshore.round(2))
print('offshore wind (GW):', S.coffshore.round(2))
print('php (GW):', S.cphp.round(2))
print('phe (GWh):', S.cphe.round(2))
print('bp (GW):', S.cbp.round(2))
print('be (GWh):', S.cbe.round(2))
print('cinter (GW):', S.cinter.round(2))
print('chvdc (GW):', S.chvdc.round(2))
print('chvac (GW):', S.chvac.round(2))


try:
    with open(f'Results/Optimisation_resultx{node}.csv', 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([S.OBJ] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.ctrans))
except FileNotFoundError:
    import os 
    os.mkdir('Results')
    with open(f'Results/Optimisation_resultx{node}.csv', 'w', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([S.OBJ] + list(S.cpv) + list(S.cwind) + list(S.cphp) + list(S.cphe) + list(S.ctrans))


from LinearStatistics import Information
Information(S)


