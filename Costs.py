import numpy as np
from numba import njit

pv_capex = 1071 # AUD/kW, GenCost 2030
pv_fom = 17 # AUD/kW p.a.
pv_vom = 0 # AUD/MWh p.a.
pv_lifetime = 30

wind_capex = 1913 # AUD/kW, GenCost 2030
wind_fom = 25 # AUD/kW p.a.
wind_vom = 0 # AUD/MWh p.a.
wind_lifetime = 25

# Short distance HVAC costs:
# transmission_capex = 4879 # AUD/MW-km 
# transmission_fom = 48.79 # AUD/MW-km p.a.
# transmission_vom = 0 # AUD/MWh p.a.
# transmission_lifetime = 25

transmission_capex = 320 # AUD/MW-km
transmission_fom = 3.2 # AUD/MW-km p.a.
transmission_vom = 0 # AUD/MWh p.a.
transmission_lifetime = 50

submarine_capex = 4000 # AUD/MW-km
submarine_fom = 40 # AUD/MW-km p.a.
submarine_vom = 0 # AUD/MWh p.a.
submarine_lifetime = 30

converter_capex = 160 # AUD/kW each
converter_fom = 1.6 # AUD/kW each p.a.
converter_vom = 0 # AUD/MWh p.a.
converter_lifetime = 30

storage_capexP = 1136 # AUD/kW # 
stoarge_capexE = 34 # AUD/kWh
storage_fom = 8.21 / 0.75 # AUD/kW p.a.
storage_vom = 0.3 / 0.75 # AUD/MWh p.a.
storage_replace = 112000 / 0.75 # AUD per replace
replace = 20 # every 20 years
storage_lifetime = 60

battery_capexP = 560 # AUD/kW
battery_capexE = 450 # AUD/kWh
battery_fom = 3500 # AUD/MW
battery_vom = 0 # AUD/MWh p.a.
battery_lifetime = 20

hydro_purchase = 50 # AUD/MWh

DR = 0.05 # real discount rate

# a list to store all cost assumptions
UnitCosts = np.array([pv_capex,pv_fom,pv_vom,pv_lifetime,wind_capex,wind_fom,wind_vom,wind_lifetime,transmission_capex,transmission_fom,transmission_vom,transmission_lifetime,
                       storage_capexP,stoarge_capexE,storage_fom,storage_vom,storage_replace,replace,storage_lifetime,battery_capexP,battery_capexE,battery_fom,battery_lifetime,
                       hydro_purchase,submarine_capex,submarine_fom,submarine_vom,submarine_lifetime,converter_capex,converter_fom,converter_vom,converter_lifetime,DR])

@njit()
def annulization(capex, fom, vom, life, dr, p, e):

    """ Calculate annulized costs for capacity p and annual generation e.

        capex: $/kW
        fom: $/kW p.a.
        vom: $/MWh p.a.
        p: GW
        e: MWh """
        

    pv = (1-(1+dr)**(-1*life))/dr

    return p * pow(10,6) * capex / pv + p * pow(10,6) * fom + e * vom

@njit()
def annulization_transmission(capex, fom, vom, life, dr, p, e, d):

    """ Calculate annulized costs for capacity p and annual generation e, for transmission lines only.

        capex: $/MW-km
        fom: $/MW-km p.a.
        vom: $/MWh p.a.
        p: GW
        e: MWh
        d: km """
        
    pv = (1-(1+dr)**(-1*life))/dr

    return p * pow(10,3) * d * capex / pv + p * pow(10,3) * d * fom + e * vom

@njit()
def calculate_costs(S, Discharge, Hydro):
    '''FIX TRANSMISSION COSTS'''
    PV_costs = annulization(S.UnitCosts[0],S.UnitCosts[1],S.UnitCosts[2],S.UnitCosts[3],S.UnitCosts[-1],sum(S.CPV),S.GPV.sum()/S.years)
    wind_costs = annulization(S.UnitCosts[4],S.UnitCosts[5],S.UnitCosts[6],S.UnitCosts[7],S.UnitCosts[-1],sum(S.CWind),S.GWind.sum()/S.years)
    
    transmission_costs = 0
    for i in range(len(S.CDC)-1):
        transmission_costs += annulization_transmission(S.UnitCosts[8],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],S.CDC[i],S.TDCabs.sum(axis=0)[i]/S.years,S.DCdistance[i])
    # VIC-TAS needs to be calculated seperately
    transmission_costs += annulization_transmission(S.UnitCosts[24],S.UnitCosts[25],S.UnitCosts[26],S.UnitCosts[27],S.UnitCosts[-1],S.CDC[-1],S.TDCabs.sum(axis=0)[-1]/S.years,S.DCdistance[-1])
    # Converter station cost, a pair of stations per HVDC line. Excluding VIC-TAS
    converter_costs = 2 * annulization(S.UnitCosts[28],S.UnitCosts[29],S.UnitCosts[30],S.UnitCosts[31],S.UnitCosts[-1],sum(S.CDC[:-1]),0)
    transmission_costs += converter_costs

    pv_phes = (1-(1+S.UnitCosts[-1])**(-1*S.UnitCosts[18]))/S.UnitCosts[-1]
    phes_costs = (S.UnitCosts[12] * S.CPHP.sum() * pow(10,6) + S.UnitCosts[13] * S.CPHS.sum() * pow(10,6)) / pv_phes \
                    + S.UnitCosts[14] * S.CPHP.sum() * pow(10,6) + S.UnitCosts[15] * Discharge.sum() / S.years \
                    + S.UnitCosts[16] * ((1+S.UnitCosts[-1])**(-1*S.UnitCosts[17]) + (1+S.UnitCosts[-1])**(-1*S.UnitCosts[17]*2)) / pv_phes
                        
    pv_battery = (1-(1+S.UnitCosts[-1])**(-1*S.UnitCosts[22]))/S.UnitCosts[-1] # 19, 20, 21, 22
    battery_costs = (S.UnitCosts[19] * S.CBP.sum() * pow(10,6) + S.UnitCosts[20] * S.CBS.sum() * pow(10,6)) / pv_battery \
                    + S.UnitCosts[21] * S.CBP.sum() * pow(10,3)
                                
    hydro_costs = S.UnitCosts[23] * Hydro

    costs = PV_costs + wind_costs + transmission_costs + phes_costs + battery_costs + hydro_costs

    return costs
