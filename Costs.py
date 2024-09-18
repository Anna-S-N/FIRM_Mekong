import numpy as np
from numba import njit

pv_capex = 876 # USD/kW, Mean global cost, IRENA Renewable Power Generation Costs in 2022: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
pv_fom = 3.6 # USD/kW p.a. Median Asia cost, IRENA Renewable Power Generation Costs in 2022: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
pv_vom = 0 # USD/MWh p.a.
pv_lifetime = 30

wind_capex = 1274 # USD/kW, Mean global onshore wind cost, IRENA Renewable Power Generation Costs in 2022: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
wind_fom = 29.5 # USD/kW p.a., No great data recent for ASEAN, so just used same assumption as 7th ASEAN Energy Outlook:https://asean.org/wp-content/uploads/2023/04/The-7th-ASEAN-Energy-Outlook-2022.pdf
wind_vom = 0 # USD/MWh p.a.
wind_lifetime = 25

# Short distance HVAC costs:
# transmission_capex = 4879 # AUD/MW-km 
# transmission_fom = 48.79 # AUD/MW-km p.a.
# transmission_vom = 0 # AUD/MWh p.a.
# transmission_lifetime = 25

# AUD to USD conversion 1 : 0.7
# Transmission costs frm AEMO Transmission Cost Database: https://aemo.com.au/en/energy-systems/major-publications/integrated-system-plan-isp/2024-integrated-system-plan-isp/current-inputs-assumptions-and-scenarios/transmission-cost-database
transmission_capex = 320 * 0.7 # USD/MW-km
transmission_fom = 3.2 * 0.7 # USD/MW-km p.a.
transmission_vom = 0 # USD/MWh p.a.
transmission_lifetime = 50

submarine_capex = 4000 * 0.7 # USD/MW-km
submarine_fom = 40 * 0.7 # USD/MW-km p.a.
submarine_vom = 0 # USD/MWh p.a.
submarine_lifetime = 30

converter_capex = 160 * 0.7 # USD/kW each
converter_fom = 1.6 * 0.7 # USD/kW each p.a.
converter_vom = 0 # USD/MWh p.a.
converter_lifetime = 30

storage_capexP = 1136 * 0.7 # USD/kW # 
stoarge_capexE = 34 * 0.7 # USD/kWh
storage_fom = 8.21 # USD/kW p.a.
storage_vom = 0.3 # USD/MWh p.a.
storage_replace = 112000 # USD per replace
replace = 50 # every 50 years
storage_lifetime = 100

battery_capexP = 45 # USD/kW, median Initial Capital Cost DC + EPC costs for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
battery_capexE = 221 + 70 # USD/kWh, median Initial Capital Cost AC for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
battery_fom = 5.25 # USD/kWh, median O&M for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
battery_vom = 0 # AUD/MWh p.a.
battery_lifetime = 20

nuclear_purchase = 190 # USD/MWh, based on illustrative midpoint from Lazard LCOE+ for US Nuclear: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
hydro_purchase = 50 # USD/MWh
import_purchase = 50 # USD/MWh, assume imports are hydro

DR = 0.05 # real discount rate: WACC between 5-6% in ASEAN countries currently https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2022/Sep/IRENA_Renewable_energy_outlook_ASEAN_2022.pdf

# a list to store all cost assumptions
UnitCosts = np.array([pv_capex,pv_fom,pv_vom,pv_lifetime,wind_capex,wind_fom,wind_vom,wind_lifetime,transmission_capex,transmission_fom,transmission_vom,transmission_lifetime,
                       storage_capexP,stoarge_capexE,storage_fom,storage_vom,storage_replace,replace,storage_lifetime,battery_capexP,battery_capexE,battery_fom,battery_lifetime,
                       hydro_purchase,submarine_capex,submarine_fom,submarine_vom,submarine_lifetime,converter_capex,converter_fom,converter_vom,converter_lifetime,
                       import_purchase,nuclear_purchase,DR])

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
def calculate_costs(S, GDischarge, GHydro, GImports, GBaseload):
    '''FIX TRANSMISSION COSTS - SOME NEED CONVERTER STATIONS LIKE MY_I'''
    PV_costs = annulization(S.UnitCosts[0],S.UnitCosts[1],S.UnitCosts[2],S.UnitCosts[3],S.UnitCosts[-1],sum(S.CPV),S.GPV.sum()/S.years)
    wind_costs = annulization(S.UnitCosts[4],S.UnitCosts[5],S.UnitCosts[6],S.UnitCosts[7],S.UnitCosts[-1],sum(S.CWind),S.GWind.sum()/S.years)
    
    transmission_costs = 0
    for i in range(len(S.CHVDC)):
        transmission_costs += annulization_transmission(S.UnitCosts[8],S.UnitCosts[9],S.UnitCosts[10],S.UnitCosts[11],S.UnitCosts[-1],S.CHVDC[i],S.TDCabs.sum(axis=0)[i]/S.years,S.DCdistance[i])
    
    # Converter station cost, a pair of stations per HVDC line. Excluding VIC-TAS
    converter_costs = 2 * annulization(S.UnitCosts[28],S.UnitCosts[29],S.UnitCosts[30],S.UnitCosts[31],S.UnitCosts[-1],sum(S.CHVDC),0)
    transmission_costs += converter_costs

    pv_phes = (1-(1+S.UnitCosts[-1])**(-1*S.UnitCosts[18]))/S.UnitCosts[-1]
    phes_costs = (S.UnitCosts[12] * S.CPHP.sum() * pow(10,6) + S.UnitCosts[13] * S.CPHS.sum() * pow(10,6)) / pv_phes \
                    + S.UnitCosts[14] * S.CPHP.sum() * pow(10,6) + S.UnitCosts[15] * GDischarge.sum() / S.years \
                    + S.UnitCosts[16] * ((1+S.UnitCosts[-1])**(-1*S.UnitCosts[17]) + (1+S.UnitCosts[-1])**(-1*S.UnitCosts[17]*2)) / pv_phes
                        
    pv_battery = (1-(1+S.UnitCosts[-1])**(-1*S.UnitCosts[22]))/S.UnitCosts[-1] # 19, 20, 21, 22
    battery_costs = (S.UnitCosts[19] * S.CBP.sum() * pow(10,6) + S.UnitCosts[20] * S.CBS.sum() * pow(10,6)) / pv_battery \
                    + S.UnitCosts[21] * S.CBS.sum() * pow(10,6)
                                
    hydro_costs = S.UnitCosts[23] * GHydro
    import_costs = S.UnitCosts[32] * GImports
    baseload_costs = S.UnitCosts[33] * GBaseload

    costs = PV_costs + wind_costs + transmission_costs + phes_costs + battery_costs + hydro_costs + import_costs + baseload_costs
    tech_costs = np.array([PV_costs, wind_costs, transmission_costs, phes_costs, battery_costs, hydro_costs, import_costs, baseload_costs], dtype=np.float64)

    return costs, tech_costs
