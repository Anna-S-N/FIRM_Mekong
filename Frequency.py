from scipy.fft import rfft, irfft, rfftfreq
import pandas as pd
import numpy as np
import pylab as plt

def find_maximum_intervals(storage_timeseries_np):
    maximum = max(storage_timeseries_np)
    indices = []
    for i in range(0,len(storage_timeseries_np)):
        if abs(storage_timeseries_np[i]-maximum) < 0.0001:
            indices.append(i)
    return indices

# Specify the LPGM file name
lpgm_file_name = "Results/freq/LPGM_SEAsia_KH_3_180_8_PHS25.csv"

sample_spacing = 1  # [1/Hour]

cutoff_frequencies = {
    "intraday": [0.5, 1/20],
    "overnight": [1/20, 1/28],
    "monthly": [1/28, 1/(24*30)],
    "seasonal": [1/(24*30), 1/(24*30*12)]#,
    #"longterm": [1/(24*30*12), -1/(24*30*12*10)]
}

taxonomy = ["intraday", "overnight", "monthly", "seasonal"]# , "longterm"]

energy_capacity_results = pd.DataFrame()
energy_capacity_results["Demand [MWh/capita/year]"] = [0]  # Placeholder, as demand is not used in this version

# Read the specified LPGM file
lpgm_df = pd.read_csv(lpgm_file_name)

time = np.linspace(0, 8760*10, 8760*10)
storage_timeseries_np = lpgm_df["PHES-Storage (MWh)"].to_numpy()

W = rfftfreq(storage_timeseries_np.size, d=time[1] - time[0])
storage_frequency = rfft(storage_timeseries_np)

category_storage_frequencies_df = pd.DataFrame()
category_storage_timeseries_df = pd.DataFrame()

minimums = {}

for category in taxonomy:
    if category != "test":
        energy_capacity_results[category] = np.zeros(1)

    cut_storage_frequency = storage_frequency.copy()
    cutoff_low = cutoff_frequencies[category][1]
    cutoff_high = cutoff_frequencies[category][0]

    cut_storage_frequency[W > cutoff_high] = 0
    cut_storage_frequency[W <= cutoff_low] = 0

    category_storage_frequencies_df[category] = cut_storage_frequency
    category_storage_timeseries_df[category] = irfft(cut_storage_frequency)

    minimums[category] = min(category_storage_timeseries_df[category])



    # Apportion dc offset across categories based on average value of minimums for a rolling window with period equal to the category
    category_energy_capacities = {}
    for category_i in range(0,len(taxonomy)):
        category = taxonomy[category_i]
        category_energy_capacities[category] = 0
        if category != "test":
            #next_category = taxonomy[category_i + 1]
            window_period = 87600#int(1/(cutoff_frequencies[category][1])) if category != "longterm" else 87600
            i = 0
            rolling_mins = []
            numpy_array_cat = category_storage_timeseries_df[category].to_numpy()
            while i + window_period < len(time):
                #print(category,i,i+window_period)
                rolling_mins.append(min(numpy_array_cat[i:i+window_period]))
                i+=window_period
        
            #average_mins = sum(rolling_mins) / i
            
            rolling_mins_np = np.array(rolling_mins)
            peak_count = 10*365*24
            negative_offset = 0
            #print(rolling_mins_np[rolling_mins_np < -1 * negative_offset])
            #print(max(rolling_mins))
            print(len(rolling_mins), len(rolling_mins_np[rolling_mins_np < -1 * 2]))

            while peak_count >= len(rolling_mins):
                negative_offset+=1
                peak_count = len(rolling_mins_np[rolling_mins_np < -1 * negative_offset])
            
            negative_offset = -1 * (np.average(rolling_mins_np) - np.std(rolling_mins_np))

            print("Neg offset", negative_offset, np.average(rolling_mins_np), 2*np.std(rolling_mins))                
            category_storage_timeseries_df[category] = numpy_array_cat + negative_offset
            #numpy_array_long = category_storage_timeseries_df["longterm"].to_numpy()
            #category_storage_timeseries_df["longterm"] = numpy_array_long - negative_offset

            j=0
            rolling_maxs = []
            numpy_array_cat = category_storage_timeseries_df[category].to_numpy()
            while j + window_period < len(time):
                #print(category,i,i+window_period)
                rolling_maxs.append(max(numpy_array_cat[i:i+window_period]))
                j+=window_period

            rolling_maxs_np = np.array(rolling_maxs)
            peak_count = 10*365*24
            positive_cap = 0
            #print("Rolling maxs", min(rolling_maxs_np))

            """ while peak_count >= len(rolling_maxs_np):
                positive_cap+=1
                peak_count = len(rolling_maxs_np[rolling_maxs_np > positive_cap]) """
            
            positive_cap = np.average(rolling_maxs_np) + np.std(rolling_mins_np)

            #category_energy_capacities[category] = min(positive_cap, max((max(storage_timeseries_np) - sum([category_energy_capacities[x] for x in taxonomy[0:category_i]])), 0))
            category_energy_capacities[category] = positive_cap

    total_preprocess_capacity = sum([category_energy_capacities[x] for x in taxonomy])
    total_original_capacity = max(storage_timeseries_np)
    for category in taxonomy:
        category_energy_capacities[category] = (category_energy_capacities[category] / total_preprocess_capacity) * total_original_capacity
    #category_energy_capacities["longterm"] = max(storage_timeseries_np) - sum([category_energy_capacities[x] for x in taxonomy if x != "longterm"])
    print(category_energy_capacities["monthly"])
    #assert category_energy_capacities["longterm"] >= 0
            

    # Find energy capacity of each category that requires fewest changes to the time series'
    """ energy_capacity_total = max(storage_timeseries_np)
    maximum_bool_np = (storage_timeseries_np == energy_capacity_total)
    exceeded_count_df = pd.DataFrame()
    maximum_capacity_df = pd.DataFrame()

    for category in taxonomy:
        category_np = category_storage_timeseries_df[category].to_numpy()   
        category_maximums_np = category_np[maximum_bool_np]
        exceeded_count_list = []
        maximum_capacity_df[category] = category_maximums_np

        for maximum in category_maximums_np:
            exceeded_count_value = len(category_np[category_np > maximum])
            exceeded_count_list.append(exceeded_count_value)
        
        exceeded_count_df[category] = exceeded_count_list
    
    exceeded_count_df["Total"] = exceeded_count_df[list(exceeded_count_df.columns)].sum(axis=1)
    fewest_changes = min(exceeded_count_df["Total"])
    energy_capacity_interval = exceeded_count_df.index[exceeded_count_df["Total"]==fewest_changes].to_list()[0]
    category_energy_capacities = maximum_capacity_df.iloc[energy_capacity_interval]
    """
    # Adjust the category time series to prevent them exceeding energy capacity
    test=1
    for i in range(0,len(time)):
        carryover = 0
        if i == test:
            print(category_storage_timeseries_df.iloc[i])
        adjusted_value = max(min(category_energy_capacities["intraday"], category_storage_timeseries_df.at[i,"intraday"]),0)
        carryover = category_storage_timeseries_df.at[i,"intraday"] - adjusted_value
        category_storage_timeseries_df.at[i,"intraday"] = adjusted_value

        category_storage_timeseries_df.at[i,"overnight"] += carryover
        adjusted_value = max(min(category_energy_capacities["overnight"], category_storage_timeseries_df.at[i,"overnight"]),0)
        carryover = category_storage_timeseries_df.at[i,"overnight"] - adjusted_value
        category_storage_timeseries_df.at[i,"overnight"] = adjusted_value

        category_storage_timeseries_df.at[i,"monthly"] += carryover
        adjusted_value = max(min(category_energy_capacities["monthly"], category_storage_timeseries_df.at[i,"monthly"]),0)
        carryover = category_storage_timeseries_df.at[i,"monthly"] - adjusted_value
        category_storage_timeseries_df.at[i,"monthly"] = adjusted_value

        category_storage_timeseries_df.at[i,"seasonal"] += carryover
        adjusted_value = max(min(category_energy_capacities["seasonal"], category_storage_timeseries_df.at[i,"seasonal"]),0)
        carryover = category_storage_timeseries_df.at[i,"seasonal"] - adjusted_value
        category_storage_timeseries_df.at[i,"seasonal"] = adjusted_value

        #category_storage_timeseries_df.at[i,"longterm"] += carryover
        #adjusted_value = max(min(category_energy_capacities["longterm"], category_storage_timeseries_df.at[i,"longterm"]),0)
        #carryover = category_storage_timeseries_df.at[i,"longterm"] - adjusted_value
        #category_storage_timeseries_df.at[i,"longterm"] = adjusted_value

        category_storage_timeseries_df.at[i,"seasonal"] += carryover
        adjusted_value = max(min(category_energy_capacities["seasonal"], category_storage_timeseries_df.at[i,"seasonal"]),0)
        carryover = category_storage_timeseries_df.at[i,"seasonal"] - adjusted_value
        category_storage_timeseries_df.at[i,"seasonal"] = adjusted_value

        category_storage_timeseries_df.at[i,"monthly"] += carryover
        adjusted_value = max(min(category_energy_capacities["monthly"], category_storage_timeseries_df.at[i,"monthly"]),0)
        carryover = category_storage_timeseries_df.at[i,"monthly"] - adjusted_value
        category_storage_timeseries_df.at[i,"monthly"] = adjusted_value

        category_storage_timeseries_df.at[i,"overnight"] += carryover
        adjusted_value = max(min(category_energy_capacities["overnight"], category_storage_timeseries_df.at[i,"overnight"]),0)
        carryover = category_storage_timeseries_df.at[i,"overnight"] - adjusted_value
        category_storage_timeseries_df.at[i,"overnight"] = adjusted_value

        category_storage_timeseries_df.at[i,"intraday"] += carryover
        adjusted_value = max(min(category_energy_capacities["intraday"], category_storage_timeseries_df.at[i,"intraday"]),0)
        carryover = category_storage_timeseries_df.at[i,"intraday"] - adjusted_value
        category_storage_timeseries_df.at[i,"intraday"] = adjusted_value

    category_storage_frequencies_df["Original"] = storage_frequency
    category_storage_timeseries_df["Original"] = storage_timeseries_np
    category_storage_frequencies_df["Frequencies"] = W

    category_storage_frequencies_df.to_csv(f"Results/frequencies_{demand}.csv")
    category_storage_timeseries_df.to_csv(f"Results/energy_profile_{demand}.csv")

    for category in taxonomy:
        energy_capacity_results.at[demand-5,category] = max(np.array(category_storage_timeseries_df[category]))/1000 # GWh

energy_capacity_results.to_csv("Results/energy_capacity_results.csv")


    
