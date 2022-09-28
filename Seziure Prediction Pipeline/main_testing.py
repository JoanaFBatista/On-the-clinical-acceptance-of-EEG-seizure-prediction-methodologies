# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:02:04 2022

@author: User
"""
from import_data import import_data
from aux_fun import splitting, select_final_result
from testing  import main_test
from plot_results import fig_performance, fig_performance_per_patient, fig_final_performance, fig_feature_selection
from save_results import save_results, save_final_results, save_test_results
import pickle


# %% --- Configurations ---
patients_list=['402']
approach = 'A'
 
# --- Train options ---
n_seizures_train = 3
SPH = 10 #10min
SOP_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
window_size=5 #5s
 
# %% Import Data | Siplitting Data | Train | Test

information_general = {}
information_train = {}
information_test = {}
final_information = []

for patient in patients_list:
    
    print(f'\n--------- Patient:{patient} ---------')
    
    # --- Import data ---
    features, datetimes, features_channels, feature_names, channel_names, seizure_info = import_data(patient)
    
    # --- Splitting data (train|test) ---
    data,datetimes,seizure_info=splitting(features, datetimes, seizure_info, n_seizures_train)
    
    # --- Load Models ---
    with open(f'../Models/Approach {approach}/model_patient{patient}', 'rb') as models_file:
        models = pickle.load(models_file)
    
    # --- Load Training Information ---
    with open(f'../Results/Approach {approach}/Patient {patient}/information_train_patient_{patient}', 'rb') as info:
        info_train = pickle.load(info)
        
    # --- Select the best SOP ---
    idx_final = select_final_result(info_train)
    SOP_final=info_train[idx_final][0]
    
    # --- Test ---
    info_te={}
    info_te['seizure']=seizure_info['test']
    info_te['features']=features_channels
    info_test, surr_ss_list = main_test(patient, approach, data['test'], datetimes['test'], info_te, SPH, SOP_final, window_size, models)
     
      
    # --- Select & save final train result ---    
    info_general = [patient, n_seizures_train, len(data['test']), SPH]
    final_information.append(info_general + info_train[idx_final] + info_test[idx_final])
    info_train[idx_final][0] = f'{info_train[idx_final][0]}*' # mark final result
    
    # --- Save results (patient) --- 
    information_general[patient] = info_general
    information_train[patient] = info_train
    information_test[patient] = info_test
    
    # --- Figure: performance (patient) ---
    fig_performance(patient, SOP_list, idx_final, info_test, info_train, approach)

# %% SAVE FINAL RESULTS
 
# --- Save results (excel) ---
save_results(information_general, information_train, information_test, approach)

# -- Save test results (excel) ---
save_test_results(final_information, approach)

# --- Figure: performance per patient (all SOPs) ---
fig_performance_per_patient(information_test, approach)

# --- Save final results (excel) ---
save_final_results(final_information, approach)

# --- Figure: final performance per patient (selected SOPs) ---
fig_final_performance(final_information,approach)

#--- Figure: relative frequency of the selected features ---
fig_feature_selection(final_information, approach, feature_names, channel_names, features_channels)

