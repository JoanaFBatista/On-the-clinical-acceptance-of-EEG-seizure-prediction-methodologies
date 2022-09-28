# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:15:30 2022

@author: User
"""

from plot_explainability import plot_FP, plot_shap_values
from import_data import import_data
from aux_fun import splitting, remove_SPH
import numpy as np
import pandas
import os

patients_list =  ['402']
approach = 'A'
for patient in patients_list:
         
    # --- Train options ---
    n_seizures_train = 3
    SPH = 10 #10min
        
    print(f'\n--------- Patient:{patient} ---------')
    
    # --- Import data ---
    features, datetimes, features_channels, feature_names, channel_names, seizure_info = import_data(patient)
        
    # --- Splitting data (train|test) ---
    data,datetimes,seizure_info=splitting(features, datetimes, seizure_info, n_seizures_train)
    
    # --- Onset time seizures ---
    onset_time_train=[seizure_info['train'][i][0] for i in range(0,len(seizure_info['train']))]
    onset_time_test=[seizure_info['test'][i][0] for i in range(0,len(seizure_info['test']))]
    onset_time={}
    onset_time['train'] = np.array([pandas.to_datetime(time, unit='s') for time in onset_time_train])
    onset_time['test'] = np.array([pandas.to_datetime(time, unit='s') for time in onset_time_test])
    

    # --- Remove SPH ---
    data_list_test = [remove_SPH(data['test'][seizure], datetimes['test'][seizure], SPH, seizure_info['test'][seizure]) for seizure in range(0,len(data['test']))]
    times_list_test = [datetimes['test'][seizure][:data_list_test[seizure].shape[0]] for seizure in range(0,len(data['test']))]
    
    # --- FP Plot/SHAP values ---
    plot_FP(patient,times_list_test, onset_time, approach)
    plot_shap_values(data_list_test, features_channels, patient, len(onset_time_train), approach)
    