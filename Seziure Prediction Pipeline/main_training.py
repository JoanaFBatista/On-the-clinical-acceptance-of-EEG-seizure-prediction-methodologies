# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:03:56 2021

@author: User
"""

from import_data import import_data
from aux_fun import splitting
from training import main_train
import pickle
import os

# %% --- Configurations ---
patient=input('Enter patient ID:')
approach = 'Approach A'
 
# --- Train options ---
n_seizures_train = 3
SPH = 10 #10min
SOP_list = [10, 20, 35] #[10, 15, 20, 25, 30, 35, 40, 45, 50]
window_size=5 #5s

# %% Import Data | Siplitting Data | Train 

print(f'\n--------- Patient:{patient} ---------')

# --- Import data ---
features, datetimes, features_channels, feature_names, channel_names, seizure_info = import_data(patient)

# --- Splitting data (train|test) ---
data,datetimes,seizure_info=splitting(features, datetimes, seizure_info, n_seizures_train)

# --- Train ---
info_tr={};
info_tr['seizure']=seizure_info['train']
info_tr['features']=features_channels

info_train=main_train(patient, approach, data['train'], datetimes['train'], info_tr, SPH, SOP_list, window_size)

# --- Save train information ---

#verifies if the given  directory path exists, if not creates one
path_results=f'../Results/{approach}/Patient {patient}'
isExist = os.path.exists(path_results)
if not isExist:
  os.makedirs(path_results)
  
fw=open(f'../Results/{approach}/Patient {patient}/information_train_patient_{patient}', 'wb')
pickle.dump(info_train,fw)
fw.close()

