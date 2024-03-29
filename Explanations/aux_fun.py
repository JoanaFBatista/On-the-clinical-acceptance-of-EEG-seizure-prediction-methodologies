# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:00:36 2021

@author: User
"""

from datetime import datetime as dt
from datetime import timedelta as t
import numpy as np
from sklearn import preprocessing, feature_selection, svm, metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas



def remove_SPH (data, datetimes, SPH, info_seizure):
    onset_time=pandas.to_datetime(int(info_seizure[0]), unit='s')
    #onset_time=dt.fromtimestamp(int(info_seizure[0]))
    dt_SPH = onset_time - t(minutes=SPH)
    idx = np.where(datetimes<=dt_SPH)
    data=data[idx]
        
    return data

def construct_target (data, datetimes, SOP):
    target=[]
    target=np.zeros((data.shape[0],), dtype=int)
    
    datetimes=datetimes[:len(data)]
    last_time=datetimes[-1]
    dt_SOP = last_time - t(minutes=SOP)
    idx_SOP = np.where(datetimes>=dt_SOP)
    target[idx_SOP]=1

    return target


def splitting(data_list, datetimes_list, seizure_info_list, n_seizures_train):
    
    data = {}
    datetimes = {}
    seizure_info = {}
    
    # Train
    data['train'] = data_list[:n_seizures_train]
    datetimes['train'] = datetimes_list[:n_seizures_train]
    seizure_info['train'] = seizure_info_list[:n_seizures_train]
    
    # Test
    data['test'] = data_list[n_seizures_train:]
    datetimes['test'] = datetimes_list[n_seizures_train:]
    seizure_info['test'] = seizure_info_list[n_seizures_train:]
    
    return data, datetimes, seizure_info
    


def class_balancing(target):
    
    # Define majority & minority classes (class with more samples vs. class with less samples)
    idx_class0 = np.where(target==0)[0]
    idx_class1 = np.where(target==1)[0]
    if len(idx_class1)>=len(idx_class0):
        idx_majority_class = idx_class1
        idx_minority_class = idx_class0
    elif len(idx_class1)<len(idx_class0):
        idx_majority_class = idx_class0
        idx_minority_class = idx_class1
    
    # Define number of samples of each group
    n_groups = len(idx_minority_class)
    n_samples = len(idx_majority_class)
    min_samples = n_samples//n_groups
    remaining_samples = n_samples%n_groups
    #distributes the number of min_samples across all groups, and one more across a number of groups equal to the remaining samples
    n_samples_per_group = [min_samples+1]*remaining_samples + [min_samples]*(n_groups-remaining_samples)
    
    # Select one sample from each group of the majority class
    idx_selected = []
    begin_idx = 0
    for i in n_samples_per_group:
        end_idx = begin_idx + i
        
        idx_group = idx_majority_class[begin_idx:end_idx]
        idx = np.random.choice(idx_group)
        idx_selected.append(idx)

        begin_idx = end_idx
        
    # Add samples from the minority class
    [idx_selected.append(idx) for idx in idx_minority_class]

    # Sort selected indexes to keep samples order
    idx_selected = np.sort(idx_selected)
    
    return idx_selected


def standardization(data):
    
    # Define scaler
    scaler = preprocessing.StandardScaler()
    # Apply fit
    scaler.fit(data)
    
    return scaler


def select_features(data, target, n_features):
    
    # Define feature selection
    selector = feature_selection.SelectKBest(score_func = feature_selection.f_classif, k = n_features)  
    # Apply feature selection
    selector.fit(data, target)

    return selector


def get_selected_features(features_len, selector):
        
    #return a mask of the features selected
    mask=selector.get_support(indices=True)
    selected_features_index=np.array(mask)
    
    return selected_features_index


def classifier(data, target, c_value):
    
    # Define svm model
    svm_model = svm.LinearSVC(C = c_value, dual = False)
    # Appy fit
    svm_model.fit(data, target)
    
    return svm_model


def performance(target, prediction):
    
    tn, fp, fn, tp = metrics.confusion_matrix(target, prediction).ravel()
    sensitivity = tp/(tp+fn)  
    specificity = tn/(tn+fp)
    metric = np.sqrt(sensitivity * specificity)
   
    return sensitivity, specificity, metric


def select_final_result(info_train):
    
    # Find highest train metric (the best SOP)
    metrics = [info[5] for info in info_train] 
    idx_best = np.argmax(metrics)
 
    return idx_best