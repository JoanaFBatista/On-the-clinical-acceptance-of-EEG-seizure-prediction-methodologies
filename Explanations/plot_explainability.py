# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:42:38 2022

@author: User
"""
import shap
import matplotlib.pyplot as plt
from aux_fun import get_selected_features, select_final_result 
import numpy as np
import pickle
import pandas
import datetime
import matplotlib.dates as md
import os

#%% --- Plotting Firing Power output of all classifiers throughout time ---
def plot_FP(patient, datetimes_test, onset_times, approach):
    
    #verifies if the given  directory path exists, if not creates one
    path_patient=f'../Results/Approach {approach}/Patient {patient}'
    isExist = os.path.exists(path_patient)
    if not isExist:
      os.makedirs(path_patient)
    
    # --- Load test predictions ---
    with open(f'../Results/Approach {approach}/Patient {patient}/test_final_prediction_patient{patient}', 'rb') as test_prediction_patient:
        test_prediction = pickle.load(test_prediction_patient)
         
    target=test_prediction['target']
    firing_power=test_prediction['firing_power']
    firing_power_all_classifiers=test_prediction['firing_power_all_classifiers']
    alarms=test_prediction['alarm']

    onset_times_train=onset_times['train']
    onset_times_test=onset_times['test']
    num_seizures_train=len(onset_times_train)
    
    # --- Load vigilance vectors ---
    directory='../Data/Vigilance_Vectors'
    vigilance=np.load(directory+"/pat_"+str(patient)+"_vigilance",allow_pickle=True)
    vigilance_datetimes=np.load(directory+"/pat_"+str(patient)+"_datetimes",allow_pickle=True)
    
    # --- Correct vigilance datetimes ---
    vigilance_timestamp=[]
    vigilance_time=[]
    for i in range(0,len(vigilance)):
        vigilance[i]=np.abs(vigilance[i]-1)
        vigilance[i]=np.clip(vigilance[i],0.05,0.95)
        
        times_stamp = np.array([time.timestamp() for time in vigilance_datetimes[i]])
        vigilance_timestamp.append(times_stamp)
        
        times_corr = np.array([pandas.to_datetime(time, unit='s') for time in vigilance_timestamp[i]])
        vigilance_time.append(times_corr)
    
    # --- Filling the missing data ---
    datetimes_new=[]
    firing_power_output_new=[]
    target_new=[]
    alarms_new=[]
    firing_power_all_new=[]
    
    for i in range(0,len(datetimes_test)):
               
        datetimes_new_i=[]
        firing_power_output_new_i=[]
        target_new_i=[]
        alarms_new_i=[]
        
        firing_power_all_new_i=[[] for i in range(0,len(firing_power_all_classifiers[i]))]    
        for j in range(0,len(datetimes_test[i])-1):
            time_difference=datetimes_test[i][j+1]-datetimes_test[i][j]
            time_difference=time_difference.seconds
        
            datetimes_new_i.append(datetimes_test[i][j])
            firing_power_output_new_i.append(firing_power[i][j])
            target_new_i.append(target[i][j])
            alarms_new_i.append(alarms[i][j])
            
            # iterate the 31 classifiers
            for k in range(0,len(firing_power_all_classifiers[i])):
                firing_power_all_new_i[k].append(firing_power_all_classifiers[i][k][j])    
                
            if time_difference<=5:
                pass
            else:
                new_datetime=datetimes_test[i][j]+datetime.timedelta(0,5)
                while(time_difference>5):
                    datetimes_new_i.append(new_datetime)
                    target_new_i.append(np.NaN)
                    alarms_new_i.append(np.NaN)
                    firing_power_output_new_i.append(np.NaN)
                    
                    time_difference=datetimes_test[i][j+1]-new_datetime
                    time_difference=time_difference.seconds
                    new_datetime=new_datetime+datetime.timedelta(0,5)
                    
                    # iterate the 31 classifiers
                    for k in range(0,len(firing_power_all_classifiers[i])):
                        firing_power_all_new_i[k].append(np.NaN)
        
        datetimes_new_i.append(datetimes_test[i][-1])
        firing_power_output_new_i.append(firing_power[i][-1])
        target_new_i.append(target[i][-1])
        alarms_new_i.append(alarms[i][-1])
        
        # iterate the 31 classifiers
        for k in range(0,len(firing_power_all_classifiers[i])):
            firing_power_all_new_i[k].append(firing_power_all_classifiers[i][k][-1])    
                        
        
        datetimes_new.append(datetimes_new_i)
        firing_power_output_new.append(firing_power_output_new_i)
        target_new.append(target_new_i)
        alarms_new.append(alarms_new_i)
        firing_power_all_new.append(firing_power_all_new_i)
        
        
    # --- Plotting Firing Power output throughout time ---   
    for i in range (0,len(datetimes_test)):

        plt.figure(figsize=(20, 15)) 
        
        # --- Plotting final FP and FP threshold --- 
        plt.plot(datetimes_new[i],firing_power_output_new[i],'k',alpha=0.7, label='Firing Power')
        plt.plot(datetimes_new[i],np.linspace(0.5, 0.5, len(datetimes_new[i])),linestyle='--',
                  color='black',alpha=0.7, label='Alarm threshold')
       
        plt.grid()
        plt.ylim(0,1)
        plt.xlim(datetimes_new[i][0],datetimes_new[i][len(datetimes_new[i])-1])
        
        # --- Plotting FP of all classifiers --- 
        for k in range(0,len(firing_power_all_classifiers[i])):
            plt.plot(datetimes_new[i],firing_power_all_new[i][k], color='black',alpha=0.15)
        
        # --- Mark the predicted alarms ---
        lb='Alarm'
        for alarm_index in np.where(np.array(alarms_new[i])==1)[0]:
            plt.plot(datetimes_new[i][alarm_index], firing_power_output_new[i][alarm_index],
                      marker='^', color='maroon',markersize=30, label=lb)
            lb=''

        plt.fill_between(datetimes_new[i], 0.5, np.array(firing_power_output_new[i]), where=np.array(firing_power_output_new[i])>0.5,
                          facecolor='brown', alpha=0.5, label='FP above alarm threshold')
        
        # --- Colour preictal period ---               
        a=np.where(np.diff(target_new[i])==1)
        
        if len(a[0])==0:
            idx_1=np.where(np.array(target_new[i])==1)[0]
            idx_alarm=[idx for idx in idx_1 if target_new[i][idx-1]!=1]
            
            plt.fill_between(datetimes_new[i], 0, 1, where=np.array(datetimes_new[i])>=
                              np.array(datetimes_new[i][idx_alarm[0]]),
                              facecolor='moccasin', alpha=0.5, label='Preictal period')

            plt.axvline(x = datetimes_new[i][idx_alarm[0]], color = 'k',
                        alpha = 0.7, linestyle='--',linewidth=0.8)   
        else:
            plt.fill_between(datetimes_new[i], 0, 1, where=np.array(datetimes_new[i])>
                              np.array(datetimes_new[i][np.where(np.diff(target_new[i])==1)[0][0]]),
                              facecolor='moccasin', alpha=0.5, label='Preictal period')
            
            plt.axvline(x = datetimes_new[i][np.where(np.diff(target_new[i])==1)[0][0]], color = 'k',
                        alpha = 0.7, linestyle='--',linewidth=0.8)
            
        # --- Plot vigilance state ---
        plt.plot(vigilance_time[i], vigilance[i], color = 'darkred',alpha=0.4, label='Vigilance state')
        
    
        plt.gcf().autofmt_xdate()
        xfmt = md.DateFormatter('%H:%M:%S')
        ax=plt.gca()
        ax.xaxis.set_major_formatter(xfmt)
        ax.yaxis.set_ticks([0,0.05,0.2,0.4,0.6,0.8,0.95,1.0])
        ax.yaxis.set_ticklabels(["0","sleep","0.2","0.4","0.6","0.8","awake","1.0"])
        str_date_time_train =[onset_times_train_i.strftime("%d-%m-%Y, %H:%M:%S") for onset_times_train_i in onset_times_train]
        str_date_time_test=onset_times_test[i].strftime("%d-%m-%Y, %H:%M:%S")
        
        plt.title("Patient "+str(patient)+", Seizure "+str(i+num_seizures_train+1), fontsize=20)
        plt.legend(bbox_to_anchor =(0.5,-0.17), loc='lower center', ncol=7 ,fontsize=15)
        
        
        plt.text(0.1,0.03, "TRAIN - Seizure 1: "+str(str_date_time_train[0])+", Seizure 2: "+str(str_date_time_train[1])+", Seizure 3: "+str(str_date_time_train[2])+"\nTEST - Seizure "+str(i+num_seizures_train+1)+": "+str(str_date_time_test), fontsize=15, transform=plt.gcf().transFigure)
        
        plt.savefig(f'../Results/Approach {approach}/Patient {patient}/Firing Power output throughout time (patient {patient}, Seizure {i+num_seizures_train+1})', bbox_inches='tight')
        plt.close()    
        

#%% --- Beeswarm of average Shap values ---
def plot_shap_values(data_test, features_channels, patient, num_seizures_train, approach):
    
    # --- Load Models ---
    with open(f'../Models/Approach {approach}/model_patient{patient}', 'rb') as models_file:
        models = pickle.load(models_file)
    # --- Load Training Information ---
    with open(f'../Results/Approach {approach}/Patient {patient}/information_train_patient_{patient}', 'rb') as info:
        info_train = pickle.load(info)
        
    # --- Select & save final train result ---
    idx_final = select_final_result(info_train)
    SOP_final=info_train[idx_final][0]
    final_model=models[SOP_final]
    
    # --- Concatenate seizures ---
    data_test=np.concatenate(data_test)
    
    # --- Shap values of all classifiers ---
    n_classifiers=len(final_model['svm'])
    
    shap_data={}
    for i in range(0,len(features_channels)):
        shap_data[i]=[]
    
    idx=[]

        
    for classifier in range(0, n_classifiers):
        scaler=final_model['scaler'][classifier]
        selector=final_model['selector'][classifier]
        idx.append(selector.get_support(indices=True))

        set_test=scaler.transform(data_test)
        set_test=selector.transform(set_test)
        idx_features=get_selected_features(len(features_channels), selector)
        features_channels_new=[features_channels[idx] for idx in idx_features ]
           
        explainer=shap.Explainer(final_model['svm'][classifier], set_test, feature_names=features_channels_new)
        shap_values=explainer(set_test)
        
        for j in range(0, len(idx_features)):
            shap_data[idx_features[j]].append(shap_values.values[:,j])
    
    idx=np.array(idx)
    count_i={}
    for i in range(0,len(features_channels)):
        count = np.count_nonzero(idx == i)
        count_i[i]=count
    idx_features_sorted=sorted(count_i.items(), key=lambda count_i: count_i[1], reverse=True)
    
    idx_most_k_selected_features=idx_features_sorted[:len(idx_features)]
    
    sh={}
    for j in idx_most_k_selected_features:
        sh[j[0]]=shap_data[j[0]]
    
    shap_avg_data=np.zeros([len(data_test),len(idx_features)])
    for i in range(0,len(sh)):
        k=list(sh.keys())[i]
        shap_avg_data[:,i]=np.average(sh[k], axis=0)
            
    # --- Average Shap values ---
    shap_values.values=shap_avg_data
    shap_values.feature_names=[features_channels[u] for u in list(sh.keys())]
    
    # --- Plotting average Shap values ---
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=11, color=plt.get_cmap("Reds"), show=False)
    plt.title("Beeswarm summary of Shap Values: Patient "+str(patient))
    plt.savefig(f'../Results/Approach {approach}/Patient {patient}/Beeswarm Shap values.png', bbox_inches='tight')
    plt.close()
       