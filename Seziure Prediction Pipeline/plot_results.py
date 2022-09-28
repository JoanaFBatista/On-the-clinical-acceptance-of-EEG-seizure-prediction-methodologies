import matplotlib.pyplot as plt 
import numpy as np
import pickle
from aux_fun import get_selected_features


def fig_performance(patient, SOP_list, idx_final, info_test, info_train, approach):
    
    # Information
    ss = [info[-6] for info in info_test]
    fprh = [info[-5] for info in info_test]
    labels = [info[0] for info in info_train]
    #labels=[str(SOP) for SOP in SOP_list] 
    labels[idx_final]=f'{labels[idx_final]}*'     

    # Figure
    fig = plt.figure(figsize=(20, 15))
    
    x = np.arange(len(ss))
    width = 0.3
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='#E3797D', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.xlabel('SOP')
    plt.ylabel('SS', color='#E3797D', fontweight='bold')
    plt.ylim([0, 1.05])
    
    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='#82A69C', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='#82A69C', fontweight='bold')
    
    plt.xticks(x, labels=labels)
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.annotate('* final result', xy=(1,0), xycoords='axes fraction', xytext=(-60,-50), textcoords='offset points')

    plt.title(f'Performance (patient {patient})', fontsize=20)

    plt.savefig(f'../Results/Approach {approach}/Patient {patient}/Performance (patient {patient})', bbox_inches='tight')
    plt.close()    


def fig_final_performance(final_information, approach):
    
    # Information
    ss = [info[-6] for info in final_information]
    fprh = [info[-5] for info in final_information]
    sop = [info[4] for info in final_information]
    
    labels = [f'patient {info[0]}' for info in final_information]
    p_values = [info[-1] for info in final_information]
    labels = [f'*{labels[info]}'  if p_values[info]<0.05 else f'{labels[info]}' for info in range(len(final_information))] # validated patients

    # Figure
    fig = plt.figure(figsize=(40, 20))
    
    x = np.arange(len(final_information))
    width = 0.3
    spare_width = 0.5
    
    # Sensitivity
    ss_bar = plt.bar(x-width/2, ss, width, color='#E3797D', label = 'SS')
    plt.bar_label(ss_bar, fmt='%.2f', padding=3)
    plt.ylabel('SS', color='#E3797D', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.xticks(x, labels=labels, rotation=90)
    plt.xlim(x[0]-spare_width,x[-1]+spare_width)

    # FPR/h
    plt.twinx() # second axis
    fprh_bar = plt.bar(x+width/2, fprh, width, color='#82A69C', label = 'FPR/h')
    plt.bar_label(fprh_bar, fmt='%.2f', padding=3)
    plt.ylabel('FPR/h', color='#82A69C', fontweight='bold')
    
    plt.table(cellText=[sop], rowLabels=['SOP'], cellLoc='center', bbox=[0, -0.25, 1, 0.05], edges='horizontal') # BBOX: [shift on x-axis, gap between plot & table, width, height]
    plt.subplots_adjust(bottom=0.25)
    
    ss_final = [np.mean(ss), np.std(ss)]
    fprh_final = [np.mean(fprh), np.std(fprh)]    
    print(f'\n\n--- FINAL TEST PERFORMANCE (selected SOPs - mean) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (mean) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 1, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)

    ss_final = [np.median(ss), np.percentile(ss, 75) - np.percentile(ss, 25)]
    fprh_final = [np.median(fprh), np.percentile(fprh, 75) - np.percentile(fprh, 25)]
    print(f'\n\n--- FINAL TEST PERFORMANCE (selected SOPs - median) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (median) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 0.88, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)

    plt.annotate('* statistically validated', xy=(1,0), xycoords='axes fraction', xytext=(-115,-130), textcoords='offset points')

    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.title('Performance per patient (Best SOP)')
    
    plt.savefig(f'../Results/Approach {approach}/Performance')
    plt.close()
        
# performance per patient for all sops     
def fig_performance_per_patient(information_test, approach):
    
    # Information
    patients = information_test.keys()
    ss_mean = []
    ss_std = []
    fprh_mean = []
    fprh_std = []
    for patient in patients:
        ss = [info[5] for info in information_test[patient]]
        fprh = [info[6] for info in information_test[patient]]
        
        ss_mean.append(np.mean(ss)) #mean of all sops
        ss_std.append(np.std(ss))
        fprh_mean.append(np.mean(fprh))
        fprh_std.append(np.std(fprh))
    labels = patients #labels = [f'patient {patient}' for patient in patients]

    # Figure
    fig = plt.figure(figsize=(40, 20))
    
    x = np.arange(len(patients))
    width = 0.3
    
    # Sensitivity
    plt.bar(x-width/2, ss_mean, width, yerr=ss_std, color='#E3797D', label = 'SS', error_kw=dict(elinewidth=0.5, capsize=5))
    plt.ylabel('SS', color='#E3797D', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.xlabel('Patient')
    plt.xticks(x, labels=labels, rotation=90)
    [plt.annotate(str(round(ss_mean[i],2)),(x[i]-width,ss_mean[i]+plt.ylim()[1]/100),fontsize=7.5) for i in range(len(ss_mean))]

    # FPR/h
    plt.twinx() # second axis
    plt.bar(x+width/2, fprh_mean, width, yerr=fprh_std, color='#82A69C', label = 'FPR/h', error_kw=dict(elinewidth= 0.5, capsize=5))
    plt.ylabel('FPR/h', color='#82A69C', fontweight='bold')
    plt.ylim(bottom=0)
    [plt.annotate(str(round(fprh_mean[i],2)),(x[i],fprh_mean[i]+plt.ylim()[1]/100),fontsize=7.5) for i in range(len(fprh_mean))]
 
    fig.legend(loc = 'upper right', bbox_to_anchor=(1,1), bbox_transform=plt.gca().transAxes)
    plt.title('Performance per patient (All SOPs)')    

    ss_final = [np.mean(ss_mean), np.std(ss_mean)]
    fprh_final = [np.mean(fprh_mean), np.std(fprh_mean)]    
    print(f'\n\n--- FINAL TEST PERFORMANCE (all SOPs - mean) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (mean) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 1, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)
    
    ss_final = [np.median(ss_mean), np.percentile(ss_mean, 75) - np.percentile(ss_mean, 25)]
    fprh_final = [np.median(fprh_mean), np.percentile(fprh_mean, 75) - np.percentile(fprh_mean, 25)]
    print(f'\n\n--- FINAL TEST PERFORMANCE (all SOPs - median) --- \nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} | FPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}')
    text = f'-- Final result (median) --\nSS = {ss_final[0]:.3f} ± {ss_final[1]:.3f} \nFPR/h = {fprh_final[0]:.3f} ± {fprh_final[1]:.3f}'
    plt.text(1, 0.9, text, bbox={'facecolor':'grey','alpha':0.2,'pad':8},horizontalalignment='left', verticalalignment='top',fontweight='bold',transform=plt.gca().transAxes)
    
    plt.savefig(f'../Results/Approach {approach}/Performance per patient')
    plt.close()
    


def fig_test_all(patient, preictal, target, prediction, firing_power, threshold, alarm, approach):
   
    # Figure
    fig = plt.figure(figsize=(20, 10))
    x=[0, len(target)]
    plt.plot(target,'o', color='orange', label='Target')
    plt.plot(prediction,'o', color='blue', label='SVM Prediction')
    plt.plot(firing_power, color='red', label='Firing Power')
    plt.plot(np.full(len(firing_power),threshold), color='green', label='Threshold')
    plt.plot(alarm, color='yellow', label='Alarms')
    
    plt.title(f'Test: patient {patient} | preictal={preictal}')
    plt.xlabel('Samples')    
    

    fig.savefig(f'../Results/Approach {approach}/Patient {patient}/test_pat{patient}_preictal{preictal}_all')
    plt.close(fig)

def fig_feature_selection(final_information, approach, features, channels, features_channels):
        
    patients = [info[0] for info in final_information]
    best_sop = [info[4] for info in final_information]
    
    idx_selected_features_list = []
    idx_final_selected_features_list = []
    
    for i in range(len(patients)):
        
        #Load models
        models = pickle.load(open(f'../Models/Approach {approach}/model_patient{patients[i]}','rb'))
        
        #for best SOP
        final_selector = models[best_sop[i]]['selector']
        #final_redundat_features_idx=models[best_sop[i]]['redundant features']
        # concatenate features from 31 models of best SOP
        #idx_final_selected_features = np.concatenate([get_selected_features(len(features_channels),final_redundat_features_idx[classifier],final_selector[classifier]) for classifier in range(len(final_selector))]) # concatenate features from 31 models of best SOP
        idx_final_selected_features = np.concatenate([get_selected_features(len(features_channels),final_selector[classifier]) for classifier in range(len(final_selector))]) # concatenate features from 31 models of best SOP
        idx_final_selected_features_list.append(idx_final_selected_features)
        
        #for all SOPs
        # concatenate features from 31 models of all SOPs
        idx_selected_features = np.concatenate([get_selected_features(len(features_channels),models[SOP]['selector'][classifier]) for SOP in models for classifier in range(len(models[SOP]['selector']))]) 
        idx_selected_features_list.append(idx_selected_features)

        # Figure: selected features (patient)
        fig_fs(features, channels, features_channels, idx_selected_features, f'patient{patients[i]}',patients[i], approach)
        # Figure: selected features (patient, Final SOP)
        fig_fs(features, channels, features_channels, idx_final_selected_features, f'final_patient{patients[i]}',patients[i], approach)
        
    # # Figure: final selected features (all patients)
    # fig_fs(features, channels, features_channels, np.concatenate(idx_final_selected_features_list),'final',None, approach)
    # # Figure: all selected features (all patients & SOPs)
    # fig_fs(features, channels, features_channels, np.concatenate(idx_selected_features_list),'all',None, approach)
    
    
def fig_fs(features, channels, features_channels, idx_selected_features, fig_name,patient, approach):
    if patient is not None:
        path_to_save=f'../Results/Approach {approach}/Patient {patient}'
    else:
        path_to_save=f'../Results/Approach {approach}'
    
    features_freq = {feature:0 for feature in features}
    channels_freq = {channel:0 for channel in channels}
    features_channels_freq = {feature_channel:0 for feature_channel in features_channels}
    for i in idx_selected_features:
        idx_feature = i//len(channels)
        feature_name = features[idx_feature]
        features_freq[feature_name] += 1 
        
        idx_channel = i%len(channels)
        channel_name = channels[idx_channel]
        channels_freq[channel_name] += 1
        
        feature_channel_name=features_channels[i]
        features_channels_freq[feature_channel_name] += 1   
        
    # Translate from number of occurrences to relative frequency
    features_freq = {name: n_occur/len(idx_selected_features) for name,n_occur in features_freq.items()}
    channels_freq = {name: n_occur/len(idx_selected_features) for name,n_occur in channels_freq.items()}
    features_channels_freq = {name: n_occur/len(idx_selected_features) for name,n_occur in features_channels_freq.items()}

    # Figure: selected frequency & channels
    features_channels_freq_to_keep={}
    for key in features_channels_freq.keys():
        if features_channels_freq[key]!=0:
            features_channels_freq_to_keep[key]=features_channels_freq[key]
    
    plt.figure(figsize=(50, 10))
    plt.bar(features_channels_freq_to_keep.keys(), features_channels_freq_to_keep.values(), color='#E3797D')
    plt.title('Relative frequency of selected features and channels')
    plt.tick_params(axis='x') # set the x-axis label size
    plt.xticks(rotation=90)
    plt.savefig(path_to_save+f'/feature_selection_detailed_{fig_name}', bbox_inches='tight')
    plt.close()

    # Figure: selected frequency + channels
    x=np.arange(len(features_freq.keys()))
    fig, ax = plt.subplots(2,1,figsize=(20, 10))
    ax[0].bar(features_freq.keys(), features_freq.values(), color='#E3797D')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(features_freq.keys(),rotation=90)
    ax[0].set_title('Relative frequency of selected features')
    ax[1].bar(channels_freq.keys(), channels_freq.values(), color='#82A69C')
    ax[1].set_title('Relative frequency of selected channels')
    fig.tight_layout()
    plt.savefig(path_to_save+f'/feature_selection_{fig_name}')
    plt.close()
