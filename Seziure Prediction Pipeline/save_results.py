import xlsxwriter as xw

def save_results(information_general, information_train, information_test, approach):
    
    # Create xlsx
    path = f'../Results/Approach {approach}/Results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})

    patients = list(information_general.keys())
    for patient in patients:
        info_general = information_general[patient]
        info_train = information_train[patient]
        info_test = information_test[patient]
        
        # Create sheet
        ws = wb.add_worksheet(f'pat_{patient}')

        # Header format
        format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
        format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
        format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
        
        # Insert Header
        header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
        header_train = ['#Features','Cost','SS samples','SP samples','Metric']
        header_test = ['SS samples','SP samples','Threshold','#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','tt','p-value']
            
        row = 0
        col = 0
        ws.write_row(row, col, header_general, format_general)
        col = len(header_general)
        ws.write_row(row, col, header_train, format_train)
        col = col + len(header_train)
        ws.write_row(row, col, header_test, format_test)
    
        # Insert data
        row = 1
        col = 0
        ws.write_row(row, col, info_general)
        
        info = [info_train[i]+info_test[i] for i in range(len(info_train))]
        col = len(info_general)
        for i in info:
            ws.write_row(row, col, i)
            row += 1

    wb.close()
    


def save_final_results(final_information, approach):
    
    # Create xlsx
    path = f'../Results/Approach {approach}/Final_results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})
    ws = wb.add_worksheet('Final results')
    
    # Header format
    format_general = wb.add_format({'bold':True, 'bg_color':'#F2F2F2'})
    format_train = wb.add_format({'bold':True, 'bg_color':'#AFF977'})
    format_test = wb.add_format({'bold':True, 'bg_color':'#A8C2F6'})
    
    # Insert Header
    header_general = ['Patient','#Seizures train','#Seizures test','SPH','SOP']
    header_train = ['#Features','Cost','SS samples','SP samples','Metric']
    header_test = ['SS samples','SP samples','Threshold','#Predicted','#False Alarms','SS','FPR/h','SS surrogate mean','SS surrogate std','tt','p-value']
            
    row = 0
    col = 0
    ws.write_row(row, col, header_general, format_general)
    col = len(header_general)
    ws.write_row(row, col, header_train, format_train)
    col = col + len(header_train)
    ws.write_row(row, col, header_test, format_test)
    
    # Insert data
    row = 1
    col = 0
    for i in final_information:
        ws.write_row(row, col, i)
        row += 1

    wb.close()
    
    
def save_test_results(final_information, approach):
    p_value=[final_information[i][20] for i in range(0,len(final_information))]
    stat=['yes' if i<0.05 else 'no' for i in p_value]
    idx=[0,2,3,4,13,14,15,16]
    info_test=[[final_information[row][col] for col in idx] for row in range(len(final_information)) ]
    for row in range(len(info_test)):
        info_test[row].append(stat[row])
        
    
    # Create xlsx
    path = f'../Results/Approach {approach}/Test_results.xlsx'
    wb = xw.Workbook(path, {'nan_inf_to_errors': True})
    ws = wb.add_worksheet('Test results')
    
    # Header format
    format_general = wb.add_format({'bold':True, 'bg_color':'#AFF977'})

    
    # Insert Header
    header_general = ['Patient','#Seizures test','SPH','SOP','#Predicted','#False Alarms','SS','FPR/h','Statistically valid']
            
    row = 0
    col = 0
    ws.write_row(row, col, header_general, format_general)
    
    # Insert data
    row = 1
    col = 0
    for i in info_test:
        ws.write_row(row, col, i)
        row += 1

    wb.close()




