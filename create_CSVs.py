'''
This module creates temporary csv files for training data for testing purpose
and create a txt file of golden means and standard deviations based on the
training data set
Author: Oscar Song
Input: Read from a folder name of the GeneralLog.txt files
Ouput: Write 9 csv files and a golden_stat.csv
'''

# Import libraries and other supporting modules

from sys import argv
from dataFilter import DataFilter
from genLogtoTable import GeneralLog
from datetime import datetime

import time
import numpy as np
import csv
import os

#Import libraries for the GUI implementation
#Source: https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter

from Tkinter import *
import tkFileDialog
import tkMessageBox
import ttk
import threading



MONTH = 31
path = os.path.dirname(os.path.realpath(__file__)) + '/'
golden_stats = []   #A dict to store the golden mean and sigma from training data
train_data = []     # A matrix with all the weights for data training
foldername = ''

# Declare objects to convert excel file into tsv file and filter out the data
converter = GeneralLog()
filter = DataFilter()

# The vector of areas
areas = ['ARMs', 'Barcode/Racks', 'Cuvettes Movement', 'Fluidics', 'LLD (Liquid Level Detection)', 'ORU', 'Others', 'SW']


'''
This method is activated when chosen to train and sort multiple generalLog
text files into a list of vectors to feed into the autoencoder
Author: Oscar Song
Input: the folder name for pure training data
Output: A 4 Dimentional list sort by areas then by general log into dates and weights
'''
def extract_and_sort(folder_name):
    #Convert the excel file into a tsv file and put it into a sorted matrix
    files_list = converter.get_all_files(path, folder_name)
    matrice_list = converter.create_matrix(files_list)

    #Summing weights of the same day for each general Log matrix
    matrice_summed_list = []

    for matrix_unsummed in matrice_list:
        matrix_summed = filter.sumDay(matrix_unsummed)
        matrix_truncated = filter.truncate_matrix(matrix_summed)
        matrice_summed_list.append(matrix_truncated)


    # Further break down each matrix by different areas
    total_list = []
    for area in areas:
        area_list = []
        for matrix in matrice_summed_list:
            raw_matrix_for_area = list(filter.segAreaTable(matrix, area))
            area_list.append(raw_matrix_for_area)
        total_list.append(area_list)

    return total_list



def batch_list_Train(total_list):
    #Ultimately have a list of list of vectors. Each sublist in the list is for
    #an area and is a list of size 31 vectors for that area

    trainingdata = []
    for area in range(len(areas)):
        traindata_for_area = []
        for DWmatrix in total_list[area]:
            # Turn a matrix into a list of size 31 vectors
            if len(DWmatrix) != MONTH:
                continue
            batch = [row[1] for row in DWmatrix]
            traindata_for_area.append(batch)

        temp_mean, temp_sigma = filter.golden(traindata_for_area)
        golden_stats.append((temp_mean,temp_sigma))
        trainingdata.append(traindata_for_area)


    return trainingdata


'''
Description: This method takes in a string of filename and write to a csv
'''
def write_csv(filename, data):


    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)






#####################GUI methods ##################
'''
Description: This method is triggered when the user clicks Browse
Output: the file name that user has selected with full path
Author: Oscar Song (borrowed from source)
'''
def open_file():
    global foldername
    #Ask the user to select a training folder
    #Source: https://stackoverflow.com/questions/9491195/browse-a-windows-directory-gui-using-python-2-7
    foldername = tkFileDialog.askdirectory(initialdir=path, title='Please select a training folder')

    # Clean and fill the entry with filename
    entry.delete(0, END)
    entry.insert(0, foldername)

    foldername = foldername.split('/')[len(foldername.split('/'))-1]
    return foldername


'''
Description: Change the frame for the GUI
Author: Oscar Song
'''
def raise_frame(frame):

    frame.tkraise()

'''
Description: Triggered when 'Process now' is clicked
Triggers the back-end autoencoder processing
Author: Oscar Song
'''
def raise_frame_f2():
    # Shows a Error message box if no general log is inputted yet
    if len(foldername) == 0:
        tkMessageBox.showerror("Error", "Please select training folder first!")

    #Change to frame 2 and execute backend using autoencoder
    else:
        raise_frame(f2)
        def execute():
            progress.grid(row=1,column=0)
            progress.start()

            train_list = extract_and_sort(foldername)
            train_data = batch_list_Train(train_list)
            write_csv('golden_stat.csv', golden_stats)

            #Create path if not exist yet
            directory = path+'/training_folder'
            if not os.path.exists(directory):
                os.makedirs(directory)

            #Write a csv for each area
            for i in range(len(areas)):
                completeName = os.path.join(directory, 'train_data_'+str(i)+'.csv')
                write_csv(completeName, train_data[i])

            progress.stop()
            progress.grid_forget()
            raise_frame(f3)                                # Switch GUI when finished

        threading.Thread(target=execute).start()    # Multi-threading to avoiding GUI freeze





###############################################









#####Main method#######
#Initialize the GUI setups
root = Tk()
root.title('ProDx Data Preparation')
root.geometry("800x150")

f1 = Frame(root)
f2 = Frame(root)
f3 = Frame(root)


for frame in (f1, f2, f3):
    frame.grid(row=0, column=0, sticky='news')


raise_frame(f1)

# Page that allows the user to select training folder
Label(f1,text="Select training folder").grid(row=3, column=0, sticky='e')
entry = Entry(f1, width=40, textvariable=foldername)
entry.grid(row=4,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=open_file).grid(row=4, column=0, sticky='ew', padx=8, pady=4)

Button(f1, text="Process Now", width=32, command=raise_frame_f2).grid(sticky='ew', padx=10, pady=10)

#Page showing that writing is in process
lab_progress = Label(f2, text="Writing to csv files... ")
lab_progress.grid(row=3, column=0, pady=2,padx=2)
progress_var = IntVar(f2)
progress = ttk.Progressbar(f2, mode="indeterminate", orient='horizontal',length=500,variable=progress_var, takefocus='True')

#Notify the user that data is prepared
Label(f3, text="Data prepared").grid(row=3, column=0, sticky='e')


root.mainloop()
