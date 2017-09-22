'''
This module calls all the other modules to do pre-autoencoder data processing,
then feeds data into a trained autoencoder to determine the anomaly level, and
use that to finally compute the status for each area and output a graph
Input:  A general log from a specific ACL TOP machine
Output: Graph of anomaly level for each area on the web browser
Author: Oscar Song, Sunjay Ravishankqr, Oshin Mundada
'''

# Import libraries and other supporting modules

from dataFilter import DataFilter
from genLogtoTable import GeneralLog
from autoencoder import Network
from autoencoder import FullyConnectedLayer
from FinalComputation import FinalComputation

import autoencoder
import time
import numpy as np
import os

#Import libraries for the GUI implementation
#Source: https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
from tkFileDialog import askopenfilename
from Tkinter import *
from PIL import Image, ImageTk

import tkMessageBox
import ttk
import threading


#Declaring global variables
MONTH = 31
path = os.path.dirname(os.path.realpath(__file__)) + '/' #Current directory to run the program in
parameters_path = path+'/areas/'
filename = ''       #To store the filename the user will choose
serial_number = ''
final_matrix = []   #To store the output of the Autoencoder to pass into FinalComputation
# Declare objects to read general log file into a matrix, filter out the data, and create graphs
converter = GeneralLog()    #Instance for reading general log
filter = DataFilter()       #Instance for filtering out the matrix
grapher = FinalComputation()    #Instance for creating the final graphs

# The list of areas
areas = ['ARMs', 'Barcode/Racks', 'Cuvettes Movement', 'Fluidics', 'LLD (Liquid Level Detection)', 'ORU', 'Others', 'SW']

'''
This method takes in a date-weight matrix that is summed
and return a list of dates paired with vectors of weights, each is shifted by one day.
Input: a date-weight matrix
Output: a list of vectors of tuples
Author: Oscar Song
'''
def sliding_window(matrix):
    tuple_list = []
    for i in range(len(matrix)-31):                                 # Loop from beginning to 31 days before the end
        weight_vector = [matrix[j][1] for j in range(i,i+MONTH)]    # Get a list of weights for the month from this point onwards
        date = matrix[i+MONTH][0]                                   # Take the date
        tuple = (date, weight_vector)                               # Match the date string to a list of weights into a tuple
        tuple_list.append(tuple)                                    # Append the tuple into a list

    if len(tuple_list) > 31:                                        # Cut the list to only the last month
        return tuple_list[len(tuple_list)-31:]
    else:                                                           # It is ok if the list has less than 31 days
        return tuple_list

'''
This method extract a single generalLog file into a sorted matrixTest
Input: the name of the general log txt file
Output: A 3 Dimentional list sorted by areas, each is a 2D matrix
Author: Oscar Song
'''
def extract_log(log_file):
    matrix_raw = converter.create_one_matrix(log_file)  # Create a matrix out of the log file
    matrix_summed = filter.sumDay(matrix_raw)           # Sum up weights of the same day

    matrix_area_list = []
    for area in areas:                                  # For each area, append the relevant data to it
        raw_matrix_for_area = list(filter.segAreaTable(matrix_summed, area))    # Segregate the area
        tuple_list_for_area = sliding_window(raw_matrix_for_area)               # Use sliding window() to get weight list of each day for last 31 days
        matrix_area_list.append(tuple_list_for_area)    # Append to the list

    return matrix_area_list


'''
Description: This method reads the golden_stat.csv and
sort the values into a list by areas
Input: a filename of the file containing the golden stats
Output: a list of size-2 lists
Author: Oscar Song
'''
def read_golden_stat(filename):
    f = open(filename, 'r')              # Open the file
    lines = f.read().split('\n')         # Read from the file by rows
    golden_stat = []
    for line in lines:                      # Read line by line
        row = line.split(',')
        if row ==['']: continue
        parsed_row = [float(x) for x in row]    # Store the row as a list of floats
        golden_stat.append(parsed_row)          # Append to a larger list

    return golden_stat




#####################GUI methods ##################
'''
Description: This method is triggered when the user clicks Browse
Output: the file name that user has selected with full path
Author: Oscar Song (borrowed from source)
'''
def open_file():
    global filename
    #
    filename = askopenfilename()

    # Clean and fill the entry with filename
    entry.delete(0, END)
    entry.insert(0, filename)

    return filename


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
    if len(filename) == 0:
        tkMessageBox.showerror("Error", "Please import file first!")

    #Change to frame 2 and execute backend using autoencoder
    else:
        raise_frame(f2)
        progress.grid(row=4,column=0,pady=2,padx=2)              # Set the progress display on GUI
        '''
        Description: A wrapper method to work with multi-thread. Use Autoencoder to put z_scores and
                     respective area and dates into the global variable final_matrix
        Author: Oscar Song
        '''
        def execute():
            global serial_number
            genLog = filename.split('/')[len(filename.split('/'))-1]    #Parse for only the file name
            s_num = genLog.split('.')[0].split('_')             #Get rid of the extension
            serial_number = str('_'.join(s_num[2:4]))                   #Extract the serial number

            golden_stat = read_golden_stat('golden_stat.csv')        # Read the csv for data normalization later
            data_list = extract_log(filename)                        # data_list is a list of list of tuples of a date and a size-31 vector

            for area in range(len(areas)):                                  # Go into an area first. data_list[i] is a list of tuples

                nn = autoencoder.load(parameters_path+'area_'+ str(area), 0.5)       # Load the trained parameters into the autoencoder
                lab_progress.config(text="Autoencoder running: "+areas[area])   #Update the progress label
                progress.config(maximum=len(data_list[area]))                   # Set the max of progress bar to the size of what's to be processed

                counter = 0                                                 # A counter to update the progress bar

                for date, data_raw_batch in data_list[area]:                # extract the items of the tuple

                    data_batch = filter.norm_test(data_raw_batch, golden_stat[area])    # Normalize the data
                    data_batch_shared = autoencoder.shared(data_batch)                  # Convert data_batch to Theano shared variable
                    data_batch_sharedx, data_batch_sharedy = data_batch_shared          # Separate output of shared function into input to autoencoder and expected output

                    z_score = nn.zscore(data_batch_sharedx)                             # Retrive the z_score from the trained autoencoder
                    final_matrix.append([areas[area], date, z_score.eval().tolist()])   #Format into matrix

                    counter+=1                                              # Update the progress bar using progress_var
                    progress_var.set(counter)

                progress.stop()

            grapher.loadData(final_matrix)                 # Load the filled final matrix into


            progress.grid_forget()                         # Once this batch of data is processed, stop progress bar
            raise_frame(f3)                           # Switch GUI to frame 3
            root.geometry("800x800")                  # Enlarge the window

        threading.Thread(target=execute).start()    # Multi-threading to avoiding GUI freeze



def go_back():
    raise_frame(f1)
    root.geometry("600x200")






'''
Function Name: graphArea()
Purpose: takes the input value from the combo box and when the graph button is clicked
will display the image with the specific functional area. Throws warnings when a specific
area has not been graphed during final computation.
'''
def graphArea(selected_area):

    value = selected_area

    if selected_area == 'Barcode/Racks':
        value = 'BarcodeRacks'

    if value == init_statement:
        tkMessageBox.showwarning("Warning", "No Functional Area Selected! Please select a functional area.")
    else:
        canvas.delete("all")
        canvas.configure(background='light grey')
        try:
            grapher.createGraph(selected_area,serial_number)
            img = Image.open(serial_number + '_' + value + '.png')
            png = ImageTk.PhotoImage(img)
            canvas.image = png
            canvas.configure(background='white')
            canvas.create_image(0,0,image=png, anchor='nw')
        except Exception as e:
            tkMessageBox.showwarning("Error", "Could not show graph! Graphed area not present in directory!")

###############################################






root = Tk()
root.title('ProDx')
root.geometry("600x200")

f1 = Frame(root)
f2 = Frame(root)
f3 = Frame(root)


for frame in (f1, f2, f3):
    frame.grid(row=0, column=0, sticky='news')


raise_frame(f1)

Label(f1,text="Import general log file").grid(row=3, column=0, sticky='e')
entry = Entry(f1, width=40, textvariable=filename)
entry.grid(row=4,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=open_file).grid(row=4, column=0, sticky='ew', padx=8, pady=4)

Button(f1, text="Process Now", command=raise_frame_f2).grid(row=5,column=0)

Label(f2, text="").grid(row=0, column=0)
Label(f2, text="").grid(row=1, column=0)
Label(f2, text="").grid(row=2, column=0)
lab_progress = Label(f2, text="Autoencoder running area: ")
lab_progress.grid(row=3, column=0, pady=2,padx=2)
progress_var = IntVar(f2)
progress = ttk.Progressbar(f2, mode="determinate", orient='horizontal',length=600,variable=progress_var)



##################
#create the logo image and place it on frame
image = Image.open('il.jpeg')
photo = ImageTk.PhotoImage(image)
logo = Label(f3, image=photo)
logo.image = photo
logo.grid(row = 0, column = 0, rowspan=3, sticky='e')

#label for the frame
Label(f3, text="ACLTOP Functional Area Graphs", font=("Helvetica", 18)).grid(row=0, column=3, rowspan=1, sticky='n', pady = 4)

#string variable for default combobox statement
default = StringVar()
init_statement = "Select Functional Area"
default.set(init_statement)

#create the combobox and place it on the grid
box = ttk.Combobox(f3,values=areas, textvariable = default)
box.grid(row=2, column=3, sticky='n', pady=4)

#create the graph button and place it on the grid
button = ttk.Button(f3, text="Display Area", command=lambda: graphArea(box.get()))
button.grid(row=2,column=4, sticky='e', padx = 4, pady=4)
Button(f3,text="Go back", command=go_back).grid(row=0,column=4)
#create the canvas on which the graph will be displayed and place it
canvas = Canvas(f3, width=800, height=600, bg='light grey')
canvas.grid(row = 3, column=0, columnspan=7, sticky='e')

#Exit the program properly
def shutdown_ttk_repeat():
    root.eval('::ttk::CancelRepeat')
    root.quit()

root.protocol("WM_DELETE_WINDOW", shutdown_ttk_repeat)
root.mainloop()
