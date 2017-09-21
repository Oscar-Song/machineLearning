'''
This is the main module of the MachineLearning package
It calls and coordinate all the other supporting python
files.
Author: Oscar Song, Sunjay Ravishankqr, Janelle Lines, Oshin Mundada
Input:  A GeneralLog.txt file
Output: A graph for each area with reconstruction error and status stated
'''

# Import libraries and other supporting modules

from sys import argv
from dataFilter import DataFilter
from genLogtoTable import GeneralLog

#from autoencoder import Autoencoder
import progressbar
from autoencoder import FullyConnectedLayer
import autoencoder
from autoencoder import Network


import numpy as np
import csv
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import math


import os

#Import libraries for the GUI implementation
#Source: https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter

from tkFileDialog import askopenfilename
from Tkinter import *
import tkFileDialog
import tkMessageBox
import ttk
import threading

path = os.path.dirname(os.path.realpath(__file__)) + '/'
param_path = path+'areas/'

filename = ''
#foldername = ''


# Declare objects to convert excel file into tsv file and filter out the data
converter = GeneralLog()    #An object to convert the general logs into a matrices
filter = DataFilter()       #An object with methods to call on to filter the data

bad_test_folder = ['ARMs', 'Barcode', 'Cuvettes', 'Fluidics', 'LLD', 'ORU', 'Others', 'SW']
# The vector of areas
areas = ['ARMs', 'Barcode/Racks', 'Cuvettes Movement', 'Fluidics', 'LLD (Liquid Level Detection)', 'ORU', 'Others', 'SW']


'''
This method extract a single generalLog file into a sorted matrixTest
Author: Oscar Song
Input: the name of the general log txt file
Output: A 3 Dimentional list sorted by areas, each is a 2D matrix
'''
def extract_log(log_file):
    matrix_raw = converter.create_one_matrix(log_file)
    matrix_summed = filter.sumDay(matrix_raw)
    matrix_truncated = filter.truncate_matrix(matrix_summed)
    matrix_area_list = []
    for area in areas:
        raw_matrix_for_area = list(filter.segAreaTable(matrix_truncated, area))
        matrix_area_list.append([row[1] for row in raw_matrix_for_area])

    return matrix_area_list


def normalize_batch(total_list, golden_stat):
    #Ultimately have a list of list of vectors. Each sublist in the list is for
    #an area and is a list of size 31 vectors for that area

    batch_list = []

    for area in range(len(areas)):
        # Turn a matrix into a list of size 31 vector
        batch_list.append(filter.norm_train(total_list[area],golden_stat[area]))

    return batch_list

def batch_list_Test(vector,golden_stat):
    #Ultimately have a list of size 31 vectors. Each vector is of an area and is normalized
    normed_batch = filter.norm_test(vector, golden_stat)
    return normed_batch


'''
Description: This method reads the sorted training
data of a certain area from a csv file
Input: a filename of the file containing the training data
Output: a list of size-31 lists
Author: Oscar Song
'''
def read_csv(filename):

    f = open(filename, 'r')
    lines = f.read().split('\n')
    train_batch_list = []
    #For each line, take the value and convert to float before store in a list
    for line in lines:
        row = line.split(',')
        if row == ['']:
            continue
        parsed_row = [float(x) for x in row]
        train_batch_list.append(parsed_row)
    return train_batch_list

'''
Description: Write data to a csv for testing purpose
'''
def write_csv(name_file, data, area):

    filename = name_file +'_'+ str(area) +'.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(data)



def prepare_csv(area_str,good_folder, bad_folder,train_percent_str,validation_percent_str,dropout_ratio):
    bad_folder = bad_folder.split('/')[len(bad_folder.split('/'))-1]

    train_percent = float(train_percent_str)
    validation_percent = float(validation_percent_str)
    area = areas.index(area_str)


    golden_stat = read_csv(filename)
    train_matrices = []
    for i in range(len(areas)):
        train_matrices.append(read_csv(good_folder+'/train_data_'+str(i)+'.csv' ))

    len_of_data = len(train_matrices[area])
    train_number = int(math.floor(train_percent * len_of_data))
    validation_number = int(math.floor(validation_percent * len_of_data))

    raise_frame(histogram)
    root.geometry("550x450")
    good_file_list = train_matrices[area][(train_number+validation_number):]
    bad_file_list = converter.get_all_files(path+bad_folder+'/', bad_test_folder[area])
    golden_stat = read_csv(filename)

    prog_csv.grid(row=1,column=0)
    plotButton.grid(row=1,column=2)

    '''
    This method is used to test the performance of a pre-trained autoencoder neural
    network. The autoencoder is loaded from the current directory, and then tested
    on a batch of pre-labeled "good" generalLog.txt files and a corresponding batch
    of pre-labeled "bad" generalLog.txt files. The z-score for each file is
    calculated, and two lists of z-scores (one for the good files, one for the bad)
    are created. These lists are then stored in two separate csv's.
    Inputs: area number, list of good data, list of bad data, set of golden statistics,
    for z-scores to separate good and bad data
    '''
    def make_csv_hist(area, good_file_list, bad_file_list, golden_stat,dropout_ratio):
        z_plots_good = []
        z_plots_bad = []

        nn = autoencoder.load(path+'/areas/area_'+str(area),float(dropout_ratio))

        csv_lab_prog.configure(text="Reading good files...")
        prog_csv.configure(maximum=len(good_file_list)-1)

        for i in range(len(good_file_list)):
            test_batchnorm = batch_list_Test(good_file_list[i], golden_stat[area])
            test_batch_shared = autoencoder.shared(test_batchnorm)
            test_batch_sharedx, test_batch_sharedy = test_batch_shared
            z_score = nn.zscore(test_batch_sharedx)
            z_plots_good.append(z_score.eval())

            prog_var_csv.set(i)

        write_csv('good_z', z_plots_good, area)
        prog_csv.stop()

        csv_lab_prog.configure(text="Reading bad files...")
        prog_csv.configure(maximum=len(bad_file_list)-1)


        for i in range(len(bad_file_list)):
            test_list = extract_log(bad_file_list[i])
            test_batchnorm_bad = batch_list_Test(test_list[area], golden_stat[area])
            if len(test_batchnorm_bad) != 31:
                continue
            test_batch_shared = autoencoder.shared(test_batchnorm_bad)
            test_batch_sharedx, test_batch_sharedy = test_batch_shared

            z_score = nn.zscore(test_batch_sharedx)
            z_plots_bad.append(z_score.eval())

            prog_var_csv.set(i)

        write_csv('bad_z', z_plots_bad, area)
        csv_lab_prog.configure(text="Finished preparation")
        prog_csv.stop()

        plotButton.configure(state="normal")
    threading.Thread(target=lambda: make_csv_hist(area,good_file_list,bad_file_list,golden_stat,dropout_ratio)).start()



def plot_hist(area_str,threshold_str, lower_bound_str = "0", upper_bound_str = "10000"):
    if threshold_str == "":
        tkMessageBox.showerror("Error", "Complete all field!")
    else:
        area = areas.index(area_str)
        threshold = float(threshold_str)
        lower_bound = float(lower_bound_str)
        upper_bound = float(upper_bound_str)
        '''
        This is a method used to plot a histogram of z-scores for both "good" and "bad"
        pre-labeled data files. This enables the user to determine a proper threshold
        for the z-scores that adequately separates good and bad files. The false
        positive and false negative rates for a particular threshold are also calculated
        in this method. It is assumed that the make_csv_hist script has already been
        run.
        Inputs: area, threshold, lower_bound (lowest z-score used to plot histogram, default = 0),
        upper_bound (highest z-score used to plot histogram, default = 10000)
        '''

        def plot_histogram(area, threshold, lower_bound = 0, upper_bound = 10000):
            z_plots_good = []
            z_plots_bad = []
            z_plots_good_plot=[]
            z_plots_bad_plot=[]

            print "Running through good files"
            good_list_zscores = 'good_z_'+str(area)+'.csv'
            z_plots_good = read_csv(good_list_zscores)
            false_positive_count=0
            correct_good_count=0
            i = 1
            for z in z_plots_good[0]:
                if z > threshold:
                    #print "False Positive, i=", i
                    #print "z-score=", z
                    #print train_matrices[area][10198:][i-1]
                    false_positive_count = false_positive_count + 1
                else:
                    #print "Correct"
                    #print "z-score=", z
                    correct_good_count = correct_good_count+1
                if z <= upper_bound and z >= lower_bound:
                    z_plots_good_plot.append(z)
                i = i+1


            print "Running through bad files"
            bad_list_zscores = 'bad_z_'+str(area)+'.csv'
            z_plots_bad = read_csv(bad_list_zscores)
            false_negative = 0
            count_correct_bad = 0

            i = 1
            for z in z_plots_bad[0]:

                if z < threshold:
                    #print "False Negative, i=",i
                    #print "z-score=", z
                    #print bad_batch[i-1]
                    false_negative = false_negative + 1
                else:
                    #print "Correct"
                    #print "z-score=", z
                    count_correct_bad = count_correct_bad+1
                if z <= upper_bound and z >= lower_bound:
                    z_plots_bad_plot.append(z)
                i = i+1

            false_negative_rate = float(false_negative)/(float(count_correct_bad)+float(false_negative))
            false_positive_rate = float(false_positive_count)/(float(correct_good_count)+float(false_positive_count))

            false_pos_lab.grid(row=6,column=0)
            false_neg_lab.grid(row=6,column=2)
            false_pos_lab.configure(text="False positive rate: "+str(false_positive_rate))
            false_neg_lab.configure(text="False negative rate: "+str(false_negative_rate))

            root.geometry("800x800")
            f = Figure(figsize=(4,4), dpi=100)
            a = f.add_subplot(111)

            colors = ['green','red']
            a.hist((z_plots_good_plot,z_plots_bad_plot), label = ('good_data', 'bad_data'), color=colors)
            a.set_xlabel('Reconstruction Error')
            a.set_ylabel('Number of instances')
            a.set_title("Histogram for "+areas[area])
            a.grid(True)
            a.legend()

            # Merge matplotlib plot with tKinter to show the graph on the GUI
            canvas= FigureCanvasTkAgg(f, master=histogram)
            plot_widget = canvas.get_tk_widget()
            plot_widget.grid(row=7, column=0)

        threading.Thread(target=lambda: plot_histogram(area,threshold,lower_bound,upper_bound)).start()

'''
Description: This method runs the actual training process using threading
Input: all the necessary paramenters for the autoencoder
Output: trained parameters of the autoencoder will be written
Author: Oscar Song(GUI part), Janelle Lines(Wrapper method train_autoencoder())
'''
def start_prog(foldername,area_str,mini_batch_size_str,epoch_str,eta_str,lm_str,train_percent_str,validation_percent_str,dropout_ratio_str):

    #Output error message when necessary
    if area_str == '[Select area to show]' or mini_batch_size_str == '' or epoch_str == '' or eta_str == '' or lm_str == '' or train_percent_str =='':
        tkMessageBox.showerror("Error", "Complete all field!")
    else:
        # Change frame and convert paramenters
        raise_frame(prog)
        area = areas.index(area_str)
        mini_batch_size = int(mini_batch_size_str)
        epoch = int(epoch_str)
        eta = float(eta_str)
        lm = float(lm_str)
        train_percent = float(train_percent_str)
        validation_percent = float(validation_percent_str)
        dropout_ratio = float(dropout_ratio_str)

        #Format the frame for better visualization
        progress.grid(row=5,column=0)
        Label(prog,text="").grid(row=6,column=0)
        Label(prog,text="").grid(row=7,column=0)
        Label(prog,text="").grid(row=8,column=0)
        Label(prog,text="").grid(row=9,column=0)
        Label(prog,text="").grid(row=10,column=0)
        Label(prog,text="").grid(row=11,column=0)
        Label(prog,text="").grid(row=12,column=0)
        Label(prog,text="").grid(row=13,column=0)
        Label(prog,text="").grid(row=14,column=0)
        Label(prog,text="").grid(row=15,column=0)
        graph_button.grid(row=16, column=2)

        # The wrapper method that does the actual training
        # The progress is shown via a progress bar
        def train_autoencoder(foldername, area,mini_batch_size,epoch,eta,lm,train_percent,validation_percent,dropout_ratio):
            progress.config(maximum=5)
            lab_progress.config(text="Extracting golden stat...")

            golden_stat = []   #A dict to store the golden mean and sigma from training data
            train_matrices = []
            golden_stat = read_csv(filename)

            progress_var.set(1)
            lab_progress.config(text="Extracting csv files...")
            for i in range(len(areas)):
                train_matrices.append(read_csv(foldername+'/train_data_'+str(i)+'.csv' ))
            train_batch = normalize_batch(train_matrices,golden_stat)

            progress_var.set(2)
            lab_progress.config(text="Training autoencoder...")

            len_of_data = len(train_batch[area])

            train_number = int(math.floor(train_percent * len_of_data))
            validation_number = int(math.floor(validation_percent * len_of_data))




            print "Amount of training data:", len(train_batch[area])
            training_data = autoencoder.shared(train_batch[area][:train_number])
            validation_data = autoencoder.shared(train_batch[area][train_number:(train_number+validation_number)])
            test_data = autoencoder.shared(train_batch[area][(train_number+validation_number):])




            net = Network([
                FullyConnectedLayer(n_in=31, n_out=21,p_dropout=dropout_ratio),
                FullyConnectedLayer(n_in=21, n_out=9,p_dropout=dropout_ratio),
                FullyConnectedLayer(n_in=9, n_out=21,p_dropout=dropout_ratio),
                FullyConnectedLayer(n_in=21, n_out=31, p_dropout=dropout_ratio)],
                mini_batch_size)


            results = net.SGD(training_data, epoch, mini_batch_size, eta, validation_data, test_data,lmbda=lm, monitor =True)


            progress_var.set(4)
            lab_progress.config(text="Saving parameters...")

            net.save(path+'/areas/area_newsort_'+str(area))

            progress_var.set(5)
            progress.stop()
            lab_progress.config(text="Done")
            graph_button.configure(state="normal",command=lambda:show_plot(results,epoch))

        # Use a thread to avoid GUI freezing
        threading.Thread(target=lambda: train_autoencoder(foldername, area,mini_batch_size,epoch,eta,lm,train_percent,validation_percent,dropout_ratio)).start()    # Multi-threading to avoiding GUI freeze

'''
Description: Show the plotted graph of the training result on the frame
User may see the decrease of cost and z_scores
Arthur: Janelle Lines
'''
def show_plot(results,epoch):
    root.geometry("650x600")
    raise_frame(result)

    epochs = range(0,epoch)
    # Use matplotlib to plot the graph
    f, ax = plt.subplots(3, sharex = True)
    ax[0].set_title("Training Cost")
    ax[0].set_ylabel('Training Cost')
    #ax[0].set_xlabel('Epoch')
    ax[0].plot(epochs,results[0])
    ax[1].set_title("Mean Training Z-score")
    ax[1].set_ylabel('Training Z-Score')
    #ax[1].set_xlabel('Epoch')
    ax[1].plot(epochs,results[1])
    ax[2].set_title("Mean Validation Z-score")
    ax[2].set_ylabel('Validation Z-Score')
    ax[2].set_xlabel('Epoch')
    ax[2].plot(epochs,results[2])

    # Merge matplotlib plot with tKinter to show the graph on the GUI
    canvas= FigureCanvasTkAgg(f, master=result)
    plot_widget = canvas.get_tk_widget()
    plot_widget.grid(row=0, column=0)


######Methods for GUI###########
'''
Description: Allow the user to open a certain file
'''
def open_file():
    global filename

    filename = askopenfilename()

    file_entry.delete(0, END)
    file_entry.insert(0, filename)

    return filename

'''
Description: Allow the user to open a certain folder
'''
def open_folder(entry):

    #Ask the user to select a training folder
    #Source: https://stackoverflow.com/questions/9491195/browse-a-windows-directory-gui-using-python-2-7
    foldername = tkFileDialog.askdirectory(initialdir=path, title='Please select a training folder')

    # Clean and fill the entry with filename
    entry.delete(0, END)
    entry.insert(0, foldername)

    foldername = foldername.split('/')[len(foldername.split('/'))-1]
    return foldername

#Change the GUI frame
def raise_frame(frame):

    frame.tkraise()

#Change the GUI frame as well as the size of the window
def get_back():
    raise_frame(setting)
    root.geometry("650x450")
    plotButton.configure(state="disabled")
    graph_button.configure(state="disabled")



#Declare the GUI
root = Tk()
root.title('Train Setting')
root.geometry("650x450")

#Declare the frames
home = Frame(root)
setting = Frame(root)
prog = Frame(root)
result = Frame(root)
histogram = Frame(root)
for frame in (home, setting, prog, result, histogram):
    frame.grid(row=0, column=0, sticky='news')

#Start with frame home and configure the frame
raise_frame(home)
Label(home,text="").grid(row=0,column=0)
Label(home, text="Choose files to import").grid(row=1, column=0, sticky='e',columnspan=15)
Label(home,text="").grid(row=2,column=0)

#Allow user to browse the golden_stat file
Label(home,text="Import golden_stat file").grid(row=3, column=0)
Button(home, text="Browse", width=10, command=open_file).grid(row=4, column=0,padx=8, pady=4)
file_entry = Entry(home, width=30, textvariable=filename)
file_entry.grid(row=4,column=1,padx=2,pady=2,sticky='we',columnspan=20)

#Allow user to select folder of the training data
Label(home,text="Import training data folder").grid(row=5, column=0)
folder_entry = Entry(home, width=30)
Button(home, text="Browse", width=10,command=lambda:open_folder(folder_entry)).grid(row=6, column=0, padx=8, pady=4)
folder_entry.grid(row=6,column=1,padx=2,pady=2,sticky='we',columnspan=20)

Label(home,text="").grid(row=7,column=0)
Label(home,text="").grid(row=8,column=0)

'''#Allow user to select folder of the good data for histogram
Label(home,text="Import good test data folder").grid(row=9, column=0)
good_hist_entry = Entry(home, width=30)
Button(home, text="Browse", width=10,command=lambda: open_folder(good_hist_entry)).grid(row=10, column=0, padx=8, pady=4)
good_hist_entry.grid(row=10,column=1,padx=2,pady=2,sticky='we',columnspan=20)'''

#Allow user to select folder of the bad data for histogram
Label(home,text="Import bad test data folder").grid(row=11, column=0)
bad_hist_entry = Entry(home, width=30)
Button(home, text="Browse", width=10,command=lambda:open_folder(bad_hist_entry)).grid(row=12, column=0, padx=8, pady=4)
bad_hist_entry.grid(row=12,column=1,padx=2,pady=2,sticky='we',columnspan=20)

Label(home,text="").grid(row=13,column=0)
Label(home,text="").grid(row=14,column=0)
Button(home, text="Next",command=lambda:raise_frame(setting)).grid(row=15, column=10,sticky='ew', padx=10, pady=10)

#Allow user to select area to train from an option menu
Label(setting,text="Select Functional Area").grid(row=0, column=0, sticky='e')
var = StringVar(setting)
var.set('[Select area to show]')
option = OptionMenu(setting, var, *areas)
option.grid(row=0, column=1)

#Allow user to set mini_batch_size from an entry
Label(setting,text="Mini batch size(eg:18)").grid(row=1, column=0, sticky='e')
mini_entry = Entry(setting, width=10)
mini_entry.grid(row=1,column=1,padx=2,pady=2,sticky='we',columnspan=10)
mini_entry.insert(0,"10")

#Allow user to set epoch from an entry
Label(setting,text="Epoch(eg:30)").grid(row=2, column=0, sticky='e')
epoch_entry = Entry(setting, width=10)
epoch_entry.grid(row=2,column=1,padx=2,pady=2,sticky='we',columnspan=10)
epoch_entry.insert(0,"30")

#Allow user to set eta from an entry
Label(setting,text="Learning rate(eg:0.0005)").grid(row=3, column=0, sticky='e')
eta_entry = Entry(setting, width=10)
eta_entry.grid(row=3,column=1,padx=2,pady=2,sticky='we',columnspan=10)
eta_entry.insert(0,"0.0005")

#Allow user to set lm from an entry
Label(setting,text="L-2 Regularization Parameter (eg:0.1)").grid(row=4, column=0, sticky='e')
lm_entry = Entry(setting, width=10)
lm_entry.grid(row=4,column=1,padx=2,pady=2,sticky='we',columnspan=10)
lm_entry.insert(0,"0.0")

#Allow user to set train number from an entry
Label(setting,text="Percentage of good files used for training (eg:0.60)").grid(row=5, column=0, sticky='e')
train_entry = Entry(setting, width=10)
train_entry.grid(row=5,column=1,padx=2,pady=2,sticky='we',columnspan=10)
train_entry.insert(0,"0.60")



#Allow user to set validationnumber from an entry
Label(setting,text="Percentage of good files used for validation (eg:0.20)").grid(row=6, column=0, sticky='e')
valid_entry = Entry(setting, width=10)
valid_entry.grid(row=6,column=1,padx=2,pady=2,sticky='we',columnspan=10)
valid_entry.insert(0,"0.20")

'''
#Allow user to set test number from an entry
Label(setting,text="Number of Test Files (eg:400)").grid(row=7, column=0, sticky='e')
test_entry = Entry(setting, width=10)
test_entry.grid(row=7,column=1,padx=2,pady=2,sticky='we',columnspan=10)
'''


#Allow user to set dropout ratio from an entry
Label(setting,text="Dropout ratio(eg:0.5)").grid(row=8, column=0, sticky='e')
dropout_entry = Entry(setting, width=10)
dropout_entry.grid(row=8,column=1,padx=2,pady=2,sticky='we',columnspan=10)
dropout_entry.insert(0,"0.5")




#Button to start training
Button(setting, text="Back",command=lambda:raise_frame(home)).grid(row=12, column=0,sticky='ew', padx=10, pady=10)
Button(setting, text="Next",
command=lambda:start_prog(folder_entry.get(),var.get(),mini_entry.get(),epoch_entry.get(),eta_entry.get(),lm_entry.get(),train_entry.get(),valid_entry.get(),dropout_entry.get())).grid(row=12, column=10, sticky='ew', padx=8, pady=4)

#Progress bar for training

Label(prog).grid(row=0,column=0)
Label(prog).grid(row=1,column=0)
Label(prog).grid(row=2,column=0)
Label(prog).grid(row=3,column=0)
lab_progress = Label(prog, text="Training Autoencoder in progress ...")
lab_progress.grid(row=4, column=0, pady=2,padx=2, columnspan=15)
progress_var = IntVar(prog)
progress = ttk.Progressbar(prog, mode="determinate", orient='horizontal',length=400,variable=progress_var)
Button(prog, text="Back",command=lambda:raise_frame(setting)).grid(row=16, column=0)
graph_button = Button(prog, text="View Training result",state="disabled")

#Allow the user to go back and train again
Button(result, text="Train Again", command=get_back).grid(row=16,column=0,sticky='ew', padx=10, pady=10)
Button(result, text="Prepare CSVs for histogram", command=lambda: prepare_csv(var.get(),folder_entry.get(), bad_hist_entry.get(),train_entry.get(),valid_entry.get(),dropout_entry.get())).grid(row=17,column=0,sticky='ew', padx=10, pady=10)

csv_lab_prog = Label(histogram, text="Training Autoencoder in progress ...")
csv_lab_prog.grid(row=0, column=0, pady=2,padx=2, columnspan=15)
prog_var_csv = IntVar(histogram)
prog_csv = ttk.Progressbar(histogram, mode="determinate", orient='horizontal',length=400,variable=prog_var_csv)

#Allow user to set threshold from an entry
Label(histogram,text="Threshold(eg:0.315)").grid(row=3, column=0, sticky='e')
threshold_entry = Entry(histogram, width=10)
threshold_entry.grid(row=3,column=2,padx=2,pady=2,sticky='we',columnspan=10)

#Allow user to set lower bound from an entry
Label(histogram,text="Lower bound for z-scores (eg:0)").grid(row=4, column=0, sticky='e')
lower_entry = Entry(histogram, width=10)
lower_entry.grid(row=4,column=1,padx=2,pady=2,sticky='we',columnspan=10)
lower_entry.insert(0,"0")

#Allow user to set upper bound from an entry
Label(histogram,text="Upper bound for z-scores (eg:10000)").grid(row=5, column=0, sticky='e')
upper_entry = Entry(histogram, width=10)
upper_entry.grid(row=5,column=1,padx=2,pady=2,sticky='we',columnspan=10)
upper_entry.insert(0,"10000")

plotButton = Button(histogram, text="Plot histogram",command=lambda:plot_hist(var.get(),threshold_entry.get(),lower_entry.get(),upper_entry.get()), state="disabled" )
plotBackButton = Button(histogram,text="Train Again",command=get_back).grid(row=2,column=2,sticky='ew', padx=10, pady=10)



false_pos_lab = Label(histogram)
false_neg_lab = Label(histogram)
#Allow the GUI and script to exit properly
def shutdown_ttk_repeat():
    root.eval('::ttk::CancelRepeat')
    root.quit()

root.protocol("WM_DELETE_WINDOW", shutdown_ttk_repeat)
root.mainloop()
