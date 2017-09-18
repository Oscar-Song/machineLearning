'''
This module, with a trained autoencoder imported,
is used to plot z_scores of good and bad data to determine the
right threshold for ok, Monitoring, or TBA status
Author: Oscar Song
Input:  A folder with good files and a folder with files known to be TBA for a certain area
Output: A plot of z_scores of different files
'''

# Import libraries and other supporting modules

from sys import argv
from dataFilter import DataFilter
from genLogtoTable import GeneralLog
from datetime import datetime

import progressbar
import autoencoder
from time import sleep
from autoencoder import Network
from autoencoder import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

#script, good_test_folder, monitor_test_folder, bad_test_folder = argv

MONTH = 31
path = os.path.dirname(os.path.realpath(__file__)) + '/'
good_test_folder = 'dataSmall'
param_path = path+'areas/'
tba_path = path+ 'TBA_folders/'
bad_test_folder = ['ARMs', 'Barcode', 'Cuvettes', 'Fluidics', 'LLD', 'ORU', 'Others', 'SW']
#tbm_path = path+ '/TBM_folders/'


# Declare objects to convert excel file into tsv file and filter out the data
converter = GeneralLog()
filter = DataFilter()
# Initialize starting time
t0 = time.clock()

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


def batch_list_Test(vector,golden_stat):
    #Ultimately have a list of size 31 vectors. Each vector is of an area and is normalized
    normed_batch = filter.norm_test(vector, golden_stat)
    #print "Test List normalized: ", time.clock()-t0
    return normed_batch

def read_golden_stat(filename):
    f = open(filename, 'r')
    lines = f.read().split('\n')
    golden_stat = []
    for line in lines:
        row = line.split(',')
        if row ==['']: continue
        parsed_row = [float(x) for x in row]
        golden_stat.append(parsed_row)
    print "Finished reading from golden_stat.csv: ", time.clock()-t0
    return golden_stat

def write_temp_csv(tmp_data, area):

    filename = 'tmp_data_'+ area+'.csv'
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(tmp_data)


def plot_histogram(area, good_file_list, bad_file_list, golden_stat):
    z_plots_good = []
    z_plots_bad = []

    good_csv = []
    bad_csv = []
    print 'load from ' + param_path+'area_'+str(area)
    nn = autoencoder.load(param_path+'area_'+str(area))


    print "Running through good files"
    bar = progressbar.ProgressBar(maxval=len(good_file_list), \
        widgets=[progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
    bar.start()
    for i in range(len(good_file_list)):
        test_list = extract_log(good_file_list[i])

        good_csv.append(test_list[area])

        test_batch = batch_list_Test(test_list[area], golden_stat[area])

        test_batch_shared = autoencoder.shared(test_batch)
        test_batch_sharedx, test_batch_sharedy = test_batch_shared

        z_score = nn.zscore(test_batch_sharedx)
        z_plots_good.append(z_score.eval())

        bar.update(i+1)
        sleep(0.1)

    bar.finish()


    print "Running through bad files"
    bar = progressbar.ProgressBar(maxval=len(bad_file_list), \
        widgets=[progressbar.Bar('=','[',']'),' ', progressbar.Percentage()])
    bar.start()

    for i in range(len(bad_file_list)):
        test_list = extract_log(bad_file_list[i])

        test_batch = batch_list_Test(test_list[area], golden_stat[area])
        if len(test_batch) != 31:
            continue

        bad_csv.append(test_list[area])


        test_batch_shared = autoencoder.shared(test_batch)
        test_batch_sharedx, test_batch_sharedy = test_batch_shared

        z_score = nn.zscore(test_batch_sharedx)
        z_plots_bad.append(z_score.eval())

        bar.update(i+1)
        sleep(0.1)

    bar.finish()

    write_temp_csv(good_csv, str(area)+'_good')
    write_temp_csv(bad_csv, str(area)+'_bad')

    colors = ['green','red']
    plt.hist((z_plots_good,z_plots_bad), label = ('good_data', 'bad_data'), color=colors)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Number of instances')
    plt.title("Histogram for "+areas[area])
    plt.grid(True)
    plt.legend()
    plt.show()




golden_stat = read_golden_stat('golden_stat.csv')

good_file_list = converter.get_all_files(path, good_test_folder)

for i in range(7,len(areas)):
    print areas[i]
    bad_file_list = converter.get_all_files(tba_path, bad_test_folder[i])
    print "length of bad file list is: ", len(bad_file_list)
    plot_histogram(i, good_file_list, bad_file_list, golden_stat)
