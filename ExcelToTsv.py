'''
Program Name: ExcelToTSV.py
Purpose: A simple script that converts the .xlsm macro file into a readable
.tsv file for parsing during the data filtering step of our workflow.
Author: Sunjay Ravishankar

'''

import csv
import xlrd
from datetime import datetime
from datetime import time


class Converter(object):
    def __init__(self,filename):
        self.filename = filename
    def convert(self):
        #open the macro workbook
        macro_workbook = xlrd.open_workbook(self.filename)

        #get the LOGBOOK VIEWER sheet
        log_sheet = macro_workbook.sheet_by_index(0)

        #open the macro.csv file for writing
        outFile = 'nData.tsv'
        with open(outFile, 'w+') as f:

            #loop through the rows starting from the first row with information
            for row in xrange(1,log_sheet.nrows):

                #maintain a column counter makes sure no comma placed at end of line
                col_counter = 1

                #loop through each cell
                for cell in log_sheet.row_values(row):

                    #checks to convert the excel date format to a normal readable format
                    if col_counter == 4:
                        excel_date = int(cell)
                        date = str(datetime.fromordinal(datetime(1900,1,1).toordinal() + excel_date - 2))
                        cell = date[:10]
			        #check to convert excel time formal to readable time format
                    elif col_counter == 5:
                        excel_time = float(cell)
                        normal_time = int(excel_time * 24 * 3600)
                        my_time = time(normal_time / 3600, (normal_time % 3600) / 60, normal_time % 60)
                        cell = str(my_time)


			        #check to make sure not to place an unnecessary comma
                    #Does not work with the umlaut symbol

                    if col_counter == log_sheet.ncols:
                        f.write(str(cell))
                    else:
                        f.write(str(cell) + '\t' )
                    col_counter += 1
                f.write('\n')
        return outFile
