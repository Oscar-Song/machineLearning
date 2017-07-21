'''
This is a class with all its methods dedicated in filters data and reducing the
dimension of their given matrix.
Author: Oscar Song, Janelle Lines, Oshin Mundada
'''

# Use the numpy library for matrix formulation
import numpy as np
import pandas as pd

DATE = 0
MACRO_AREA = 1
WEIGHT = 2

class DataFilter(object):

    # The constructor method
    def __init__(self):
        pass            # No members needed for this class

    '''
    Input: a .csv file
    Output: a 4D array
    Author: Oscar Song,
    Description: This script takes in a complete csv puts it
    into a 4d array with all the unnecessary rows and columns
    filtered out.
    '''
    def filterTable(self, filename):
        # Declare necessary vectors
        date = []
        area = []
        weight = []

        DAW = []
        with open(filename) as f:
            next(f)                 # Skip the header row of csv
            for line in f:
                row = line.split('\t')
                eType = row[2]      # Read in the eType
                if eType != 'ERROR' or row[7] == '':    # Skip rows with wrong type or unspecified area
                    continue
                else:

                    list = []
                    list.append(row[3])
                    list.append(row[7])
                    list.append(float(row[10]))
                DAW.append(list)

        #DAW= np.column_stack((date,area,weight))  # Combine the columns

        return DAW

    '''
    Input: A DAW matrix
    Output: A DAW matrix with weight summed up for each date and there's no duplicate weight
    Author: Oshin Mundada, Janelle Lines
    '''
    def sumDay(self, DAWmat):

        df = pd.DataFrame(DAWmat, columns=['Date','Area','Weight'])
        gro = df.groupby(['Date','Area'], as_index=False)['Weight'].sum()
        matrix = gro.values.tolist()
        return matrix

    '''
    Input: A 3D array and a string
    Output: A 2D array
    Author: Oshin Mundada
    Description: This function returns the date-weight matrix from the date-area-weight matrix and the required area.
    '''
    def segAreaTable(self,DAWmat,area):
        dwmat=[]
        search = area; #assigns the area to search
        for sublist in DAWmat:
            if sublist[MACRO_AREA] == search:
                dwmat.append([sublist[DATE],sublist[WEIGHT]]) #appends only date and weight to new listt i.e date-weight matrix
        return dwmat
