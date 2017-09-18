'''
Description: This is a class with all of its methods dedicated
to filtering data and reducing the dimension of the input matrix.
Author: Oscar Song, Janelle Lines, Oshin Mundada
Date: July 21, 2017
'''

# Use the numpy library for matrix formulation
import numpy as np
import pandas as pd

DATE = 0
MACRO_AREA = 1
WEIGHT = 2
MONTH = 31

class DataFilter(object):

    # The constructor method
    def __init__(self):
        pass            # No members needed for this class


    '''
    Input: A DAW list (Date, Weight, Area)
    Output: A DAW list with weights summed up for each date and area. The
    output matrix doesn't have duplicate weights for a given day and area.
    Author: Oshin Mundada, Janelle Lines
    '''
    def sumDay(self, DAWmat):
        # Use Panda to auto-sort the matrix
        df = pd.DataFrame(DAWmat, columns=['Date','Area','Weight'])
        gro = df.groupby(['Date','Area'], as_index=False)['Weight'].sum()
        matrix = gro.values.tolist()
        return matrix

    '''
    Description: This method truncate the matrix to the last 31 days
    Input: A three-column list
    Output: A three-column list with at most 31 dates
    Author: Oscar Song
    '''
    def truncate_matrix(self, DAWmat):
        month = 31
        counter = 0
        index = len(DAWmat)-1
        #Looping backwards from the end of the matrix
        for i in range(len(DAWmat)-1, 1, -1):
            if DAWmat[i][DATE] != DAWmat[i-1][DATE]:    #Increment the counter when distinct date is found
                counter += 1
            if counter == month:    #Break the loop if 31 days is recorded
                index = i
                break
        return DAWmat[index:]   #Only take the last month of the matrix


    '''
    Description: This function returns the date-weight matrix for a specific
    area from the date-area-weight matrix and the required area.
    Input: A list of size-3 lists
    Output: A list of size-2 lists
    Author: Oshin Mundada
    Modifier: Oscar Song
    '''
    def segAreaTable(self,DAWmat,area):
        dwmat=[]
        #Go through each row in the inputted matrix
        for row in DAWmat:
            #Add the row if there's a match of area and
            if row[DATE] not in [new_row[0] for new_row in dwmat] and row[MACRO_AREA]==area:
                dwmat.append([row[DATE], row[WEIGHT]])
            #Add new date and default weight if not the area
            elif row[DATE] not in [new_row[0] for new_row in dwmat]:
                dwmat.append([row[DATE], 0.0])
            #Update the weight if the right area if found
            elif row[MACRO_AREA] == area:
                dwmat[len(dwmat)-1][1] = row[WEIGHT]

        return dwmat


    '''
    Description: This method takes in a list training data (list of length 31 lists) and does
    data normalization on it, based on the mean and standard deviation given in
    the meanstd (tuple) input. The output is a second list of normalized training
    data (list of length 31 lists).
    Author: Janelle Lines, Oscar Song
    Input: A list of list
    Output: A pair of floats
    '''

    def golden(self, data):
        flat_vec = [item for sublist in data for item in sublist]
        mu =np.mean(flat_vec)
        sigma = np.std(flat_vec)
        return mu, sigma

    '''
    Description: This method takes in the t-score and squash it between 0 to 1
    using the sigmoid function
    Author: Janelle Lines
    '''
    def sigmoid(self, z):

        return 1.0/(1.0+np.exp(-z))

    '''
    Description: This method takes in a numpy array and does data normalization
    on it before output it as a list
    Input: A list of lists and a list
    Output: A list of list
    Author: Janelle Lines, Oscar Song
    '''
    def norm_train(self, vec, meanstd):
        #Extract the mean and sigma from the meanstd list
        mu =meanstd[0]
        sigma = meanstd[1]
        newvec = []
        #Set conditional statement in case of sigma==0
        if sigma != 0:
            for item in vec:
                #For each item in the list, apply data normalization
                sublist = list(map(lambda x: self.sigmoid((x-mu)/sigma),item))
                newvec.append(sublist)

        else:
            for item in vec:
                #For sigma==0, set all items to 0.5 by default
                sublist = [0.5]*len(item)
                newvec.append(sublist)
        return newvec

    '''
    Description: This method takes in a numpy array and does
    data normalization on it before output it as a list
    Author: Janelle Lines, Oscar Song
    Input: A list and a list
    Output: A list
    '''
    def norm_test(self, vec, meanstd):

        mu =meanstd[0]
        sigma = meanstd[1]
        if sigma != 0:
            return list(map(lambda x: self.sigmoid((x-mu)/sigma),vec))
        else:
            return [0.5]*len(vec)
