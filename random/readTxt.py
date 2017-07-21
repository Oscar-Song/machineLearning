#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import pandas as pd
import numpy as np
np.set_printoptions(threshold='nan')
#import tensorflow as tf

# Import data set
path = "/Users/oscarwlsong/Desktop/testPython/generalLog.csv"

def extract_data(filename):
    # initialize arrays to hold the necessary datas
    sCode = []
    eType = []
    dataTime = []
    funcArea = []
    sUserName = []
    sDescription = []
    sFilename = []
    nLineNumber = []
    nSubCode = []
    eCPU = []

    with open(filename) as f:
        next(f)
        for line in f:
            row = line.split(',')
            sCode.append(int(row[0]))
            eType.append(row[1])
            dataTime.append(row[2])
            funcArea.append(row[3])
            sUserName.append(row[4])
            sDescription.append(row[5])
            sFilename.append(row[6])
            nLineNumber.append(int(row[7]))
            nSubCode.append(int(row[8]))
            eCPU.append(int(row[9]))
'''
        I don't think that we will need the sCode,nLineNumber,nSubCode, and eCPU

        refer to this video for more implementation later
         https://www.youtube.com/watch?v=0xVqLJe9_CY
         We can't move on unless we know which vector can we disregard and
        which ones can we merge as a features matrix
'''
