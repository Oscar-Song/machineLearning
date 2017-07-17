#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import pandas as pd
import numpy as np
np.set_printoptions(threshold='nan')
#import tensorflow as tf

# Import data set
path = "/Users/oscarwlsong/Desktop/testPython/generalLog.csv"

df = pd.read_csv(path, sep=',', header=None)
data = df.values

print data
