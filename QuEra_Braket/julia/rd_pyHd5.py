#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# test reading of HD5 saved in Julia
from pprint import pprint
from pyToolbox.Util_H5io4 import read4_data_hdf5

inpF="data/emu123.mxmGrid.h5"
bigD,meta=read4_data_hdf5(inpF,verb=1)
for x in bigD:
    print('\nkey=',x); pprint(bigD[x])
    
# inspect 1st row
P=bigD["probs"]
print('P',P)
print('Done-Python')
