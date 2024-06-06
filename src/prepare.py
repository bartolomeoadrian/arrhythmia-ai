import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import wfdb as wf
import glob
from biosppy.signals import ecg
from wfdb import processing
import scipy
from scipy import *
from os import path
# sns.set()

new_len = 277
alldata = np.empty(shape=[0, new_len])
print(alldata.shape)

all_csv = glob.glob('./data/mitbih/csv_files/mitbih/*.csv')

for j in all_csv:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    alldata = np.append(alldata, csvrows, axis=0)
print(alldata.shape)

#### remove unidentified beats and save all data

no_anno = np.where(alldata[:,-2]==0.0)[0]
print(no_anno.shape)
alldata = np.delete(alldata, no_anno,0)
print(alldata.shape)

with open('./data/all_data.csv', "wb") as fin:
    np.savetxt(fin, alldata, delimiter=",", fmt='%f')