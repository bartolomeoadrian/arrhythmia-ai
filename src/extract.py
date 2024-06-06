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

def extract_data():
	data_files = glob.glob('./data/mitbih/*.atr')
	data_files = [i[:-4] for i in data_files]
	data_files.sort()
	return data_files

length = 275
records = extract_data()
print('Total files: ', len(records))


good_beats = ['N','L','R','B','A','a','J','S','V','r',
			 'F','e','j','n','E','/','f','Q','?']

for file_path in records:
	file_pathpts = file_path.split('/')
	fn = file_pathpts[-1]
	print('Loading file:', file_path)

	# Read in the data
	if path.exists(file_path+".hea"):
		record = wf.rdsamp(file_path)
		annotation = wf.rdann(file_path, 'atr')

		# Print some meta informations
		print('    Sampling frequency used for this record:', record[1].get('fs'))
		print('    Shape of loaded data array:', record[0].shape)
		print('    Number of loaded annotations:', len(annotation.num))

		# Get the ECG values from the file.
		data = record[0].transpose()

		clas = np.array(annotation.symbol)
		rate = np.zeros_like(clas, dtype='float')
		for clasid, clasval in enumerate(clas):
			if (clasval == 'N'):
				rate[clasid] = 1.0 # Normal
			elif (clasval == 'L'):
				rate[clasid] = 2.0 # LBBBB
			elif (clasval == 'R'):
				rate[clasid] = 3.0 # RBBBB
			elif (clasval == 'V'):
				rate[clasid] = 4.0 # Premature Ventricular contraction
			elif (clasval == 'A'):
				rate[clasid] = 5.0 # Atrial Premature beat
			elif (clasval == 'F'):
				rate[clasid] = 6.0 # Fusion ventricular normal beat
			elif (clasval == 'f'):
				rate[clasid] = 7.0 # Fusion of paced and normal beat
			elif (clasval == '/'):
				rate[clasid] = 8.0 # paced beat

		rates = np.zeros_like(data[0], dtype='float')
		rates[annotation.sample] = rate

		indices = np.arange(data[0].size, dtype='int')

		# Manipulate both channels
		for channelid, channel in enumerate(data):
			chname = record[1].get('sig_name')[channelid]
			print('    ECG channel type:', chname)

			# Find rpeaks in the ECG data. Most should match with
			# the annotations.
			out = ecg.ecg(signal=channel, sampling_rate=360, show=False)

			# Split into individual heartbeats. For each heartbeat
			# record, append classification.

			beats = []
			for ind, ind_val in enumerate(out['rpeaks']):

				start,end = ind_val-length//2, ind_val+length//2
				if start < 0:
					start = 0
				diff = length - len(channel[start:end])
				if diff > 0:
					padding = np.zeros(diff, dtype='float')
					padded_channel = np.append(padding, channel[start:end])
					beats.append(padded_channel)
				else:
					beats.append(channel[start:end])

				# Get the classification value that is on
				# or near the position of the rpeak index.
				from_ind = 0 if ind_val < 10 else ind_val - 10
				to_ind = ind_val + 10
				clasval = rates[from_ind:to_ind].max()

				# Standardize the data
				beats[ind] = ((beats[ind] - np.mean(beats[ind])) / np.std(beats[ind]))

				# Append the classification to the beat data.
				beats[ind] = np.append(beats[ind], clasval)

				# Append the record number to the beat data.
				beats[ind] = np.append(beats[ind], fn[-3:])

			# Save to CSV file.
			savedata = np.array(beats[:], dtype=float)
			outfn = './data/mitbih/csv_files/'+fn+'_'+chname+'.csv'
			print('Generating ', outfn)
			with open(outfn, "wb") as fin:
				np.savetxt(fin, savedata, delimiter=",", fmt='%f')
	else:
		print('    ECG '+file_path+' not complete, skipping')