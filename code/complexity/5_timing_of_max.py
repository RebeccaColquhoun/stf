
import sys
sys.path.append('..')


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import pickle
import obspy
from scipy.signal import find_peaks
import os
import pandas as pd

#from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import curve_fit

import scipy.integrate

from functions_load_stf import *
from functions_end_stf import find_end_stf



combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')


window = 2


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def get_data_and_prepare(row, dataset, get_stf):
	times_to_return = []
	momentrates_to_return = []

	if dataset == 'scardec_moy' or dataset == 'scardec_opt':
		name = row[dataset[:-4]]
	else:
		name = row[dataset]

	if name == '0' or name == 0:
		return None, None

	if dataset == 'scardec_moy':
		momentrate, time = get_stf(name, 'fctmoy')
	elif dataset == 'scardec_opt':
		momentrate, time = get_stf(name, 'fctopt')
	else:
		momentrate, time = get_stf(name)

	if dataset == 'sigloch':
		momentrate_list = momentrate
		time_list = time
		print(momentrate_list[0])
	elif dataset == 'isc':
		momentrate_list = momentrate
		time_list = [time]
	else:
		momentrate_list = [momentrate]
		time_list = [time]
		
	count = 0
	for momentrate, time in zip(momentrate_list, time_list):
		if time[0] == time[1]:
			time = time[1:]

		if dataset != 'sigloch':
			save_key = row.event
			dataset_name = dataset
		else:
			dataset_name = dataset + '_' + str(count)
			save_key = row.event + '_' + str(count)

		# if save_key in to_ignore:
		# 	continue
		#print(time[0:10])
		
		momentrate = np.array(momentrate)

		time = np.array(time)


		detected_end_time, detected_end, detected_start_time, detected_start = find_end_stf(momentrate, time, dataset)
		if dataset == 'sigloch':
			print(row.event)
			print(detected_start_time, detected_start)
			print(detected_end_time, detected_end)
		time = time[detected_start:detected_end] # shift to start STF at zero
		time = time - time[0]
		momentrate = momentrate[detected_start:detected_end]
		#momentrate = momentrate - momentrate[0]
		# max_len = max(max_len, len(momentrate))
		norm_momentrate = momentrate #/ max(momentrate)
		norm_time = time #/ max(time)
		#axs[0].plot(norm_time, norm_momentrate)

		interp_momentrate_stf = np.interp(np.linspace(0, max(time), int(max(time)*100)), norm_time, norm_momentrate)
		#interp_momentrate = np.zeros(20000)
		#if max(time) < 200:
		#	interp_momentrate[0:int(max(time)*100)] = interp_momentrate_stf
		#else:
		#	interp_momentrate = interp_momentrate_stf[0:20000]
		#interp_momentrate[interp_momentrate < 0] = 0
		#print('interp_momentrate', interp_momentrate_stf)
		times_to_return.append(norm_time)#np.linspace(0, max(time), int(max(time)*100)))
		momentrates_to_return.append(norm_momentrate)#interp_momentrate_stf)
	return times_to_return, momentrates_to_return

def peaks_plot_overall(momentrate, momentrate_unsmoothed, time, peaks, name, dataset):
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	ax1.plot(time, momentrate, 'g-')
	ax2.plot(time, momentrate_unsmoothed, 'k-', alpha=0.5)
	ax1.plot(time[peaks], momentrate[peaks], 'ro')

	ax1.set_xlabel('Time')
	ax1.set_ylabel('Smoothed Moment Rate', color='g')
	ax2.set_ylabel('Unsmoothed Moment Rate', color='k')

	plt.title(f'{name} {dataset}')
	plt.savefig(f'{output_dir}{name}/{dataset}_peaks.png')
	plt.close()


smoothing_param = 10000 # 1/smoothing_param is the fraction of the length of the moment rate to smooth over
r2_limit = 0.95
height_threshold = 0.1 #peaks must be this high

output_dir = f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks_all/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

columns = ['event', 'scardec_opt_max_timing',  'scardec_moy_max_timing', 'ye_max_timing', 'usgs_max_timing', 'sigloch_0_max_timing', 'sigloch_1_max_timing', 'isc_max_timing']

peaks_dict = {}
count_datasets = {}
for i, row in combined.iterrows():
	# if i > 10:
	# 	break
	# 	continue
	print(row.event)
	# if row.event not in ['20011012_1502']:
	# 	continue
	# if i > 100:
	# 	break
	if not os.path.exists(output_dir+f'{row.event}/'):
		os.makedirs(output_dir+f'{row.event}/')
	#with open(f"/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks_all/R2_{r2_limit*10:02}/{row.event}/fits.txt", "w") as file:
	event_dict = {}
	#file.write(f"Processing {row.event}\n")	
	for dataset, get_stf in zip(['scardec_opt', 'scardec_moy', 'ye', 'usgs', 'sigloch', 'isc'], [get_scardec_stf, get_scardec_stf, get_ye_stf, get_usgs_stf, get_sigloch_stf, get_isc_stf]):
		#print(dataset)
		times_to_use, momentrates_to_use = get_data_and_prepare(row, dataset, get_stf)
		if times_to_use is None:
			#print('is none')
			continue
		else:
			if dataset not in count_datasets:
				count_datasets[dataset] = 1
			else:
				count_datasets[dataset] += 1
		sigloch_count = 0
		for time, momentrate in zip(times_to_use, momentrates_to_use):
			try:
				if dataset == 'sigloch':
					momentrate = 10**momentrate
					momentrate_sigloch = momentrate.copy()
					#file.write(f'Processing {dataset}_{sigloch_count}\n')
					name_to_save = f'{dataset}_{sigloch_count}'
					# print(name_to_save)
					# print(momentrate[0])
				else:
					#file.write(f'Processing {dataset}\n')
					name_to_save = dataset
				if np.argmax(momentrate) == 0:
					continue
				# plt.plot(time, momentrate)
				# plt.show()
			
				#smoothing_points = len(momentrate) // smoothing_param
				#if smoothing_points == 0:
				#	smoothing_points = 1
				momentrate_unsmoothed = momentrate.copy()
				#momentrate = smooth(momentrate, smoothing_points)
				momentrate = momentrate - momentrate[0]
				max_mr = max(momentrate)
				momentrate = momentrate / 10**int(np.log10(max_mr))

				# num_points_below_zero = np.sum(momentrate < 0)
				# if num_points_below_zero < 0.1 * len(momentrate):
				# 	continue
				# plt.plot(time, momentrate)
				# plt.show()
				popt_list = []
				r_squared_list = []
				#total_moment = scipy.integrate.simpson(momentrate, dx = time[1]-time[0])
				norm_momentrate = momentrate / max(momentrate)
				norm_time = time / max(time)
				interp_momentrate = np.interp(np.linspace(0, 1, 10000), norm_time, norm_momentrate)
				interp_momentrate[interp_momentrate < 0] = 0
				# plt.plot(np.linspace(0, 1, 10000), interp_momentrate)
				# plt.show()
				# break
				peak_timing = np.argmax(interp_momentrate)

				event_dict[name_to_save + '_max_timing'] = peak_timing

				#peaks_plot_overall(momentrate, momentrate_unsmoothed, time, peaks, row.event, name_to_save)
				#print(f'{dataset} peaks: {num_peaks}')
				if dataset == 'sigloch' and sigloch_count > 0:
					print('sigloch_1')


			except Exception as e:
				print(f'Error: {e}')
				#file.write(f'Error: {e}\n')
				continue
		sigloch_count += 1

	peaks_dict[row.event] = event_dict
	#print(peaks_dict.keys())
	if i % 100 == 0:
		print('Writing to file: ', i)
		df = pd.DataFrame.from_dict(peaks_dict, orient='index', columns=columns)
		print(df)
		df.to_csv(f'{output_dir}max_timing_unsmoothed.csv', header=False, mode='a')
		peaks_dict = {}


print(count_datasets)
print('Writing to file: ', i)
df = pd.DataFrame.from_dict(peaks_dict, orient='index', columns=columns)
print(df)
df.to_csv(f'{output_dir}max_timing_unsmoothed.csv', header=False, mode='a')
peaks_dict = {}

# final_df = pd.read_csv(f'{output_dir}peaks_dict.csv', names=columns)
# print(final_df)


