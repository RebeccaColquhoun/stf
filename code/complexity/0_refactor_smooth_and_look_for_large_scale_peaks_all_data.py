
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

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.optimize import curve_fit

import scipy.integrate

from functions_load_stf import *
from functions_end_stf import find_end_stf



combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')


window = 2


# Define a sum of Gaussian functions
def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    #print('params', params)
    for i in range(0, len(params), 3):
        #print(params[i:i+3])
        amp = params[i]
        mean = params[i+1]
        stddev = params[i+2]
        y += amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    return y

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def plot_gaussians(popt, ax, time, momentrate, r_squared):
    if np.array_equal(popt, [0, 0, 0]):
        ax.plot(time, momentrate*1e18,
                    label='Data',
                    linestyle='-',
                    color='black',
                    zorder = 1)
        return
    num_gaussians = len(popt) // 3
    params = []
    for i in range(num_gaussians):
        if i < 10:
            ls = '--'
        else:
            ls = ':'
        amp = popt[0 + i*3]
        mean = popt[1 + i*3]
        stddev = popt[2 + i*3]
        params.append((amp, mean, stddev))
        x = time
        y = amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
        ax.plot(time, y*1e18,
                        #label=f'Gaussian {i+1}',
                        linestyle=ls,
                        zorder = 10)

    ax.plot(time, momentrate*1e18,
                        label='Data',
                        linestyle='-',
                        color='black')
    ax.plot(time, multi_gaussian(time, *popt)*1e18,
                            label=fr'Sum of {num_gaussians} Gaussians, $R^2$ = {r_squared:.3f}',
                            linestyle='-',
                            zorder = 1,
                            color='red')
    ax.legend()
    return


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


		detected_end_time, detected_end, detected_start_time, detected_start = find_end_stf(momentrate, time)
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
		times_to_return.append(np.linspace(0, max(time), int(max(time)*100)))
		momentrates_to_return.append(interp_momentrate_stf)
	return times_to_return, momentrates_to_return


def fit_gaussians(gaussian_num, time, momentrate):
	initial_guess = []
	if gaussian_num > 1:
		for i in range(gaussian_num):
			initial_guess.extend([1, (max(time)/(gaussian_num+1)) * (i+1), 1])
	else:
		initial_guess = [max(momentrate), time[np.argmax(momentrate)], 1]

	bounds = ([0 if i % 3 == 0 else -np.inf for i in range(len(initial_guess))],
			[np.inf for _ in range(len(initial_guess))])
	try:
		popt, pcov = curve_fit(multi_gaussian,
						time,
						momentrate,
						p0=initial_guess,
						bounds=bounds,
						maxfev=5000)
		y_fit = multi_gaussian(time, *popt)
		residuals = momentrate - y_fit
		ss_res = np.sum(residuals**2)
		ss_tot = np.sum((momentrate - np.mean(momentrate))**2)
		r_squared = 1 - (ss_res / ss_tot)
		
	except RuntimeError:
		r_squared = 0
		popt = [0, 0, 0]
		print('Optimal parameters not found')
	return r_squared, popt


def gaussians_plot_overall(popt_list, r_squared_list, time, momentrate, event, r2_limit, dataset, max_mr = 0):
	if len(popt_list) == 1:
		fig, axs = plt.subplots(1,1, figsize=(10, 10), sharex=True)
		for i in range(0, len(popt_list)):
			popt = popt_list[i]
			plot_gaussians(popt, axs, time, momentrate, r_squared_list[0])
	elif len(popt_list) < 6:
		# print('in elif')
		fig, axs = plt.subplots(len(popt_list),1, figsize=(10, 10), sharex=True)
		for j in range(0, len(popt_list)):
			popt = popt_list[j]
			plot_gaussians(popt, axs[j], time, momentrate, r_squared_list[j])
	else:
		# print(len(popt_list))
		# print(event)
		if len(popt_list) % 2 == 0:
			#print('in if')
			fig, axs = plt.subplots(((len(popt_list) + 1) // 2) + 1, 1, figsize=(10, 10), sharex=True)
		else:
			#print('in else')
			fig, axs = plt.subplots(((len(popt_list)) // 2) + 1, 1, figsize=(10, 10), sharex=True)
		for j in range(0, len(popt_list), 2):
			popt = popt_list[j]
			plot_gaussians(popt, axs[j//2], time, momentrate, r_squared_list[j])
		# Plot the last element in popt_list
		if len(popt_list) % 2 == 0:
			popt = popt_list[-1]
			plot_gaussians(popt, axs[-1], time, momentrate, r_squared_list[j])


	if r_squared_list[-1] < r2_limit:
		plt.suptitle(f'{dataset} -- {event} - R-squared < {r2_limit}')
	else:
		plt.suptitle(f'{dataset} -- {event}')

	# for ax in axs:
	# 	ax.set_ylim(-1, 1)

	# Add a shared y-axis label by adding a subplot
	fig.add_subplot(111, frame_on=False)
	plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
	plt.xlabel("Time (s)")
	plt.ylabel(f"Moment Rate (x10^{max_mr} Nm/s)")
	plt.tight_layout()
	plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks_all/R2_{r2_limit*10:02}/{event}/{dataset}.png')
	# if len(popt_list) > 4:
	#plt.show()
	plt.close()


a = {'event1':{'a':1, 'b':2}, 'event2':[{'a':5, 'b':6}, {'a':7, 'b':8}]}


a


smoothing_points = 1
r2_limit = 0.95

output_dir = f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks_all/R2_{r2_limit*10:02}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

columns = ['event', 'scardec_opt_r2', 'scardec_opt_num_gaussians', 'scardec_moy_r2', 'scardec_moy_num_gaussians', 'ye_r2', 'ye_num_gaussians', 'usgs_r2', 'usgs_num_gaussians', 'isc_r2', 'isc_num_gaussians', 'sigloch_0_r2', 'sigloch_0_num_gaussians', 'sigloch_1_r2', 'sigloch_1_num_gaussians', 'sigloch_2_r2', 'sigloch_2_num_gaussians', 'sigloch_3_r2', 'sigloch_3_num_gaussians', 'sigloch_4_r2', 'sigloch_4_num_gaussians', 'sigloch_5_r2', 'sigloch_5_num_gaussians', 'sigloch_6_r2', 'sigloch_6_num_gaussians', 'sigloch_7_r2', 'sigloch_7_num_gaussians', 'sigloch_8_r2', 'sigloch_8_num_gaussians', 'sigloch_9_r2', 'sigloch_9_num_gaussians']

gaussians_dict = {}
for i, row in combined.iterrows():
	# 	continue
	print(row.event)
	# if row.event not in ['20011012_1502']:
	# 	continue
	# if i >= 100:
	# 	break
	if not os.path.exists(output_dir+f'{row.event}/'):
		os.makedirs(output_dir+f'{row.event}/')
	with open(f"/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks_all/R2_{r2_limit*10:02}/{row.event}/fits.txt", "w") as file:
		event_dict = {}
		file.write(f"Processing {row.event}\n")	
		for dataset, get_stf in zip(['scardec_opt', 'scardec_moy', 'ye', 'usgs', 'sigloch', 'isc'], [get_scardec_stf, get_scardec_stf, get_ye_stf, get_usgs_stf, get_sigloch_stf, get_isc_stf]):
			#print(dataset)
			times_to_use, momentrates_to_use = get_data_and_prepare(row, dataset, get_stf)
			if times_to_use is None:
				#print('is none')
				continue
			sigloch_count = 0
			for time, momentrate in zip(times_to_use, momentrates_to_use):
				try:
					if dataset == 'sigloch':
						momentrate = 10**momentrate
						momentrate_sigloch = momentrate.copy()
						file.write(f'Processing {dataset}_{sigloch_count}\n')
						name_to_save = f'{dataset}_{sigloch_count}'
						# print(name_to_save)
						# print(momentrate[0])
					else:
						file.write(f'Processing {dataset}\n')
						name_to_save = dataset
					if np.argmax(momentrate) == 0:
						continue
					# plt.plot(time, momentrate)
					# plt.show()
				
					smoothing_points = len(momentrate) // 20
					if smoothing_points == 0:
						smoothing_points = 1
					momentrate = smooth(momentrate, smoothing_points)
					momentrate = momentrate - momentrate[0]
					max_mr = max(momentrate)
					momentrate = momentrate / 10**int(np.log10(max_mr))

					num_points_below_zero = np.sum(momentrate < 0)
					if num_points_below_zero > 0.1 * len(momentrate):
						continue
					# plt.plot(time, momentrate)
					# plt.show()
					popt_list = []
					r_squared_list = []
					total_moment = scipy.integrate.simpson(momentrate, dx = time[1]-time[0])
					for gaussian_num in range(1, 11):
						#print(gaussian_num)
						file.write(f'Fitting {gaussian_num} Gaussians\n')
						#print(momentrate[0])
						r_squared, popt = fit_gaussians(gaussian_num, time, momentrate)
						r_squared_list.append(r_squared)
						popt_list.append(popt)
						#print(r_squared)
						#print(popt)

						num_gaussians = len(popt) // 3
						params = []
						for j in range(num_gaussians):
							amp = popt[j*3]
							mean = popt[j*3 + 1]
							stddev = popt[j*3 + 2]
							params.append((amp, mean, stddev))
							x = time
							y = amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
							file.write(f"Gaussian {i+1} - Amplitude: {amp}, Mean: {mean}, Standard Deviation: {stddev}\n")
						file.write(f'R-squared: {r_squared}\n')
						#print(params)
						moment_gaussian = scipy.integrate.simpson(multi_gaussian(time, *popt), dx = time[1]-time[0])
						file.write(f'Proportion moment from Gaussian fit: {moment_gaussian/total_moment}\n')
						if r_squared > r2_limit:
							break

					event_dict[name_to_save + '_r2'] = r_squared_list[-1]
					event_dict[name_to_save + '_num_gaussians'] = len(popt)//3

					if r_squared_list[-1] < r2_limit:
						#print(row.event, f'r2 < {r2_limit}')
						file.write(f'!!!!!!!!!!!! R-squared < {r2_limit} !!!!!!!!!!!!!!!!!!!!\n')
						#print(popt_list)

					gaussians_plot_overall(popt_list, r_squared_list, time, momentrate, row.event, r2_limit, name_to_save, int(np.log10(max_mr)))
				except Exception as e:
					print(f'Error: {e}')
					file.write(f'Error: {e}\n')
					continue
			sigloch_count += 1
	#print(event_dict)
	gaussians_dict[row.event] = event_dict
	if i % 100 == 0:
		print('Writing to file: ', i)
		df = pd.DataFrame.from_dict(gaussians_dict, orient='index', columns=columns)
		df.to_csv(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks_all/R2_{r2_limit*10:02}/gaussians_dict.csv', header=False, mode='a')
		gaussians_dict = {}



