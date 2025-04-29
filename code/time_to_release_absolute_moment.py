# 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import pickle
import obspy
from scipy.signal import find_peaks
import os
import pandas as pd
from scipy.integrate import cumulative_trapezoid as cumtrapz


import cmcrameri.cm as cmc

from matplotlib import patches
import seaborn as sns


def get_scardec_stf(scardec_name, wanted_type = 'fctopt'):
	time = []
	momentrate = []

	event = os.listdir(f'/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec/{scardec_name}')
	starts = [n for n, l in enumerate(event) if l.startswith(wanted_type)]
	with open(f'/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec/{scardec_name}/{event[starts[0]]}') as f:
		lines = f.read().splitlines()

	lines = lines[2:]
	for line in lines:
		split = line.split(' ')
		split = [s for s in split if s not in ['', ' ', '\n']]
		time.append(float(split[0]))
		momentrate.append(float(split[1]))

	momentrate = np.array(momentrate)
	time = np.array(time)
	return momentrate, time

# 
def get_ye_stf(ye_name):
	data_path = '/home/earthquakes1/homes/Rebecca/phd/stf/data/Ye_et_al_2016/'
	momentrate = []
	time = []

	with open(data_path + str(ye_name), 'r') as f:
		data = f.readlines()
		for line in data:
			line = line.strip()
			line = line.rstrip()
			if line[0] not in ['0','1','2','3','4','5','6','7','8','9']:
				continue
			line = line.split()
			time.append(float(line[0]))
			momentrate.append(float(line[1]))
	momentrate = np.array(momentrate)
	time = np.array(time)
	return momentrate, time

# 
def get_usgs_stf(usgs_name):
	data_path = '/home/earthquakes1/homes/Rebecca/phd/stf/data/USGS/'
	momentrate = []
	time = []

	with open(data_path + str(usgs_name), 'r') as f:
		data = f.readlines()
		for line in data:
			line = line.strip()
			line = line.rstrip()
			if line[0] not in ['0','1','2','3','4','5','6','7','8','9']:
				continue
			line = line.split()
			time.append(float(line[0]))
			momentrate.append(float(line[1]))

	momentrate = np.array(momentrate)
	time = np.array(time)

	if usgs_name == '19950205_225105.txt' or usgs_name == '20041226_005853.txt':
		momentrate = momentrate
	elif int(usgs_name[0:4]) < 2021:
		momentrate = momentrate / 10**7 # convert to Nm from dyne cm
	elif int(usgs_name[0:4]) == 2021:
		if int(usgs_name[4:6]) < 5:
			momentrate = momentrate / 10**7
	else:
		momentrate = momentrate
	return momentrate, time

# 
def get_sigloch_stf(sigloch_name):
	data_path = '/home/siglochnas1/shared/AmplitudeProjects/pdata_processed/psdata_events/'
	momentrate = []
	time = []

	file_path = data_path + str(sigloch_name) + '/outfiles/ampinv.stf.xy'

	with open(file_path, 'r') as file:
		content = file.read()
		content = content.split('\n')
		greater_than_count = content.count('>')
		if greater_than_count > 0:
			time = [list(np.arange(0, 25.6, 0.1))]
			momentrate = [[]]
			for i in range(greater_than_count-1):
				time.append(list(np.arange(0, 25.6, 0.1)))
				momentrate.append([])


		stf_count = 0
		for c in content:
			if c not in ['<', '>', '']:
				split = c.split()
				#time[stf_count].append(float(split[0]))
				momentrate[stf_count].append(10**float(split[1]))
			else:
				stf_count += 1

	# time = np.arange(0, 25.6, 0.1)
	# time = np.array(time)
	return momentrate, time

# 
def get_isc_stf(isc_name):
	isc_save_path = '/home/earthquakes1/homes/Rebecca/phd/stf/data/isc/'
	with open(f'{isc_save_path}{isc_name}/{isc_name}.txt', 'rb') as f:
		stf_list = pickle.load(f)
	with open(f'{isc_save_path}{isc_name}/{isc_name}_norm_info.txt', 'rb') as f:
		norm_dict = pickle.load(f)

	time = np.arange(0, 25.6, 0.1)
	momentrate = np.array(stf_list)*norm_dict['mo_norm']*10**8,
	#print(momentrate)
	return momentrate[0], time

# 
combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')

# 
combined.columns = ['event', 'scardec', 'ye', 'isc', 'sigloch', 'usgs', 'mag']

# 
def myround(x, base=1):
	return base * round(x/base)

# 
def find_end_stf(momentrate, time, dataset = ''):
	not_zero = np.where(momentrate > 0)[0]
	#print(max(momentrate))
	start = min(not_zero)
	end = max(not_zero)

	detected_end = end
	detected_end_time = time[end]

	time = time[:end]
	momentrate = momentrate[:end]

	less_than_10 = np.where(momentrate <= 10*max(momentrate)/100)[0]

	if dataset == 'sigloch':
		start = np.where(momentrate > 0.05 * max(momentrate))[0][0]
	else:
		start = min(not_zero)
	#print(less_than_10)
	total_moment = scipy.integrate.simpson(momentrate[start:end],
										dx = time[1]-time[0])
	#print(less_than_10)
	for i in less_than_10:
		if i <= start:
			continue
		if i == 0:
			continue
		moment = scipy.integrate.simpson(momentrate[start:i],
										dx = time[1]-time[0])
		#print(i, moment/total_moment)
		if moment >= 0.5 * total_moment:
			#print('inif')
			#print(f'first time where < 10% of total momentrate and 50% of moment released: {time[i]} s')
			detected_end_time = time[i]
			detected_end = i
			#print(f'proportion of moment released: {(moment/total_moment)*100:.2f}%')
			break
	return detected_end_time, detected_end, time[start], start
	#return time[end], end

# 
# looks for time value of root
def f3(end_time, total_moment, time_opt, momentrate_opt, start, points_before_zero, proportion = 0.1):
	dx = time_opt[1]-time_opt[0]
	end_window = (end_time/dx)+points_before_zero
	end = int(np.floor(end_window))
	if start == end:
		end += 1
	short = scipy.integrate.simpson(momentrate_opt[start:end], dx = dx)
	return short-(total_moment*proportion)

# 
def moment_in_different_windows(window = None, window_prop = None, combined=None):
#def moment_in_different_windows(window = None, window_prop = None, combined=None):
	#window = 1
	#window_prop = None
	if combined is None:
		combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')
		combined.columns = ['event', 'scardec', 'ye', 'isc', 'sigloch', 'usgs', 'mag']
	if window is None and window_prop is None:
		window_prop = 1

	simpson = []

	simpson_short = []

	durations = []

	magnitudes = []

	datasets = []

	names = []

	depths = []
	
	to_ignore = ['20051203_1610_1', '20071226_2204_2', '20030122_0206_1', '20090929_1748_0', '20120421_0125_1', '20110311_2011_2']

	for i, row in combined.iterrows():

		for dataset, get_stf in zip(['scardec_opt', 'scardec_moy', 'ye', 'usgs', 'sigloch', 'isc'], [get_scardec_stf, get_scardec_stf, get_ye_stf, get_usgs_stf, get_sigloch_stf, get_isc_stf]):
		#for dataset, get_stf in zip(['sigloch'], [get_sigloch_stf]):

			if dataset == 'scardec_moy' or dataset == 'scardec_opt':
				name = row[dataset[:-4]]
			else:
				name = row[dataset]

			if name == '0' or name == 0:
				continue

			if dataset == 'scardec_moy':
				momentrate, time = get_stf(name, 'fctmoy')
			elif dataset == 'scardec_opt':
				momentrate, time = get_stf(name, 'fctopt')
			else:
				momentrate, time = get_stf(name)

			if dataset != 'sigloch':
				momentrate_list = [momentrate]
				time_list = [time]
			else:
				momentrate_list = momentrate
				time_list = time

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

				if save_key in to_ignore:
					continue
				
				momentrate = np.array(momentrate)

				time = np.array(time)
				detected_end_time, detected_end, detected_start_time, detected_start = find_end_stf(momentrate, time, dataset)
				time = time[detected_start:detected_end] # shift to start STF at zero
				
				start = 0
				#end = len(momentrate)
				duration = time[-1] - time[0]
				#durations.append(duration)
				momentrate = momentrate[detected_start:detected_end]
				start = 0
				#end = len(momentrate)
				duration = time[-1] - time[0]
				durations.append(duration)
				end = len(momentrate)
				dx = time[1]-time[0]
				
				simpson.append(scipy.integrate.simpson(momentrate[start:end], dx = time[1]-time[0]))

				if window_prop is None: #using static time window
					end_window = int(round((window/dx), 0))    #int((end-start)*(window/duration))
				else: #based on proportion of duration
					end_window = int((duration)*window_prop)

				# print(duration, window, end_window, end, dx)
				# print(start, start+end_window)
				# print(dx * window)

				if window < dx:
					ynew = np.interp(np.linspace(0, dx*2, window), time[0:2], momentrate[0:2])
					simpson_short.append(scipy.integrate.simpson(ynew[0:1]))

				#if duration == end_window:
				
				else:
					simpson_short.append(scipy.integrate.simpson(momentrate[start:start + end_window], dx = time[1]-time[0]))

				if row.event in catalog['event'].values:
					event_cat = catalog[catalog['event']==row.event]
					print(row.event)
					print(event_cat)
					magnitudes.append(event_cat['magnitude'].values[0])
					depths.append(event_cat['depth/km'].values[0])
				else:
					magnitudes.append(row.mag)
					depths.append(np.nan)

				datasets.append(dataset_name)
				names.append(name)
	return names, simpson, simpson_short, durations, magnitudes, datasets, depths


def myround(x, base=1):
	return base * round(x/base)

# 
catalog = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_m55_catalog.csv', sep = '|')
cols = catalog.columns
column_names = []
for c in cols:
	column_names.append(c.strip().rstrip().lower())
column_names[0] = 'catalog_id'
catalog.columns = column_names
catalog['year'] = catalog.apply(lambda x: x['time'][:4], axis = 1)
catalog['month'] = catalog.apply(lambda x: x['time'][5:7], axis = 1)
catalog['day'] = catalog.apply(lambda x: x['time'][8:10], axis = 1)
catalog['hour'] = catalog.apply(lambda x: x['time'][11:13], axis = 1)
catalog['minute'] = catalog.apply(lambda x: x['time'][14:16], axis = 1)

catalog['event'] = catalog.apply(lambda x: x['year'] + x['month'] + x['day'] + '_' + x['hour'] + x['minute'], axis = 1)

catalog['int_magnitude'] = catalog.apply(lambda x: myround(x['magnitude']), axis = 1)
catalog.drop(columns = ['contributor', 'contributorid', 'magauthor', 'eventlocationname', 'author', 'catalog', 'time'], inplace = True)
catalog = catalog[['event', 'catalog_id', 'year', 'month', 'day', 'hour', 'minute', 'latitude', 'longitude', 'depth/km', 'magnitude', 'int_magnitude', 'magtype']]

 

#def moment_in_different_windows(window = None, window_prop = None, combined=None):
#def moment_in_different_windows(window = None, window_prop = None, combined=None):
window = 1
window_prop = None

if combined is None:
	combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')
	combined.columns = ['event', 'scardec', 'ye', 'isc', 'sigloch', 'usgs', 'mag']
if window is None and window_prop is None:
	window_prop = 1



to_ignore = ['20051203_1610_1', '20071226_2204_2', '20030122_0206_1', '20090929_1748_0', '20120421_0125_1', '20110311_2011_2']
results = {}
for target_mag in [5.2]:#np.arange(4, 7.1, 0.25):
	results[str(target_mag)] = {}
	simpson = []
	simpson_short = []
	durations = []
	magnitudes = []
	datasets = []
	names = []
	depths = []

	reaches_target = []
	print(target_mag)
	for i, row in combined.iterrows():

		for dataset, get_stf in zip(['scardec_opt', 'scardec_moy', 'ye', 'usgs', 'sigloch', 'isc'], [get_scardec_stf, get_scardec_stf, get_ye_stf, get_usgs_stf, get_sigloch_stf, get_isc_stf]):
		#for dataset, get_stf in zip(['sigloch'], [get_sigloch_stf]):

			if dataset == 'scardec_moy' or dataset == 'scardec_opt':
				name = row[dataset[:-4]]
			else:
				name = row[dataset]

			if name == '0' or name == 0:
				continue

			if dataset == 'scardec_moy':
				momentrate, time = get_stf(name, 'fctmoy')
			elif dataset == 'scardec_opt':
				momentrate, time = get_stf(name, 'fctopt')
			else:
				momentrate, time = get_stf(name)

			if dataset != 'sigloch':
				momentrate_list = [momentrate]
				time_list = [time]
			else:
				momentrate_list = momentrate
				time_list = time

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

				if save_key in to_ignore:
					continue
				
				momentrate = np.array(momentrate)

				time = np.array(time)
				detected_end_time, detected_end, detected_start_time, detected_start = find_end_stf(momentrate, time, dataset)
				time = time[detected_start:detected_end] # shift to start STF at zero
				time = time - time[0]
				
				start = 0
				#end = len(momentrate)
				duration = time[-1] - time[0]
				#durations.append(duration)
				momentrate = momentrate[detected_start:detected_end]
				start = 0
				#end = len(momentrate)
				duration = time[-1] - time[0]
				
				end = len(momentrate)
				dx = time[1]-time[0]
				simpson_result = scipy.integrate.simpson(momentrate[start:end], dx = time[1]-time[0])
				if simpson_result < 10**16:
					continue

				simpson.append(simpson_result)

				durations.append(duration)
				# Calculate the cumulative integral of mr
				cumulative_integral = cumtrapz(momentrate[start:end], time[start:end], initial=0)
				#print(cumulative_integral)

				# Find the time at which the cumulative integral equals 10**20
				target_value = 10**((3/2)*target_mag + 9.1)
				if cumulative_integral[-1] >= target_value:
					target_index = np.where(cumulative_integral >= target_value)[0][0]
					target_time = time[target_index]
				else:
					target_time = None
					target_index = None

				reaches_target.append(target_time)

				#print(f'The time at which the integral of mr equals {target_value} is approximately {target_time} seconds ({target_mag}).')


				if row.event in catalog['event'].values:
					event_cat = catalog[catalog['event']==row.event]
					#print(row.event)
					#print(event_cat)
					magnitudes.append(event_cat['magnitude'].values[0])
					depths.append(event_cat['depth/km'].values[0])
				else:
					magnitudes.append(row.mag)
					depths.append(np.nan)

				datasets.append(dataset_name)
				names.append(name)


	# 
	simpson = np.array(simpson)
	reaches_target = np.array(reaches_target)
	durations = np.array(durations)
	magnitudes = np.array(magnitudes)
	depths = np.array(depths)


	# 
	colors = cmc.batlow(np.linspace(0, 1, 5))
	unique_datasets = ['scardec', 'usgs', 'sigloch', 'ye', 'isc']
	dataset_colors = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}

	# 
	datasets_for_colors = []
	for d in datasets:
		datasets_for_colors.append(d.split('_')[0])

	# 
	len(names), len(simpson), len(durations), len(magnitudes), len(datasets), len(depths)

	# 
	db = pd.DataFrame({'name': names,
					'magnitude': magnitudes, 
					'depth': depths,
					'simpson': simpson, 
					'reaches_target': reaches_target, 
					'duration': durations,
					'dataset': datasets,
					'dataset_for_colors': datasets_for_colors})


	plt.scatter(db['magnitude'], db['reaches_target'], color=[dataset_colors[d.split('_')[0]] for d in db['dataset']], alpha=0.2)
	plt.ylabel(f'Time to release moment equivalent to M{target_mag}')
	plt.xlabel('Magnitude')
	# Fit a line of best fit
	# Remove rows with NaN values in 'magnitude' or 'reaches_target' before fitting
	clean_db = db.dropna(subset=['magnitude', 'reaches_target'])
	print(len(clean_db))
	m, b = np.polyfit(clean_db['magnitude'].astype('float'), clean_db['reaches_target'].astype('float'), 1)
	# Calculate Spearman correlation coefficient
	spearman_r, spearman_p = stats.spearmanr(clean_db['magnitude'], clean_db['reaches_target'])
	#print(f'Spearman correlation: {spearman_r}, p-value: {spearman_p}')
	plt.plot(db['magnitude'], m*db['magnitude'] + b, color='red', linestyle='--', label=f'Best fit: y = {m:.2f}x + {b:.2f} (Spearman r = {spearman_r:.2f}, p = {spearman_p:.2f})')
	plt.legend()
	plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/time_for_absolute_moment/time_to_reach_target_{target_mag}.png')
	plt.close()
	# 
	plt.scatter(db['magnitude'], db['reaches_target'], color=[dataset_colors[d.split('_')[0]] for d in db['dataset']], alpha=0.2)
	plt.ylabel(f'Time to release moment equivalent to M{target_mag}')
	plt.xlabel('Magnitude')
	clean_db = db.dropna(subset=['magnitude', 'reaches_target'])
	clean_db['reaches_target'] = pd.to_numeric(clean_db['reaches_target'], errors='coerce')
	for dataset in clean_db['dataset'].unique():
		subset = clean_db[clean_db['dataset'] == dataset]
		subset = subset[subset['reaches_target'] > 0]
		plt.scatter(subset['magnitude'], subset['reaches_target'], color=[dataset_colors[d.split('_')[0]] for d in subset['dataset']], alpha=0.2)
		m, b = np.polyfit(subset['magnitude'].astype('float'), subset['reaches_target'].astype('float'), 1)
		spearman_r, spearman_p = stats.spearmanr(subset['magnitude'], subset['reaches_target'])
		if dataset.startswith('scardec'):
			dataset_color = 'scardec'
		#print(f'Spearman correlation: {spearman_r}, p-value: {spearman_p}')
		plt.plot(subset['magnitude'], m*subset['magnitude'] + b, color=dataset_colors[dataset.split('_')[0]], linestyle='--', label=f'{dataset} Best fit: y = {m:.2f}x + {b:.2f} (Spearman r = {spearman_r:.2f}, p = {spearman_p:.2f})')
		results[str(target_mag)][dataset+'_m'] = m
		results[str(target_mag)][dataset+'_b'] = b
		results[str(target_mag)][dataset+'_spearman_r'] = spearman_r
		results[str(target_mag)][dataset+'_spearman_p'] = spearman_p
	print(results)
	plt.legend()
	#plt.show()
	plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/time_for_absolute_moment/time_to_reach_target_{target_mag}_each_dataset_line.png')
	plt.close()

results_db = pd.DataFrame(results)
results_db.to_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/time_to_reach_target_results.csv')
print(results_db)


# 



