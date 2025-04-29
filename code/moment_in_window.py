import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import pickle
import os
import pandas as pd
import cmcrameri.cm as cmc

def myround(x, base=1):
	return base * round(x/base)


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


# looks for time value of root
def f3(end_time, total_moment, time_opt, momentrate_opt, start, points_before_zero, proportion = 0.1):
	dx = time_opt[1]-time_opt[0]
	end_window = (end_time/dx)+points_before_zero
	end = int(np.floor(end_window))
	if start == end:
		end += 1
	short = scipy.integrate.simpson(momentrate_opt[start:end], dx = dx)
	return short-(total_moment*proportion)


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
					# print(row.event)
					# print(event_cat)
					magnitudes.append(event_cat['magnitude'].values[0])
					depths.append(event_cat['depth/km'].values[0])
				else:
					magnitudes.append(row.mag)
					depths.append(np.nan)

				datasets.append(dataset_name)
				names.append(name)
	return names, simpson, simpson_short, durations, magnitudes, datasets, depths

combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')


combined.columns = ['event', 'scardec', 'ye', 'isc', 'sigloch', 'usgs', 'mag']
for max_depth_lim in [6378]:
	for min_depth_lim in [0]:#, 30, 70]:
		if min_depth_lim >= max_depth_lim:
			continue
		for time in [1, 2, 3, 4, 5, 10]:
			names, simpson, simpson_short, durations, magnitudes, datasets, depths = moment_in_different_windows(window = time)


			simpson = np.array(simpson)
			simpson_short = np.array(simpson_short)
			durations = np.array(durations)
			magnitudes = np.array(magnitudes)
			depths = np.array(depths)


			colors = cmc.batlow(np.linspace(0, 1, 5))
			unique_datasets = ['scardec', 'usgs', 'sigloch', 'ye', 'isc']
			dataset_colors = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}


			datasets_for_colors = []
			for d in datasets:
				datasets_for_colors.append(d.split('_')[0])

			db = pd.DataFrame({'name': names,
								'magnitude': magnitudes, 
								'depth': depths,
								'simpson': simpson, 
								'simpson_short': simpson_short, 
								'dataset': datasets_for_colors})


			subset = db[db['simpson'] > 10**16]
			subset = subset[subset['simpson_short'] > 0]

			spearman_r, spearman_p = stats.spearmanr(subset['simpson'], subset['simpson_short'])
			print(f'Spearman correlation: {spearman_r}, p-value: {spearman_p}')

			m, b = np.polyfit(np.log10(subset['simpson']), np.log10(subset['simpson_short']), 1)
			#m, b = np.polyfit(simpson, simpson_short, 1)


			colors = cmc.batlow(np.linspace(0, 1, 5))
			unique_datasets = ['scardec', 'usgs', 'sigloch', 'ye', 'isc']
			dataset_colors = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}
			
			shallow = db[db['depth'] < max_depth_lim]
			shallow = shallow[shallow['depth'] > min_depth_lim]

			plt.figure(figsize=(10, 6))
			for dataset in unique_datasets:
				subset = shallow[shallow['dataset'] == dataset]
				subset = subset[subset['simpson'] > 10**16]
				subset = subset[subset['simpson_short'] > 0]
				if dataset != 'usgs':
					subset_no_nan = subset.dropna()
					subset = subset_no_nan[abs(subset_no_nan['magnitude']-2/3*(np.log10(subset_no_nan['simpson'])-9.1)) < 1]
				plt.scatter(subset['simpson'], subset['simpson_short'], label=dataset, alpha=0.5, facecolors='none', edgecolors=dataset_colors[dataset])

				spearman_r, spearman_p = stats.spearmanr(subset['simpson'], subset['simpson_short'])
				print(f'Spearman correlation: {spearman_r}, p-value: {spearman_p}')

				m, b = np.polyfit(np.log10(subset['simpson']), np.log10(subset['simpson_short']), 1)
				#m, b = np.polyfit(simpson, simpson_short, 1)

				plt.plot(np.array([10e16, 10e19, 10e23]),
						10**(m * np.log10(np.array([10e16, 10e19, 10e23])) + b),
						c=dataset_colors[dataset],
						linestyle = '-')
				print(dataset)
				ratio = subset['simpson_short']/subset['simpson']
				print(np.histogram(ratio, bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
				print(np.count_nonzero(ratio > 0.9))
				print(len(subset))



			plt.xlabel('Moment in full STF')
			plt.ylabel(f'Moment in first {time} seconds')

			plt.plot([10**16, 10**25], [10**16, 10**25], ls='--', c='k')
			plt.plot([10**16, 10**25], 0.5*np.array([10**16, 10**25]), ls=':', c='k')

			subset = shallow[shallow['simpson'] > 10**16]
			subset = subset[subset['simpson_short'] > 0]

			spearman_r, spearman_p = stats.spearmanr(subset['simpson'], subset['simpson_short'])
			print(f'Spearman correlation: {spearman_r}, p-value: {spearman_p}')

			m, b = np.polyfit(np.log10(subset['simpson']), np.log10(subset['simpson_short']), 1)
			#m, b = np.polyfit(simpson, simpson_short, 1)

			plt.plot(np.array([10e15, 10e19, 10e24]),
					10**(m * np.log10(np.array([10e15, 10e19, 10e24])) + b),
					c='hotpink',
					label = fr'Best fit: $\log_{{10}}(M_0)$ = {m:.2f}$\log_{{10}}(M_0)$ + {b:.2f}' + '\n' + f'(Spearman r = {spearman_r:.2f}, p = {spearman_p:.2f})',
					linestyle = '-')
			# for fraction in np.arange(0, 1.0, 0.1):
			# 	plt.plot(np.array([10e15, 10e19, 10e24]),
			# 		fraction*np.array([10e15, 10e19, 10e24]),
			# 		c='k',
			# 		label = '90% of ',
			# 		linestyle = ':')

			plt.title(f'{min_depth_lim} < Depth < {max_depth_lim} km')
			plt.yscale('log')
			plt.xscale('log')

			#plt.xlim(5,10)

			plt.legend()
			plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_in_absolute_time/moment_in_{time}_seconds_min_depth_{min_depth_lim}_max_depth_{max_depth_lim}_90.png')
			plt.close()

			#-----------------------

			datasets_for_colors_2 = []
			for d in datasets:
				if d.split('_')[0] == 'sigloch':
					datasets_for_colors_2.append(d.split('_')[0])
				else:
					datasets_for_colors_2.append(d)


			db_2 = pd.DataFrame({'name': names, 'depth': depths, 'magnitude': magnitudes, 'simpson': simpson, 'simpson_short': simpson_short, 'dataset': datasets_for_colors_2})


			colors = cmc.batlow(np.linspace(0, 1, 5))
			unique_datasets = ['scardec', 'usgs', 'sigloch', 'ye', 'isc']
			dataset_colors = {dataset: colors[i] for i, dataset in enumerate(unique_datasets)}

			unique_datasets_2 = ['scardec_opt', 'scardec_moy', 'usgs', 'sigloch', 'ye', 'isc']

			fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True, sharex=True, figsize=(7, 8))
			shallow = db_2[db_2['depth'] < max_depth_lim]
			shallow = shallow[shallow['depth'] > min_depth_lim]
			
			
			for i, dataset in enumerate(unique_datasets_2):
				ax = axes[i//2, i%2]
				subset = shallow[shallow['dataset'] == dataset]
				subset = subset[subset['simpson'] > 10**16]
				subset = subset[subset['simpson_short'] > 0]
				if dataset != 'usgs':
					subset_no_nan = subset.dropna()
					subset = subset_no_nan[abs(subset_no_nan['magnitude']-2/3*(np.log10(subset_no_nan['simpson'])-9.1)) < 1]
				ax.scatter(subset['simpson'], subset['simpson_short'], label=dataset, alpha=0.7, facecolors='none', edgecolors=dataset_colors[dataset.split('_')[0]])
				#ax.set_xlim(5, 10)
				ax.set_yscale('log')
				ax.set_xscale('log')


				spearman_r, spearman_p = stats.spearmanr(subset['simpson'], subset['simpson_short'])
				print(f'Spearman correlation: {spearman_r}, p-value: {spearman_p}')

				m, b = np.polyfit(np.log10(subset['simpson']), np.log10(subset['simpson_short']), 1)
				#m, b = np.polyfit(simpson, simpson_short, 1)

				ax.plot(np.array([10e16, 10e19, 10e23]),
						10**(m * np.log10(np.array([10e16, 10e19, 10e23])) + b),
						c=dataset_colors[dataset.split('_')[0]],
						label = fr'$\log_{{10}}(M_0)$ = {m:.2f}$\log_{{10}}(M_0)$ + {b:.2f}' + '\n' + f'(Spearman r = {spearman_r:.2f}, p = {spearman_p:.2f})',
						linestyle = '-')



				ax.plot([10**16, 10**25], [10**16, 10**25], ls='--', c='k')
				# for fraction in np.arange(0, 1.0, 0.1):
				# 	ax.plot([10**16, 10**18, 10**20, 10**22, 10**24, 10**25], fraction*np.array([10**16, 10**18, 10**20, 10**22, 10**24, 10**25]), ls=':', c='k')

				ax.legend(fontsize = 8)

				if i%2 == 0:
					ax.set_ylabel(f'Moment in first {time} seconds')
				if i//2 == 2:
					ax.set_xlabel('Total Moment in STF')
			#fig.suptitle(f'{min_depth_lim} < Depth < {max_depth_lim} km')
			plt.tight_layout()
			plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_in_absolute_time/moment_in_{time}_seconds_dataset_min_depth_{min_depth_lim}_max_depth_{max_depth_lim}.png', bbox_inches='tight', dpi=300)
			plt.close()
			#plt.show()