
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import pandas as pd
import pickle
import cmcrameri.cm as cmc
import seaborn as sns
import math
from scipy.optimize import curve_fit


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

    momentrate = momentrate / 10**7 # convert to Nm from dyne cm
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


combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined_scardec_ye_usgs_sigloch_isc_mag.csv')


combined.columns = ['event', 'scardec', 'ye', 'isc', 'sigloch', 'usgs', 'mag']


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

def gaussian(x, amp=1, mean=0.5, stddev=1):
	return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def triangle(x, center = 0.5, peak = 1, start = 0, end = 1):
	#start is x value where y = 0
	#end is x value where y = 0

	x = np.array(x)
	dx = 1/len(x)
	y = np.zeros(len(x))

	left_m = peak/(center-start)
	right_m = peak/(end-center)

	#print(left_m, right_m)

	y[int(start/dx):int(center/dx)] = left_m*x[int(start/dx):int(center/dx)]
	y[int(start/dx):int(center/dx)] = y[int(start/dx):int(center/dx)] - y[int(start/dx)]
	y[int(center/dx):int(end/dx)] = -right_m*x[int(center/dx):int(end/dx)]
	y[int(center/dx):int(end/dx)] = y[int(center/dx):int(end/dx)] + abs(y[int(center/dx)]) + peak
	#print(y[int(len(x)*((end-center)/(end-start)))])
	# y[:int(len(right_m)] = y[:int(len(right_m)] - min(y[:int(len(x)/2)])
	# y[int(len(x)/2):] = y[int(len(x)/2):] - min(y[int(len(x)/2):])

	return y


def boxcar(x, start=0, end=0, peak=1):
	length = len(x)
	start_index = int(start*len(x))
	end_index = int(end*len(x))
	y = np.zeros(len(x))
	y[start_index:length-end_index] = peak

	return y


def trapezium(x, start = 0.3333, end = 0.6666, peak = 1, start_height = 0, end_height = 0, end_ramp = 0, end_ramp_height = 0):

	length = len(x)
	end_ramp_index = int(length*end_ramp)
	start_flat_index = int(length*start)
	end_flat_index = int(length*(1-end))
	
	#print(length, start_index, end_index)

	y = np.zeros(length)
	
	y[:end_ramp_index] = np.linspace(start_height, end_ramp_height, end_ramp_index)
	
	y[end_ramp_index:start_flat_index] = np.linspace(end_ramp_height, peak, start_flat_index-end_ramp_index)
	y[start_flat_index:length-end_flat_index] = peak
	y[length-end_flat_index:] = peak - np.linspace(end_height, peak, end_flat_index) + end_height
	return y



def sine_boxcar(x, start = 0.3333, end = 0.3333, peak = 1, start_height = 0, end_height = 0):
	length = len(x)
	start_index = int(length*start)
	end_index = int(length*end)
	#print(length, start_index, end_index)

	y = np.zeros(length)

	y[:start_index] = (np.sin(np.linspace(-math.pi/2, math.pi/2, start_index))+1)*((peak-start_height)/2)+start_height
	y[start_index:length-end_index] = peak
	y[length-end_index:] = ((np.sin(np.linspace(-math.pi/2, math.pi/2, end_index)))[::-1]+1)*((peak-end_height)/2)+end_height
	return y

#for dataset_wanted in ['scardec_opt', 'scardec_moy', 'ye', 'usgs', 'sigloch', 'isc']:

to_ignore = ['20051203_1610_1', '20071226_2204_2', '20030122_0206_1', '20090929_1748_0', '20120421_0125_1', '20110311_2011_2']

columns_to_save = ['event', 'dataset', 'mag', 'interp_momentrate']
df_interp = pd.DataFrame(columns = columns_to_save)
max_len = 0
for i, row in combined.iterrows():
	if i % 100 == 0:
		print(i)
	# if i < 360:
	# 	continue

	#fig, axs = plt.subplots(2, 1, figsize=(10, 10))


	for dataset, get_stf in zip(['scardec_opt', 'scardec_moy', 'ye', 'usgs', 'sigloch', 'isc'], [get_scardec_stf, get_scardec_stf, get_ye_stf, get_usgs_stf, get_sigloch_stf, get_isc_stf]):
		# if dataset != dataset_wanted:
		# 	continue
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
			#print(time[0:10])
			
			momentrate = np.array(momentrate)

			time = np.array(time)
			detected_end_time, detected_end, detected_start_time, detected_start = find_end_stf(momentrate, time, dataset)
			time = time[detected_start:detected_end] # shift to start STF at zero
			time = time - time[0]
			momentrate = momentrate[detected_start:detected_end]
			max_len = max(max_len, len(momentrate))
			norm_momentrate = momentrate / max(momentrate)
			norm_time = time / max(time)
			#axs[0].plot(norm_time, norm_momentrate)

			interp_momentrate = np.interp(np.linspace(0, 1, 10000), norm_time, norm_momentrate)
			interp_momentrate[interp_momentrate < 0] = 0

			#axs[1].plot(np.linspace(0, 1, 100), interp_momentrate)

			df_interp = pd.concat([df_interp, pd.DataFrame([[row.event, dataset_name, row.mag, interp_momentrate]], columns = columns_to_save)], ignore_index = True)
			count += 1


interval = 6
boundaries = [5] #np.arange(5, 9.1, interval)

colors = cmc.batlow((np.linspace(0, len(boundaries), len(boundaries)+1)/len(boundaries)))
max_len = 0
for i, b in enumerate(boundaries):
	bigger_than = df_interp[((df_interp.mag > b))]
	filtered = bigger_than[bigger_than.mag < b+interval]
	max_len = max(max_len, len(filtered))
	subset = filtered.interp_momentrate.values
	if len(subset) < 100:
		marker = ':'
	else:
		marker = '-'


x = np.linspace(0, 1, 10000)
styles = ['dotted', 'dashed',  'dashdot', (0, (1, 1)), (0, (5, 10))]


obs_y = np.median(list(subset), axis = 0)
obs_x = np.linspace(0, 1, len(subset[0]))

print('trapezium')
left_flat_list = []
right_flat_list = []
height_list = []
start_height_list = []
end_height_list = []
sum_list = []
ramp_right_list = []
end_ramp_height_list = []

points_to_test = 10

for left_flat in np.linspace(0.3, 0.5, points_to_test):
	for right_flat in np.linspace(0.4, 0.7, points_to_test):
		for height in np.linspace(0.7, 0.9, points_to_test):
			for start_height in np.linspace(0, 0.15, points_to_test):
				for end_height in np.linspace(0, 0.3, points_to_test):
					for ramp_right in np.linspace(0.0, 0.3, points_to_test):
						for end_ramp_height in np.linspace(0, 0.5, points_to_test):
							trap_y = trapezium(x, left_flat, right_flat, height, start_height = start_height, end_height = end_height, end_ramp = ramp_right, end_ramp_height = end_ramp_height)
							sum_list.append(np.sum((trap_y - obs_y)**2))
							left_flat_list.append(left_flat)
							right_flat_list.append(right_flat)
							height_list.append(height)
							start_height_list.append(start_height)
							end_height_list.append(end_height)
							ramp_right_list.append(ramp_right)
							end_ramp_height_list.append(end_ramp_height)


trap_df = pd.DataFrame({'left_flat': left_flat_list, 'right_flat': right_flat_list, 'height': height_list, 'start_height': start_height_list, 'end_height': end_height_list, 'ramp_right': ramp_right_list, 'end_ramp_height': end_ramp_height_list, 'r2': sum_list})

best_trap_left_flat = trap_df.sort_values('r2').head(1).left_flat.values[0]
best_trap_right_flat = trap_df.sort_values('r2').head(1).right_flat.values[0]
best_trap_height = trap_df.sort_values('r2').head(1).height.values[0]
best_trap_start_height = trap_df.sort_values('r2').head(1).start_height.values[0]
best_trap_end_height = trap_df.sort_values('r2').head(1).end_height.values[0]
best_trap_ramp_right = trap_df.sort_values('r2').head(1).ramp_right.values[0]
best_trap_end_ramp_height = trap_df.sort_values('r2').head(1).end_ramp_height.values[0]


print('triangle')
center_list = []
height_list = []
start_posn_list = []
end_posn_list = []
sum_list = []

for center in np.linspace(0.3, 0.7, 20):
	for height in np.linspace(0.7, 0.9, 20):
		for start_posn in np.linspace(-0.4, 0.4, 20):
			for end_posn in np.linspace(0.6, 1.4, 20):
				tri_y = triangle(x, center = center, peak = height, start = start_posn, end = end_posn)
				sum_list.append(np.sum((tri_y - obs_y)**2))
				center_list.append(center)
				height_list.append(height)
				start_posn_list.append(start_posn)
				end_posn_list.append(end_posn)
				
tri_df = pd.DataFrame({'center': center_list, 'height': height_list, 'start_posn': start_posn_list, 'end_posn': end_posn_list, 'r2': sum_list})

best_tri_center = tri_df.sort_values('r2').head(1).center.values[0]
best_tri_height = tri_df.sort_values('r2').head(1).height.values[0]
best_tri_start = tri_df.sort_values('r2').head(1).start_posn.values[0]
best_tri_end = tri_df.sort_values('r2').head(1).end_posn.values[0]

print('boxcar')
start_list = []
end_list = []
height_list = []
sum_list = []

for start in np.linspace(0., 0.5, 20):
	for end in np.linspace(0., 0.5, 20):
		for height in np.linspace(0.7, 1, 20):
			box_y = boxcar(x, start = start, end = end, peak = height)
			sum_list.append(np.sum((box_y - obs_y)**2))
			start_list.append(start)
			height_list.append(height)
			end_list.append(end)

boxcar_df = pd.DataFrame({'start': start_list, 'end': end_list, 'height': height_list, 'r2': sum_list})

best_box_height = boxcar_df.sort_values('r2').head(1).height.values[0]
best_box_start = boxcar_df.sort_values('r2').head(1).start.values[0]
best_box_end = boxcar_df.sort_values('r2').head(1).end.values[0]


print('sineboxcar')
start_list = []
end_list = []
height_list = []
start_height_list = []
end_height_list = []
sum_list = []

for start in np.linspace(0., 1, 20):
	for end in np.linspace(0., 1, 20):
		for height in np.linspace(0.7, 1, 20):
			for start_height in np.linspace(0, 0.15, 20):
				for end_height in np.linspace(0, 0.3, 20):
					sinebox_y = sine_boxcar(x, start = start, end = end, peak = height)
					sum_list.append(np.sum((sinebox_y - obs_y)**2))
					start_list.append(start)
					height_list.append(height)
					end_list.append(end)
					start_height_list.append(start_height)
					end_height_list.append(end_height)
			
sineboxcar_df = pd.DataFrame({'start': start_list, 'end': end_list, 'height': height_list, 'start_height': start_height_list, 'end_height': end_height_list, 'r2': sum_list})

best_sinebox_height = sineboxcar_df.sort_values('r2').head(1).height.values[0]
best_sinebox_start = sineboxcar_df.sort_values('r2').head(1).start.values[0]
best_sinebox_end = sineboxcar_df.sort_values('r2').head(1).end.values[0]
best_sinebox_start_height = sineboxcar_df.sort_values('r2').head(1).start_height.values[0]
best_sinebox_end_height = sineboxcar_df.sort_values('r2').head(1).end_height.values[0]




#sineboxcar_df.sort_values('r2').head(10)

print('gaussian')
	
amp_list = []
mean_list = []
std_list = []
sum_list = []

for mean in np.linspace(0., 1, 100):
	for std in np.linspace(0., 5, 100):
		for height in np.linspace(0, 1, 100):
			gauss_y = gaussian(x, amp = height, mean = mean, stddev = std)
			sum_list.append(np.sum((gauss_y - obs_y)**2))
			amp_list.append(height)
			mean_list.append(mean)
			std_list.append(std)
			
gaussian_df = pd.DataFrame({'mean_val': mean_list, 'std_val': std_list, 'height_val': amp_list, 'r2': sum_list})

#gaussian_df.sort_values('r2').head(10)
best_gauss_height = gaussian_df.sort_values('r2').head(1).height_val.values[0]
best_gauss_mean = gaussian_df.sort_values('r2').head(1).mean_val.values[0]
best_gauss_std = gaussian_df.sort_values('r2').head(1).std_val.values[0]




x = np.linspace(0, 1, 10000)
styles = ['dotted', 'dashed',  'dashdot', (0, (1, 1)), (0, (5, 10))]

obs_y = np.median(list(subset), axis = 0)
obs_x = np.linspace(0, 1, len(subset[0]))
plt.plot(obs_x, obs_y, label = 'Average observed', color = 'k')

gaussian_y = gaussian(x, amp = best_gauss_height, mean = best_gauss_mean, stddev = best_gauss_std)
plt.plot(x, gaussian_y, label = f'Gaussian - {np.sum((gaussian_y - obs_y)**2):.2f}', linestyle=styles.pop(0), linewidth = 2.5)

triangle_y = triangle(x, center = best_tri_center, peak = best_tri_height, start = best_tri_start, end = best_tri_end)
#triangle_y[0: int(0.14*len(x))] = 1.13*x[0:int(0.14*len(x))] + 0.03
plt.plot(x, triangle_y, label = f'Triangle - {np.sum((triangle_y - obs_y)**2):.2f}', linestyle=styles.pop(0), linewidth = 2.5)

boxcar_y = boxcar(x, start = best_box_start, end = best_box_end, peak = best_box_height)
plt.plot(x, boxcar_y, label = f'Boxcar - {np.sum((boxcar_y - obs_y)**2):.2f}', linestyle=styles.pop(0), linewidth = 2.5)

trap_y = trapezium(x, best_trap_left_flat, best_trap_right_flat, best_trap_height, start_height = best_trap_start_height, end_height = best_trap_end_height, end_ramp = best_trap_ramp_right, end_ramp_height = best_trap_end_ramp_height)
plt.plot(x, trap_y, label = f'Trapezium - {np.sum((trap_y - obs_y)**2):.2f}', linestyle=styles.pop(0), linewidth = 2.5)

sine_boxcar_y = sine_boxcar(x, start = 0.6, end = 0.42, peak = 0.8, start_height = 0, end_height = 0.17)

#sine_boxcar_y = sine_boxcar(x, start = best_sinebox_start, end = best_sinebox_end, peak = best_sinebox_height, start_height = best_sinebox_start_height, end_height = best_sinebox_end_height)
plt.plot(x, sine_boxcar_y, label = f'Sine Boxcar - {np.sum((sine_boxcar_y - obs_y)**2):.2f}', linestyle=styles.pop(0), linewidth = 2.5)

plt.ylabel('Normalized Moment Rate')
plt.xlabel('Normalized Time')
plt.legend()

plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/average_STF/average_observed_and_models_all_v2.png')
plt.close()

with open(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/average_STF/average_best_fitting_models.txt', 'a') as f:
	f.write(f'ALL \n')
	f.write(f'Gaussian \n')
	f.write(f'height: {best_gauss_height}, mean: {best_gauss_mean}, std: {best_gauss_std}\n')
	f.write(f'2-norm {np.sum((gaussian_y - obs_y)**2):.2f}\n')
	f.write(f'Triangle \n')
	f.write(f'center: {best_tri_center}, height: {best_tri_height}, start: {best_tri_start}, end: {best_tri_end}\n')
	f.write(f'2-norm {np.sum((triangle_y - obs_y)**2):.2f}\n')
	f.write(f'Boxcar \n')
	f.write(f'start: {best_box_start}, end: {best_box_end}, height: {best_box_height}\n')
	f.write(f'2-norm {np.sum((boxcar_y - obs_y)**2):.2f}\n')
	f.write(f'Trapezium \n')
	f.write(f'left flat: {best_trap_left_flat}, right flat: {best_trap_right_flat}, height: {best_trap_height}, start height: {best_trap_start_height}, end height: {best_trap_end_height}, ramp right: {best_trap_ramp_right}, end ramp height: {best_trap_end_ramp_height}\n')
	f.write(f'2-norm {np.sum((trap_y - obs_y)**2):.2f}\n')
	f.write(f'Sine Boxcar \n')
	f.write(f'start: {best_sinebox_start}, end: {best_sinebox_end}, height: {best_sinebox_height}, start height: {best_sinebox_start_height}, end height: {best_sinebox_end_height}\n')
	f.write(f'2-norm {np.sum((sine_boxcar_y - obs_y)**2):.2f}\n')
	f.write('\n\n\n')

