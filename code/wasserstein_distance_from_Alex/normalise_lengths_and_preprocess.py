import os
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt

import pandas as pd
from scipy.interpolate import interp1d

from sklearn import preprocessing

combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined.csv')

data_dir = '/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec'
output_file = '/home/earthquakes1/homes/Rebecca/phd/stf/code/wasserstein_distance_from_Alex/norm_processed_data.csv'

# Get list of fctopt* files
file_pattern = os.path.join(data_dir, '*/fctopt*')
files = glob.glob(file_pattern)

# Read column 2 from each file and store in a list
all_data = []

all_data.append(['scardec_name', 'magnitude'] + [f'{i}' for i in range(100)])
# for file in files:
#     data = []
#     data.append(file.split('/')[-1])
#     with open(file, 'r') as f:
#         lines = f.readlines()[2:]  # Skip the first 2 rows
#         column_data = [line.split()[1] for line in lines if line.strip()]
#         data.extend(column_data)
#         all_data.append(data)

# # Find the length of the longest list
# max_length = min(len(data) for data in all_data)
# max_length_index = next(i for i, data in enumerate(all_data) if len(data) == max_length)
# print(f"Max length: {max_length}, Row number: {max_length_index}, data: {all_data[max_length_index][0]}")

for scardec_name in os.listdir('/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec'):
    db = combined[combined['scardec_name']==scardec_name]

    time_opt = []
    momentrate_opt = []

    time_moy = []
    momentrate_moy = []

    event = os.listdir(f'/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec/{scardec_name}')
    starts = [n for n, l in enumerate(event) if l.startswith('fctopt')]
    with open(f'/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec/{scardec_name}/{event[starts[0]]}') as f:
        lines = f.read().splitlines()

    lines = lines[2:]
    for line in lines:
        split = line.split(' ')
        split = [s for s in split if s not in ['', ' ', '\n']]
        time_opt.append(float(split[0]))
        momentrate_opt.append(float(split[1]))

    momentrate_opt = np.array(momentrate_opt)

    not_zero = np.where(momentrate_opt > 0)[0]

    # Interpolate to 100 points
    interp_func = interp1d(not_zero, momentrate_opt[not_zero], kind='linear', fill_value="extrapolate")
    new_indices = np.linspace(not_zero[0], not_zero[-1], 100)
    momentrate_opt = interp_func(new_indices)

    momentrate_opt = momentrate_opt / np.sum(momentrate_opt)

    normalised_momentrate_opt = preprocessing.StandardScaler(momentrate_opt.reshape(-1, 1)) #.fit_transform(momentrate_opt.reshape(-1, 1)).reshape(-1)

    data = [scardec_name, int(db['scardec_magnitude'].values[0])]
    data.extend(normalised_momentrate_opt)
    all_data.append(data)
    # fig, axs = plt.subplots(2)
    # axs[0].plot(momentrate_opt)
    # axs[0].hlines(np.mean(momentrate_opt), 0, 100, colors='r')
    # axs[1].plot(normalised_momentrate_opt)
    # plt.show()

# Save all_data as CSV
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in all_data:
        csvwriter.writerow(row)