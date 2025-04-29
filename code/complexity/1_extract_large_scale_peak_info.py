import os

import pandas as pd

gaussians_info = {}

r2_limit = 0.8
smoothing_points = 1

if smoothing_points > 1:
    built_dir_name = f'R2_{int(r2_limit*10):02}/smoothed_{smoothing_points}'
else:
    built_dir_name = f'R2_{int(r2_limit*10):02}/'

dir_name = f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/large_scale_peaks/{built_dir_name}/'

combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined.csv')

for fn in os.listdir(dir_name):
# Open the file in read mode
    if fn.endswith(".txt"):
        with open(dir_name + fn, 'r') as file:
        # Read each line in the file one by one
            for line in file:
                # Process the line
                #print(line.strip())
                if line.startswith('Processing'):
                    eq_name = line.split(' ')[1]
                    eq_name = eq_name.strip('\n')
                    db = combined[combined['scardec_name']==eq_name]
                    gaussians_info[eq_name] = {'depth':db.depth.values[0], 'moment': db.moment.values[0], 'magnitude': db.scardec_magnitude.values[0]}
                    #print(db.columns)
                elif line.startswith('Fitting'):
                    number_of_gaussians = int(line.split(' ')[1])
                elif line.startswith('R-squared'):
                    #gaussians_r[number_of_gaussians] = float(line.split(':')[1])
                    col = 'r_squared_' + str(number_of_gaussians)
                    r_squared = float(line.split(':')[1])
                    gaussians_info[eq_name][col] = r_squared
                # elif line.startswith('Proportion'):
                #     gaussians_moment[number_of_gaussians] = float(line.split(':')[1])

            gaussians_info[eq_name]['n_gaussians'] = number_of_gaussians
            gaussians_info[eq_name]['r_squared'] = r_squared
        #print(gaussians_r)
        # Convert the gaussians_info dictionary to a pandas DataFrame
        gaussians_df = pd.DataFrame.from_dict(gaussians_info, orient='index')

        # Save the DataFrame to a CSV file
        #gaussians_df.to_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/gaussians_info.csv', index_label='eq_name')
gaussians_df = gaussians_df.reindex(columns=(['magnitude', 'moment', 'depth', 'n_gaussians', 'r_squared'] + list([a for a in gaussians_df.columns if a not in ['moment', 'depth', 'n_gaussians', 'r_squared']]) ))
print(gaussians_df)
built_dir_name = built_dir_name.replace('/', '_')
gaussians_df.to_csv(f'/home/earthquakes1/homes/Rebecca/phd/stf/data/results/{built_dir_name}gaussians_info.csv', index_label='eq_name')