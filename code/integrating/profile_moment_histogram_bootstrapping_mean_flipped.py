import os

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


def plot_scatter_figures(proportion, root_times, moments, durations):
    root_times = np.array(root_times)
    moments = np.array(moments)
    durations = np.array(durations)

    plt.scatter(root_times, durations, c = np.log10(moments), cmap = cmc.batlow, alpha = 0.5)
    plt.ylabel('Duration (s)')
    plt.xlabel(f'time to release {proportion*100}% of moment (s)')
    plt.colorbar(label = 'log10(moment)')
    plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_intervals/time_for_{proportion*100}_percent_moment_against_duration.png')
    plt.close()

    plt.scatter(root_times, np.log10(moments), c = durations, cmap = cmc.batlow, alpha = 0.5)
    plt.ylabel('log10(moment)')
    plt.xlabel(f'time to release {proportion*100}% of moment (s)')
    plt.colorbar(label = 'Duration (s)')
    plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_intervals/time_for_{proportion*100}_percent_moment_against_moment.png')
    plt.close()

    plt.scatter(root_times/durations, np.log10(moments), c = durations, cmap = cmc.batlow, alpha = 0.5)
    plt.ylabel('log10(moment)')
    plt.xlabel(f'proportion of duration to release {proportion*100}% of moment')
    plt.colorbar()
    plt.xlim(0, 1)
    plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_intervals/fraction_of_duration_for_{proportion*100}_percent_moment_against_moment.png')
    plt.close()

def myround(x, base=5):
    return base * round(x/base)

combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined.csv')

def get_stf(scardec_name, wanted_type = 'fctopt'):
    db = combined[combined['scardec_name']==scardec_name]

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
    return momentrate, time, db

def get_mag(scardec_name):
    db = combined[combined['scardec_name']==scardec_name]
    mag = db['mag'].values[0]
    return mag
# looks for time value of root

def f3(end_time, total_moment, time_opt, momentrate_opt, start, points_before_zero, proportion = 0.1):
    dx = time_opt[1]-time_opt[0]
    end_window = (end_time/dx)+points_before_zero
    end = int(np.floor(end_window))
    short = scipy.integrate.simpson(momentrate_opt[start:end], dx = dx)
    return short-(total_moment*proportion)

def plot_hist_figures(proportion, root_times, durations):
    root_times = np.array(root_times)
    durations = np.array(durations)

    plt.hist(root_times/durations, bins = 100)

    plt.ylabel('Frequency')
    plt.xlabel(f'Proportion of duration to release {proportion*100}% of moment')
    plt.xlim(0, 1)
    plt.show()
    #plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_intervals/histogram_fraction_of_duration_for_{proportion*100}_percent_moment.png')
    #plt.close()

def bootstrap(data, n=1000, proportion = 1):
    """Bootstrap resampling of data.

    Args:
        data: 1D array of data to be resampled.
        n: number of resamples to take.

    Returns:
        resampled data.
    """

    rng = np.random.default_rng()

    resampled_data = rng.choice(data, (n, int(len(data)*proportion)), replace = True)

    return resampled_data
all_proportions = [[0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7], [0.25, 0.75], [0.2, 0.8], [0.15, 0.85], [0.1, 0.9], [0.05, 0.95]]
all_durations = []
all_root_times = []
all_moments = []
all_relative_root_times = []
for proportions_list in all_proportions:
    for proportion in proportions_list:
        print(proportion)
        durations = []
        root_times = []
        relative_root_times = []

        diff = []
        moments = []

        for scardec_name in os.listdir('/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec'):
            #print(scardec_name)
            momentrate_opt, time_opt, db = get_stf(scardec_name)
            mag = get_mag(scardec_name)

            not_zero = np.where(momentrate_opt > 0)[0]

            dx = time_opt[1]-time_opt[0]

            start = min(not_zero)
            end = max(not_zero)
            points_before_zero = abs(min(time_opt)/dx)

            duration = time_opt[end] - time_opt[start]
            durations.append(duration)

            start_time = time_opt[start]
            end_time = time_opt[end]

            total_moment = scipy.integrate.simpson(momentrate_opt[start:end], dx = time_opt[1]-time_opt[0])
            moments.append(total_moment)
            root, r = scipy.optimize.bisect(f3,
                                            start_time+dx,
                                            end_time,
                                            rtol = 1e-6,
                                            full_output = True,
                                            args = (total_moment,
                                                    time_opt,
                                                    momentrate_opt,
                                                    start,
                                                    points_before_zero,
                                                    proportion,))
            root_idx = np.floor(root/dx)
            root_time = root_idx*dx
            root_times.append(root_time)
            relative_root_times.append(root_time-start_time)

            if root_time-start_time > duration:
                print('root time greater than duration, proportion:', proportion)
                print(scardec_name)

        root_times = np.array(root_times)
        durations = np.array(durations)
        moments = np.log10(np.array(moments))
        relative_root_times = np.array(relative_root_times)

        rel_root_times = relative_root_times/durations

        bs = bootstrap(rel_root_times, proportion = 0.5)

        bs_means = np.mean(bs, axis = 1)

        bs_means = np.sort(bs_means)

        lower = bs_means[25]
        upper = bs_means[975]

        print(f'lower: {lower}, mean: {np.mean(rel_root_times)}, upper: {upper}')
        n, bins = np.histogram(rel_root_times, bins=np.arange(0, 1.01, 0.01))

        if proportion < 0.5:
            P = n/sum(n)
        else:
            Q = (n/sum(n))[::-1]

        plt.ylabel('Frequency')
        plt.xlabel('Proportion of duration to release proportion of moment')
        plt.xlim(0, 1)

        if proportion < 0.5:
            plt.stairs(P, bins, alpha = 0.5, color = 'midnightblue', label = f'{proportion*100:.0f}%', fill = True)
            plt.axvline(x = lower, color = 'lightskyblue')#, linestyle = ':')
            plt.axvline(x = upper, color = 'lightskyblue')#, linestyle = ':')
            plt.axvline(x = np.mean(rel_root_times), color = 'dodgerblue')
        else:
            plt.stairs(Q, bins, alpha = 0.5, color = 'darkgreen', label = f'{proportion*100:.0f}%', fill = True)
            plt.axvline(x = 1-lower, color = 'palegreen')
            plt.axvline(x = 1-upper, color = 'palegreen')
            plt.axvline(x = 1-np.mean(rel_root_times), color = 'limegreen')
    plt.ylabel('Frequency')
    plt.xlabel('Proportion of duration to release % of moment')
    plt.xlim(0, 1)
    plt.legend()
    #plt.show()
    plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_intervals/bootstrapping_duration_proportions/flipped_histogram_fraction_of_duration_for_{(1-proportion)*100}_{(proportion)*100}_percent_moment_bs_mean.png')
    plt.close()