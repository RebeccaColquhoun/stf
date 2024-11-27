import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import rel_entr


def myround(x, base=5):
    '''rounds to nearest base (e.g. if base is 5, rounds to nearest 5)'''
    return base * round(x / base)


def get_stf(scardec_name, combined, wanted_type='fctopt'):
    db = combined[combined['scardec_name'] == scardec_name]

    time = []
    momentrate = []

    event = os.listdir(f'/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec/{scardec_name}')
    starts = [n for n, l in enumerate(event) if l.startswith(wanted_type)]
    if not starts:
        raise ValueError(f"No files starting with {wanted_type} found in {scardec_name}")
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


def f3(end_time, total_moment, time_opt, momentrate_opt, start, points_before_zero, proportion=0.1):
    '''looks for time value of root'''
    dx = time_opt[1] - time_opt[0]
    end_window = (end_time / dx) + points_before_zero
    end = int(np.floor(end_window))
    short = scipy.integrate.simpson(momentrate_opt[start:end], dx=dx)
    return short - (total_moment * proportion)


def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension

    Args:
        p, q: two discrete probability distributions

    Returns:
        Hellinger distance between p and q. This is distributed between 0 and 1.
    """
    p = np.array(p)  # these need to be probability distributions
    q = np.array(q)

    # calculate the square of the difference of ith distr elements
    s = (np.sqrt(p) - np.sqrt(q)) ** 2

    # calculate sum of squares
    sosq = sum(s)

    return np.sqrt(sosq) / math.sqrt(2)


data_path = '/home/earthquakes1/homes/Rebecca/phd/stf/data/scardec'
combined = pd.read_csv('/home/earthquakes1/homes/Rebecca/phd/stf/data/combined.csv')

all_proportions = [[0.45, 0.55],
                   [0.4, 0.6],
                   [0.35, 0.65],
                   [0.3, 0.7],
                   [0.25, 0.75],
                   [0.2, 0.8],
                   [0.15, 0.85],
                   [0.1, 0.9],
                   [0.05, 0.95]]
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

        for scardec_name in os.listdir(data_path):
            # Get the moment rate and time for the current scardec_name
            momentrate_opt, time_opt, db = get_stf(scardec_name, combined)

            # Find indices where moment rate is greater than zero
            not_zero = np.where(momentrate_opt > 0)[0]

            dx = time_opt[1] - time_opt[0]

            start = min(not_zero)
            end = max(not_zero)
            points_before_zero = abs(min(time_opt) / dx)

            # Calculate the duration of the event
            duration = time_opt[end] - time_opt[start]
            durations.append(duration)

            start_time = time_opt[start]
            end_time = time_opt[end]

            # Calculate the total moment using Simpson's rule
            total_moment = scipy.integrate.simpson(momentrate_opt[start:end], dx=time_opt[1] - time_opt[0])
            moments.append(total_moment)

            # Find the root time using the bisection method
            root, r = scipy.optimize.bisect(f3,
                                            start_time + dx,
                                            end_time,
                                            rtol=1e-6,
                                            full_output=True,
                                            args=(total_moment,
                                                  time_opt,
                                                  momentrate_opt,
                                                  start,
                                                  points_before_zero,
                                                  proportion,))
            root_idx = np.floor(root / dx)
            root_time = root_idx * dx
            root_times.append(root_time)
            relative_root_times.append(root_time - start_time)

            if root_time - start_time > duration:
                print('root time greater than duration, proportion:', proportion)
                print(scardec_name)

        root_times = np.array(root_times)
        relative_root_times = np.array(relative_root_times)
        durations = np.array(durations)
        moments = np.log10(np.array(moments))

        rel_root_times = relative_root_times / durations

        # Create a histogram of the relative root times
        n, bins = np.histogram(rel_root_times, bins=np.arange(0, 1.01, 0.01))

        if proportion < 0.5:
            P = n / sum(n)
        else:
            Q = (n / sum(n))[::-1]

        plt.ylabel('Frequency')
        plt.xlabel('Proportion of duration to release proportion of moment')
        plt.xlim(0, 1)

        if proportion > 0.5:
            tf = (P != 0) & (Q != 0)
            z_p = np.zeros(len(bins) - 1)
            z_q = np.zeros(len(bins) - 1)
            z_p[tf] = P[tf]
            z_q[tf] = Q[tf]
            plt.stairs(P, bins, linestyle='--', color='tab:blue')
            plt.stairs(Q, bins, linestyle='--', color='tab:orange')
            plt.stairs(z_p, bins, color='tab:blue', label=f'P: {myround((1-proportion)*100):.0f}%')
            plt.stairs(z_q, bins, color='tab:orange', label=f'Q: {myround(proportion*100):.0f}%')

            # Calculate KL divergence and Hellinger distance
            kl_divergence = sum(rel_entr(P[tf], Q[tf]))
            print(kl_divergence)
            kl_divergence_2 = sum(rel_entr(Q[tf], P[tf]))
            print(kl_divergence_2)
            hellinger_distance = hellinger_explicit(P, Q)
            print(hellinger_distance)

            plt.title(f'KL(P||Q): {kl_divergence:.5f}, KL(Q||P): {kl_divergence_2:.5f} \n Hellinger distance: {hellinger_distance:.5f}')
            plt.legend()
            # plt.show()

    plt.savefig(f'/home/earthquakes1/homes/Rebecca/phd/stf/figures/moment_intervals/flipped_histogram_fraction_of_duration_for_{myround((1-proportion)*100)}_{myround(proportion*100)}_percent_moment_mean_std.png')
    plt.close()
    plt.close()
