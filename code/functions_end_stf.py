import numpy as np
import scipy
import pandas as pd

# def find_end_stf(momentrate, time, threshold_limit = None):
#     if threshold_limit == None:
#         threshold_limit = 0.5
#     else:
#         threshold_limit = float(threshold_limit)
#     not_zero = np.where(momentrate > 0)[0]
#     #print(max(momentrate))
#     start = min(not_zero)
#     end = max(not_zero)

#     detected_end = end
#     detected_end_time = time[end]

#     time = time[:end]
#     momentrate = momentrate[:end]
#     # print('mr', momentrate)
#     # print('time', time)

#     less_than_10 = np.where(momentrate <= 10*max(momentrate)/100)[0]
#     #print(less_than_10)
#     print(less_than_10)
#     #print(start)
#     total_moment = scipy.integrate.simpson(momentrate[start:end],
#                                         dx = time[start+1]-time[start])
#     #print(less_than_10)
#     for i in less_than_10:
#         if i <= start:
#             continue
#         if i == 0:
#             continue
#         # print('i', i)
#         # print('start', start)
#         # print('mr short', momentrate[start:i])
#         moment = scipy.integrate.simpson(momentrate[start:i],
#                                         dx = time[start+1]-time[start])
#         #print(i, moment/total_moment)
#         # print(threshold_limit)
#         # print(type(threshold_limit))
#         # print(type(total_moment))
#         # print(type(moment))
#         if moment >= threshold_limit * total_moment:
#             print('inif')
#             #print(f'first time where < 10% of total momentrate and 50% of moment released: {time[i]} s')
#             detected_end_time = time[i]
#             detected_end = i
#             #print(f'proportion of moment released: {(moment/total_moment)*100:.2f}%')
#             break
#     return detected_end_time, detected_end, time[start], start
#     #return time[end], end
    
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
def find_99_end_stf(momentrate, time):
    not_zero = np.where(momentrate > 0)[0]
    #print(max(momentrate))
    start = min(not_zero)
    end = max(not_zero)

    detected_end = end
    detected_end_time = time[end]

    time = time[:end]
    momentrate = momentrate[:end]

    less_than_10 = np.where(momentrate <= 10*max(momentrate)/100)[0]
    # print(less_than_10)
    # print(start)
    total_moment = scipy.integrate.simpson(momentrate[start:end],
                                        dx = time[start+1]-time[start])
    #print(less_than_10)
    for i in less_than_10:
        if i <= start:
            continue
        if i == 0:
            continue
        moment = scipy.integrate.simpson(momentrate[start:i],
                                        dx = time[start+1]-time[start])
        #print(i, moment/total_moment)
        if moment >= 0.99 * total_moment:
            #print('inif')
            #print(f'first time where < 10% of total momentrate and 50% of moment released: {time[i]} s')
            detected_end_time = time[i]
            detected_end = i
            #print(f'proportion of moment released: {(moment/total_moment)*100:.2f}%')
            break

    return detected_end_time, detected_end, time[start], start
    #return time[end], end
    
def find_zero_end_stf(momentrate, time):
    not_zero = np.where(momentrate > 0)[0]
    #print(max(momentrate))
    start = min(not_zero)
    end = max(not_zero)

    detected_end = end
    detected_end_time = time[end]

    return detected_end_time, detected_end, time[start], start
    #return time[end], end