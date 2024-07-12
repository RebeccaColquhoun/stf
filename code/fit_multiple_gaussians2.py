from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def gaussian(x, peakPosition, width):
    g = np.exp(-((x - peakPosition) / (0.60056120439323 * width)) ** 2)
    return g

centers = np.array([64,59,23,79,29,93])#%randi(100, 1, numGaussians);
sigmas = np.array([6,11,7,5,11,3])#%randi(20, 1, numGaussians);
amplitudes = np.array([40,20,16,26,20,36])#%randi([10, 40], 1, numGaussians);

x = np.linspace(0, 150, 1000)
y = np.zeros(len(x))

with open(f"cat_and_stf/621560715.txt", 'rb') as f:
    y = pickle.load(f)
x = np.linspace(0, len(y), 256)
numGaussians = 3

# tActual = pd.DataFrame({'amplitudes':amplitudes, 'mean_posn':centers, 'width':sigmas})
# # Now sort parameters in order of increasing mean, just so it's easier to think about (though it's not required).
# tActual = tActual.sort_values('mean_posn')
# tActual = tActual.reset_index(drop=True)

# for k in range(0, numGaussians):
#     this_gaussian = tActual.amplitudes[k] * gaussian(x, tActual.mean_posn[k], tActual.width[k])
#     #this_gaussian = gaussian(x, tActual.mean_posn[k], tActual.width[k])
#     y = y + this_gaussian
#     plt.plot(x, this_gaussian)
def func_plot(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        plt.plot(x, amp * np.exp( -((x - ctr)/wid)**2))
    return y

def func(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y
#for n in range(1, 15):
guess = []
for i in range(4):
    guess += [30*i, 10, 10]   
#print(guess)
popt, pcov = curve_fit(func, x, y, p0=guess, maxfev = 10000000)
#print(popt)
fit = func(x, *popt)
print(fit)
theError = np.linalg.norm(fit - y)
print(theError)
print(min(y))
plt.plot(x, y)
plt.plot(x, fit , color='r', linestyle=':')
plt.show()