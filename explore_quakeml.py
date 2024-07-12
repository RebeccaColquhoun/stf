import obspy
import pickle
import matplotlib.pyplot as plt
import scipy
import numpy as np
import os

#eq_list = os.listdir('cat_and_stf/*.txt')

files = os.listdir('cat_and_stf/')
eq_list = [i[:9] for i in files if i.endswith('.txt')]
eq_list = list(set(eq_list))
print(eq_list)
#eq_list = ['616636559']

moments  = []
mags_mo = []
mags = []

# def plot_stf(stf, norm, eq_no):
#     fig, axs = plt.subplots()
#     axs.set_title(eq_no)
#     axs.plot(np.array(stf)*norm*1E7)
#     axs.set_label('Moment rate Nm/s')
#     axs.set_xlabel('Time (s)')
#     if show == True:
#         plt.show()
#     if save == True:
#         #plt.savefig($FIGURES+'/stfs/{eq_no}.pdf')


for eq_no in eq_list:#:
    cat = obspy.read_events(f"cat_and_stf/{eq_no}.xml")
    #print(cat)
    with open(f"cat_and_stf/{eq_no}.txt", 'rb') as f:
        stf = pickle.load(f)
    with open(f"cat_and_stf/{eq_no}_norm_info.txt", 'rb') as f:
        norm_info = pickle.load(f)
    #print(norm_info)
    #print(stf)
    #plot_stf(stf, norm)
    #plt.show()
    stf_pos = [s if s>0 else 0 for s in stf ]
    moment = scipy.integrate.trapz(np.array(stf)*norm_info['mo_norm']*1E7)#, np.linspace(0, 256))#, norm_info['N_samp']))
    #print(np.sum(np.array(stf)*1.2E10*0.1))
    for i in cat[0].magnitudes:
        if i.creation_info.author=='ISC-PPSM':
            mags_mo.append(10**((i.mag*1.5)+9.1))
            mags.append(i.mag)
            print(f"{moment:.2e}")
            moments.append(moment)
print(stf_pos)
fig, ax = plt.subplots(1,1)
ax.scatter(mags_mo, moments)
ax.plot([10**17, 10**20], [10**17, 10**20], color = 'orange')
ax2 = ax.twiny()
ax2.scatter(mags, moments)

ax.set_xlim([1E17, 1E20])
ax2.set_xlim([(17-9.1)/1.5, (20-9.1)/1.5])

ax.set_yscale('log')
ax.set_xscale('log')


ax.set_ylabel('Moment: integral(stf*mo_norm*10E7)')
ax.set_xlabel('Moment: 10**((Mw*1.5)+9.1)')
ax2.set_xlabel('Mw (ISC_PPSM)')
plt.show()