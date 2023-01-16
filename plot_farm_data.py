"""Plot zeta for DS8 for different wind farm sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics as sk
from read_NWP_data import *

#array to store results for histogram
zeta = np.zeros((5,240))
farm_diameters = [10,15,20,25,30]

#load farm data for different sizes
for i in range(5):
    farm_diameter = farm_diameters[i]
    for no in range(10):
        zeta[i,24*no:24*(no+1)] = np.load(f'data_hubh/zeta_DS{no}_{farm_diameter}.npy')

for i in [0, 2, 4]:
    plt.hist(zeta[i,:], density=True, range=(0,80), bins=25, histtype=u'step', label=str(farm_diameters[i])+' km')

plt.ylabel('Density')
plt.xlabel(r'$\zeta$')
plt.legend()
plt.savefig('plots/zeta_histogram.png')
plt.close()

fig, ax = plt.subplots(ncols=4, figsize=[16,4])

#empty arrays to store results
M = np.ones((6,24))
beta = np.ones((6,24))

#load results for different roughness lengths
z0 = ['0p05', '0p1', '0p35', '0p7', '1p4']
for i in range(5):
    M[i+1,:] = np.load(f'data_hubh/M_DS1_20_{z0[i]}.npy')
    beta[i+1,:] = np.load(f'data_hubh/beta_DS1_20_{z0[i]}.npy')

#plots results
for i in range(4):
    for j in range(6):
        ax[i].plot(1-beta[:,i*6:i*6+j], M[:,i*6:i*6+j]-1)
        ax[i].set_xlim([0,0.3])
        ax[i].set_ylim([0,5])
        ax[i].set_xlabel(r'$1-\beta$')
ax[0].set_ylabel(r'$M-1$')
plt.savefig('plots/M_vs_beta_plots.png')