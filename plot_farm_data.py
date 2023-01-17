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
        zeta[i,24*no:24*(no+1)] = np.load(f'data/zeta_DS{no}_{farm_diameter}.npy')

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
    M[i+1,:] = np.load(f'data/M_DS1_20_{z0[i]}.npy')
    beta[i+1,:] = np.load(f'data/beta_DS1_20_{z0[i]}.npy')

#plots results
for i in range(4):
    for j in range(6):
        ax[i].plot(1-beta[:,i*6:i*6+j], M[:,i*6:i*6+j]-1)
        ax[i].set_xlim([0,0.3])
        ax[i].set_ylim([0,5])
        ax[i].set_xlabel(r'$1-\beta$')
ax[0].set_ylabel(r'$M-1$')
plt.savefig('plots/M_vs_beta_plots.png')
plt.close()

#calculate difference between volume-averaged and hub height averaged beta values
error_beta = 0
error_zeta = 0
for farm_diameter in [10, 15, 20, 25, 30]:
    for i in range(10):
        beta_vol = np.load(f'data/beta_DS{i}_{farm_diameter}.npy')
        beta_hubh = np.load(f'data_hubh/beta_DS{i}_{farm_diameter}.npy')
        zeta_vol = np.load(f'data/zeta_DS{i}_{farm_diameter}.npy')
        zeta_hubh = np.load(f'data_hubh/zeta_DS{i}_{farm_diameter}.npy')
        error_beta += sk.mean_absolute_percentage_error(beta_vol, beta_hubh)
        error_zeta += sk.mean_absolute_error(zeta_vol, zeta_hubh)
error_beta = error_beta/50
error_zeta = error_zeta/50
print(error_beta)
print(error_zeta)

fig, ax = plt.subplots(ncols=3, figsize = [12,3])
farm_diameters = [10, 20, 30]
for i in range(3):
    for no in range(10):
        zeta = np.load(f'data/zeta_DS{no}_{farm_diameters[i]}.npy')
        cf0 = np.load(f'data/cf0_DS{no}_{farm_diameters[i]}.npy')
        ax[i].scatter(cf0, zeta, label=f'DS{no}')
        ax[i].set_ylabel(r'$\zeta$')
        ax[i].set_xlabel(r'$C_{f0}$')
ax[0].set_ylim([0,80])
ax[1].set_ylim([0,50])
ax[2].set_ylim([0,40])
ax[2].legend(ncol=2)
plt.tight_layout()
plt.savefig('plots/zeta_cf0.png')
