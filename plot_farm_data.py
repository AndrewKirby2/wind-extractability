"""Plot zeta for DS8 for different wind farm sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk

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

for no in range(10):
    zeta = np.load(f'data/zeta_DS{no}_30.npy')
    cf0 = np.load(f'data/cf0_DS{no}_30.npy')
    fr0 = np.load(f'data/fr0_DS{no}_30.npy')
    cond = np.logical_and(zeta>0, zeta<40)
    cbar = plt.scatter(fr0[cond], cf0[cond], c=zeta[cond], vmin=0, vmax=40)
plt.ylabel(r'$C_{f0}$')
plt.xlabel(r'$Fr_0$')
plt.colorbar()
plt.legend()
plt.savefig(f'plots/zeta_30km_fr_iter.png')
plt.close()

for no in range(10):
    print(f'Mean absolute percentage errors for DS{no} (%)')

    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Size (km)','zeta','beta','1/(1-beta)','M-1'))
    #calculate difference between old and new methods
    for farm_diameter in [10,15,20,25,30]:
        #load data from new calculations
        zeta_new = np.load(f'data/zeta_DS{no}_{farm_diameter}.npy')
        beta_new = np.load(f'data/beta_DS{no}_{farm_diameter}.npy')
        M_new = np.load(f'data/M_DS{no}_{farm_diameter}.npy')
        #load data from old calculations
        tmp = np.atleast_1d(np.load(f'../../part1_turbines/UM_farm_data/data/zeta_DS{no}.npy', allow_pickle=True))
        zeta_old = tmp[0][farm_diameter]
        tmp = np.atleast_1d(np.load(f'../../part1_turbines/UM_farm_data/data/beta_DS{no}.npy', allow_pickle=True))
        beta_old = tmp[0][farm_diameter]
        tmp = np.atleast_1d(np.load(f'../../part1_turbines/UM_farm_data/data/M_DS{no}.npy', allow_pickle=True))
        M_old = tmp[0][farm_diameter]
        print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(farm_diameter,round(100*sk.mean_absolute_percentage_error(zeta_new,zeta_old),1),
            round(100*sk.mean_absolute_percentage_error(beta_new,beta_old),1),
            round(100*sk.mean_absolute_percentage_error(1.0/(1.0-beta_new),1.0/(1.0-beta_old)),1),
            round(100*sk.mean_absolute_percentage_error(1.0-M_new,1.0-M_old),1)))
