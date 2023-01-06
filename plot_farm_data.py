"""Plot zeta for DS8 for different wind farm sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk

#load farm data for different sizes
for no in range(10):
    zeta = np.load(f'data/zeta_DS{no}_30.npy')
    cf0 = np.load(f'data/cf0_DS{no}_30.npy')
    inv_fr0 = 1.0/np.load(f'data/fr0_DS{no}_30.npy')
    print(no, inv_fr0)
    #unstable conditions - set inverse froude number to 0
    inv_fr0 = np.nan_to_num(inv_fr0)

    plt.scatter(inv_fr0, cf0, c=zeta, vmin=0, vmax=50)
plt.xlim([-1,2])
plt.ylabel(r'$C_{f0}$')
plt.xlabel(r'$1/Fr_{0}$')
cbar = plt.colorbar()
cbar.set_label(r'$\zeta$')

plt.savefig('plots/zeta_cf0_30km.png')

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
