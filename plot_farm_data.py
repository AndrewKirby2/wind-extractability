"""Plot zeta for DS8 for different wind farm sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk

#load farm data for different sizes
for farm_diameter in [10,15,20,25,30]:
    zeta = np.load(f'data/zeta_DS8_{farm_diameter}.npy')
    plt.plot(zeta, label=str(farm_diameter)+' km')
plt.legend()
plt.ylabel(r'$\zeta$')
plt.xlabel('Time (h)')

plt.savefig('plots/zeta_DS8.png')

for no in range(10):
    print(f'Mean absoulte percentage errors for DS{no} (%)')

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
