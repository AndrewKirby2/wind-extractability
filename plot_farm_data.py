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

for cf0_min in np.arange(0.001,0.0025,0.0001):
    print(cf0_min)
    cf0_max = cf0_min + 0.0001
    norm = mpl.colors.Normalize(vmin=0, vmax=35)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
    cmap.set_array([])

    for no in range(10):
        print(no)
        zeta = np.load(f'data/zeta_DS{no}_30.npy')
        cf0 = np.load(f'data/cf0_DS{no}_30.npy')
        cond1 = np.logical_and(cf0>cf0_min,cf0<cf0_max)
        cond2 = np.logical_and(zeta>0,zeta<35)
        cond = np.logical_and(cond1, cond2)
        var_dict = load_NWP_data(f'DS{no}', 30)
        rig_profile, rig_heights = farm_vertical_profile(var_dict, 'rig_mn_0', 30)
        for i in range(24):
            if cond[i]:
                print(cf0[i])
                plt.plot( rig_profile[i,:], rig_heights, color=cmap.to_rgba(zeta[i]))

    plt.ylabel(r'Height (m)')
    plt.xlabel(r'$Ri_g$')
    cbar = plt.colorbar(cmap)
    cbar.set_label('$\zeta$')
    plt.xlim([-2, 2])
    plt.ylim([0, 1000])
    plt.savefig(f'plots/zeta_30km_cf0_{np.round(cf0_min,4)}_to_{np.round(cf0_max,4)}.png')
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
