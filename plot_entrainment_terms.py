"""Plot the advection terms of zeta
"""

import numpy as np
import matplotlib.pyplot as plt
from read_NWP_data import *

for farm_diameter in [10,15,20,25,30]:
    for DS_no in range(10):
        entrainment_term = np.load(f'data_zeta_components_hcv250/entrainment_term_DS{DS_no}_{farm_diameter}.npy')
        zeta = np.load(f'data/zeta_DS{DS_no}_{farm_diameter}.npy')
        beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
        tauw0 = np.load(f'data/tauw0_DS{DS_no}_{farm_diameter}.npy')
        cf0 = np.load(f'data/cf0_DS{DS_no}_{farm_diameter}.npy')
        uf0 = np.load(f'data/uf0_DS{DS_no}_{farm_diameter}.npy')
        tautop0 = np.load(f'data/tautop0_DS{DS_no}_{farm_diameter}.npy')
        tautop = np.load(f'data/tautop_DS{DS_no}_{farm_diameter}.npy')
        M = np.load(f'data/M_DS{DS_no}_{farm_diameter}.npy')
        entrainment = tauw0*(1-beta)/250
        stress_ratio = tautop0/tauw0
        model = tauw0*(1-beta)
        plt.scatter(tautop0/tauw0, entrainment_term, c=1-beta, vmin=0.05, vmax=0.15)
cbar = plt.colorbar()
plt.ylim([-5,30])
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('plots/entrainment_terms.png')
plt.close()

for time_no in range(24):
    print(time_no)
    var_dict = load_NWP_data('DS1', 20)
    wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], 20, 100)
    taux0_profile, tau_heights = farm_vertical_profile(var_dict, 'taux_mn_0', 20)
    tauy0_profile, tau_heights = farm_vertical_profile(var_dict, 'tauy_mn_0', 20)
    tauxf0 = taux0_profile*np.cos(wind_dir_0[:,np.newaxis]) + tauy0_profile*np.sin(wind_dir_0[:,np.newaxis])
    plt.plot(tauxf0[time_no,:]/tauxf0[time_no,0], tau_heights, linestyle='--')

    for z0 in ['0p05', '0p1', '0p35', '0p7', '1p4']:
        print(z0)
        M = np.load(f'data/M_DS1_20_{z0}.npy')
        var_dict = load_NWP_data('DS1', 20, z0)
        wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], 20, 100)
        taux_profile, tau_heights = farm_vertical_profile(var_dict, 'taux_mn', 20)
        tauy_profile, tau_heights = farm_vertical_profile(var_dict, 'tauy_mn', 20)
        tauxf = taux_profile*np.cos(wind_dir[:,np.newaxis]) + tauy_profile*np.sin(wind_dir[:,np.newaxis])
        beta = np.load(f'data/beta_DS1_20_{z0}.npy')
        plt.plot(tauxf[time_no,:]/tauxf[time_no,0], tau_heights/np.sqrt(M[time_no]), label=z0)

    plt.xlim([0,1])
    plt.ylim([0,1000])
    plt.legend()
    plt.savefig(f'plots/DS1_stress_profiles_{time_no}.png')
    plt.close()
