"""Plot zeta predictions and compare 
with recorded values
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

for DS_no in range(10):
    for farm_diameter in [10, 15, 20, 25, 30]:

        adv = np.load(f'data_zeta_components_hcv250/advection_term_DS{DS_no}_{farm_diameter}.npy')
        pgf = np.load(f'data_zeta_components_hcv250/pgf_term_DS{DS_no}_{farm_diameter}.npy')

        beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
        cf0 = np.load(f'data/cf0_DS{DS_no}_{farm_diameter}.npy')

        #average farm length
        L = np.pi*farm_diameter*1000/4

        adv_pgf_model = (1/cf0)*(250/L)*(1-beta**2)

        plt.scatter((adv+pgf)*(1-beta), adv_pgf_model, c='b', marker='x', s=10)
        plt.xlim([0,5])
        plt.ylim([0,5])
        plt.plot([0,5],[0,5], c='b')

plt.xlabel(r'Recorded $\frac{\Delta M_{adv} + \Delta M_{pgf}}{M_{F0}}$')
plt.ylabel(r'Predicted $\frac{\Delta M_{adv} + \Delta M_{pgf}}{M_{F0}}$')
plt.savefig(f'plots/adv_pgf_model.png')

plt.close()

for DS_no in range(10):
    for farm_diameter in [10, 15, 20, 25, 30]:

        ent = np.load(f'data_zeta_components_hcv250/entrainment_term_DS{DS_no}_{farm_diameter}.npy')

        beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
        M = np.load(f'data/M_DS{DS_no}_{farm_diameter}.npy')
        tautop0 = np.load(f'data/tautop0_DS{DS_no}_{farm_diameter}.npy')
        tauw0 = np.load(f'data/tauw0_DS{DS_no}_{farm_diameter}.npy')


        ent_model = M + M*beta*(tautop0/tauw0-1) - tautop0/tauw0

        plt.scatter(ent*(1-beta), ent_model, c=tautop0/tauw0, vmin=0, vmax=1, marker='x', s=10)
        plt.xlim([0,5])
        plt.ylim([0,5])
        plt.plot([0,5],[0,5], c='b')

plt.xlabel(r'Recorded $\frac{\Delta M_{ent}}{M_{F0}}$')
plt.ylabel(r'Predicted $\frac{\Delta M_{ent}}{M_{F0}}$')
cbar = plt.colorbar()
cbar.set_label(r'$\tau_{t0}/\tau_{w0}$')
plt.savefig(f'plots/ent_model.png')