"""Plot the advection terms of zeta
"""

import numpy as np
import matplotlib.pyplot as plt
from read_NWP_data import *
import scipy.stats as sp

advection_pgf = np.zeros(240)
advection = np.zeros(240)
inv_cf0 = np.zeros(240)

for DS_no in range(10):
    advection_term = np.load(f'data_zeta_components_hcv250/advection_term_DS{DS_no}_30.npy')
    pgf_term = np.load(f'data_zeta_components_hcv250/pgf_term_DS{DS_no}_30.npy')
    advection_pgf[DS_no*24:(DS_no+1)*24] = advection_term + pgf_term
    cf0 = np.load(f'data/cf0_DS{DS_no}_30.npy')
    zeta = np.load(f'data/zeta_DS{DS_no}_30.npy')
    beta = np.load(f'data/beta_DS{DS_no}_30.npy')
    inv_cf0[DS_no*24:(DS_no+1)*24] = 1/cf0
    model = (1/cf0)*(250./30e3)*(1+beta)
    plt.scatter(1/cf0, advection_term+pgf_term, c=beta, vmin=0.85, vmax=0.95, label='DS'+str(DS_no))
#plt.legend()
cbar = plt.colorbar()
cbar.set_label(r'$\beta$')
plt.ylim([0,50])
plt.xlim([0,1500])
plt.xlabel(r'$1/C_{f0}$')
plt.ylabel(r'Advection and pgf contribution to $\zeta$')
plt.savefig('plots/advection_pgf_beta_30.png')
plt.close()

for DS_no in range(10):
    advection_term = np.load(f'data_zeta_components_hcv250/advection_term_DS{DS_no}_30.npy')
    pgf_term = np.load(f'data_zeta_components_hcv250/pgf_term_DS{DS_no}_30.npy')
    advection[DS_no*24:(DS_no+1)*24] = advection_term
    cf0 = np.load(f'data/cf0_DS{DS_no}_30.npy')
    zeta = np.load(f'data/zeta_DS{DS_no}_30.npy')
    inv_cf0[DS_no*24:(DS_no+1)*24] = 1/cf0
    plt.scatter(1/cf0, advection_term, label='DS'+str(DS_no))
    #for i in range(24):
    #    plt.annotate(str(i),(1/cf0[i],advection_term[i]))
plt.legend()
plt.ylim([0,50])
plt.xlim([0,1500])
plt.xlabel(r'$1/C_{f0}$')
plt.ylabel(r'Advection contribution to $\zeta$')
plt.savefig('plots/advection_cf0_30.png')
plt.close()

print(sp.pearsonr(inv_cf0, advection))
print(sp.spearmanr(inv_cf0, advection))
print(sp.kendalltau(inv_cf0, advection))
