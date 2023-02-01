"""Plot the advection terms of zeta
"""

import numpy as np
import matplotlib.pyplot as plt
from read_NWP_data import *

advection = np.zeros(240)
inv_cf0 = np.zeros(240)

for DS_no in range(10):
    advection_term = np.load(f'data_zeta_components_hcv250/advection_term_DS{DS_no}_30.npy')
    advection[DS_no*24:(DS_no+1)*24] = advection_term
    cf0 = np.load(f'data/cf0_DS{DS_no}_30.npy')
    zeta = np.load(f'data/zeta_DS{DS_no}_30.npy')
    inv_cf0[DS_no*24:(DS_no+1)*24] = 1/cf0
    plt.scatter(1/cf0, advection_term, label='DS'+str(DS_no))
    for i in range(24):
        plt.annotate(str(i),(1/cf0[i],advection_term[i]))
plt.legend()
plt.ylim([0,50])
plt.savefig('plots/advection_cf0_30.png')
plt.close()
