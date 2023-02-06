import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from read_NWP_data import *

DS_no = 1
M = np.load('data/M_DS5_30.npy')
cf0 = np.load('data/cf0_DS5_30.npy')
beta = np.load('data/beta_DS5_30.npy')
tauw0 = np.load('data/tauw0_DS5_30.npy')
var_dict = load_NWP_data(f'DS{DS_no}', 30)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], 30, 100)
taux0_profile, tau_heights = farm_vertical_profile(var_dict, 'taux_mn_0', 30)
tauy0_profile, tau_heights = farm_vertical_profile(var_dict, 'tauy_mn_0', 30)
tauxf0 = taux0_profile*np.cos(wind_dir_0[:,np.newaxis]) + tauy0_profile*np.sin(wind_dir_0[:,np.newaxis])

tauxf0_hcv = np.zeros(24)
tauxf0_beta_hcv = np.zeros(24)

for i in range(24):
    #interpolate velocity at turbine hub height
    sc = sp.interpolate.CubicSpline(tau_heights, tauxf0[i,:])
    tauxf0_hcv[i] = sc(250)
    tauxf0_beta_hcv[i] = sc(beta[i]*250)

print(tauxf0_hcv/tauw0)
model_numerator = 1 + ((1/cf0) * (1e3/(np.pi*30e3)) * (1 - beta**2)) - (tauxf0_hcv/tauw0)
model_denominator = 1 - (tauxf0_beta_hcv/tauw0)
plt.plot(M)
plt.plot(model_numerator/model_denominator)
plt.savefig(f'plots/DS{DS_no}_M.png')