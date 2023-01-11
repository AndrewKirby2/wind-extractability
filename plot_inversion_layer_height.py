"""Plots to validate calculation
of farm data
"""

import numpy as np
from read_NWP_data import *
import matplotlib.pyplot as plt

DS_no = 1
var_dict = load_NWP_data(f'DS{DS_no}', 30)

theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', 30)
layer_heights, theta_profile_interp, interp_heights = neutral_layer_height(theta_profile, theta_heights)
print(layer_heights)
print(np.load(f'data/zeta_DS{DS_no}_30.npy'))

for i in range(24):
    plt.figure()
    plt.plot(theta_profile[i,:]-theta_profile[i,0], theta_heights)
    plt.axhline(layer_heights[i])
    plt.xlim([-1,7.5])
    plt.ylim([0,1000])
    plt.savefig(f'plots/neutral_layer_height_DS{DS_no}_{i}.png')
    plt.close()

var_dict = load_NWP_data(f'DS{DS_no}', 30)

#information about farm CV
cv_height = 250
hubh = 100

wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
u_mean_0 = CV_average(var_dict, 'u_mn_0', farm_diameter, cv_height)
v_mean_0 = CV_average(var_dict, 'v_mn_0', farm_diameter, cv_height)

# calculate farm-layer-averaged streamwise velocity U_F
uf_0 = u_mean_0*np.cos(wind_dir_0) + v_mean_0*np.sin(wind_dir_0)

theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', 30)
layer_heights, theta_profile_interp, interp_heights = neutral_layer_height(theta_profile, theta_heights)

fr = calculate_fr_number(uf_0, hubh, layer_heights, theta_profile_interp, interp_heights)
print(fr)