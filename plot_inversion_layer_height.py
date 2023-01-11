"""Plots to validate calculation
of farm data
"""

import numpy as np
from read_NWP_data import *
import matplotlib.pyplot as plt

DS_no = 2
var_dict = load_NWP_data(f'DS{DS_no}', 30)

theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', 30)
layer_heights = neutral_layer_height(theta_profile, theta_heights)

for i in range(24):
    plt.figure()
    plt.plot(theta_profile[i,:]-theta_profile[i,0], theta_heights)
    plt.axhline(layer_heights[i])
    plt.xlim([-1,7.5])
    plt.ylim([0,1000])
    plt.savefig(f'plots/neutral_layer_height_DS{DS_no}_{i}.png')
    plt.close()

#information about farm CV
cv_height = 250
hubh = 100

var_dict = load_NWP_data(f'DS{DS_no}', 30)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', 30)
neu_layer_height = neutral_layer_height(theta_profile, theta_heights)
print(neu_layer_height)
fr = calculate_fr_number(var_dict, neu_layer_height, wind_dir_0, hubh, farm_diameter)
print(fr)