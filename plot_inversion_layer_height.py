"""Plots to validate calculation
of farm data
"""

import numpy as np
from read_NWP_data import *
import matplotlib.pyplot as plt

DS_no = 8
var_dict = load_NWP_data(f'DS{DS_no}', 30)

theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', 30)
layer_heights = neutral_layer_height(theta_profile, theta_heights)

for i in range(24):
    plt.figure()
    plt.plot(theta_profile[i,:]-theta_profile[i,0], theta_heights)
    plt.axhline(layer_heights[i])
    plt.xlim([-1,7.5])
    plt.ylim([0,2000])
    plt.savefig(f'plots/neutral_layer_height_DS{DS_no}_{i}.png')
    plt.close()
