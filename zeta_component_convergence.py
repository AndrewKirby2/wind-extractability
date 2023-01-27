import numpy as np
from read_NWP_data import *
from calculate_zeta_components import *
import sklearn.metrics as sk

farm_diameter = 30
DS_no = 9
var_dict = load_NWP_data(f'DS{DS_no}', farm_diameter)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, 250)
wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, 250)

X_top_adv_0, X_top_adv = calculate_X_advection_top(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
X_side_adv_0, X_side_adv = calculate_X_advection_side(var_dict, farm_diameter, 250, wind_dir_0, wind_dir, n_disc=100, n_vert=10)
X_adv_0 = X_top_adv_0 + X_side_adv_0
X_adv = X_top_adv + X_side_adv

zeta = np.load(f'data/zeta_DS{DS_no}_{farm_diameter}.npy')
beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
tauw0 = np.load(f'data/tauw0_DS{DS_no}_{farm_diameter}.npy')
uf0 = np.load(f'data/uf0_DS{DS_no}_{farm_diameter}.npy')

zeta_adv = (250/tauw0) * (X_adv - X_adv_0) / (1 - beta)

for disc in np.arange(15,100,5):

    zeta_prev = zeta_adv

    X_top_adv_0, X_top_adv = calculate_X_advection_top(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
    X_side_adv_0, X_side_adv = calculate_X_advection_side(var_dict, farm_diameter, 250, wind_dir_0, wind_dir, n_disc=100, n_vert=disc)
    X_adv_0 = X_top_adv_0 + X_side_adv_0
    X_adv = X_top_adv + X_side_adv

    zeta_adv = (250/tauw0) * (X_adv - X_adv_0) / (1 - beta)

    print(disc, sk.mean_absolute_error(zeta_prev, zeta_adv))
