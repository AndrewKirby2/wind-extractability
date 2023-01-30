import numpy as np
from read_NWP_data import *
from calculate_zeta_components import *

cv_height = 250
farm_diameter = 30
hubh = 100

for farm_diameter in [10, 15, 20, 25, 30]:
    print(farm_diameter)
    for DS_no in range(10):
        print(DS_no)
        var_dict = load_NWP_data(f'DS{DS_no}', farm_diameter)
        wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
        wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, hubh)
        X_top_rey_0, X_top_rey = calculate_X_reynolds(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
        X_top_adv_0, X_top_adv = calculate_X_advection_top(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
        X_side_adv_0, X_side_adv = calculate_X_advection_side(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
        X_adv_0 = X_top_adv_0 + X_side_adv_0
        X_adv = X_top_adv + X_side_adv
        pres_term_0, pres_term = calculate_PGF(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
        accel_0, accel = calculate_acceleration(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
        C_0, C = calculate_coriolis_term(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
        zeta = np.load(f'data/zeta_DS{DS_no}_{farm_diameter}.npy')
        beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
        tauw0 = np.load(f'data/tauw0_DS{DS_no}_{farm_diameter}.npy')
        uf0 = np.load(f'data/uf0_DS{DS_no}_{farm_diameter}.npy')

        top_rey = (250/tauw0) * (X_top_rey - X_top_rey_0) / (1 - beta)
        adv = (250/tauw0) * (X_adv - X_adv_0) / (1 - beta)
        pgf =  (250/tauw0) * (pres_term - pres_term_0) / (1 - beta)
        time = (250/tauw0) * (accel - accel_0) / (1 - beta)
        coriolis = (250/tauw0) * (C - C_0) / (1 - beta)

        np.save(f'data_zeta_components_hcv250/entrainment_term_DS{DS_no}_{farm_diameter}.npy', top_rey)
        np.save(f'data_zeta_components_hcv250/advection_term_DS{DS_no}_{farm_diameter}.npy', adv)
        np.save(f'data_zeta_components_hcv250/pgf_term_DS{DS_no}_{farm_diameter}.npy', pgf)
        np.save(f'data_zeta_components_hcv250/acceleration_term_DS{DS_no}_{farm_diameter}.npy', time)
        np.save(f'data_zeta_components_hcv250/coriolis_term_DS{DS_no}_{farm_diameter}.npy', coriolis)


farm_diameter = 30
DS_no = 4
var_dict = load_NWP_data(f'DS{DS_no}', farm_diameter)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, hubh)
accel_0, accel = calculate_acceleration(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir)
uf0 = np.load(f'data/uf0_DS{DS_no}_{farm_diameter}.npy')
beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
plt.plot(accel_0)
plt.plot(accel)
plt.grid(True)
plt.savefig(f'plots/accel_{DS_no}_{farm_diameter}.png')
plt.close()
plt.plot(uf0)
plt.plot(beta*uf0)
plt.grid(True)
plt.savefig(f'plots/velocities_{DS_no}_{farm_diameter}.png')
plt.close()

farm_diameter = 10
for DS_no in range(0):
    var_dict = load_NWP_data(f'DS{DS_no}', farm_diameter)
    wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
    wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, hubh)
    uf0 = np.load(f'data/uf0_DS{DS_no}_{farm_diameter}.npy')
    beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
    plt.polar(wind_dir_0, range(24))
    plt.plot(wind_dir, range(24))
    plt.savefig(f'plots/DS{DS_no}_wind_dir.png')
    plt.close()
    plt.plot(uf0)
    plt.plot(beta*uf0)
    plt.savefig(f'plots/DS{DS_no}_wind_speeds.png')
    plt.close()
    plt.plot(beta)
    plt.savefig(f'plots/DS{DS_no}_beta.png')
    plt.close()