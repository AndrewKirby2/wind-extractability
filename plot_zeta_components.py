import numpy as np
from read_NWP_data import *
from calculate_zeta_components import *

farm_diameter = 30
DS_no = 1
var_dict = load_NWP_data(f'DS{DS_no}', farm_diameter)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, 250)
wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, 250)
X_top_rey_0, X_top_rey = calculate_X_reynolds(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
X_top_adv_0, X_top_adv = calculate_X_advection_top(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
X_side_adv_0, X_side_adv = calculate_X_advection_side(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
pres_term_0, pres_term = calculate_PGF(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
accel_0, accel = calculate_acceleration(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
zeta = np.load(f'data/zeta_DS{DS_no}_{farm_diameter}.npy')
beta = np.load(f'data/beta_DS{DS_no}_{farm_diameter}.npy')
tauw0 = np.load(f'data/tauw0_DS{DS_no}_{farm_diameter}.npy')
uf0 = np.load(f'data/uf0_DS{DS_no}_{farm_diameter}.npy')

top_rey = (250/tauw0) * (X_top_rey - X_top_rey_0) / (1 - beta)
top_adv = (250/tauw0) * (X_top_adv - X_top_adv_0) / (1 - beta)
side_adv =  (250/tauw0) * (X_side_adv - X_side_adv_0) / (1 - beta)
pgf =  (250/tauw0) * (pres_term - pres_term_0) / (1 - beta)
time = (250/tauw0) * (accel - accel_0) / (1 - beta)

plt.plot(top_rey, label='Reynolds stress top surface')
plt.plot(top_adv, label='Advection top surface')
plt.plot(side_adv, label='Advection side surfaces')
plt.plot(pgf, label='Pressure gradient forcing')
plt.plot(-time, label='Acceleration term')
plt.plot(top_rey+top_adv+side_adv+pgf-time, c='k')
plt.plot(zeta, c='k', linestyle='--')
print(top_rey+top_adv+side_adv+pgf-time)
print(zeta)
plt.legend()
plt.savefig(f'plots/zeta_components_DS{DS_no}.png')
plt.close()

plt.scatter(uf0, pgf)
plt.savefig('plots/pres_term.png')