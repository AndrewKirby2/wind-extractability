import numpy as np
from read_NWP_data import *
from calculate_zeta_components import *

cv_height = 250
hubh = 100

entrainment = np.zeros((5,240))
advection = np.zeros((5,240))
pgf = np.zeros((5,240))
acceleration = np.zeros((5,240))
coriolis = np.zeros((5,240))
farm_diameters = [10, 15, 20, 25, 30]

for i in range(5):
    for DS_no in range(10):
        
        entrainment[i,24*DS_no:24*(DS_no+1)] = np.load(f'data_zeta_components_hcv500/entrainment_term_DS{DS_no}_{farm_diameters[i]}.npy')
        advection[i,24*DS_no:24*(DS_no+1)] = np.load(f'data_zeta_components_hcv500/advection_term_DS{DS_no}_{farm_diameters[i]}.npy')
        pgf[i,24*DS_no:24*(DS_no+1)] = np.load(f'data_zeta_components_hcv500/pgf_term_DS{DS_no}_{farm_diameters[i]}.npy')
        acceleration[i,24*DS_no:24*(DS_no+1)] = np.load(f'data_zeta_components_hcv500/acceleration_term_DS{DS_no}_{farm_diameters[i]}.npy')
        coriolis[i,24*DS_no:24*(DS_no+1)] = np.load(f'data_zeta_components_hcv500/coriolis_term_DS{DS_no}_{farm_diameters[i]}.npy')

    data = [advection[i,:], entrainment[i,:], pgf[i,:], -acceleration[i,:], -coriolis[i,:]]
    ax = plt.axes()
    plt.boxplot(data)
    plt.ylabel(r'Contribution to $\zeta$')
    plt.ylim([-10,50])
    ax.set_xticklabels(['Advection', 'Entrainment', 'PGF', 'Acceleration', 'Coriolis'])
    plt.savefig(f'plots/zeta_components_{farm_diameters[i]}km.png')
    plt.close()

data = [entrainment[0,:], entrainment[1,:], entrainment[2,:], entrainment[3,:], entrainment[4,:]]
ax = plt.axes()
plt.boxplot(data)
plt.ylim([-10,25])
plt.ylabel(r'Contribution to $\zeta$')
plt.xlabel('Farm diameter (km)')
ax.set_xticklabels([10,15,20,25,30])
plt.savefig('plots/entrainment_farm_size.png')
plt.close()

data = [advection[0,:], advection[1,:], advection[2,:], advection[3,:], advection[4,:]]
ax = plt.axes()
plt.boxplot(data)
plt.ylim([-10,50])
ax.set_xticklabels([10,15,20,25,30])
x = np.linspace(10,30,100)
y = 10*np.median(advection[0,:])/x
plt.plot(np.linspace(1,5,100),y)
plt.ylabel(r'Contribution to $\zeta$')
plt.xlabel('Farm diameter (km)')
plt.savefig('plots/advection_farm_size.png')
plt.close()

data = [pgf[0,:], pgf[1,:], pgf[2,:], pgf[3,:], pgf[4,:]]
ax = plt.axes()
plt.boxplot(data)
plt.ylim([-10,25])
ax.set_xticklabels([10,15,20,25,30])
x = np.linspace(10,30,100)
y = 10*np.median(pgf[0,:])/x
plt.plot(np.linspace(1,5,100),y)
plt.ylabel(r'Contribution to $\zeta$')
plt.xlabel('Farm diameter (km)')
plt.savefig('plots/pgf_farm_size.png')
plt.close()


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