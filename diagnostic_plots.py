"""Plots to validate calculation
of farm data
"""

import numpy as np
from read_NWP_data import *
import matplotlib.pyplot as plt

DS_no = 'DS4'
farm_diameter = 30
hubh = 100
cv_height=250

var_dict = load_NWP_data(DS_no, farm_diameter)
u = var_dict['u_mn']
v = var_dict['v_mn']
u_0 = var_dict['u_mn_0']
v_0 = var_dict['v_mn_0']
taux = var_dict['taux_mn']
tauy = var_dict['tauy_mn']
taux_0 = var_dict['taux_mn_0']
tauy_0 = var_dict['tauy_mn_0']
theta_0 = var_dict['theta_mn_0']

#farm parameters
mperdeg = 111132.02
grid = var_dict['u_mn'][0]
zh = grid.coords('level_height')[0].points

#discretisation for interpolation
n_lats = 200
n_lons = 200

lats = np.linspace(-1,1,n_lats)
lons = np.linspace(359,361, n_lons)
u = u.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
v = v.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
u_0 = u_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
v_0 = v_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
taux = taux.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
tauy = tauy.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
taux_0 = taux_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
tauy_0 = tauy_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

theta_0 = theta_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

#mask all data points outside of wind farm CV
mask = np.full(u[:,:,:,:].shape, True)
c_lat = 0.0135 # centre of domain 
c_lon = 360.0135 # centre of domain
count = 0
for i, lat in enumerate(lats):
    dlat = lat - c_lat
    for j, lon in enumerate(lons):
        dlon = lon - c_lon
        d = np.sqrt(dlat*dlat + dlon*dlon)
        if d <= (1000*farm_diameter/2./mperdeg):
            mask[:,:,i,j] = False
            count += 1

#average valid points in the horizontal direction
umean = np.mean(np.ma.array(u.data[:,:,:,:], mask=mask), axis=(2,3))
vmean = np.mean(np.ma.array(v.data[:,:,:,:], mask=mask), axis=(2,3))
umean_0 = np.mean(np.ma.array(u_0.data[:,:,:,:], mask=mask), axis=(2,3))
vmean_0 = np.mean(np.ma.array(v_0.data[:,:,:,:], mask=mask), axis=(2,3))

tauxmean = np.mean(np.ma.array(taux.data[:,:,:,:], mask=mask), axis=(2,3))
tauymean = np.mean(np.ma.array(tauy.data[:,:,:,:], mask=mask), axis=(2,3))
tauxmean_0 = np.mean(np.ma.array(taux_0.data[:,:,:,:], mask=mask), axis=(2,3))
tauymean_0 = np.mean(np.ma.array(tauy_0.data[:,:,:,:], mask=mask), axis=(2,3))

mask = np.full(theta_0[:,:,:,:].shape, True)
c_lat = 0.0135 # centre of domain 
c_lon = 360.0135 # centre of domain
count = 0
for i, lat in enumerate(lats):
    dlat = lat - c_lat
    for j, lon in enumerate(lons):
        dlon = lon - c_lon
        d = np.sqrt(dlat*dlat + dlon*dlon)
        if d <= (1000*farm_diameter/2./mperdeg):
            mask[:,:,i,j] = False
            count += 1

thetamean_0 = np.mean(np.ma.array(theta_0.data[:,:,:,:], mask=mask), axis=(2,3))

umean_func = CV_average(var_dict, 'u_mn', farm_diameter, cv_height)
umean0_func = CV_average(var_dict, 'u_mn_0', farm_diameter, cv_height)
vmean_func = CV_average(var_dict, 'v_mn', farm_diameter, cv_height)
vmean0_func = CV_average(var_dict, 'v_mn_0', farm_diameter, cv_height)
taux_func = surface_average(var_dict, 'taux_mn', farm_diameter)
tauy_func = surface_average(var_dict, 'tauy_mn', farm_diameter)
taux0_func = surface_average(var_dict, 'taux_mn_0', farm_diameter)
tauy0_func = surface_average(var_dict, 'tauy_mn_0', farm_diameter)
wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, hubh)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
print(umean_func[23])
print(umean0_func[23])
print(vmean_func[23])
print(vmean0_func[23])
print(wind_dir[23])
print(wind_dir_0[23])
print(taux_func[23])
print(taux0_func[23])
print(tauy_func[23])
print(tauy0_func[23])
print(calculate_farm_data(DS_no, farm_diameter))

zeta = np.load(f'data/zeta_{DS_no}_{farm_diameter}.npy')
cf0 = np.load(f'data/cf0_{DS_no}_{farm_diameter}.npy')

for i in range(24):
    plt.figure()
    plt.title(str(round(zeta[i],1))+'    '+str(round(cf0[i],4)))
    plt.plot(np.sqrt(tauxmean_0[i,:]**2+tauymean_0[i,:]**2), zh, label='taux_0', color='blue')
    plt.xlabel('Shear stress (m/s)')
    plt.ylabel('Height (m)')
    plt.ylim([0,1000])
    plt.xlim([-0.1,0.5])
    plt.savefig(f'plots/stress_profiles_{DS_no}_{i}.png')
    plt.close()

grid = var_dict['theta_mn_0'][0]
zh = grid.coords('level_height')[0].points

for i in range(24):
    plt.figure()
    plt.title(str(round(zeta[i],1))+'    '+str(round(cf0[i],4)))
    plt.plot(thetamean_0[i,:]-thetamean_0[i,0], zh, label='taux', color='blue')
    plt.xlabel(r'$\theta$ (K)')
    plt.ylabel('Height (m)')
    plt.ylim([0,1000])
    plt.xlim([0,10])
    plt.savefig(f'plots/theta_profiles_{DS_no}_{i}.png')
    plt.close()