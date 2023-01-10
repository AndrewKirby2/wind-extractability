"""Plots to validate calculation
of farm data
"""

import numpy as np
from read_NWP_data import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp

fig, ax = plt.subplots(dpi=300)
norm = mpl.colors.Normalize(vmin=15, vmax=70)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

for no in range(10):
    print(no)
    DS_no = f'DS{no}'
    farm_diameter = 10
    zeta = np.load(f'data/zeta_{DS_no}_{farm_diameter}.npy')
    cf0 = np.load(f'data/cf0_{DS_no}_{farm_diameter}.npy')
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
    u_0 = u_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
    v_0 = v_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
    taux_0 = taux_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
    tauy_0 = tauy_0.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
    wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
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
    umean_0 = np.mean(np.ma.array(u_0.data[:,:,:,:], mask=mask), axis=(2,3))
    vmean_0 = np.mean(np.ma.array(v_0.data[:,:,:,:], mask=mask), axis=(2,3))
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

    for i in range(24):
        grid = var_dict['u_mn'][0]
        zh_u = grid.coords('level_height')[0].points

        grid = var_dict['theta_mn_0'][0]
        zh_th = grid.coords('level_height')[0].points
        delta_theta = np.gradient(thetamean_0[i,:], zh_th)
        N_sq = (9.81/thetamean_0[i,0])*delta_theta
        sc = sp.interpolate.CubicSpline(zh_u, umean_0[i,:])
        umean_i = sc(zh_th)
        sc = sp.interpolate.CubicSpline(zh_u, vmean_0[i,:])
        vmean_i = sc(zh_th)
        u_mag = umean_i*np.cos(wind_dir_0[i]) + vmean_i*np.sin(wind_dir_0[i])
        inv_fr = (np.sqrt(N_sq)*100)/u_mag
        inv_fr = np.nan_to_num(inv_fr)
        sc = sp.interpolate.CubicSpline(zh_th, inv_fr)
        inv_fr = sc(np.linspace(0,250,251))
        inv_fr_bar = np.sum(inv_fr)/250.
        if zeta[i] < 45 and zeta[i] > 0:
            ax.scatter(inv_fr_bar, cf0[i], color=cmap.to_rgba(zeta[i]), vmin=15, vmax=70)
                

ax.set_xlabel(r'$\overline{1/Fr_0}$')
ax.set_ylabel(r'$C_{f0}$')
ax.set_ylim([0.0005,0.003])
ax.set_xlim([0,0.3])
cbar = fig.colorbar(cmap)
cbar.set_label(r'$\zeta$')
plt.tight_layout()
plt.savefig(f'plots/zeta_10km_fr_ave.png')
plt.close()