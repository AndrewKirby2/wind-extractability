"""Calculate the different components
which contribute to zeta
See  https://doi.org/10.1017/jfm.2022.979
for more details
"""

from read_NWP_data import *
import numpy as np
import scipy.interpolate as sp

def calculate_X_reynolds(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir, n_disc=200):
    """  Calculates the inflow of momentum X due to
    Reynolds stress at the top surface of control volume
    (note that this per unit volume)

    Parameters
    ----------
    var_dict : iris Cube
        Data for variables with and without farm present
    farm_diamater: int
        wind farm diameter in kilometres
    hubh: int
        wind turbine hub height
    wind_dir_0 : numpy array (size 24)
        hubh wind direction without turbines in radians
    wind_dir : numpy array (size 24)
        hubh wind direction with turbines in radians
    n_disc : int
        number of discresation points

    Returns
    -------
    X_top_0 : numpy array (size 24)
        net momentum injection per unit volume without turbines(N/m^3)
    X_top : numpy array (size 24)
        net momentum injection per unit volume with turbines (N/m^3)
    """
    #farm parameters
    mperdeg = 111132.02
    grid = var_dict['taux_mn'][0]
    zh = grid.coords('level_height')[0].points

    #discretisation for interpolation
    n_lats = n_disc
    n_lons = n_disc

    #extract data from NWP simulations
    taux_mn = var_dict['taux_mn']
    taux_mn_0 = var_dict['taux_mn_0']
    tauy_mn = var_dict['tauy_mn']
    tauy_mn_0 = var_dict['tauy_mn_0']

    #grid interpolation
    lats = np.linspace(-1,1,n_lats)
    lons = np.linspace(359,361, n_lons)
    taux_mn = taux_mn.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    taux_mn_0 = taux_mn_0.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    tauy_mn = tauy_mn.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    tauy_mn_0 = tauy_mn_0.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())

    #mask all data points outside of wind farm CV
    mask = np.full(tauy_mn[:,:,:].shape, True)
    c_lat = 0.0135 # centre of domain 
    c_lon = 360.0135 # centre of domain
    count = 0
    for i, lat in enumerate(lats):
        dlat = lat - c_lat
        for j, lon in enumerate(lons):
            dlon = lon - c_lon
            d = np.sqrt(dlat*dlat + dlon*dlon)
            if d <= (1000*farm_diameter/2./mperdeg):
                mask[:,i,j] = False
                count += 1

    #calculate shear stress at the top surface in the hub height wind direction with turbines present
    taux_mn_0 = np.mean(np.ma.array(taux_mn_0.data[:,:,:], mask=mask), axis=(1,2))
    tauy_mn_0 = np.mean(np.ma.array(tauy_mn_0.data[:,:,:], mask=mask), axis=(1,2))
    tau_xf_0 = taux_mn_0.data*np.cos(wind_dir_0) + tauy_mn_0.data*np.sin(wind_dir_0)

    #calculate shear stress at the top surface in the hub height wind direction with turbines present
    taux_mn = np.mean(np.ma.array(taux_mn.data[:,:,:], mask=mask), axis=(1,2))
    tauy_mn = np.mean(np.ma.array(tauy_mn.data[:,:,:], mask=mask), axis=(1,2))
    tau_xf = taux_mn.data*np.cos(wind_dir) + tauy_mn.data*np.sin(wind_dir)

    X_top_0 = tau_xf_0 / cv_height
    X_top = tau_xf / cv_height

    return X_top_0, X_top

def calculate_X_advection_top(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir, n_disc=200):
    """  Calculates the inflow of momentum X due to
    advection through top of control volume
    (note that this per unit volume)

    Parameters
    ----------
    var_dict : iris Cube
        Data for variables with and without farm present
    farm_diamater: int
        wind farm diameter in kilometres
    hubh: int
        wind turbine hub height
    wind_dir_0 : numpy array (size 24)
        hubh wind direction without turbines in radians
    wind_dir : numpy array (size 24)
        hubh wind direction with turbines in radians
    n_disc : int
        number of grid points in longitude and
        latitude for interpolation

    Returns
    -------
    X_top_0 : numpy array (size 24)
        net momentum injection per unit volume without turbines(N/m^3)
    X_top : numpy array (size 24)
        net momentum injection per unit volume with turbines (N/m^3)
    """

    #farm parameters
    mperdeg = 111132.02
    grid = var_dict['taux_mn'][0]
    zh = grid.coords('level_height')[0].points

    #discretisation for interpolation
    n_lats = n_disc
    n_lons = n_disc

    #extract data from NWP simulations
    u_mn = var_dict['u_mn']
    u_mn_0 = var_dict['u_mn_0']
    v_mn = var_dict['v_mn']
    dens_mn = var_dict['dens_mn']
    v_mn_0 = var_dict['v_mn_0']
    w_mn = var_dict['w_mn']
    w_mn_0 = var_dict['w_mn_0']
    dens_mn_0 = var_dict['dens_mn_0']

    #calculate momentum advection through top surface
    lats = np.linspace(-1,1,n_lats)
    lons = np.linspace(359,361, n_lons)
    u_mn = u_mn.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    v_mn = v_mn.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    w_mn = w_mn.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    dens_mn = dens_mn.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())

    u_mn_0 = u_mn_0.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    v_mn_0 = v_mn_0.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    w_mn_0 = w_mn_0.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())
    dens_mn_0 = dens_mn_0.interpolate([('grid_latitude', lats),('grid_longitude', lons),('level_height', cv_height)], iris.analysis.Linear())

    #calculate momentum flux at the top of control surface
    X_adv_top = -dens_mn*w_mn*(u_mn*np.cos(wind_dir[:, np.newaxis, np.newaxis]) 
                        + v_mn*np.sin(wind_dir[:, np.newaxis, np.newaxis]))
    X_adv_top_0 = -dens_mn_0*w_mn_0*(u_mn_0*np.cos(wind_dir_0[:, np.newaxis, np.newaxis])
                         + v_mn_0*np.sin(wind_dir_0[:, np.newaxis, np.newaxis]))

    #mask all data points outside of wind farm CV
    mask = np.full(u_mn[:,:,:].shape, True)
    c_lat = 0.0135 # centre of domain 
    c_lon = 360.0135 # centre of domain
    count = 0
    for i, lat in enumerate(lats):
        dlat = lat - c_lat
        for j, lon in enumerate(lons):
            dlon = lon - c_lon
            d = np.sqrt(dlat*dlat + dlon*dlon)
            if d <= (1000*farm_diameter/2./mperdeg):
                mask[:,i,j] = False
                count += 1
    
    #average momentum flux at the top surface of control volume
    X_adv_top = np.mean(np.ma.array(X_adv_top.data[:,:,:], mask=mask), axis=(1,2))
    X_adv_top_0 = np.mean(np.ma.array(X_adv_top_0.data[:,:,:], mask=mask), axis=(1,2))

    #calculate momentum flux per unit volume
    X_adv_top = X_adv_top / cv_height
    X_adv_top_0 = X_adv_top_0 / cv_height

    return X_adv_top_0, X_adv_top

def calculate_X_advection_side(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir, n_disc=15):
    """  Calculates the inflow of momentum X due to
    advection through top of control volume
    (note that this per unit volume)

    Parameters
    ----------
    var_dict : iris Cube
        Data for variables with and without farm present
    farm_diamater: int
        wind farm diameter in kilometres
    hubh: int
        wind turbine hub height
    wind_dir_0 : numpy array (size 24)
        hubh wind direction without turbines in radians
    wind_dir : numpy array (size 24)
        hubh wind direction with turbines in radians
    n_disc : int
        number of grid points in azimuthal and vertical
        direction for interpolation
    
    Returns
    -------
    X_side_0 : numpy array (size 24)
        net momentum injection per unit volume without turbines(N/m^3)
    X_side : numpy array (size 24)
        net momentum injection per unit volume with turbines (N/m^3)
    """

    #farm parameters
    mperdeg = 111132.02
    c_lat = 0.0135 # centre of domain 
    c_lon = 360.0135 # centre of domain

    #discretisation for interpolation
    n_azi = n_disc
    n_vert = 250
    heights = np.linspace(0, cv_height, n_vert)

    #extract data from NWP simulations
    u_mn = var_dict['u_mn']
    u_mn_0 = var_dict['u_mn_0']
    v_mn = var_dict['v_mn']
    dens_mn = var_dict['dens_mn']
    v_mn_0 = var_dict['v_mn_0']
    dens_mn_0 = var_dict['dens_mn_0']

    #calculate momentum advection through top surface
    angles = np.linspace(0,2*np.pi*(n_azi-1)/n_azi, n_azi)
    lons = c_lon + (1000*farm_diameter/2./mperdeg)*np.cos(angles)
    lats = c_lat + (1000*farm_diameter/2./mperdeg)*np.sin(angles)

    mom_in = np.zeros((24,n_azi))
    mom_in_0 = np.zeros((24,n_azi))

    for i in range(n_azi):
        u = u_mn.interpolate([('grid_latitude', lats[i]),('grid_longitude', lons[i]),('level_height', heights)], iris.analysis.Linear())
        v = v_mn.interpolate([('grid_latitude', lats[i]),('grid_longitude', lons[i]),('level_height', heights)], iris.analysis.Linear())
        dens = dens_mn.interpolate([('grid_latitude', lats[i]),('grid_longitude', lons[i]),('level_height', heights)], iris.analysis.Linear())

        u_0 = u_mn_0.interpolate([('grid_latitude', lats[i]),('grid_longitude', lons[i]),('level_height', heights)], iris.analysis.Linear())
        v_0 = v_mn_0.interpolate([('grid_latitude', lats[i]),('grid_longitude', lons[i]),('level_height', heights)], iris.analysis.Linear())
        dens_0 = dens_mn_0.interpolate([('grid_latitude', lats[i]),('grid_longitude', lons[i]),('level_height', heights)], iris.analysis.Linear())

        m_in = - dens.data * u.data * np.cos(angles[i]) - dens.data * v.data * np.sin(angles[i])
        m_in_0 = - dens_0.data * u_0.data * np.cos(angles[i]) - dens_0.data * v_0.data * np.sin(angles[i])

        mom_in_tmp = m_in * (u.data * np.cos(wind_dir[:,np.newaxis]) + v.data * np.sin(wind_dir[:,np.newaxis]))
        mom_in_0_tmp = m_in_0 * (u_0.data * np.cos(wind_dir_0[:,np.newaxis]) + v_0.data * np.sin(wind_dir_0[:,np.newaxis]))

        mom_in[:,i] = np.mean(mom_in_tmp, axis=1)
        mom_in_0[:,i] = np.mean(mom_in_0_tmp, axis=1)
    
    X_side = np.mean(mom_in, axis=1)/ (farm_diameter*1000/4.0)
    X_side_0 = np.mean(mom_in_0, axis=1)/ (farm_diameter*1000/4.0)
    return X_side_0, X_side

farm_diameter = 30
DS_no = 0
var_dict = load_NWP_data(f'DS{DS_no}', farm_diameter)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, 250)
wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, 250)
var_dict['u_mn'].data[:,:,:,:] = 10.0
var_dict['v_mn'].data[:,:,:,:] = 0.0
var_dict['dens_mn'].data[:,:,:,:] = 1.0
lons = var_dict['u_mn'].coords('grid_longitude')[0].points
for i in range(np.size(lons)):
    if lons[i] > 360.0135:
        var_dict['u_mn'].data[:,:,:,i] = 0.0
wind_dir = np.zeros(24)
#X_adv_0, X_adv = calculate_X_advection_side(var_dict, farm_diameter, 250, wind_dir_0, wind_dir, n_disc=10)
#print(X_adv)
