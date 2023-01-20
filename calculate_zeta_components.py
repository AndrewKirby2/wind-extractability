"""Calculate the different components
which contribute to zeta
See  https://doi.org/10.1017/jfm.2022.979
for more details
"""

from read_NWP_data import *
import numpy as np

def calculate_X_reynolds(var_dict, farm_diameter, cv_height, wind_dir_0, wind_dir):
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

    Returns
    -------
    X_top_0 : numpy array (size 24)
    net momentum injection per unit volume (N/m^3)
    """
    #farm parameters
    mperdeg = 111132.02
    grid = var_dict['taux_mn'][0]
    zh = grid.coords('level_height')[0].points

    #discretisation for interpolation
    n_lats = 200
    n_lons = 200

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

farm_diameter = 30
var_dict = load_NWP_data('DS3', farm_diameter)
wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, 250)
wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, 250)
print(calculate_X_reynolds(var_dict, farm_diameter, 250, wind_dir_0, wind_dir))