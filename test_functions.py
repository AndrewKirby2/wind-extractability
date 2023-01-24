"""Tests for functions analysing NWP data"""

import numpy as np
import numpy.testing as npt
from read_NWP_data import *
from calculate_zeta_components import *

def test_X_adv_side():
    """Test calculation of momentum advection at
    top surface
    """
    var_dict = load_NWP_data('DS8', 30)
    wind_dir_0 = np.linspace(0,2*np.pi,24)
    wind_dir = np.zeros(24)
    var_dict['u_mn_0'].data[:,:,:,:] = 10.0
    var_dict['v_mn_0'].data[:,:,:,:] = 10.0
    var_dict['dens_mn_0'].data[:,:,:,:] = 1.0
    X_side_0, X_side = calculate_X_advection_side(var_dict, 30, 250, wind_dir_0, wind_dir, n_disc=10)
    npt.assert_almost_equal(X_side_0, np.zeros(24))
    var_dict['u_mn'].data[:,:,:,:] = 10.0
    var_dict['v_mn'].data[:,:,:,:] = 0.0
    var_dict['dens_mn'].data[:,:,:,:] = 1.0
    lons = var_dict['u_mn'].coords('grid_longitude')[0].points
    for i in range(np.size(lons)):
        if lons[i] > 360.0135:
            var_dict['u_mn'].data[:,:,:,i] = 0.0
    X_side_0, X_side = calculate_X_advection_side(var_dict, 30, 250, wind_dir_0, wind_dir, n_disc=15)
    npt.assert_allclose(X_side, 4.24413181e-3*np.ones(24), rtol=0.005)

def test_X_adv_top():
    """Test calculation of momentum advection at
    top surface
    """
    var_dict = load_NWP_data('DS8', 30)
    wind_dir_0 = np.ones(24)
    wind_dir = np.zeros(24)
    u_mn_0 = var_dict['u_mn']
    var_dict['u_mn'].data[:,:,:,:] = 10.0
    var_dict['w_mn'].data[:,:,:,:] = -0.1
    var_dict['dens_mn'].data[:,:,:,:] = 1.0
    X_top_0, X_top = calculate_X_advection_top(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
    npt.assert_almost_equal(X_top, 4e-3*np.ones(24))
    var_dict['u_mn_0'].data[:,:,:,:] = 10.0
    var_dict['v_mn_0'].data[:,:,:,:] = 5.0
    var_dict['w_mn_0'].data[:,:,:,:] = -0.1
    var_dict['dens_mn_0'].data[:,:,:,:] = 0.75
    X_top_0, X_top = calculate_X_advection_top(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
    npt.assert_almost_equal(X_top_0, 2.883113395e-3*np.ones(24))

def test_X_top():
    """Test calculation of Reynolds stress at
    top surface
    """
    var_dict = load_NWP_data('DS8', 30)
    wind_dir_0 = np.ones(24)
    wind_dir = np.zeros(24)
    taux_mn_0 = var_dict['taux_mn']
    zh = var_dict['taux_mn'].coords('level_height')[0].points
    for i in range(40):
        var_dict['taux_mn'].data[:,i,:,:] = zh[i]/1000.0
        var_dict['taux_mn_0'].data[:,i,:,:] = zh[i]/2000.0
        var_dict['tauy_mn_0'].data[:,i,:,:] = zh[i]/1000.0
    X_top_0, X_top = calculate_X_reynolds(var_dict, farm_diameter, 250, wind_dir_0, wind_dir)
    npt.assert_array_equal(X_top, 1e-3*np.ones(24))
    npt.assert_almost_equal(X_top_0, 1.111622138e-3*np.ones(24))

def test_dummy():
    npt.assert_array_equal(np.array([0]),np.array([0]))

def test_wind_direction():
    """ Test wind direction calculation is correct for
    positive and negative wind directions
    """
    var_dict = load_NWP_data('DS5',20)
    u = var_dict['u_mn']
    v = var_dict['v_mn']
    u.data[:,:,:,:] = 4.0
    v.data[:,:,:,:] = 3.0
    wind_dir = hubh_wind_dir(var_dict, u, v, 20, 100)
    npt.assert_almost_equal(wind_dir, 0.6435011088*np.ones(24))
    v.data[:,:,:,:] = -3.0
    wind_dir = hubh_wind_dir(var_dict, u, v, 20, 100)
    npt.assert_almost_equal(wind_dir, -0.6435011088*np.ones(24))

def test_cv_average_linear():
    """ Test CV average for linear profiles
    """
    var_dict = load_NWP_data('DS7',30)
    zh = var_dict['u_mn'].coords('level_height')[0].points
    for i in range(40):
        var_dict['u_mn'].data[:,i,:,:] = 4.0*zh[i]/250
    varmean_cv = CV_average(var_dict, 'u_mn', 30, 250)
    npt.assert_almost_equal(varmean_cv, 2.0*np.ones(24), decimal=5)
    for i in range(40):
        var_dict['u_mn_0'].data[:,i,:,:] = -8.0*zh[i]/250
    varmean_cv = CV_average(var_dict, 'u_mn_0', 30, 250)
    npt.assert_almost_equal(varmean_cv, -4.0*np.ones(24), decimal=5)

def test_cv_average_quadratic():
    """ Test CV average for quadratic profile
    """
    var_dict = load_NWP_data('DS7',30)
    zh = var_dict['u_mn'].coords('level_height')[0].points
    for i in range(40):
        var_dict['u_mn'].data[:,i,:,:] = zh[i]**2/10000
    varmean_cv = CV_average(var_dict, 'u_mn', 30, 250)
    npt.assert_almost_equal(varmean_cv, 2.083333*np.ones(24), decimal=5)

def test_cv_average_log():
    """ Test CV average for logarithmic profile
    """
    var_dict = load_NWP_data('DS0',10)
    zh = var_dict['u_mn'].coords('level_height')[0].points
    for i in range(40):
        var_dict['v_mn_0'].data[:,i,:,:] = 10*np.log(zh[i])/np.log(100)
    varmean_cv = CV_average(var_dict, 'v_mn_0', 10, 250)
    npt.assert_allclose(varmean_cv, 9.81872*np.ones(24), rtol=0.005)

def test_surface_average():
    """ Test surface average function
    """
    var_dict = load_NWP_data('DS3',20)
    taux_0 = var_dict['taux_mn_0']
    taux_0.data[:,0,:,:] = 5.0
    varmean_surface = surface_average(var_dict, 'taux_mn_0', 20)
    npt.assert_array_equal(varmean_surface, 5.0*np.ones(24))

def test_top_surface_average():
    """ Test surface average function
    """
    var_dict = load_NWP_data('DS5',10)
    zh = var_dict['u_mn'].coords('level_height')[0].points
    for i in range(40):
        var_dict['v_mn_0'].data[:,i,:,:] = 10*np.log(zh[i])/np.log(100)
    varmean_cv = top_surface_average(var_dict, 'v_mn_0', 10, 250)
    npt.assert_allclose(varmean_cv, 11.98970*np.ones(24), rtol=0.005)
    zh = var_dict['theta_mn_0'].coords('level_height')[0].points
    for i in range(40):
        var_dict['theta_mn_0'].data[:,i,:,:] = 280+0.005*zh[i]
    varmean_cv = top_surface_average(var_dict, 'theta_mn_0', 10, 100)
    npt.assert_allclose(varmean_cv, 280.5*np.ones(24), rtol=0.005)

def test_vertical_profile():
    """Test vertical profile function
    """
    var_dict = load_NWP_data('DS7', 10)
    theta_0 = var_dict['theta_mn_0']
    theta_0.data[:,0,:,:] = 275.0
    theta_0.data[:,6,:,:] = 280.0
    theta_profile, heights = farm_vertical_profile(var_dict, 'theta_mn_0', 10)
    npt.assert_array_equal(theta_profile[:,0], 275.0*np.ones(24))
    theta_profile, heights = farm_vertical_profile(var_dict, 'theta_mn_0', 10)
    npt.assert_array_equal(theta_profile[:,6], 280.0*np.ones(24))
