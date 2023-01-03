"""Tests for functions analysing NWP data"""

import numpy as np
import numpy.testing as npt
from read_NWP_data import *

def test_dummy():
    npt.assert_array_equal(np.array([0]),np.array([0]))

def test_wind_direction():
    """ Test wind direction calculation is correct for
    positive and negative wind directions
    """
    var_dict = load_NWP_data('DS5',20)
    u = var_dict['u']
    v = var_dict['v']
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
    zh = var_dict['u'].coords('level_height')[0].points
    for i in range(40):
        var_dict['u'].data[:,i,:,:] = 4.0*zh[i]/250
    varmean_cv = CV_average(var_dict, 'u', 30, 250)
    npt.assert_almost_equal(varmean_cv, 2.0*np.ones(24), decimal=5)
    for i in range(40):
        var_dict['u_0'].data[:,i,:,:] = -8.0*zh[i]/250
    varmean_cv = CV_average(var_dict, 'u_0', 30, 250)
    npt.assert_almost_equal(varmean_cv, -4.0*np.ones(24), decimal=5)

def test_cv_average_quadratic():
    """ Test CV average for quadratic profile
    """
    var_dict = load_NWP_data('DS7',30)
    zh = var_dict['u'].coords('level_height')[0].points
    for i in range(40):
        var_dict['u'].data[:,i,:,:] = zh[i]**2/10000
    varmean_cv = CV_average(var_dict, 'u', 30, 250)
    npt.assert_almost_equal(varmean_cv, 2.083333*np.ones(24), decimal=5)

def test_cv_average_log():
    """ Test CV average for logarithmic profile
    """
    var_dict = load_NWP_data('DS0',10)
    zh = var_dict['u'].coords('level_height')[0].points
    for i in range(40):
        var_dict['v_0'].data[:,i,:,:] = 10*np.log(zh[i])/np.log(100)
    varmean_cv = CV_average(var_dict, 'v_0', 10, 250)
    npt.assert_allclose(varmean_cv, 9.81872*np.ones(24), rtol=0.005)

def test_surface_average():
    """ Test surface average function
    """
    var_dict = load_NWP_data('DS3',20)
    taux_0 = var_dict['taux_0']
    taux_0.data[:,0,:,:] = 5.0
    varmean_surface = surface_average(var_dict, 'taux_0', 20)
    npt.assert_array_equal(varmean_surface, 5.0*np.ones(24))
