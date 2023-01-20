import iris
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import code
import numpy as np
import scipy as sp
import time


def load_NWP_data(DS_no, farm_diameter, z0='0p1'):
  """Load data from UM simulation with wind farm parameterisation

  Parameters
  ----------
  DS_no : str
    Data set number i.e. 'DS0'
  farm_diameter : int
    Wind farm diameter in kilometres
  z0 : str
    Wind farm roughness length (m)
  
  Returns
  -------
  var_dict : iris Cube
    Data for variables with and without farm present
  """

  #times and variable names
  fn_times = ['000','006', '012', '018']
  fn_vars = ['u_mn', 'v_mn', 'w_mn', 'taux_mn', 'tauy_mn', 'dens_mn', 'theta_mn']

  #create dictionary of extract data
  var_dict = dict()

  #Firstly, without farm present
  #location of datasets
  ddir = f'../../part1_turbines/UM_farm_data/Datasets/{DS_no}/wfm_run_z0_ctl_{DS_no}/d2/'

  #loop through each variable
  for i in range(len(fn_vars)):
    var_dict[fn_vars[i]+'_0']=[]
    #loop through each time
    for j in range(len(fn_times)):
      fnm = ddir+fn_vars[i]+'_'+fn_times[j]+'.nc'
      var_dict[fn_vars[i]+'_0'].append(iris.load(fnm)[0])
    var_dict[fn_vars[i]+'_0'] = iris.cube.CubeList(var_dict[fn_vars[i]+'_0'])
    var_dict[fn_vars[i]+'_0'] = var_dict[fn_vars[i]+'_0'].concatenate_cube()

  #Secondly, with farm present
  #location of datasets
  ddir = f'../../part1_turbines/UM_farm_data/Datasets/{DS_no}/wfm_run_z0_{z0}_d{str(farm_diameter)}/d2/'

  #loop through each variable
  for i in range(len(fn_vars)):
    var_dict[fn_vars[i]]=[]
    #loop through each time
    for j in range(len(fn_times)):
      fnm = ddir+fn_vars[i]+'_'+fn_times[j]+'.nc'
      var_dict[fn_vars[i]].append(iris.load(fnm)[0])
    var_dict[fn_vars[i]] = iris.cube.CubeList(var_dict[fn_vars[i]])
    var_dict[fn_vars[i]] = var_dict[fn_vars[i]].concatenate_cube()

  return var_dict

def hubh_wind_dir(var_dict, u, v, farm_diameter, hubh):
  """Calculates the wind direction at the turbine hub height

  Parameters
  ----------
  var_dict : iris Cube
    Data for variables with and without farm present
  u : iris Cube
    u velocity data
  v : iris Cube
    v velocity data
  farm_diamater: int
    wind farm diameter in kilometres
  hubh: int
    wind turbine hub height
  
  Returns
  -------
  ang : numpy array (size 24)
    wind direction in radians
  """
  
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

  #add 0 surface velocity
  umean_full = np.zeros((24,41))
  umean_full[:,0] = 0
  umean_full[:,1:] = umean
  vmean_full = np.zeros((24,41))
  vmean_full[:,0] = 0
  vmean_full[:,1:] = vmean
  zh_full = np.zeros(41)
  zh_full[0] = 0
  zh_full[1:] = zh

  
  ang = np.zeros(24)

  #loop over each 1hr time period
  for i in range(24):
    #interpolate velocity at turbine hub height
    sc = sp.interpolate.CubicSpline(zh_full, umean_full[i,:])
    umean_hubh = sc(hubh)
    sc = sp.interpolate.CubicSpline(zh_full, vmean_full[i,:])
    vmean_hubh = sc(hubh)

    #calucate wind direction at turbine hub height
    ang[i] = np.angle(complex(umean_hubh,vmean_hubh), deg=False)

  return ang

def CV_average(var_dict, var, farm_diameter, cv_height):
  """Calculate control volume average of quantity

  Parameters
  ----------
  var_dict : iris Cube
    dictionary containing data extracted from NWP simulations
  var : str
    name of variable to be averaged i.e. 'u' or 'u_0'
  farm_diameter : int
    wind farm diameter in kilometres
  cv_height : int
    height of control volume in metres

  Returns
  -------
  varmean_cv : numpy array (size 24)
    control volume averaged quantities for each hour
    time period across 24 hour period
  """
  
  #farm parameters
  mperdeg = 111132.02
  grid = var_dict[var][0]
  zh = grid.coords('level_height')[0].points

  #discretisation for interpolation
  n_lats = 200
  n_lons = 200

  lats = np.linspace(-1,1,n_lats)
  lons = np.linspace(359,361, n_lons)
  variable = var_dict[var]
  variable = variable.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

  #mask all data points outside of wind farm CV
  mask = np.full(variable[:,:,:,:].shape, True)
  c_lat = 0.0135 # centre of domain (lats[-1] is last value)
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
  varmean = np.mean(np.ma.array(variable.data[:,:,:,:], mask=mask), axis=(2,3))

  #add 0 surface velocity
  varmean_full = np.zeros((24,41))
  if var == 'theta_mn' or var == 'theta_mn_0':
    varmean_full = varmean
    zh_full = zh
  else:
    varmean_full[:,0] = 0
    varmean_full[:,1:] = varmean
    zh_full = np.zeros(41)
    zh_full[0] = 0
    zh_full[1:] = zh

  #array to store cv averages
  varmean_cv = np.zeros(24)

  #loop over each 1hr time period
  for i in range(24):

    #interpolate velocity at turbine hub height
    z_interp = np.linspace(0, cv_height, round(cv_height+1))
    sc = sp.interpolate.CubicSpline(zh_full, varmean_full[i,:])
    var_interp = sc(z_interp)

    #integrate over control volume
    varmean_cv[i] = sp.integrate.trapz(var_interp, z_interp)/cv_height
  
  return varmean_cv

def surface_average(var_dict, var, farm_diameter):
  """Calculate surface average of quantity

  Parameters
  ----------
  var_dict : iris Cube
    dictionary containing data extracted from NWP simulations
  var : str
    name of variable to be averaged i.e. 'u' or 'u_0'
  farm_diameter : int
    wind farm diameter in kilometres

  Returns
  -------
  varmean_cv : numpy array (size 24)
    control surface averaged quantities for each hour
    time period across 24 hour period
  """
  
  #farm parameters
  mperdeg = 111132.02
  grid = var_dict[var][0]
  zh = grid.coords('level_height')[0].points

  #discretisation for interpolation
  n_lats = 200
  n_lons = 200

  lats = np.linspace(-1,1,n_lats)
  lons = np.linspace(359,361, n_lons)
  variable = var_dict[var]
  variable = variable.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

  #mask all data points outside of wind farm CV
  mask = np.full(variable[:,:,:,:].shape, True)
  c_lat = 0.0135 # centre of domain (lats[-1] is last value)
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

  #average valid points in the horizontal direction at the lowest model level
  varmean_surface = np.mean(np.ma.array(variable.data[:,0,:,:], mask=mask[:,0,:,:]), axis=(1,2))
  
  return varmean_surface.data

def farm_vertical_profile(var_dict, var, farm_diameter):
  """Calculate vertical profiles of horizontally averaged
  fields within wind farm area

  Parameters
  ----------
  var_dict : iris Cube
    dictionary containing data extracted from NWP simulations
  var : str
    name of variable to be averaged i.e. 'u' or 'u_0'
  farm_diameter : int
    wind farm diameter in kilometres
  cv_height : int
    height of control volume in metres

  Returns
  -------
  var_profile : numpy array (size (24, variable) )
    vertical profile of horizontally averaged variable
  heights : numpy array (size (variable) )
    heights above surface for vertical profile
  """
  #farm parameters
  mperdeg = 111132.02
  grid = var_dict[var][0]
  heights = grid.coords('level_height')[0].points

  #discretisation for interpolation
  n_lats = 200
  n_lons = 200

  lats = np.linspace(-1,1,n_lats)
  lons = np.linspace(359,361, n_lons)
  variable = var_dict[var]
  variable = variable.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

  #mask all data points outside of wind farm CV
  mask = np.full(variable[:,:,:,:].shape, True)
  c_lat = 0.0135 # centre of domain (lats[-1] is last value)
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

  #average valid points in the horizontal direction at the lowest model level
  var_profile = np.mean(np.ma.array(variable.data[:,:,:,:], mask=mask[:,:,:,:]), axis=(2,3))

  return var_profile, heights

def neutral_layer_height(theta_profile, theta_heights):
  """Estimates height of the neutral layer which is
  used for the calculation of the Froude number
  (where N_sq is greater than 1e-6)

  Parameters
  ----------
  theta_profile : numpy array (size (24,41))
    vertical profile of theta
  theta_heights : numpy array (size 41)
    height above surface for theta vertical profile

  Returns:
  layer_height : numpy array (size 24)
    heights of the neutral layer height
  theta_profile_interp : numpy array (size (24,4001))
    interpolated theta vertical profile
  interp_heights : numpy array (size 4001)
    heights for interpolated theta vertical profile
  """
  #array to store results
  layer_height = np.zeros(24)
  theta_profile_interp = np.zeros((24,4001))
  #loop over time periods
  for i in range(24):
    sc = sp.interpolate.CubicSpline(theta_heights, theta_profile[i,:])
    interp_heights = np.arange(0,4001,1)
    theta_profile_interp[i,:] = sc(interp_heights)
    theta_grad = np.gradient(theta_profile[i,:], theta_heights)
    N_sq = (9.81/theta_profile_interp[i,0])*theta_grad
    #find index where theta is first 0.2K higher than surface value
    index = np.argmax(N_sq > 1e-6)
    layer_height[i] = theta_heights[index]
  return layer_height

def calculate_fr_number(var_dict, neutral_layer_height, wind_dir_0, hubh, farm_diameter):
  """ Calculate Froude number based on iterative procedure
  described in Vosper et. al. 2009 DOI: 10.1002/qj.407
  """
  #maximum change in z_av
  delta_z_max = 5
  max_iter = 2000
  #calculate u and v vertical profiles
  u_profile, u_heights = farm_vertical_profile(var_dict, 'u_mn_0', farm_diameter)
  v_profile, v_heights = farm_vertical_profile(var_dict, 'v_mn_0', farm_diameter)
  theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', farm_diameter)
  #array to store results
  froude_number = np.zeros(24)
  #loop over time periods
  for i in range(24):
    #initial guess of bulk averaging layer depth
    z_av = max(neutral_layer_height[i], hubh)

    #height for integration
    z_int = np.linspace(0, z_av, int(z_av))

    #interpolate and integrate
    sc = sp.interpolate.CubicSpline(u_heights, u_profile[i,:])
    u_interp = sc(z_int)
    sc = sp.interpolate.CubicSpline(v_heights, v_profile[i,:])
    v_interp = sc(z_int)
    u_av = sp.integrate.trapz(u_interp, z_int)/z_av
    v_av = sp.integrate.trapz(v_interp, z_int)/z_av
    sc = sp.interpolate.CubicSpline(theta_heights, theta_profile[i,:])

    #calculate bulk average bouyancy
    theta_top = sc(z_av)
    N_sq = (9.80665/theta_profile[i,0])*(theta_top-theta_profile[i,0])/z_av
    if N_sq < 0:
      n_av = 1e-3
    else:
      n_av = np.sqrt(N_sq)

    #depth averaged wind in hub height direction
    wind = u_av*np.cos(wind_dir_0[i]) + v_av*np.sin(wind_dir_0[i])

    #maximum change per iteration
    if wind/n_av > delta_z_max:
      z_av_new = max(neutral_layer_height[i], hubh) + delta_z_max
    elif wind/n_av < -delta_z_max:
      z_av_new = max(neutral_layer_height[i], hubh) - delta_z_max
    else:
      z_av_new = max(neutral_layer_height[i], hubh) + wind/(max(n_av,1e-3))

    #iterate until change is less than 0.25% or max_iter iteractions
    iter_count = 0
    while (np.abs(z_av_new-z_av)/z_av > 0.0025 and iter_count < max_iter):

      z_av = z_av_new
      #height for integration
      z_int = np.linspace(0, z_av, int(z_av))

      #interpolate and integrate
      sc = sp.interpolate.CubicSpline(u_heights, u_profile[i,:])
      u_interp = sc(z_int)
      sc = sp.interpolate.CubicSpline(v_heights, v_profile[i,:])
      v_interp = sc(z_int)
      u_av = sp.integrate.trapz(u_interp, z_int)/z_av
      v_av = sp.integrate.trapz(v_interp, z_int)/z_av
      sc = sp.interpolate.CubicSpline(theta_heights, theta_profile[i,:])

      #calculate bulk average bouyancy
      theta_top = sc(z_av)
      N_sq = (9.80665/theta_profile[i,0])*(theta_top-theta_profile[i,0])/z_av
      if N_sq < 0:
        n_av = 1e-3
      else:
        n_av = np.sqrt(N_sq)

      #depth averaged wind in hub height direction
      wind = u_av*np.cos(wind_dir_0[i]) + v_av*np.sin(wind_dir_0[i])

      #maximum change per iteration
      z_av_new = max(neutral_layer_height[i], hubh) + wind/(max(n_av,1e-3))
      if z_av_new - z_av > delta_z_max:
        z_av_new = z_av + delta_z_max
      elif z_av_new - z_av < -delta_z_max:
        z_av_new = z_av - delta_z_max

      iter_count += 1
      if iter_count == max_iter:
        print('Froude number calculation not converged',i)
    
    froude_number[i] = wind / (n_av*hubh)

  return froude_number

  

def top_surface_average(var_dict, var, farm_diameter, cv_height):
  """Calculate average of quantity across top surface
    of control volume

  Parameters
  ----------
  var_dict : iris Cube
    dictionary containing data extracted from NWP simulations
  var : str
    name of variable to be averaged i.e. 'u' or 'u_0'
  farm_diameter : int
    wind farm diameter in kilometres
  cv_height : int
    height of control volume in metres

  Returns
  -------
  varmean_cv : numpy array (size 24)
    control surface averaged quantities for each hour
    time period across 24 hour period
  """
  #farm parameters
  mperdeg = 111132.02
  grid = var_dict[var][0]
  zh = grid.coords('level_height')[0].points

  #discretisation for interpolation
  n_lats = 200
  n_lons = 200

  lats = np.linspace(-1,1,n_lats)
  lons = np.linspace(359,361, n_lons)
  variable = var_dict[var]
  variable = variable.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

  #mask all data points outside of wind farm CV
  mask = np.full(variable[:,:,:,:].shape, True)
  c_lat = 0.0135 # centre of domain (lats[-1] is last value)
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
  varmean = np.mean(np.ma.array(variable.data[:,:,:,:], mask=mask), axis=(2,3))

  #array to store averages
  varmean_top = np.zeros(24)

  #loop over each 1hr time period
  for i in range(24):

    #interpolate variable at top of control volume
    sc = sp.interpolate.CubicSpline(zh, varmean[i,:])
    var_interp = sc(cv_height)
    varmean_top[i] = var_interp
  
  return varmean_top

def calculate_farm_data(DS_no, farm_diameter, z0='0p1'):
  """Calculate wind farm variables

  Parameters
  ----------
  DS_no : str
    Data set number i.e. 'DS0'
  farm_diameter : int
    wind farm diameter in kilometres
  z0 : str
    wind farm roughness length (m)

  Returns
  -------
  beta : numpy array (size 24)
    farm wind-speed reduction factor for each hour
    time period across 24 hour period
  M : numpy array (size 24)
    momentum availability factor for each hour
    time period across 24 hour period
  zeta : numpy array (size 24)
    wind extractability factor for each hour
    time period across 24 hour period
  cf0 : numpy array (size 24)
    natural surface friction coefficent for each hour
    time period across 24 hour period
  fr0 : numpy array (size 24)
    natural Froude number for each hour
    time period across 24 hour period
  rig_hubh : numpy array (size 24)
    time-averaged gradient Richardson number at the 
    turbine hubh for each hour time period across 
    24 hour period
  """

  #extract data
  var_dict = load_NWP_data(DS_no, farm_diameter, z0)

  #information about farm CV
  cv_height = 250
  hubh = 100

  #with farm present
  wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, hubh)
  u_mean = CV_average(var_dict, 'u_mn', farm_diameter, cv_height)
  v_mean = CV_average(var_dict, 'v_mn', farm_diameter, cv_height)
  taux_mean = surface_average(var_dict, 'taux_mn', farm_diameter)
  tauy_mean = surface_average(var_dict, 'tauy_mn', farm_diameter)
  # calculate farm-layer-averaged streamwise velocity U_F
  uf = u_mean*np.cos(wind_dir) + v_mean*np.sin(wind_dir)
  # calculate surface stress in streamwise direction
  tauw = taux_mean*np.cos(wind_dir) + tauy_mean*np.sin(wind_dir)

  #without farm present
  wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
  u_mean_0 = CV_average(var_dict, 'u_mn_0', farm_diameter, cv_height)
  v_mean_0 = CV_average(var_dict, 'v_mn_0', farm_diameter, cv_height)
  taux_mean_0 = surface_average(var_dict, 'taux_mn_0', farm_diameter)
  tauy_mean_0 = surface_average(var_dict, 'tauy_mn_0', farm_diameter)
  dens_mean_0 = CV_average(var_dict, 'dens_mn_0', farm_diameter, cv_height)
  # calculate farm-layer-averaged streamwise velocity U_F
  uf_0 = u_mean_0*np.cos(wind_dir_0) + v_mean_0*np.sin(wind_dir_0)
  # calculate surface stress in streamwise direction
  tauw_0 = taux_mean_0*np.cos(wind_dir_0) + tauy_mean_0*np.sin(wind_dir_0)
  #calculate natural friction coefficient
  cf_0 = tauw_0/(0.5*dens_mean_0*uf_0*uf_0)
  #calculate natural Froude number
  theta_profile, theta_heights = farm_vertical_profile(var_dict, 'theta_mn_0', farm_diameter)
  neu_layer_height = neutral_layer_height(theta_profile, theta_heights)
  #fr_0 = calculate_fr_number(var_dict, neu_layer_height, wind_dir_0, hubh, farm_diameter)

  #calculate farm wind-speed reduction factor \beta
  beta = uf/uf_0
  #calculate momentum availability factor M
  M = tauw/tauw_0
  #calculate wind extractability factor \zeta
  zeta = (M-1)/(1-beta)

  return beta, M, zeta, cf_0, tauw_0, uf_0

def calculate_farm_data_hubh(DS_no, farm_diameter, z0='0p1'):
  """Calculate wind farm variables using
  the turbine hub height for beta

  Parameters
  ----------
  DS_no : str
    Data set number i.e. 'DS0'
  farm_diameter : int
    wind farm diameter in kilometres
  z0 : str
    wind farm roughness length (m)

  Returns
  -------
  beta : numpy array (size 24)
    farm wind-speed reduction factor for each hour
    time period across 24 hour period
  M : numpy array (size 24)
    momentum availability factor for each hour
    time period across 24 hour period
  zeta : numpy array (size 24)
    wind extractability factor for each hour
    time period across 24 hour period
  """

  #extract data
  var_dict = load_NWP_data(DS_no, farm_diameter, z0)

  #information about farm CV
  cv_height = 250
  hubh = 100

  #with farm present
  wind_dir = hubh_wind_dir(var_dict, var_dict['u_mn'], var_dict['v_mn'], farm_diameter, hubh)
  u_mean = top_surface_average(var_dict, 'u_mn', farm_diameter, hubh)
  v_mean = top_surface_average(var_dict, 'v_mn', farm_diameter, hubh)
  taux_mean = surface_average(var_dict, 'taux_mn', farm_diameter)
  tauy_mean = surface_average(var_dict, 'tauy_mn', farm_diameter)
  # calculate farm-layer-averaged streamwise velocity U_F
  uf = u_mean*np.cos(wind_dir) + v_mean*np.sin(wind_dir)
  # calculate surface stress in streamwise direction
  tauw = taux_mean*np.cos(wind_dir) + tauy_mean*np.sin(wind_dir)

  #without farm present
  wind_dir_0 = hubh_wind_dir(var_dict, var_dict['u_mn_0'], var_dict['v_mn_0'], farm_diameter, hubh)
  u_mean_0 = top_surface_average(var_dict, 'u_mn_0', farm_diameter, hubh)
  v_mean_0 = top_surface_average(var_dict, 'v_mn_0', farm_diameter, hubh)
  taux_mean_0 = surface_average(var_dict, 'taux_mn_0', farm_diameter)
  tauy_mean_0 = surface_average(var_dict, 'tauy_mn_0', farm_diameter)
  # calculate farm-layer-averaged streamwise velocity U_F
  uf_0 = u_mean_0*np.cos(wind_dir_0) + v_mean_0*np.sin(wind_dir_0)
  # calculate surface stress in streamwise direction
  tauw_0 = taux_mean_0*np.cos(wind_dir_0) + tauy_mean_0*np.sin(wind_dir_0)


  #calculate farm wind-speed reduction factor \beta
  beta = uf/uf_0
  #calculate momentum availability factor M
  M = tauw/tauw_0
  #calculate wind extractability factor \zeta
  zeta = (M-1)/(1-beta)

  return beta, M, zeta

for farm_diameter in [10,15,20,25,30]:
  print(farm_diameter)
  for no in range(0):
    print(no)
    beta, M, zeta, cf_0, tauw_0, uf_0 = calculate_farm_data(f'DS{no}', farm_diameter)
    np.save(f'data/beta_DS{no}_{farm_diameter}.npy', beta)
    np.save(f'data/M_DS{no}_{farm_diameter}.npy', M)
    np.save(f'data/zeta_DS{no}_{farm_diameter}.npy', zeta)
    np.save(f'data/cf0_DS{no}_{farm_diameter}.npy', cf_0)
    np.save(f'data/tauw0_DS{no}_{farm_diameter}.npy', tauw_0)
    np.save(f'data/uf0_DS{no}_{farm_diameter}.npy', uf_0)
  
for z0 in range(0):#['0p05', '0p1', '0p35', '0p7', '1p4']:
  print(z0)
  beta, M, zeta, cf0, tauw_0 = calculate_farm_data(f'DS1', 20, z0)
  np.save(f'data/beta_DS1_20_{z0}.npy', beta)
  np.save(f'data/M_DS1_20_{z0}.npy', M)
  np.save(f'data/zeta_DS1_20_{z0}.npy', zeta)
