import iris
import matplotlib.pyplot as plt
import code
import numpy as np
import scipy.integrate as sp


def load_NWP_data(DS_no, farm_diameter):
  """Load data from UM simulation with wind farm parameterisation

  Parameters
  ----------
  DS_no : str
    Data set number i.e. 'DS0'
  farm_diameter : int
    Wind farm diameter in kilometres
  
  Returns
  -------
  var_dict : iris Cube
    Data for variables with and without farm present
  """

  #times and variable names
  fn_times = ['000','006', '012', '018']
  fn_vars = ['u', 'v', 'w', 'p', 'ustar', 'psurf', 'taux', 'tauy', 'dens']

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

  #Secondly, without farm present
  #location of datasets
  ddir = f'../../part1_turbines/UM_farm_data/Datasets/{DS_no}/wfm_run_z0_0p1_d{str(farm_diameter)}/d2/'

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

def hubh_wind_dir(u, v, farm_diameter, hubh):
  """Calculates the wind direction at the turbine hub height

  Parameters
  ----------
  u : iris Cube
    velocities in x direction
  v : iris Cube
    velocities in y direction
  farm_diamater: int
    wind farm diameter in kilometres
  hubh: int
    wind turbine hug height
  
  Returns
  -------
  ang : float
    wind direction in radians
  """
  
  #farm parameters
  mperdeg = 111132.02
  grid = var_dict['p'][0]
  zh = grid.coords('level_height')[0].points

  #discretisation for interpolation
  n_lats = 200
  n_lons = 200

  lats = np.linspace(-1,1,n_lats)
  lons = np.linspace(359,361, n_lons)
  u = var_dict['u']
  v = var_dict['v']
  u = u.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())
  v = v.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

  #mask all data points outside of wind farm CV
  mask = np.full(u[:,:,:,:].shape, True)
  c_lat = lats[0]+(lats[-1]-lats[0])/2. # centre of domain (lats[-1] is last value)
  c_lon = lons[0]+(lons[-1]-lons[0])/2. # centre of domain
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

  #interpolate velocity at turbine hub height
  umean_hubh = np.interp(hubh, zh_full, umean_full[0,:])
  vmean_hubh = np.interp(hubh, zh_full, vmean_full[0,:])

  #calucate wind direction at turbine hub height
  ang = np.angle(complex(umean_hubh,vmean_hubh), deg=False)

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
  
  #only valid for u and v variables!
  assert var[0] == 'u' or var[0] == 'v'
  #farm parameters
  mperdeg = 111132.02
  grid = var_dict['p'][0]
  zh = grid.coords('level_height')[0].points

  #discretisation for interpolation
  n_lats = 200
  n_lons = 200

  lats = np.linspace(-1,1,n_lats)
  lons = np.linspace(359,361, n_lons)
  var = var_dict[var]
  var = var.interpolate([('grid_latitude', lats),('grid_longitude', lons)], iris.analysis.Linear())

  #mask all data points outside of wind farm CV
  mask = np.full(var[:,:,:,:].shape, True)
  c_lat = lats[0]+(lats[-1]-lats[0])/2. # centre of domain (lats[-1] is last value)
  c_lon = lons[0]+(lons[-1]-lons[0])/2. # centre of domain
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
  varmean = np.mean(np.ma.array(var.data[:,:,:,:], mask=mask), axis=(2,3))

  #add 0 surface velocity
  varmean_full = np.zeros((24,41))
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
    z_interp = np.linspace(0, cv_height, cv_height+1)
    var_interp = np.interp(z_interp, zh_full, varmean_full[i,:])

    #integrate over control volume
    varmean_cv[i] = sp.trapz(var_interp, z_interp)/cv_height
  
  return varmean_cv

#var_dict = load_NWP_data('DS5',20)
#u = var_dict['u']
#v = var_dict['v']
#print(hubh_wind_dir(u, v, 20, 100))
#print(CV_average(var_dict,'u_0',20,250))