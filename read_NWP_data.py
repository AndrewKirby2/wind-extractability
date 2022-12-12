import iris
import matplotlib.pyplot as plt
import code



ddir = '../../part1_turbines/UM_farm_data/Datasets/DS1/wfm_run_z0_ctl_DS1/d2/'

fn_times = ['000','006', '012', '018']
fn_vars = ['u', 'v', 'w', 'p', 'ustar', 'psurf', 'taux', 'tauy', 'dens']

var_dict = dict()

for i in range(len(fn_vars)):
  var_dict[fn_vars[i]]=[]
  for j in range(len(fn_times)):
    fnm = ddir+fn_vars[i]+'_'+fn_times[j]+'.nc'
    var_dict[fn_vars[i]].append(iris.load(fnm)[0])