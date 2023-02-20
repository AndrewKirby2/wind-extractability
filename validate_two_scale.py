"""Valdiate two-scale against results
of Allaerts and Meyers 2017 JFM
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp
import scipy.optimize as opt


###############################
# Case S1
###############################

vel_data = np.loadtxt('allaerts_meyers_velocity_s1.csv', delimiter=',')
vel = 12*vel_data[:,0]
heights = 100*vel_data[:,1]

sc_vel = sp.CubicSpline(heights, vel)
heights_interp = np.linspace(0,250,251)
vel_interp = sc_vel(heights_interp)


stress_data = np.loadtxt('allaerts_meyers_stress_s1.csv', delimiter=',')
stress = 1.3*(0.310**2)*stress_data[:,0]
heights = 100*stress_data[:,1]

sc_stress = sp.CubicSpline(heights, stress)
heights_interp = np.linspace(0,250,251)
stress_interp = sc_stress(heights_interp)

direction_data = np.loadtxt('allaerts_meyers_direction_s1.csv', delimiter=',')
direction = (-7.72*np.pi/180)*direction_data[:,0]
heights = 100*direction_data[:,1]

sc_direction = sp.CubicSpline(heights, direction)
heights_interp = np.linspace(0,250,251)
direction_interp = sc_direction(heights_interp)


uf0 = np.mean(vel_interp*np.cos(direction_interp))

turbine_area = np.pi*100**2/4
cp_recorded = 1.386e6/(0.5*uf0**3*turbine_area)
tauw0 = stress_interp[0]*np.cos(direction_interp[0])
cf0 = tauw0/(0.5*1.3*uf0**2)
ctstar = 0.55
array_density = turbine_area / (7.5*5.33*100**2)
h_cv = 250

def ndfm(beta):
    lhs = ctstar*(array_density/cf0)*beta**2 + beta**2
    rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - (sc_stress(h_cv)*np.cos(sc_direction(h_cv)))/tauw0
    rhs_bottom = 1 - (sc_stress(beta*h_cv)*np.cos(sc_direction(beta*h_cv)))/tauw0
    return lhs - rhs_top/rhs_bottom

beta = opt.bisect(ndfm,1,0.3)
rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - (sc_stress(h_cv)*np.cos(sc_direction(h_cv)))/tauw0
rhs_bottom = 1 - (sc_stress(beta*h_cv)*np.cos(sc_direction(beta*h_cv)))/tauw0
M = rhs_top / rhs_bottom
print((M-1)/(1-beta))

cp_model = beta**3 * 1.33**(-0.5) * ctstar**1.5
print(cp_model)
print(cp_recorded)

###############################
# Case S2
###############################

vel_data = np.loadtxt('allaerts_meyers_velocity_s2.csv', delimiter=',')
vel = 12*vel_data[:,0]
heights = 100*vel_data[:,1]

sc_vel = sp.CubicSpline(heights, vel)
heights_interp = np.linspace(0,250,251)
vel_interp = sc_vel(heights_interp)


stress_data = np.loadtxt('allaerts_meyers_stress_s2.csv', delimiter=',')
stress = 1.3*(0.315**2)*stress_data[:,0]
heights = 100*stress_data[:,1]

sc_stress = sp.CubicSpline(heights, stress)
heights_interp = np.linspace(0,250,251)
stress_interp = sc_stress(heights_interp)

direction_data = np.loadtxt('allaerts_meyers_direction_s2.csv', delimiter=',')
direction = (-11.03*np.pi/180)*direction_data[:,0]
heights = 100*direction_data[:,1]

sc_direction = sp.CubicSpline(heights, direction)
heights_interp = np.linspace(0,250,251)
direction_interp = sc_direction(heights_interp)


uf0 = np.mean(vel_interp*np.cos(direction_interp))

turbine_area = np.pi*100**2/4
cp_recorded = 1.314e6/(0.5*uf0**3*turbine_area)
tauw0 = stress_interp[0]*np.cos(direction_interp[0])
cf0 = tauw0/(0.5*1.3*uf0**2)
ctstar = 0.55
array_density = turbine_area / (7.5*5.33*100**2)
h_cv = 250

def ndfm(beta):
    lhs = ctstar*(array_density/cf0)*beta**2 + beta**2
    rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - (sc_stress(h_cv)*np.cos(sc_direction(h_cv)))/tauw0
    rhs_bottom = 1 - (sc_stress(beta*h_cv)*np.cos(sc_direction(beta*h_cv)))/tauw0
    return lhs - rhs_top/rhs_bottom

beta = opt.bisect(ndfm,1,0.3)
rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - (sc_stress(h_cv)*np.cos(sc_direction(h_cv)))/tauw0
rhs_bottom = 1 - (sc_stress(beta*h_cv)*np.cos(sc_direction(beta*h_cv)))/tauw0
M = rhs_top / rhs_bottom
print((M-1)/(1-beta))

cp_model = beta**3 * 1.33**(-0.5) * ctstar**1.5
print(cp_model)
print(cp_recorded)

###############################
# Case S4
###############################

vel_data = np.loadtxt('allaerts_meyers_velocity_s4.csv', delimiter=',')
vel = 12*vel_data[:,0]
heights = 100*vel_data[:,1]

sc_vel = sp.CubicSpline(heights, vel)
heights_interp = np.linspace(0,250,251)
vel_interp = sc_vel(heights_interp)


stress_data = np.loadtxt('allaerts_meyers_stress_s4.csv', delimiter=',')
stress = 1.3*(0.306**2)*stress_data[:,0]
heights = 100*stress_data[:,1]

sc_stress = sp.CubicSpline(heights, stress)
heights_interp = np.linspace(0,250,251)
stress_interp = sc_stress(heights_interp)

direction_data = np.loadtxt('allaerts_meyers_direction_s2.csv', delimiter=',')
direction = (-17.5*np.pi/180)*direction_data[:,0]
heights = 100*direction_data[:,1]

sc_direction = sp.CubicSpline(heights, direction)
heights_interp = np.linspace(0,250,251)
direction_interp = sc_direction(heights_interp)


uf0 = np.mean(vel_interp*np.cos(direction_interp))

turbine_area = np.pi*100**2/4
cp_recorded = 1.147e6/(0.5*uf0**3*turbine_area)
tauw0 = stress_interp[0]*np.cos(direction_interp[0])
cf0 = tauw0/(0.5*1.3*uf0**2)
ctstar = 0.55
array_density = turbine_area / (7.5*5.33*100**2)
h_cv = 250

def ndfm(beta):
    lhs = ctstar*(array_density/cf0)*beta**2 + beta**2
    rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - (sc_stress(h_cv)*np.cos(sc_direction(h_cv)))/tauw0
    rhs_bottom = 1 - (sc_stress(beta*h_cv)*np.cos(sc_direction(beta*h_cv)))/tauw0
    return lhs - rhs_top/rhs_bottom

beta = opt.bisect(ndfm,1,0.3)
rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - (sc_stress(h_cv)*np.cos(sc_direction(h_cv)))/tauw0
rhs_bottom = 1 - (sc_stress(beta*h_cv)*np.cos(sc_direction(beta*h_cv)))/tauw0
M = rhs_top / rhs_bottom
print((M-1)/(1-beta))

cp_model = beta**3 * 1.33**(-0.5) * ctstar**1.5
print(cp_model)
print(cp_recorded)