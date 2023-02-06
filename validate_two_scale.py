import numpy as np
import scipy.interpolate as sp
import scipy.optimize as opt

data = np.loadtxt('velocity_magnitude.csv', delimiter=',')
vel = 10.5*data[:,0]
heights = 1000*data[:,1]

def theta(z):
    theta = -0.21537*(0.1301-1.0812e-3*z)
    return theta

def tauxf0(z):
    tau = 0.169*(1 - z/630)*np.cos(theta(z))
    return tau

sc = sp.CubicSpline(heights, vel)
heights_interp = np.linspace(0,2.5*119,100)
vel_interp = sc(heights_interp)

theta_interp = theta(heights_interp)
print(theta_interp)

u = vel_interp*np.cos(theta_interp)

uf0 = np.mean(u)

cf0 = 3.04e-3
ctstar = 0.75
array_density = 0.0314
h_cv = 3*119

def ndfm(beta):
    lhs = ctstar*(array_density/cf0)*beta**2 + beta**2
    rhs_top = 1 + (1/cf0)*(h_cv/15e3)*(1 - beta**2) - tauxf0(h_cv)/tauxf0(0)
    rhs_bottom = 1 - tauxf0(beta*h_cv)/tauxf0(0)
    return lhs - rhs_top/rhs_bottom

beta = opt.bisect(ndfm,1,0.3)

cp = beta**3 * 1.33**(-0.5) * ctstar**1.5
cp_isolated = 1.33**(-0.5) * 0.75**1.5
print(cp/cp_isolated)