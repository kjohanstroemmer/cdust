import numpy as np
import pandas as pd
import pymultinest as pmn
import glob
import sys

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import w0waCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import wCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d

from time import time
from numpy import median



############ CONSTANTS
cc = 299792458
G = 6.67408*10**(-11)
og = 5e-5
############

############ DATA
#JLA (zcmb dz mb dmb x1 dx1 color dcolor hm)
jla = pd.read_csv('/home/kj/dust/sn_data/jla_lcparams.txt',
 delim_whitespace = True, header = 0)
#Binned JLA 
jb = pd.read_csv('/home/kj/dust/sn_data/dist_binned.txt',
 delim_whitespace=True, header=None, names=['zb', 'mub'])
jb_cm = np.loadtxt('/home/kj/dust/sn_data/covmat_binned.txt')
jb_cm *= 1e-6
jb_icm = np.linalg.inv(jb_cm)
#PANTHEON
#full long corrected (zcmb, mb, dmb)
pc = pd.read_csv('/home/kj/dust/sn_data/lcparam_full_long.txt',
 delim_whitespace = True, header = 0)
#binned (zcmb, mb, dmb)
pb = pd.read_csv('/home/kj/dust/sn_data/lcparam_DS17f.txt',
 delim_whitespace = True, header = 0)
pb_stat = np.diag((pb.dmb.values)**2)
pb_sys = np.loadtxt('/home/kj/dust/sn_data/syscov_panth.txt')
pb_cm = pb_stat + pb_sys
pb_icm = np.linalg.inv(pb_cm)

###### CMB DATA #######
Rcmb = 1.7382   #######
dRcmb = 0.0088  #######
z_rec = 1092    #######
#######################




########### FUNCTIONS #################









######### EXTINCTION CCM89 ##########
class ccm:
    def __init__(self, rv, av):
        self.av=av
        self.rv=rv
    def a(self, lam):
        x=1./lam
        if x >= .3 and x <= 1.1:
            a = .574* pow(x, 1.61)
        elif x >= 1.1 and x <= 3.3:
            y= x -1.82
            a=1 + .17699 *y - .50447*y**2 - 0.02427* y**3 +.72085*y **4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7            
        else:
            print("x is not in range")
        return a        

    def b(self, lam):
        x=1./lam
        if x >= .3 and x <= 1.1:
            b= -0.527 * pow(x, 1.61)
        elif x >= 1.1 and x <= 3.3:
            y= x-1.82
            b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260* y**6 - 2.09002*y**7
        else: 
            print("x is not in range")
        return b
        
    def alam(self, lam):
        alav=self.a(lam)+self.b(lam)/self.rv
        al=alav*self.av
        return al

    def kappa_lam(self, lam):
        alav=self.a(lam)+self.b(lam)/self.rv
        kappa = 1.54e3*alav
        return kappa

Rv = 3.1 #total to selective extinction
Av = 0 #extinction in V-band. Not used
exti = ccm(Rv, Av)





### COSMOLOGY FOR CMB-SCALE REDSHIFTS
def E_far(om, h0, z, w0, wa):
	w = w0 + wa*z/(1+z)
	return np.sqrt(om*(1.+z)**3.+og*(1+z)**4+(1.-om)*(1.+z)**(3.*w+3))
#Define luminosity distance. H0 removed: log(DL) = log(I)-log(H0)
def DL_far(om, h0, z_max, w0, wa):
	res = quad(lambda z: 1 / E_far(om, h0, z, w0, wa), 0, z_max)
	return c/h0/1000*(1.+z_max) * res[0]

### COSMOLOGY WITHOUT RADIATION
def E(om, z, w0, wa):
    w = w0 + wa*z/(1+z)
    return np.sqrt(om*(1.+z)**3.+(1.-om)*(1.+z)**(3.*w+3))

#Define luminosity distance. 
def DL(om, z_max, w0, wa):
    res = integrate.quad(lambda z: 1 / E(om, z, w0, wa), 0, z_max)
    return c/1000*(1.+z_max) * res[0]

#Define attenuation function
def attenuation(om, h0, z_max, w0, wa, gamma):
    h0si = h0/(3.09*10**(19)) #h0 in SI
    res = quad(lambda z: exti.kappa_lam(0.44*(1+z_max)/(1+z))*(1+z)**(gamma-1)/E(om, z, w0, wa), 0, z_max)
    return 1.086*3*h0si*cc/(8*np.pi*G)*res[0]

def mu_cov(alpha, beta):
    """ Assemble the full covariance matrix of distance modulus

    See Betoule et al. (2014), Eq. 11-13 for reference
    """
    Ceta = sum([fits.getdata(mat) for mat in glob.glob('/home/kj/dust/sn_data/jla_cov/C*.fits')])
    #Ceta = sum([fits.getdata(mat) for mat in glob.glob('/home/kajo2802/dust/sn_data/C*.fits')])
    

    Cmu = np.zeros_like(Ceta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            Cmu += (coef1 * coef2) * Ceta[i::3,j::3]

    # Add diagonal term from Eq. 13
    sigma = np.loadtxt('/home/kj/dust/sn_data/jla_cov/sigma_mu.txt')
    #sigma = np.loadtxt('/home/kajo2802/dust/sn_data/sigma_mu.txt')

    sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
    Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    
    return Cmu


######CMB CHI-SQUARE ############
def cmb_chisq(om, h0, w0, wa):
	h = h0/100
	om_cmb = om*h**2
	R = np.sqrt(om_cmb*h0**2)*DL_far(om_cmb, h0, z_rec, w0, wa)/c
	chisq = (R-Rcmb)**2/dRcmb**2
	return chisq



#redshift for use in interpolation
z50 = np.linspace(0.001, 2.26, 50)

################################################
################### MULTINEST ##################
################################################





##################### JLA ######################

def llhood_jlac_flcdm(model_param, ndim, nparam):
	om, a, b, MB, DM = [model_param[i] for i in range(5)]

	h0 = 70
	w0 = -1
	wa = 0

	
	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jlac_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*0.3
	cube[2] = cube[2]*5
	cube[3] = cube[3]*10-25
	cube[4] = cube[4]*0.5-0.25


def llhood_jlad_flcdm(model_param, ndim, nparam):
	om, a, b, MB, DM, od, g = [model_param[i] for i in range(7)]

	h0 = 70
	w0 = -1
	wa = 0

	A = []
	for k in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[k], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jla.zcmb.values)


	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th - DUST
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jlad_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*0.3
	cube[2] = cube[2]*5
	cube[3] = cube[3]*10-25
	cube[4] = cube[4]*0.5-0.25
	cube[5] = cube[5]*10-10
	cube[6] = cube[6]*9-6



def llhood_jlac_wcdm(model_param, ndim, nparam):
	om, w0, a, b, MB, DM = [model_param[i] for i in range(6)]

	h0 = 70
	wa = 0

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jlac_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*0.3
	cube[3] = cube[3]*5
	cube[4] = cube[4]*10-25
	cube[5] = cube[5]*0.5-0.25





def llhood_jlad_wcdm(model_param, ndim, nparam):
	om, w0, a, b, MB, DM, od, g = [model_param[i] for i in range(8)]

	h0 = 70
	wa = 0

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jla.zcmb.values)

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th - DUST
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jlad_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*0.3
	cube[3] = cube[3]*5
	cube[4] = cube[4]*10-25
	cube[5] = cube[5]*0.5-0.25
	cube[6] = cube[6]*10-10
	cube[7] = cube[7]*9-6





def llhood_jlac_wzcdm(model_param, ndim, nparam):
	om, w0, wa, a, b, MB, DM = [model_param[i] for i in range(7)]

	h0 = 70

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jlac_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*6-3
	cube[3] = cube[3]*0.3
	cube[4] = cube[4]*5
	cube[5] = cube[5]*10-25
	cube[6] = cube[6]*0.5-0.25


def llhood_jlad_wzcdm(model_param, ndim, nparam):
	om, w0, wa, a, b, MB, DM, od, g = [model_param[i] for i in range(9)]

	h0 = 70

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jla.zcmb.values)

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(jla.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jla.mb.values + a*jla.x1.values - b*jla.color.values - MB - mod_th - DUST
	hub_res[jla.hm.values >= 10.] += DM

	C = mu_cov(a, b)
	iC = np.linalg.inv(C)

	chisq = np.dot(hub_res.T, np.dot(iC, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jlad_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*6-4
	cube[2] = cube[2]*6-3
	cube[3] = cube[3]*0.3
	cube[4] = cube[4]*5
	cube[5] = cube[5]*10-25
	cube[6] = cube[6]*0.5-0.25
	cube[7] = cube[7]*10-10
	cube[8] = cube[8]*9-6



########################################################
######################## JLABIN ########################
########################################################



def llhood_jbc_flcdm(model_param, ndim, nparam):
	om, h0 = [model_param[i] for i in range(2)]

	w0 = -1
	wa = 0

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(jb.zb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jb.mub.values - mod_th

	chisq = np.dot(hub_res.T, np.dot(jb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jbc_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50


def llhood_jbc_wcdm(model_param, ndim, nparam):
	om, h0, w0 = [model_param[i] for i in range(3)]

	wa = 0

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(jb.zb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jb.mub.values - mod_th

	chisq = np.dot(hub_res.T, np.dot(jb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jbc_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4

def llhood_jbc_wzcdm(model_param, ndim, nparam):
	om, h0, w0, wa = [model_param[i] for i in range(4)]

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(jb.zb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jb.mub.values - mod_th

	chisq = np.dot(hub_res.T, np.dot(jb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jbc_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*6-3

def llhood_jbd_flcdm(model_param, ndim, nparam):
	om, h0, od, g = [model_param[i] for i in range(4)]

	w0 = -1
	wa = 0

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jb.zb.values)

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(jb.zb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jb.mub.values - mod_th - DUST

	chisq = np.dot(hub_res.T, np.dot(jb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jbd_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*10-10
	cube[3] = cube[3]*9-6


def llhood_jbd_wcdm(model_param, ndim, nparam):
	om, h0, w0, od, g = [model_param[i] for i in range(5)]

	wa = 0

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jb.zb.values)

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(jb.zb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jb.mub.values - mod_th - DUST

	chisq = np.dot(hub_res.T, np.dot(jb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jbd_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*10-10
	cube[4] = cube[4]*9-6



def llhood_jbd_wzcdm(model_param, ndim, nparam):
	om, h0, w0, wa, od, g = [model_param[i] for i in range(6)]


	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(jb.zb.values)

	dist_th = w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(jb.zb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = jb.mub.values - mod_th

	chisq = np.dot(hub_res.T, np.dot(jb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_jbd_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*6-3
	cube[4] = cube[4]*10-10
	cube[5] = cube[5]*9-6


########################################################################
########################################################################
########################################################################


########################################################################
######################## PANTHEON BINNED ###############################
########################################################################

def llhood_pbc_flcdm(model_param, ndim, nparam):
	om, h0, MB = [model_param[i] for i in range(3)]

	w0 = -1
	wa = 0

	
	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_pbc_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*10-25



def llhood_pbc_wcdm(model_param, ndim, nparam):
	om, h0, w0, MB = [model_param[i] for i in range(4)]

	wa = 0

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_pbc_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*10-25

def llhood_pbc_wzcdm(model_param, ndim, nparam):
	om, h0, w0, wa, MB = [model_param[i] for i in range(5)]

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_pbc_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*6-3
	cube[4] = cube[4]*10-25

####################### WITH DUST ######################

def llhood_pbd_flcdm(model_param, ndim, nparam):
	om, h0, MB, od, g = [model_param[i] for i in range(5)]

	w0 = -1
	wa = 0

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(pb.zcmb.values)

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB - DUST

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_pbd_flcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*10-25
	cube[3] = cube[3]*10-10
	cube[4] = cube[4]*9-6
 




def llhood_pbd_wcdm(model_param, ndim, nparam):
	om, h0, w0, MB, od, g = [model_param[i] for i in range(6)]

	wa = 0

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(pb.zcmb.values)

	dist_th = wCDM(h0, om, 1-om, w0).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB - DUST

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_pbd_wcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*10-25
	cube[4] = cube[4]*10-10
	cube[5] = cube[5]*9-6




def llhood_pbd_wzcdm(model_param, ndim, nparam):
	om, h0, w0, wa, MB, od, g = [model_param[i] for i in range(7)]

	A = []
	for i in range(0,50):
		At = 10**(od)*attenuation(om, h0, z50[i], w0, wa, g)
		A.append(At)
	A_inter = interp1d(z50, A)
	DUST = A_inter(pb.zcmb.values)

	dist_th = FlatLambdaCDM(h0, om).luminosity_distance(pb.zcmb.values).value
	mod_th = 5*np.log10(dist_th) + 25

	hub_res = pb.mb.values - mod_th - MB - DUST

	chisq = np.dot(hub_res.T, np.dot(pb_icm, hub_res))

	if cmb == 1:
		T = cmb_chisq(om, h0, w0, wa)
		chisq += T

	return -0.5*chisq

def prior_pbd_wzcdm(cube, ndim, nparam):
	cube[0] = cube[0]
	cube[1] = cube[1]*50+50
	cube[2] = cube[2]*6-4
	cube[3] = cube[3]*6-3
	cube[4] = cube[4]*10-25
	cube[5] = cube[5]*10-10 #log od
	cube[6] = cube[6]*9-6 #gamma





def prior_general(cube, ndim, nparam):
	cube[0] = cube[0]*2 #om
	cube[0] = cube[0]*50+50 #h0
	cube[0] = cube[0]*10-25 #MB
	cube[0] = cube[0]*6-4 #w0
	cube[0] = cube[0]*10-5 #wa
	cube[0] = cube[0]*1-0.5 #DM
	cube[0] = cube[0]*0.3 #alpha
	cube[0] = cube[0]*5 #beta
	cube[0] = cube[0]*10-10 #log od
	cube[0] = cube[0]*9-6 #gamma

#add cmb+bao priors










cmb = 0
start = time()
pmn.run(llhood_jlac_flcdm, prior_jlac_flcdm, 5, verbose = True, n_live_points = 2000)
pmn.run(llhood_jlac_wcdm, prior_jlac_wcdm, 6, verbose = True, n_live_points = 2000)
pmn.run(llhood_jlac_wzcdm, prior_jlac_wzcdm, 7, verbose = True, n_live_points = 2000)
pmn.run(llhood_jlad_flcdm, prior_jlad_flcdm, 7, verbose = True, n_live_points = 2000)
pmn.run(llhood_jlad_wcdm, prior_jlad_wcdm, 8, verbose = True, n_live_points = 2000)
pmn.run(llhood_jlad_wzcdm, prior_jlad_wzcdm, 9, verbose = True, n_live_points = 2000)

pmn.run(llhood_jbc_flcdm, prior_jbc_flcdm, 2, verbose = True, n_live_points = 2000)
pmn.run(llhood_jbc_wcdm, prior_jbc_wcdm, 3, verbose = True, n_live_points = 2000)
pmn.run(llhood_jbc_wzcdm, prior_jbc_wzcdm, 4, verbose = True, n_live_points = 2000)
pmn.run(llhood_jbd_flcdm, prior_jbd_flcdm, 4, verbose = True, n_live_points = 2000)
pmn.run(llhood_jbd_wcdm, prior_jbd_wcdm, 5, verbose = True, n_live_points = 2000)
pmn.run(llhood_jbd_wzcdm, prior_jbd_wzcdm, 6, verbose = True, n_live_points = 2000)

pmn.run(llhood_pbc_flcdm, prior_pbc_flcdm, 3, verbose = True, n_live_points = 2000)
pmn.run(llhood_pbc_wcdm, prior_pbc_wcdm, 4, verbose = True, n_live_points = 2000)
pmn.run(llhood_pbc_wzcdm, prior_pbc_wzcdm, 5, verbose = True, n_live_points = 2000)
pmn.run(llhood_pbd_flcdm, prior_pbd_flcdm, 5, verbose = True, n_live_points = 2000)
pmn.run(llhood_pbd_wcdm, prior_pbd_wcdm, 6, verbose = True, n_live_points = 2000)
pmn.run(llhood_pbd_wzcdm, prior_pbd_wzcdm, 7, verbose = True, n_live_points = 2000)
end = time()
print('sampler time', end-start)

