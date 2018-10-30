import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
import sncosmo
import emcee
import scipy.integrate as integrate
import corner
import astropy
import random
import sys
import extinction

ndim = 6
nwalkers = 30

outfile = sys.argv[1]


#this code fits wCDM to the binned PANTHEON data using a term correcting for dust in the IGM.



B = pd.read_csv('/home/kj/dust/sn_data/lcparam_DS17f.txt', delim_whitespace = True, header = 0)

data = B.sort_values(by = ['zcmb'])

z = data.zcmb.values
mb = data.mb.values
dmb = data.dmb.values



#PHYSICAL CONSTANTS
cc = 299792458 #speed of light
om = 0.309 #matter density
d_om = 0.006 #error in matter density
G = 6.67408*10**(-11) #gravitational constant
h0 = 70.511 #hubble constant
h0_si = h0/(3.09*10**(19)) # hubble constant in SI





#FUNCTIONS




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
		kappa = 1.54e4*alav
		return kappa

Rv = 3.1 #total to selective extinction
Av = 0 #extinction in V-band. Not used
exti = ccm(Rv, Av)


def E(om, z, w):
	#FlatLambdaCDM Friedmann
	return np.sqrt(om*(1.+z)**3.+(1.-om)*(1.+z)**(3.*w+3))

#Define luminosity distance. H0 removed: log(DL) = log(I)-log(H0)
def DL(om, z_max, w):
	res = integrate.quad(lambda z: 1 / E(om, z, w), 0, z_max)
	return cc/1000*(1.+z_max) * res[0]

#Define attenuation function
def attenuation(om, h0, z_max, w, gamma):
	h0si = h0/(3.09*10**(19)) #h0 in SI
	res = integrate.quad(lambda z: exti.kappa_lam(0.44*(1+z_max)/(1+z))*(1+z)**(gamma-1)/E(om, z, w), 0, z_max)
	return 1.086*3*h0si*cc/(8*np.pi*G)*res[0]


#PRIORS on parameters
#om
l_om = 0
u_om = 2
#h0
l_h0 = 50
u_h0 = 100
#w
l_w = -5
u_w = 5
#alpha 
l_a = 0
u_a = 10
#beta
l_b = 0
u_b = 10
#o_d
l_od = 0
u_od = 1
#log od
l_logod = -15
u_logod = -1
#gamma
l_g = -10
u_g = 10
#MB (absolute magnitude)
l_MB = -30
u_MB = -10



#MCMC

#log prior function
def lnprior(theta):
    om, h0, w, od, g, MB = theta
    if  l_om < om < u_om and l_h0 < h0 < u_h0 and l_w < w < u_w and l_logod < od < u_logod and l_g < g < u_g and l_MB < MB < u_MB:
        return 0.0
    return -np.inf


n_A = 0
def lnlike(theta):
	#n_A counts MCMC iterations, total number is N = nwalkers*iterations/#threads
	global n_A
	n_A += 1
	#define the parameters
	om, h0, w, od, g, MB = theta


	om = 0.309
	h0 = 70
	w = -1

	#luminosity distance: wCDM
	dist_th = astropy.cosmology.wCDM(h0, om, 1-om, w).luminosity_distance(z).value
	#distance modulus	
	mod_th = 5*np.log10(dist_th)+25 
	#attenuation term 
	A = []
	for i in range(0,len(z)):
		At = 10**(od)*attenuation(om, h0, z[i], w, g)
		A.append(At)
	#hubble residuals
	hub_res = - mod_th + mb - MB + A
	#chi-square
	chisq = np.sum(hub_res**2/dmb**2)
	
	#safety measure
	if np.isnan(chisq) == 1:
		return -np.inf
	
	print("iter", n_A, "(chisq, Om, H0, w, od, gamma, MB) is","%.3f" %chisq, "%.3f" %om, "%.3f" %h0, "%.3f" %w, "%.3f" %od, "%.3f" %g, "%.3f" %MB)
	return -0.5*chisq

def lnprob(theta):
	try:
		lp = lnprior(theta)
		like = lnlike(theta)
		if np.isnan(like) == 1:
			return -np.inf
		return lp + like
	except:
		return -np.inf

#define the fiducial values
fid_arr = [0.5, 80, -1, -5, -1, -19.3]
label_arr = ["om",  "H0", "w", "od", "gamma", "MB"]

pos_min = np.array([ l_om,  l_h0, l_w, l_logod, l_g, l_MB])
pos_max = np.array([ u_om,  u_h0, u_w, u_logod, u_g, u_MB])
psize = pos_max - pos_min
pos = [fid_arr + 1e-4*np.random.rand(ndim) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=3)
start = time()
pos, prob, state = sampler.run_mcmc(pos, 2000)

#burnin in the first 50 samples
samples = sampler.chain[:, 400:, :].reshape((-1, ndim))
end = time()
np.savetxt('chains/'+outfile, samples)
print("The sampler took", end - start, "seconds")
#output samples, (om, h0, w, od, gamma, MB)
om_t = np.median(samples[:,0])
h0_t = np.median(samples[:,1])
w_t = np.median(samples[:,2])
od_t = np.median(samples[:,3])
g_t = np.median(samples[:,4])
MB_t = np.median(samples[:,5])

fig = corner.corner(samples, labels=label_arr, truths=[om_t, h0_t, w_t, od_t, g_t, MB_t], smooth=1)	
plt.show()