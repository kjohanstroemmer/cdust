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

ndim = 6
nwalkers = 30

outfile = sys.argv[1]


#DATA
C11 = pd.read_csv('/home/kj/dust/sn_data/PANTHEON/C11m.txt', delim_whitespace = True, header = 1)
c11 = C11.sort_values(by = ['zCMB'])

z = c11.zCMB.values
mb = c11.mB.values
mberr = c11.mBERR.values
c = c11.c.values
cerr = c11.cERR.values
x1 = c11.x1.values
x1err = c11.x1ERR.values

#Absolute magnitude of SNIa
MB = -19.3

#MCMC
#priors on the parameters for 
def lnprior(theta):
    om,  h0, wa, w0, alpha, beta = theta
    if  0 < om < 2 and 50 < h0 < 100 and -3 < w0 < 3 and -10 < wa < 10 and 0 < alpha and 0 < beta:
        return 0.0
    return -np.inf

n_A = 0
def lnlike(theta):
	#n_A counts iterations of MCMC, total number is N = nwalkers*iterations/#threads
	global n_A
	n_A += 1
	#define the parameters
	om, h0, w0, wa, alpha, beta = theta
	#luminosity distance: w0wa
	dist_th = astropy.cosmology.w0waCDM(h0, om, 1-om, w0, wa).luminosity_distance(z).value
	#distance modulus
	mod_th = 5*np.log10(dist_th)+25 
	#hubble residuals
	hub_res = - mod_th + mb+alpha*x1-beta*c-MB
	#chi-square
	chisq = np.sum(hub_res**2/mberr**2)
	
	print("Current iteration is", n_A, "(chisq, Om, H0, w0, wa, alpha, beta) is","%.3f" %chisq, "%.3f" %om, "%.3f" %h0, "%.3f" %w0,"%.3f" %wa, "%.3f" %alpha, "%.3f" %beta)
	return -0.5*chisq

def lnprob(theta):
	try:
		lp = lnprior(theta)
		like = lnlike(theta)
		return lp + like
	except:
		return -np.inf

#define the fiducial values
fid_arr = [0.5, 80, -1, 0, 0.2, 3]
label_arr = ["om",  "H0", "w0", "wa", "alpha", "beta"]

pos_min = np.array([ 0,  50, -3, -10, -10, -10])
pos_max = np.array([ 2,  100, 3, 10, 10, 10])
psize = pos_max - pos_min
pos = [fid_arr + 1e-4*np.random.rand(ndim) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)
start = time()
pos, prob, state = sampler.run_mcmc(pos, 2000)

#burnin in the first 50 samples
samples = sampler.chain[:, 400:, :].reshape((-1, ndim))
end = time()
np.savetxt('chains/'+outfile, samples)
print("The sampler took", end - start, "seconds")
#output samples, (om, h0, w0, wa, alpha, beta)
om_t = np.median(samples[:,0])
h0_t = np.median(samples[:,1])
w0_t = np.median(samples[:,2])
wa_t = np.median(samples[:,3])
a_t = np.median(samples[:,4])
b_t = np.median(samples[:,5])

fig = corner.corner(samples, labels=label_arr, truths=[om_t, h0_t, w0_t, wa_t, a_t, b_t], smooth=1)	
plt.show()