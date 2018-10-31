import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import astropy



#PHYSICAL CONSTANTS
cc = 299792458 #speed of light
G = 6.67408*10**(-11) #gravitational constant





#extinction CCM89
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



#wCDM E-function
def E(om, z, w):
	return np.sqrt(om*(1.+z)**3.+(1.-om)*(1.+z)**(3.*w+3))

#Luminosity distance
def DL(om, z_max, w):
	res = integrate.quad(lambda z: 1 / E(om, z, w), 0, z_max)
	return cc/1000*(1.+z_max) * res[0]

#Dust attenuation integral
def attenuation(om, h0, z_max, w, gamma):
	h0si = h0/(3.09*10**(19)) #h0 in SI-units
	res = integrate.quad(lambda z: exti.kappa_lam(0.44*(1+z_max)/(1+z))*(1+z)**(gamma-1)/E(om, z, w), 0, z_max)
	return 1.086*3*h0si*cc/(8*np.pi*G)*res[0]



#calling the extinction class
Rv = 3.1 #total to selective extinction ratio
Av = 30000 #extinction in V-band. Not used

exti = ccm(Rv, Av)


#plotting the attenuation as function of z

z = np.linspace(0.01, 2, 100)

om = 0.309
h0 = 70
w = -1
g = -1
od = 8e-5

A = []
for i in range(0, 100):
	at = od*attenuation(om, h0, z[i], w, g)
	A.append(at)

plt.plot(z, A)
plt.title('dust correction, input parameter values: \n $\Omega_m = 0.309 \ \ H_0 = 70 \ \ w = -1 \ \ \gamma = -1 \ \ \Omega_d = 8\cdot 10^{-5}$')
plt.xlabel('z')
plt.ylabel('$\Delta\mu (=A^{rest}_{B}) [mag]$')
plt.show()