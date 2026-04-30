# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:16:14 2024

@author: Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import scipy as sci
import sympy as sym


#constants
einf=2.9
eps0=8.8541878188e-12 #F m^-1
wL=1.3e10 #Hz
wT=5.7e9 #Hz
gamma=2.8e8 #Hz^-1
Nperp=0.15
Npa=0.7
rperp=1e-9 #m
rpa=2e-9 
c=299792458#m/s
err=1
#defining the functions
def epsilon(w):
    return einf*(1+(wL**2-wT**2)/(wT**2-w**2-1j*w*gamma))

def Delta(w):
    return 4*np.pi*eps0*rpa*rperp**2/6*(Nperp-Npa)*(epsilon(w)-1)**2/(Nperp*Npa*(epsilon(w)-1+1/Npa)*(epsilon(w)-1+1/Nperp))
'''
Emission of photons
'''
def emissionIntegrand(w,Om,normalization):
    return w**3*np.abs(w-2*Om)**3*np.abs(Delta(normalization*(w-Om)))**2
def emission(Om,normalization):
    prefactor=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
    integral,err=sci.integrate.quad(emissionIntegrand,0,2*Om,args=(Om,normalization,))
    return integral*prefactor
def perfectEmission(Om):
    return 8/(2835*np.pi)*(rpa*rperp**2/c**3*(1/Npa-1/Nperp))**2*Om**7



'''
Epsilon plot
'''
def epsilonPlot():
    #plot parameters
    a=1
    b=10
    num=10000 #number of points
    wPlt=np.logspace(a,b,num)
    plt.ylabel('$\epsilon(\omega)$')
    plt.xlabel('$\omega$ (Hz)')
    plt.grid(True)
    """plot limits"""
    plt.ylim((-150,300))
    '''Real part'''
    plt.plot(wPlt,np.real(epsilon(wPlt)),label='Real Part')
    '''Imaginary part'''
    plt.plot(wPlt,np.imag(epsilon(wPlt)),label='Imaginary Part')
    plt.legend()
    plt.show()

'''
|Delta|^2 plot
'''
def delta2plot():
    a=8
    b=12
    num=6000 #number of points
    wPlt=np.logspace(a,b,num)
    plt.ylabel('|$\Delta$|$^2$')
    plt.xlabel('$\omega$ (Hz)')
    plt.grid(True)
    plt.loglog(wPlt,np.abs(Delta(wPlt))**2)    
    plt.show()

'''Emission Plot'''
def emissionPlot():
    a=8
    b=12
    num=600 #number of points
    omPlt=np.logspace(a,b,num)
    emissionPlt=np.zeros_like(omPlt)
    for i in range(omPlt.size):
        emissionPlt[i]=emission(omPlt[i],1)/emission(omPlt[i],0)
    plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.loglog(omPlt,emissionPlt)
    plt.show()
def perfectEmissionPlot():
    a=8
    b=12
    num=700 #number of points
    omPlt=np.logspace(a,b,num)
    emissionPlt2=np.zeros_like(omPlt)
    for i in range(omPlt.size):
        emissionPlt2[i]=emission(omPlt[i],1)/perfectEmission(omPlt[i])
    plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.title('Normalized photon emission for BST')
    plt.loglog(omPlt,emissionPlt2)
    plt.savefig('perfectPlot.png',dpi=400)
    plt.show()
epsilonPlot()
delta2plot()
#emissionPlot()
perfectEmissionPlot()
#print(emission(1e13,1)/perfectEmission(1e13))
