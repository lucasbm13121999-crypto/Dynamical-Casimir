# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:26:36 2025

@author: Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import scipy as sci
from numpy import sqrt
import sympy as sym


#constants
eps0=8.8541878188e-12 #F m^-1
Nperp=0.15
Npa=0.7
rperp=1e-9 #m
rpa=2e-9 
c=299792458#m/s
wT=5.7e9
gamma=2.8e8
einf=2.9
e0=einf+(1.3*10/5.7)**2-1
err=1
correctionFactor=1000
#wpa=wT*np.sqrt(1+Npa*(e0-1))/np.sqrt(1+Npa*(einf-1))
#wperp=wT*np.sqrt(1+Nperp*(e0-1))/np.sqrt(1+Nperp*(einf-1))
def perfectEmission(Om):
    return 8/(2835*np.pi)*(rpa*rperp**2/c**3*(1/Npa-1/Nperp))**2*Om**7

def epsilon(w):
    return einf+(e0-einf)/(1-(w/wT)**2+1j*w*gamma/wT**2)

def dimLessDelta(w):
    return (epsilon(w)-1)**2/((epsilon(w)-1+1/Npa)*(epsilon(w)-1+1/Nperp))
    
def emissionIntegrand(u,Om):
    return (1-u**2)**3*np.abs(dimLessDelta(u*Om))**2

def emissionBST(Om):
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om**7
    integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(Om),limit=2000)
    return integral*prefactor
'''Plots'''
    
def delta2plot(a,b,num):
    wPlt=np.logspace(a,b,num)
    plt.ylabel('|$\Delta$|$^2$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.loglog(wPlt,np.abs(dimLessDelta(wPlt))**2)    
    plt.savefig('plasmaDelta2.png',dpi=400)
    plt.show()

def spectrum(Om,num):
    wPlt=np.linspace(-1,1,num)
    specPlt=np.zeros_like(wPlt)
    for i in range(wPlt.size):
        specPlt[i]=emissionIntegrand(wPlt[i], Om)
    #plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('u')
    plt.grid(True)
    plt.title('$\Omega=$%.0E' %Om)
    plt.plot(wPlt,specPlt)
    plt.show()

spectrum(wT*1e2, 300)
#print(emissionBST(wT))
#delta2plot(9,11,300)
