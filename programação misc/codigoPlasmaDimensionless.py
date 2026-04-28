# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:16:14 2024

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
wp=1.36734e16#rad/s
gamma=5.31744e13
err=1
#picos={np.sqrt(-gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),np.sqrt(gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),np.sqrt(-gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2)),np.sqrt(gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(-gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(-gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2))}
correctionFactor=1000


'''
Emission of photons
'''
def emissionIntegrand(u,Om):
    x1=Om**2/(wp**2*Nperp)
    y1=Om*gamma/(wp**2*Nperp)
    x2=Om**2/(wp**2*Npa)
    y2=Om*gamma/(wp**2*Npa)
    return (1-u**2)**3/(((1-x1*u**2)**2+(y1*u)**2)*((1-x2*u**2)**2+(y2*u)**2))

def emissionPlasma(Om):
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om**7
    
    if Om<wp:
        integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(Om),limit=2000)
        return integral*prefactor
    elif Om>=wp:
        lim=wp/Om
        integral,err=sci.integrate.quad(emissionIntegrand,-lim,lim,args=(Om),limit=2000)
        return integral*prefactor
    
def perfectEmission(Om):
    return 8/(2835*np.pi)*(rpa*rperp**2/c**3*(1/Npa-1/Nperp))**2*Om**7

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



    
def normEmissionPlot(a,b,num=10000): 
    omPlt=np.logspace(a,b,num)
    emissionPlt2=np.zeros_like(omPlt)
    for i in range(omPlt.size):
        emissionPlt2[i]=emissionPlasma(omPlt[i])/perfectEmission(omPlt[i])
    plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.title('Normalized photon emission for Drude Model')
    plt.loglog(omPlt,emissionPlt2)
    plt.savefig('normPlasmaPlot.png',dpi=500)
    plt.show()


#Om1=[1e8,2e9,3e9,4e9,5e9,8e9,1e10,3e10,7e10,1e11,3e11,5e11,1e13]
#spectrum(100*wp, 4000)
#for i in Om1:  
    #spectrum(i, 4000)
normEmissionPlot(14,19,1000)
#print(4/3*np.pi*rpa*rperp**2)