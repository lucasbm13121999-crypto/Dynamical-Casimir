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
eps0=8.8541878188e-12 #F m^-1
Nperp=0.15
Npa=0.7
rperp=1e-9 #m
rpa=2e-9 
c=299792458#m/s
wp=1e10
gamma=wp/100
err=1
#defining the functions  
def epsilon(w):
    return 1-wp**2/(w*(w+1j*gamma))

def Delta(w):
    return 4*np.pi*eps0*rpa*rperp**2/6*(Nperp-Npa)*(epsilon(w)-1)**2/(Nperp*Npa*(epsilon(w)-1+1/Npa)*(epsilon(w)-1+1/Nperp))
'''
Emission of photons
'''
def emissionIntegrand(w,Om):
    if w==Om:
        return w**3*np.abs(w-2*Om)**3*np.abs(Delta(w-Om+1e-3))**2
    else:
        return w**3*np.abs(w-2*Om)**3*np.abs(Delta(w-Om))**2
    
def emissionPlasma(Om):
    prefactor=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
    integral,err=sci.integrate.quad(emissionIntegrand,0,2*Om,args=(Om),limit=100)
    return integral*prefactor
def perfectEmission(Om):
    return 8/(2835*np.pi)*(rpa*rperp**2/c**3*(1/Npa-1/Nperp))**2*Om**7



'''
Epsilon plot
'''
def epsilonPlot(a,b,num):
    print('nao sei fazer o grafico de epsilon!!!')

'''
|Delta|^2 plot
'''
def delta2plot(a,b,num):
    wPlt=np.logspace(a,b,num)
    plt.ylabel('|$\Delta$|$^2$')
    plt.xlabel('$\omega$ (rad/s)')
    plt.grid(True)
    plt.loglog(wPlt,np.abs(Delta(wPlt))**2)    
    plt.savefig('plasmaDelta2.png',dpi=400)
    plt.show()

'''Emission Plot'''
def emissionPlot(a,b,num=10000): 
    omPlt=np.logspace(a,b,num)
    emissionPlt2=np.zeros_like(omPlt)
    for i in range(omPlt.size):
        emissionPlt2[i]=emissionPlasma(omPlt[i])
    plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.title('Normalized photon emission for Drude Model')
    plt.loglog(omPlt,emissionPlt2)
    plt.savefig('plasmaPlot.png',dpi=400)
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
    plt.savefig('normPlasmaPlot.png',dpi=400)
    plt.show()
def findDelta2peaks(a,b,num):
    wPlt=np.logspace(a,b,num)
    d=np.abs(Delta(wPlt))**2
    peaks,err=sci.signal.find_peaks(d)
    return wPlt[peaks]
def spectrum(a,b,Om,num):
    wPlt=np.linspace(a,b,num)
    specPlt=np.zeros_like(wPlt)
    for i in range(wPlt.size):
        specPlt[i]=emissionIntegrand(wPlt[i], Om)
    #plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$\omega$ (rad/s)')
    plt.grid(True)
    plt.title('$\Omega=$%.0E' %Om)
    plt.plot(wPlt,specPlt)
    plt.savefig('spec\%.0E.png' %Om,dpi=400)
    plt.show()
#epsilonPlot(1e8,1e12,600)
#delta2plot(8,10.5,800)
#normEmissionPlot(8,13,300)
#print(findDelta2peaks(8,10.5,800))
#print(np.sqrt(wp**2*Npa))
#emissionPlot(8,13,300)
Om1=[1e8,2e9,3e9,4e9,5e9,8e9,1e10,3e10,7e10,1e11,3e11,5e11,1e13]
for i in Om1:  
    spectrum(0,2*i,i, 4000)