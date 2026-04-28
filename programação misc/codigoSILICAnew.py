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
einf=2.896
e0=einf+(1.3*10/5.7)**2-1
err=1
correctionFactor=1000
#constant epsilon equal to 3.9
epsilon=3.9
dimLessDelta=(epsilon-1)**2/((epsilon-1+1/Npa)*(epsilon-1+1/Nperp)) 

#wpa=wT*np.sqrt(1+Npa*(e0-1))/np.sqrt(1+Npa*(einf-1))
#wperp=wT*np.sqrt(1+Nperp*(e0-1))/np.sqrt(1+Nperp*(einf-1))
def perfectEmission(Om):
    return 8/(2835*np.pi)*(rpa*rperp**2/c**3*(1/Npa-1/Nperp))**2*Om**7



    
def emissionIntegrand(u,Om):
    return (1-u**2)**3*np.abs(dimLessDelta)**2

def emissionSILICA(Om):
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om**7
    integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(Om),limit=2000)
    return integral*prefactor

'''Plots'''
    
def delta2plot(a,b,num):
    wPlt=np.logspace(a,b,num)
    #plt.ylabel('|$\Delta$|$^2$')
    plt.xlabel('$\omega$ (rad/s)')
    plt.grid(True)
    plt.loglog(wPlt,np.abs(dimLessDelta)**2,label='delta')    
    plt.loglog(wPlt,np.abs(epsilon(wPlt)-1)**2,label='num')#numerador
    plt.legend(['$\propto$|$\Delta$|$^2$','|$\epsilon(\omega)-1$|$^2$'])
    #plt.savefig('plasmaDelta2.png',dpi=400)
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
    
def normEmissionPlot(a,b,num=10000): 
    omPlt=np.logspace(a,b,num)
    emissionPlt2=np.zeros_like(omPlt)
    for i in range(omPlt.size):
        emissionPlt2[i]=emissionSILICA(omPlt[i])/perfectEmission(omPlt[i])
    plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.title('Normalized photon emission for SiO$_2$')
    plt.loglog(omPlt,emissionPlt2)
    #plt.savefig('normPlasmaPlot.png',dpi=400)
    plt.show()


#print(emissionBST(wT))
spectrum(wT*1.36, 300)
#delta2plot(9,11,300)
normEmissionPlot(8,10,300)


'''checagens'''
'''
c1=4*np.pi*eps0*rpa*rperp**2/6*(1/Npa-1/Nperp)
c2=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
#print(1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2-c1**2*c2)
Om0=1e2
def lowfreq(w):
    return w**3*np.abs(w-2*Om0)**3
integral0,err0=sci.integrate.quad(lowfreq,0,2*Om0)
print((np.abs(c1*(e0-1)**2/((e0-1+1/Npa)*(e0-1+1/Nperp)))**2*c2*integral0-emissionBST(1e2))/emissionBST(1e5))
'''