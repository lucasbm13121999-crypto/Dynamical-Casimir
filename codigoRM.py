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


#fundamental constants
eps0=8.8541878188e-12 #F m^-1
c=299792458#m/s
#geometrical constants
Nperp=0.15
Npa=0.7
rperp=1e-9 #m
rpa=2e-9 
#material constants (gold)
wp=1.36734e16#rad/s
gamma=5.31744e13
#silica
eSi=2.25
#calço
d=1.6e-50
def epsilonDrude(w):
    return 1-wp**2/(w*(w+1j*gamma)+d)

def perfectEmission(Om):
    return 8/(2835*np.pi)*(rpa*rperp**2/c**3*(1/Npa-1/Nperp))**2*Om**7
def epsilonRM(w,f):
    return eSi*(epsilonDrude(w)*(1+2*f)+2*eSi*(1-f))/(epsilonDrude(w)*(1-2*f)+eSi*(2+f))

def dimLessDelta(w,f):
    return (epsilonRM(w,f)-1)**2/((epsilonRM(w,f)-1+1/Npa)*(epsilonRM(w,f)-1+1/Nperp))
    
def emissionIntegrand(u,f,Om):
    return (1-u**2)**3*np.abs(dimLessDelta(u*Om,f))**2

def emissionRM(f,Om):
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om**7
    integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(f,Om),limit=2000)
    return integral*prefactor

'''Plots'''
'''  
def delta2plot(a,b,num):
    wPlt=np.logspace(a,b,num)
    #plt.ylabel('|$\Delta$|$^2$')
    plt.xlabel('$\omega$ (rad/s)')
    plt.grid(True)
    plt.loglog(wPlt,np.abs(dimLessDelta(wPlt))**2,label='delta')    
    plt.loglog(wPlt,np.abs(epsilon(wPlt)-1)**2,label='num')#numerador
    plt.legend(['$\propto$|$\Delta$|$^2$','|$\epsilon(\omega)-1$|$^2$'])
    #plt.savefig('plasmaDelta2.png',dpi=400)
    plt.show()
'''

def emissionPlotRM(a,b,fList,num=300): 
    omPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(fList),omPlt.size))
    for i in range(len(fList)):
        for j in range(omPlt.size):
            emissionPlt1[i,j]=emissionRM(fList[i],omPlt[j])/perfectEmission(omPlt[j])
    for i in range(len(fList)):
        plt.loglog(omPlt,emissionPlt1[i])
    
    plt.ylabel('$\Gamma$ (s$^{-1}$)')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    #plt.savefig('normRMPlot.png',dpi=500)
    plt.show()
#fList=[0.1,0.49,0.5,0.51,0.9]
fList=[0.1,0.9]
print(emissionRM(0.1,1e14))
emissionPlotRM(9,11,fList)
#print(emissionBST(wT))
#spectrum(wT*1.36, 300)
#delta2plot(9,11,300)
'''checagens

c1=4*np.pi*eps0*rpa*rperp**2/6*(1/Npa-1/Nperp)
c2=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
#print(1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2-c1**2*c2)
Om0=1e2
integral0,err0=sci.integrate.quad(lowfreq,0,2*Om0)
print((np.abs(c1*(e0-1)**2/((e0-1+1/Npa)*(e0-1+1/Nperp)))**2*c2*integral0-emissionBST(1e2))/emissionBST(1e5))

def lowfreqEmission(Om):
    c1=4*np.pi*eps0*rpa*rperp**2/6*(1/Npa-1/Nperp)
    c2=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
    def lowfreq(w):
        return w**3*np.abs(w-2*Om)**3
    integral0,err0=sci.integrate.quad(lowfreq,0,2*Om)
    return np.abs(c1*(e0-1)**2/((e0-1+1/Npa)*(e0-1+1/Nperp)))**2*c2*integral0
'''