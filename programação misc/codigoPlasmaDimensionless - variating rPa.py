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

#normalization
Gexp=3.53e-23 #1/s

#constants
eps0=8.8541878188e-12 #F m^-1
c=299792458#m/s
wp=1.36734e16#rad/s
gamma=5.31744e13
err=1
vs=6e3#sound speed in m/s
excList=[0.2,0.4,0.6,0.9,0.99]
#picos={np.sqrt(-gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),np.sqrt(gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),np.sqrt(-gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2)),np.sqrt(gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(-gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(gamma*np.sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(-gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2)),-np.sqrt(gamma*np.sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2))}
correctionFactor=1000
def Om(rpa):
    return vs/rpa
def Npa(exc):
    return (1-exc**2)/(2*exc**3)*(np.log((1+exc)/(1-exc))-2*exc)
def Nperp(exc):
    return (1-Npa(exc))/2
def rperp(rpa,exc):
    return np.sqrt(1-exc**2)*rpa
    
'''
Emission of photons
'''
def emissionIntegrand(u,Om1,Npa1,Nperp1):
    x1=Om1**2/(wp**2*Nperp1)
    y1=Om1*gamma/(wp**2*Nperp1)
    x2=Om1**2/(wp**2*Npa1)
    y2=Om1*gamma/(wp**2*Npa1)
    return (1-u**2)**3/(((1-x1*u**2)**2+(y1*u)**2)*((1-x2*u**2)**2+(y2*u)**2))

def emissionPlasma(rpa,exc):
    Om1=Om(rpa)
    Npa1=Npa(exc)
    Nperp1=Nperp(exc)
    rperp1=rperp(rpa,exc)
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa1-1/Nperp1)**2*(rpa*rperp1**2)**2*Om1**7
    if Om1<wp:
        integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(Om1,Npa1,Nperp1),limit=2000)
        return integral*prefactor
    elif Om1>=wp:
        lim=wp/Om1
        #print(lim)
        integral,err=sci.integrate.quad(emissionIntegrand,-lim,lim,args=(Om1,Npa1,Nperp1),limit=2000)
        return integral*prefactor

def perfectEmission(rpa,exc):
    return 8/(2835*np.pi)*(rpa*rperp(rpa,exc)**2/c**3*(1/Npa(exc)-1/Nperp(exc)))**2*Om(rpa)**7

def lorentzEmission(exc):
    return vs**6*wp**2*(1-exc**2)**2/(9*36*gamma*c**6)*(1/Npa(exc)+1/Nperp(exc))

def emissionVaringRpa(a,b,excList,num):
    rpaPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(excList),rpaPlt.size))
    #print(emissionPlt1)
    for i in range(len(excList)):
        for j in range(rpaPlt.size):
            emissionPlt1[i,j]=emissionPlasma(rpaPlt[j],excList[i])
        plt.loglog(rpaPlt,emissionPlt1[i],label='e = '+str(excList[i]))
    plt.ylabel('$\Gamma$ $(s^{-1})$')
    plt.xlabel('$r_{\parallel}$ ($\mu$m)')
    #plt.ylim(1e-31,1e-17)
    plt.grid(True)
    plt.legend()
    plt.savefig('emissionRpa.png',dpi=400)
    plt.show()

def normEmissionVaringRpa(a,b,excList,num):
    rpaPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(excList),rpaPlt.size))
    #print(emissionPlt1)
    for i in range(len(excList)):
        for j in range(rpaPlt.size):
            emissionPlt1[i,j]=emissionPlasma(rpaPlt[j],excList[i])/perfectEmission(rpaPlt[j],excList[i])
        plt.loglog(rpaPlt,emissionPlt1[i],label=excList[i])
    plt.ylabel('$\Gamma/\Gamma_0$')
    plt.xlabel('$r_{\parallel}$ (m)')
    plt.grid(True)
    plt.legend()
    plt.show()

def normLorentzEmissionVaringRpa(a,b,excList,num):
    rpaPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(excList),rpaPlt.size))
    #print(emissionPlt1)
    for i in range(len(excList)):
        for j in range(rpaPlt.size):
            emissionPlt1[i,j]=emissionPlasma(rpaPlt[j],excList[i])/lorentzEmission(excList[i])
        plt.loglog(rpaPlt,emissionPlt1[i],label=excList[i])
    plt.ylabel('$\Gamma/\Gamma_L $')
    plt.xlabel('$r_{\parallel}$ (m)')
    plt.grid(True)
    print(plt.ylim())
    #plt.legend()
    plt.show()

def excMax(rpa,num=1000):
    excPlt=np.linspace(0.001,0.999,num)
    emissionPlt2=np.zeros_like(excPlt)
    for i in range(excPlt.size):
        emissionPlt2[i]=emissionPlasma(rpa, excPlt[i])
    plt.plot(excPlt,emissionPlt2)
    plt.ylabel('$\Gamma$ $(s^{-1})$')
    plt.xlabel('$e$')
    plt.grid(True)
    #plt.legend()
    plt.savefig('maxExc.png',dpi=400)
    plt.show()
    print(excPlt[emissionPlt2.argmax()])

emissionVaringRpa(-9, 1, excList, 300)
#emissionVaringRpa(-14, -11, excList, 300)    
#excMax(1e-9,500)
'''
normEmissionVaringRpa(-9, -4, excList, 500)
normLorentzEmissionVaringRpa(-9, -3, excList, 400)  
print(emissionPlasma(2e-9 , 0.8))   
   ''' 
    
    
    
    

'''
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
'''


'''    
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
'''
