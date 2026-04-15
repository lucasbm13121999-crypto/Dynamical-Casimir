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
from matplotlib.cm import get_cmap

#norm
emissionExp=3.53e-23


#constants
eps0=8.8541878188e-12 #F m^-1
c=299792458#m/s
wT=5.7e9
gamma=2.8e8
einf=2.896
e0=einf+(1.3*10/5.7)**2-1
eps0=8.8541878188e-12 #F m^-1
c=299792458#m/s
err=1
vs=1.5e5#carbon nanotube sound speed in m/s
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
def epsilon(w):
    return einf+(e0-einf)/(1-(w/wT)**2+1j*w*gamma/wT**2)

def dimLessDelta(w,Npa1,Nperp1):
    return (epsilon(w)-1)**2/((epsilon(w)-1+1/Npa1)*(epsilon(w)-1+1/Nperp1))

def emissionIntegrand(u,Om1,Npa1,Nperp1):
    return (1-u**2)**3*np.abs(dimLessDelta(u*Om1,Npa1,Nperp1))**2

def emissionBST(rpa,exc):
    Om1=Om(rpa)
    Npa1=Npa(exc)
    Nperp1=Nperp(exc)
    rperp1=rperp(rpa,exc)
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa1-1/Nperp1)**2*(rpa*rperp1**2)**2*Om1**7
    integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(Om1,Npa1,Nperp1),limit=2000)
    return integral*prefactor

def lowfreqEmission(rpa,exc):
    Om1=Om(rpa)
    Npa1=Npa(exc)
    Nperp1=Nperp(exc)
    rperp1=rperp(rpa,exc)
    c1=4*np.pi*eps0*rpa*rperp1**2/6*(1/Npa1-1/Nperp1)
    c2=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
    def lowfreq(w):
        return w**3*np.abs(w-2*Om1)**3
    integral0,err0=sci.integrate.quad(lowfreq,0,2*Om1)
    return np.abs(c1*(e0-1)**2/((e0-1+1/Npa1)*(e0-1+1/Nperp1)))**2*c2*integral0

def normEmissionVaringRpa(a,b,excList,num):
    rpaPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(excList),rpaPlt.size))
    #print(emissionPlt1)
    for i in range(len(excList)):
        for j in range(rpaPlt.size):
            emissionPlt1[i,j]=emissionBST(rpaPlt[j],excList[i])/emissionExp
        if len(excList)<=5:
            gradient_hex = ["#8B0000", "#FF8C00", "#FFD700", "#006400", "#00008B"]
            plt.loglog(rpaPlt*1e6,emissionPlt1[i],label='e = '+str(excList[i]),color=gradient_hex[i])
        else:
            plt.loglog(rpaPlt*1e6,emissionPlt1[i],label='e = '+str(excList[i]))    
        
    plt.ylabel('$\Gamma/\Gamma_{qs}$')
    plt.xlabel('$r_{\parallel}$ ($\mu$m)')
    #plt.ylim(1e-31,1e-17)
    plt.grid(True)
    plt.legend()
    plt.savefig('NORMemission_BST_Rpa.png',dpi=400)
    plt.show()

def emissionVaringRpa(a,b,excList,num):
    rpaPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(excList),rpaPlt.size))
    #print(emissionPlt1)
    for i in range(len(excList)):
        for j in range(rpaPlt.size):
            emissionPlt1[i,j]=emissionBST(rpaPlt[j],excList[i])#/lowfreqEmission(rpaPlt[j], excList[i])
        if len(excList)<=5:
            gradient_hex = ["#8B0000", "#FF8C00", "#FFD700", "#006400", "#00008B"]
        plt.loglog(rpaPlt,emissionPlt1[i],label='e = '+str(excList[i]),color=gradient_hex[i])
    plt.ylabel('$\Gamma $(s$^{-1}$)')
    plt.xlabel('$r_{\parallel}$ (m)')
    #plt.ylim(1e-31,1e-17)
    plt.grid(True)
    plt.legend()
    plt.savefig('NORMemission_BST_Rpa.png',dpi=400)
    plt.show()
'''  
def normEmissionVaringRpa(a,b,excList,num):
    rpaPlt=np.logspace(a,b,num)
    emissionPlt1=np.zeros((len(excList),rpaPlt.size))
    #print(emissionPlt1)
    for i in range(len(excList)):
        for j in range(rpaPlt.size):
            emissionPlt1[i,j]=emissionBST(rpaPlt[j],excList[i])/lowfreqEmission(rpaPlt[j], excList[i])
        plt.loglog(rpaPlt,emissionPlt1[i],label='e = '+str(excList[i]))
    plt.ylabel('$\Gamma/\Gamma_{QS}$')
    plt.xlabel('$r_{\parallel}$ (m)')
    #plt.ylim(1e-31,1e-17)
    plt.grid(True)
    plt.legend()
    plt.savefig('NORMemission_BST_Rpa.png',dpi=400)
    plt.show()
'''
def excMax(rpa,num=1000):
    excPlt=np.linspace(0.001,0.999,num)
    emissionPlt2=np.zeros_like(excPlt)
    for i in range(excPlt.size):
        emissionPlt2[i]=emissionBST(rpa, excPlt[i])
    plt.plot(excPlt,emissionPlt2)
    plt.ylabel('$\Gamma$ $(s^{-1})$')
    plt.xlabel('$e$')
    plt.grid(True)
    plt.legend('$r$')
    #plt.savefig('maxExc.png',dpi=400)
    plt.show()
    print(excPlt[emissionPlt2.argmax()])
def excMaxVarious(rpaList,num=300):
    
    excPlt=np.linspace(0.001,0.999,num)
    emissionPlt1=np.zeros((len(rpaList),excPlt.size))
    #print(emissionPlt1)
    for i in range(len(rpaList)):
        for j in range(excPlt.size):
            emissionPlt1[i,j]=emissionBST(rpaList[i],excPlt[j])
        #plt.plot(excPlt,emissionPlt1[i],label='r = '+str(rpaList[i]))
    for i in range(len(rpaList)):
        maxi=emissionBST(rpaList[i],excPlt[emissionPlt1[i].argmax()])
        print(maxi)
        for j in range(excPlt.size):
            emissionPlt1[i,j]=emissionPlt1[i,j]/maxi
        if len(rpaList)<=5:
            gradient_hex = ["#8B0000", "#FF8C00", "#FFD700", "#006400", "#00008B"]
            plt.plot(excPlt,emissionPlt1[i],label='r$_\parallel$ = '+str(round(rpaList[i]*1e6,2))+' ($\mu$m)',color=gradient_hex[i])
        else:
            plt.plot(excPlt,emissionPlt1[i],label='r$_\parallel$ = '+str(round(rpaList[i]*1e6,2))+' ($\mu$m)')
    
    plt.ylabel('$\Gamma/\Gamma_{max}$')
    plt.xlabel('$e$')
    plt.grid(True)
    plt.legend()
    plt.savefig('FINALEXCPLOT.png',dpi=500)
    plt.show()
    
excList=[0.2,0.4,0.6,0.9,0.99]    
excMaxVarious([1e-6,1.5e-5,1.7e-5,1.9e-5,4e-5])
#normEmissionVaringRpa(-7.3, -5.5, excList, 400)
#emissionVaringRpa(-6, -4., excList, 400)
#excMax(0.5e-6,200)
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
