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
#from matplotlib.animation import FuncAnimation


#constants
eps0=8.8541878188e-12 #F m^-1
Nperp=0.15
Npa=0.7
rperp=150e-9 #m
rpa=2*rperp 
c=299792458#m/s
wT=5.7e9
gammaCTE=2.8e8
einf=2.896
e0=einf+(1.3*10/5.7)**2
#e0=3.9 #silica
err=1
correctionFactor=1000
#wpa=wT*np.sqrt(1+Npa*(e0-1))/np.sqrt(1+Npa*(einf-1))
#wperp=wT*np.sqrt(1+Nperp*(e0-1))/np.sqrt(1+Nperp*(einf-1))
def lowfreqEmission(Om):
    c1=4*np.pi*eps0*rpa*rperp**2/6*(1/Npa-1/Nperp)
    c2=1/(9*np.pi*c**6*(4*np.pi*eps0)**2)
    def lowfreq(w):
        return w**3*np.abs(w-2*Om)**3
    integral0,err0=sci.integrate.quad(lowfreq,0,2*Om)
    return np.abs(c1*(e0-1)**2/((e0-1+1/Npa)*(e0-1+1/Nperp)))**2*c2*integral0

def epsilon(w,gamma):
    return einf+(e0-einf)/(1-(w/wT)**2+1j*w*gamma/wT**2)

def dimLessDelta(w,gamma):
    return (epsilon(w,gamma)-1)**2/((epsilon(w,gamma)-1+1/Npa)*(epsilon(w,gamma)-1+1/Nperp))
    
def emissionIntegrand(u,Om,gamma):
    return (1-u**2)**3*np.abs(dimLessDelta(u*Om,gamma))**2

def emissionBST(Om,gamma):
    prefactor=1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om**7
    integral,err=sci.integrate.quad(emissionIntegrand,-1,1,args=(Om,gamma),limit=2000)
    return integral*prefactor



'''Plots'''
    
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

def spectrum(arrayOm,num):
    for i in range(arrayOm.size):
        Om1=arrayOm[i]
        uPlt=np.linspace(0,2,num)
        specPlt=np.zeros(uPlt.size)
        for j in range(uPlt.size):
            specPlt[j]=1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om1**6*emissionIntegrand(uPlt[j]-1, Om1)
        #specPlt=specPlt/specPlt.max()
        plt.ylabel('$d\Gamma/d\omega$')
        plt.xlabel('$\omega/\Omega$')
        plt.grid(True)
        plt.plot(uPlt,specPlt, label=f'$\Omega={arrayOm[i]/wT}\,\omega_T$')
    gradient_hex = ["#8B0000", "#FF8C00", "#FFD700", "#006400", "#00008B"]
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)    
    #plt.ylim(1e-7,1e2)
    plt.tight_layout()
    plt.savefig('spectrumBST', dpi=500)
    plt.show()
    
def normEmissionPlot(a,b,num=300): 
    omPlt=np.logspace(a,b,num)
    emissionPlt2=np.zeros_like(omPlt)
    for i in range(omPlt.size):
        emissionPlt2[i]=emissionBST(omPlt[i],gammaCTE)/lowfreqEmission(omPlt[i])
    plt.ylabel('$\Gamma/\Gamma_{qs}$')
    plt.xlabel('$\Omega$ (rad/s)')
    plt.grid(True)
    plt.loglog(omPlt,emissionPlt2)
    plt.savefig('normBSTPlot.png',dpi=500)
    plt.show()

def gammaEmissionPlot(a,b,num=300):
    gammaPlt=np.logspace(a,b,num)
    emissionPlt3=np.zeros_like(gammaPlt)
    for i in range(gammaPlt.size):
        emissionPlt3[i]=emissionBST(wT,gammaPlt[i])
    plt.ylabel('$\Gamma$')
    plt.xlabel('$\gamma$ (rad/s)')
    plt.grid(True)
    plt.loglog(gammaPlt,emissionPlt3)
    plt.savefig('gammaPlt.png',dpi=500)
    plt.show()
   
    
def testeEpsilon(a,b,num=300):
    gammaList = [1,1e5,1e7, 1e8,1e9,1e10, 1e11]
    wPlt = np.linspace(a, b, num)
    
    # Criamos uma lista de arrays (um array vazio para cada gamma)
    eps_arrays = [np.zeros_like(wPlt) for _ in gammaList]
    plt.yscale('log')
    # Loop principal percorre os gammas (j é o valor, idx é o índice 0,1,2...)
    for idx, j in enumerate(gammaList):
        # Loop interno percorre os pontos do gráfico
        for i in range(wPlt.size):
            # Usa 'j' (o gamma atual) e salva no array correto
            eps_arrays[idx][i] = np.abs(dimLessDelta(wPlt[i], j))**2
    plt.ylabel(r'$|\Delta|^2$')
    plt.xlabel(r'$\omega$ (rad/s)')
    plt.grid(True)
    # Loop inteligente: percorre a lista de gammas e a lista de resultados ao mesmo tempo
    for idx, g in enumerate(gammaList):
        # eps_arrays[idx] pega o resultado correspondente àquele gamma
        # f-string formata o label automaticamente
        plt.plot(wPlt, eps_arrays[idx], label=f'$\gamma={g:.0e}$')
    
    plt.legend()
    # plt.savefig('gammaPlt.png', dpi=500)
    plt.show()
    
listOm=wT*np.array([1/2,1.3,1.90,10])

testeEpsilon(0,2*wT,2000)
#spectrum(listOm, 1000)
#print(lowfreqEmission(wT))
#print(lowfreqEmission(2*np.pi*5.2e9))
#print(emissionBST(wT))
#spectrum(wT*1.36, 300)
#delta2plot(9,11,300)
#normEmissionPlot(8,12,300)
#print(lowfreqEmission(1e7)/(1e7**6))
#print(e0)

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