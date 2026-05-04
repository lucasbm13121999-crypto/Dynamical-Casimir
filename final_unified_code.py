# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:16:14 2024

@author: Lucas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from numpy import sqrt
import sympy as sym
from matplotlib.cm import get_cmap

import matplotlib.pyplot as plt

# Configurações Otimizadas para o padrão APS PRL
plt.rcParams.update({
    # --- Fontes e Textos ---
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",       # <-- ADICIONADO: Garante que a matemática use Computer Modern mesmo sem LaTeX
    "text.usetex": False,           # Pode mudar para True se tiver o TeX Live / MiKTeX instalado
    
    # --- Tamanhos ---
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.titlesize": 10,
    
    # --- Dimensões e Resolução ---
    "figure.figsize": (3.37, 3.37/1.61),  # Largura de 1 coluna PRL
    "savefig.dpi": 600,             # 600 para cores/pcolormesh, 1200 para pure line art (se rasterizar algo)
    "savefig.format": 'pdf',
    "savefig.bbox": 'tight',        # <-- ADICIONADO: Evita que os labels sejam cortados ao salvar
    "savefig.pad_inches": 0.05,     # <-- ADICIONADO: Margem mínima para o tight
    
    # --- Linhas ---
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    
    # --- Ticks (Marcadores) ---
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.right": True,
    
    # <-- ADICIONADO: Configurações extras de estética de ticks
    "xtick.minor.visible": True,    # Habilita minor ticks no eixo X
    "ytick.minor.visible": True,    # Habilita minor ticks no eixo Y
    "xtick.major.size": 4,          # Tamanho do tick principal
    "xtick.minor.size": 2,          # Tamanho do tick menor
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
})


#Fig 2
def fixed_geometry():
    #----------------------------------------------------------------------------------
    #Constants
    #----------------------------------------------------------------------------------
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

    #Geometric Constants
    Nperp=0.15
    Npa=0.7
    rperp=75e-9 #m
    rpa=2*rperp      
    #----------------------------------------------------------------------------------
    #Calculations 
    #----------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------
    #Plots
    #----------------------------------------------------------------------------------
    def plot_combined_prl(arrayOm, a, b, num_spec=1000, num_norm=300):
        # Cria a figura e os dois eixos (ax1 = esquerda, ax2 = direita)
        # 6.75 polegadas é o padrão PRL para figuras ocupando duas colunas
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(3.375, 3.375/1.61*2))#3.375/1.61*2
        
        # ==========================================
        # PAINEL (a): spectrumPlt
        # ==========================================
        # O uso de um mapa de cores sequencial (ex: 'viridis') garante 
        # que as cores tenham brilhos diferentes quando convertidas para cinza
        colors = plt.cm.viridis(np.linspace(0, 0.8, arrayOm.size))
        for i in range(arrayOm.size):
            Om1 = arrayOm[i]
            uPlt = np.linspace(0, 2, num_spec)
            specPlt = np.zeros(uPlt.size)
            
            for j in range(uPlt.size):
                specPlt[j] = 1/(36*9*np.pi*c**6)*(1/Npa-1/Nperp)**2*(rpa*rperp**2)**2*Om1**6*emissionIntegrand(uPlt[j]-1, Om1, gamma)
            
            # Plota no eixo 1 (ax1)
            ax1.plot(uPlt, specPlt,color=colors[i], label=f'$\Omega={arrayOm[i]/wT:g}\,\omega_T$')
        ax1.set_ylim(1e-45,1e-27)
        ax1.set_ylabel(r'$\rm{d}\Gamma/\rm{d}\omega$')
        ax1.set_xlabel(r'$\omega/\Omega$')
        ax1.set_yscale('log')
        ax1.grid(True, linestyle=':', alpha=0.25) # Grade mais suave para não poluir
        ax1.set_yticks([1e-45,1e-40,1e-35, 1e-30])
        # Adicionando a legenda (ajuste a posição se necessário)
        #ax1.legend(loc='best', frameon=False, labelspacing=0.15) 
        
        # Texto '(a)' no canto superior esquerdo
        ax1.text(-0.15, 1.05, r'(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

        # ==========================================
        # PAINEL (b): normEmissionPlot
        # ==========================================
        omPlt = np.logspace(a, b, num_norm)
        emissionPlt2 = np.zeros_like(omPlt)
        
        for i in range(omPlt.size):
            emissionPlt2[i] = emissionBST(omPlt[i], gamma)/lowfreqEmission(omPlt[i])

        #mudando de unidade (rad/s -> GHz)
        fPlt_GHz=omPlt*1e-9/(2*np.pi)
        # Plota no eixo 2 (ax2) usando loglog
        ax2.loglog(fPlt_GHz, emissionPlt2,color='black') 
        
        ax2.set_ylabel(r'$\Gamma/\Gamma_{\rm{qs}}$')
        ax2.set_xlabel(r'$\Omega/(2\pi)$ (GHz)')
        ax2.grid(True, linestyle=':', alpha=0.25)
        
        # Texto '(b)' no canto superior esquerdo
        ax2.text(-0.15, 1.05, r'(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

        # ==========================================
        # AJUSTES FINAIS E EXPORTAÇÃO
        # ==========================================
        # Ajusta o espaçamento entre os subplots para evitar sobreposição de textos
        plt.tight_layout()
        #plt.subplots_adjust(hspace=0.3)
        
        plt.savefig('padrao_prl_final_plots/combined_spectrum_emission.pdf', bbox_inches='tight')
        plt.show()

    # --- Chamando a função ---
    # Substitua pelas suas chamadas reais
    listOm = wT * np.array([1/2, 1.3, 1.90, 10])
    plot_combined_prl(listOm, a=9, b=12)
#---------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
import scipy.integrate

def optimized_geometry():
    #----------------------------------------------------------------------------------
    #Constants
    #----------------------------------------------------------------------------------
    eps0=8.8541878188e-12 #F m^-1
    c=299792458#m/s
    wT=5.7e9
    gamma=2.8e8
    einf=2.896
    e0=einf+(1.3*10/5.7)**2-1
    vs=1.5e5#carbon nanotube sound speed in m/s
    
    #----------------------------------------------------------------------------------
    #Calculations 
    #----------------------------------------------------------------------------------
    #Geometric Funcions
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
        
    #----------------------------------------------------------------------------------
    #Plots
    #----------------------------------------------------------------------------------
    def plot_combined_prl(excList, a, b, rpaList, num_a=500, num_b=500):
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(3.375, 3.375/1.61*2))
        
        # ==========================================
        # PAINEL (a): Emission vs r_pa
        # ==========================================
        colorsA = plt.cm.viridis(np.linspace(0, 0.8, len(excList)))

        # CORREÇÃO LÓGICA: geomspace usa os valores reais passados na chamada da função
        rpaPlt = np.logspace(a, b, num_a)
        emissionPlt1 = np.zeros((len(excList), rpaPlt.size))
        
        for i in range(len(excList)):
            for j in range(rpaPlt.size):
                emissionPlt1[i,j] = emissionBST(rpaPlt[j], excList[i])
            # CORREÇÃO LÓGICA: Plot fora do loop 'j'
            ax1.loglog(rpaPlt, emissionPlt1[i])
            
        ax1.set_ylabel('$\Gamma$ (s$^{-1}$)')
        ax1.set_xlabel('$r_{\parallel}$ ($\mu$m)')
        ax1.grid(True, linestyle=':', alpha=0.25) 
        ax1.text(-0.15, 1.05, r'(a)', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

        # ==========================================
        # PAINEL (b): normEmission vs exc
        # ==========================================
        colorsB = plt.cm.viridis(np.linspace(0, 0.8, len(rpaList)))

        excPlt = np.linspace(0.001, 0.999, num_b)
        # CORREÇÃO LÓGICA: Variável emissionPlt2 (não Plt1) deve ser usada aqui
        emissionPlt2 = np.zeros((len(rpaList), excPlt.size))
        
        for i in range(len(rpaList)):
            for j in range(excPlt.size):
                emissionPlt2[i,j] = emissionBST(rpaList[i], excPlt[j])

        for i in range(len(rpaList)):
            # CORREÇÃO LÓGICA: Encontra o máximo direto do array calculado (mais eficiente)
            maxi = np.max(emissionPlt2[i])
            
            # Normalização (vetorial, tira a necessidade do loop j)
            emissionPlt2[i] = emissionPlt2[i] / maxi
            
            # CORREÇÃO LÓGICA: Plot fora do loop 'j'
            ax2.plot(excPlt, emissionPlt2[i])

        #ax2.set_ylabel(r'$\Gamma/\Gamma_{\rm{max}}$')
        #ax2.set_xlabel(r'$e$')
        ax2.grid(True, linestyle=':', alpha=0.25)
        ax2.text(-0.15, 1.05, r'(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
     
        # ==========================================
        # AJUSTES FINAIS E EXPORTAÇÃO
        # ==========================================
        plt.tight_layout()
        plt.savefig('optimized_geometry.pdf', bbox_inches='tight')
        plt.show()

    # --- Chamando a função ---
    excList = [0.2, 0.6, 0.9, 0.99]    
    rpaList = [1e-6, 1.7e-5, 1.9e-5, 4e-5]
    plot_combined_prl(excList, -6, -4, rpaList)

#MAIN
#fixed_geometry()
optimized_geometry()