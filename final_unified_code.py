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
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",       # <-- ADICIONADO: Garante que a matemática use Computer Modern mesmo sem LaTeX
    "text.usetex": False,           # Pode mudar para True se tiver o TeX Live / MiKTeX instalado
    
    # --- Tamanhos ---
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    
    # --- Dimensões e Resolução ---
    "figure.figsize": (3.37, 2.5),  # Largura de 1 coluna PRL
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
#----------------------------------------------------------------------------------
#Calculations
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#Plots
#----------------------------------------------------------------------------------
