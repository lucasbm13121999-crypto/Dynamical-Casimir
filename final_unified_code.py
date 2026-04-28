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

# Configurações para o padrão PRL
plt.rcParams.update({
    #"text.usetex": True,            # Usa LaTeX para renderizar
    "font.family": "serif",         # Fonte com serifa (Computer Modern)
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,                # Tamanho base para PRL
    "axes.labelsize": 10,           # Tamanho dos labels (eixos)
    "legend.fontsize": 8,           # Legendas um pouco menores
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 10,
    "figure.figsize": (3.37, 2.5),  # Largura de 1 coluna em polegadas
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    "savefig.dpi": 600,             # Alta resolução para submissão
    "savefig.format": 'pdf',    # Formato vetorial é preferível
    "text.usetex": False,
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "xtick.top": True,
    "xtick.bottom": True,
    "ytick.left": True,             # Garante tick na esquerda
    "ytick.right": True             # <-- ADICIONADO: Fechando a "caixa" de ticks
})
#----------------------------------------------------------------------------------
#Calculations
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
#Plots
#----------------------------------------------------------------------------------
