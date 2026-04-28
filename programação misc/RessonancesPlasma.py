# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:51:52 2024

@author: Lucas
"""



import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
import scipy as sci
import sympy as sym



u, Om, wp, gamma, Nperp, Npa=sym.symbols('u Om wp gamma Nperp Npa')
x1=Om**2/(wp**2*Nperp)
y1=Om*gamma/(wp**2*Nperp)
x2=Om**2/(wp**2*Npa)
y2=Om*gamma/(wp**2*Npa)
sol=sym.solveset(((1-x1*u**2)**2+(y1*u)**2)*((1-x2*u**2)**2+(y2*u)**2),u)
for i in sol.args:
    print(i)
    print('\n')

'''
PICOS:
    sqrt(-gamma*sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2))


    sqrt(gamma*sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2))


    sqrt(-gamma*sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2))


    sqrt(gamma*sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2))


    -sqrt(-gamma*sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2))


    -sqrt(gamma*sqrt(-4*Npa*wp**2 + gamma**2)/(2*Om**2) + (2*Npa*wp**2 - gamma**2)/(2*Om**2))


    -sqrt(-gamma*sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2))


    -sqrt(gamma*sqrt(-4*Nperp*wp**2 + gamma**2)/(2*Om**2) + (2*Nperp*wp**2 - gamma**2)/(2*Om**2))
'''