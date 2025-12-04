from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x1 = x[0]
    x2 = x[1]
    
    termo1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    
    termo2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    
    termo3 = 20 + np.exp(1)
    
    return termo1 + termo2 + termo3

L_INF = -8
L_SUP = 8

# grs = GlobalRandomSearch(target_func=f, max_it=500, epsilon=0.5, lim_inf=L_INF, lim_sup=L_SUP)
# grs.search()


# lrs = LocalRandomSearch(target_func=f, max_it=500, sigma=0.8, lim_inf=L_INF, lim_sup=L_SUP)
# lrs.search() 


hill = Hill(target_func=f, max_it=100, sigma=0.999999999, max_vizinhos=50, lim_inf=L_INF, lim_sup=L_SUP)
hill.search()