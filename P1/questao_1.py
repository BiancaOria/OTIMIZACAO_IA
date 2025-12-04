from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt

def f(x):
        return (x[0]**2+x[1]**2)

L_INF = -5
L_SUP = 5

"""
grs = GlobalRandomSearch(target_func=f, max_it=500, epsilon=0.5, lim_inf=L_INF, lim_sup=L_SUP)
grs.search()


lrs = LocalRandomSearch(target_func=f, max_it=500, sigma=0.8, lim_inf=L_INF, lim_sup=L_SUP)
lrs.search() 

"""

hill = Hill(target_func=f, max_it=100, sigma=0.5, max_vizinhos=20, lim_inf=L_INF, lim_sup=L_SUP)
hill.search()