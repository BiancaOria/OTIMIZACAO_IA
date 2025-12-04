from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt

def f(x):
        return (x[0]**2+x[1]**2)


"""
grs = GlobalRandomSearch(target_func=f, max_it=500, epsilon=0.5, lim_inf=-10, lim_sup=10)
grs.search()


lrs = LocalRandomSearch(target_func=f, max_it=500, sigma=0.8, lim_inf=-10, lim_sup=10)
lrs.search() 

"""

hill = Hill(
    target_func=f, 
    max_it=100, 
    sigma=0.5, 
    max_vizinhos=20, 
    lim_inf=-5, 
    lim_sup=5
)
hill.search()