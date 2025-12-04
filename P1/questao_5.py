from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x1 = x[0]
    x2 = x[1]
    return (x1 * np.cos(x1) / 20) + (2 * np.exp(-x1**2 - (x2 - 1)**2)) + (0.01 * x1 * x2)



# grs = GlobalRandomSearch(target_func=f, max_it=500, epsilon=0.5, opt='max', lim_inf=np.array([-10, -10]), lim_sup=np.array([10, 10]))
# grs.search()


# lrs = LocalRandomSearch(target_func=f, max_it=500, sigma=0.8, opt='max', lim_inf=np.array([-10, -10]), lim_sup=np.array([10, 10]))
# lrs.search() 


hill = Hill(target_func=f, max_it=100, sigma=10, max_vizinhos=10000, opt='max', lim_inf=np.array([-10, -10]), lim_sup=np.array([10, 10]))
hill.search()