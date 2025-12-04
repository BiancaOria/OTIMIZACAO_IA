from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x1 = x[0]
    x2 = x[1]
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

L_INF = np.array([-1, -1])
L_SUP = np.array([3, 3])

# grs = GlobalRandomSearch(target_func=f, max_it=500, epsilon=0.5, opt='max', lim_inf=L_INF, lim_sup=L_SUP)
# grs.search()


# lrs = LocalRandomSearch(target_func=f, max_it=500, sigma=0.999999999, opt='max', lim_inf=L_INF, lim_sup=L_SUP)
# lrs.search() 


hill = Hill(target_func=f, max_it=500, sigma=0.999999999999999999999999999, max_vizinhos=400, opt='max', lim_inf=L_INF, lim_sup=L_SUP)
hill.search()