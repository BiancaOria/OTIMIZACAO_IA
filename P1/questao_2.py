from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt

def f(x):
   return np.exp(-(x[0]**2+x[1]**2)) + 2*np.exp(-((x[0]-1.7)**2+(x[1]-1.7)**2))



# grs = GlobalRandomSearch(target_func=f, max_it=500, epsilon=0.5, opt='max', lim_inf=np.array([-2,-2]), lim_sup=np.array([4,5]))
# grs.search()


# lrs = LocalRandomSearch(target_func=f, max_it=500, sigma=0.8, opt='max', lim_inf=np.array([-2,-2]), lim_sup=np.array([4,5]))
# lrs.search() 


hill = Hill(target_func=f, max_it=100, sigma=0.5, max_vizinhos=20, opt='max', lim_inf=np.array([-2,-2]), lim_sup=np.array([4,5]))
hill.search()