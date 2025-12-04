from busca_discreta import GlobalRandomSearch, LocalRandomSearch
from hill import Hill
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def calcular_moda(lista):
    contagem = Counter(lista)
    (valor, freq) = contagem.most_common(1)[0]
    return valor, freq

def f(x):
   return np.exp(-(x[0]**2+x[1]**2)) + 2*np.exp(-((x[0]-1.7)**2+(x[1]-1.7)**2))

lim_inf = np.array([-2,-2])
lim_sup = np.array([4,5])
sigma= 0.5
max_it=1000
opt='max'
rodadas = 100

resultados_grs = []
resultados_lrs = []
resultados_hill = []

for _ in range(rodadas):
   grs = GlobalRandomSearch(target_func=f, max_it=max_it, opt=opt,lim_inf=lim_inf, lim_sup=lim_sup)
   sol_grs= grs.search()
   resultados_grs.append(round(sol_grs, 3))


   lrs = LocalRandomSearch(target_func=f, max_it=max_it, sigma=sigma, opt=opt,lim_inf=lim_inf, lim_sup=lim_sup)
   sol_lrs= lrs.search() 
   resultados_lrs.append(round(sol_lrs, 3))


   hill = Hill(target_func=f, max_it=max_it, sigma=sigma, max_vizinhos=100, opt=opt,lim_inf=lim_inf, lim_sup=lim_sup)
   sol_hill= hill.search()
   resultados_hill.append(round(sol_hill, 3))
   
moda_grs, freq_grs = calcular_moda(resultados_grs)
moda_lrs, freq_lrs = calcular_moda(resultados_lrs)
moda_hill, freq_hill = calcular_moda(resultados_hill)

print("\n========== TABELA DE MODA ==========")
print("Algoritmo\tModa\tFrequência")
print(f"GRS\t\t{moda_grs}  \t{freq_grs}")
print(f"LRS\t\t{moda_lrs}  \t{freq_lrs}")
print(f"HILL\t\t{moda_hill}   \t{freq_hill}")
print("====================================\n")     