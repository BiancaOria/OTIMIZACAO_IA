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
    x1 = x[0]
    x2 = x[1]
    y = (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)
    return y


lim_inf = -5.12
lim_sup = 5.12
sigma= 0.5
max_it=1000
rodadas = 100



resultados_grs = []
resultados_lrs = []
resultados_hill = []


for _ in range(rodadas):
    grs = GlobalRandomSearch(target_func=f, max_it=max_it,lim_inf=lim_inf, lim_sup=lim_sup)
    sol_grs= grs.search()
    resultados_grs.append(round(sol_grs, 3))


    lrs = LocalRandomSearch(target_func=f, max_it=max_it, sigma=sigma,lim_inf=lim_inf, lim_sup=lim_sup)
    sol_lrs= lrs.search() 
    resultados_lrs.append(round(sol_lrs, 3))


    hill = Hill(target_func=f, max_it=max_it, sigma=sigma, max_vizinhos=100,lim_inf=lim_inf, lim_sup=lim_sup)
    sol_hill= hill.search()
    resultados_hill.append(round(sol_hill, 3))
    
moda_grs, freq_grs = calcular_moda(resultados_grs)
moda_lrs, freq_lrs = calcular_moda(resultados_lrs)
moda_hill, freq_hill = calcular_moda(resultados_hill)

print("\n========== TABELA DE MODA ==========")
print("Algoritmo\tModa \tFrequência")
print(f"GRS\t\t{moda_grs}   \t{freq_grs}")
print(f"LRS\t\t{moda_lrs}   \t{freq_lrs}")
print(f"HILL\t\t{moda_hill} \t{freq_hill}")
print("====================================\n")      