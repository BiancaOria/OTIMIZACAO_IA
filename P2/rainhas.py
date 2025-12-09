from Tempura import SimulatedAnnealingTempura
import numpy as np
import matplotlib.pyplot as plt
import time

inicio = time.time()

solucoes_encontradas =[]
tentativas = 1
total_iteracoes = 0

print("\n>>> Iniciando mineração das 92 soluções...")

while len(solucoes_encontradas) < 92:
    
    
    sa = SimulatedAnnealingTempura(T_inicial=100_000, cooling_rate=0.98)
    solucao, custo, it = sa.search()
    solucao_lista = solucao.tolist()
    total_iteracoes += it
    
    if solucao_lista not in solucoes_encontradas :
        solucoes_encontradas.append(solucao_lista)
        print(f"Solução {len(solucoes_encontradas)} (coluna 1 a 8): {solucao + 1} | Custo: {custo} | Iterações: {it}") 
    
    tentativas +=1

fim = time.time()

sa.plot_convergence()

print(f"\nTempo total de execução: {fim - inicio:.2f} segundos.")
print(f"\nSucesso! Todas as 92 soluções encontradas.")
print(f"Iterações de algoritmos executadas: {tentativas} e iterações totais: {total_iteracoes}\n")