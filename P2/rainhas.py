from Tempura import SimulatedAnnealingTempura
import numpy as np
import matplotlib.pyplot as plt
import time

print(">>> Iniciando busca única com visualização...")
# Item 2: Definição da temperatura inicial (ex: 10.0) e decaimento (0.99)
sa = SimulatedAnnealingTempura(max_it=1000, T_inicial=10.0, cooling_rate=0.99)
solucao, conflitos = sa.search()

print(f"Busca finalizada. Conflitos: {conflitos}")
print(f"Configuração (coluna 1 a 8): {solucao + 1}") # +1 para ajustar índice 0-7 para 1-8
sa.plot_convergence()

# --- Parte 2: Desafio das 92 Soluções (Item final do trabalho) ---
# AVISO: Isso pode demorar. Desabilitamos o plot para ir rápido.
resp = input("\nDeseja rodar a busca pelas 92 soluções? (s/n): ")

if resp.lower() == 's':
    solucoes_encontradas = set()
    tentativas = 0
    inicio_tempo = time.time()
    
    print("\n>>> Iniciando mineração das 92 soluções...")
    
    while len(solucoes_encontradas) < 92:
        # Instancia sem plot (simulando rápido)
        # Usamos uma classe simplificada ou modificamos a flag de plot, 
        # mas aqui vamos apenas criar a lógica rápida sem a classe visual para não travar:
        
        # --- Lógica Rápida ("Headless") ---
        board = np.random.permutation(8)
        cost = sa.calculate_attacks(board)
        temp = 10.0
        
        for _ in range(1000): # max it
            if cost == 0: break
            
            # Perturb
            new_board = np.copy(board)
            i, j = np.random.choice(8, 2, replace=False)
            new_board[i], new_board[j] = new_board[j], new_board[i]
            new_cost = sa.calculate_attacks(new_board)
            
            delta = new_cost - cost
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                board = new_board
                cost = new_cost
            
            temp *= 0.95
            
        tentativas += 1
        
        if cost == 0:
            # Converte array numpy para tupla (hashable) para guardar no Set
            t_solucao = tuple(board)
            if t_solucao not in solucoes_encontradas:
                solucoes_encontradas.add(t_solucao)
                print(f"Encontradas: {len(solucoes_encontradas)}/92 (Tentativas totais: {tentativas})")

    fim_tempo = time.time()
    print(f"\nSucesso! Todas as 92 soluções encontradas.")
    print(f"Tempo total: {fim_tempo - inicio_tempo:.2f} segundos")
    print(f"Iterações de algoritmos executadas: {tentativas}")