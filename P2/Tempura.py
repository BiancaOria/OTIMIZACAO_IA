import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects #

class SimulatedAnnealingTempura:
    def __init__(self, max_it, T_inicial, cooling_rate):
        self.max_it = max_it
        self.T = T_inicial
        self.cooling_rate = cooling_rate
        self.N = 8 
        
        # Estado inicial: permutação aleatória (garante 1 rainha por linha/coluna)
        self.board = np.random.permutation(self.N)
        
        # Custo inicial (número de ataques)
        self.cost = self.calculate_attacks(self.board)
        self.historico = [self.cost]
        self.best_solution = np.copy(self.board)
        self.min_cost = self.cost

        # Configuração da Figura
        if plt.fignum_exists(1): plt.close(1)
        self.fig = plt.figure(1, figsize=(6,6))
        self.ax = self.fig.subplots()
        self.update_plot()

    # --- Item 1: Projeto da Função f(x) (Custo) ---
    def calculate_attacks(self, board):
        """
        Calcula o número de pares de rainhas se atacando diagonalmente.
        (Ataques lineares são impossíveis devido à representação por permutação)
        """
        h = 0 # número de ataques
        n = len(board)
        for i in range(n):
            for j in range(i + 1, n):
                # Verifica se estão na mesma diagonal
                # Diferença absoluta das colunas == Diferença absoluta das linhas
                if abs(board[i] - board[j]) == abs(i - j):
                    h += 1
        return h

    # --- Item 4: Função de Perturbação Controlada ---
    def perturb(self, current_board):
        """
        Troca duas rainhas de coluna aleatoriamente.
        Mantém a restrição de 'uma por linha, uma por coluna'.
        """
        new_board = np.copy(current_board)
        i, j = np.random.choice(self.N, 2, replace=False)
        new_board[i], new_board[j] = new_board[j], new_board[i]
        return new_board

    def update_plot(self):
        self.ax.clear()
        # Desenha o tabuleiro
        board_img = np.zeros((self.N, self.N))
        board_img[1::2, ::2] = 1
        board_img[::2, 1::2] = 1
        self.ax.imshow(board_img, cmap='gray_r', interpolation='nearest')
        
        # Desenha as rainhas
        for col, row in enumerate(self.board):
            # Invertemos o row para o plot ficar igual ao sistema de coordenadas cartesiano visual
            # ou mantemos padrão matriz (0 no topo). Vamos usar padrão matriz.
            self.ax.text(col, row, '♛', fontsize=25, ha='center', va='center', color='gold', path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])
            
        self.ax.set_title(f"Ataques (h): {self.cost} | Temp: {self.T:.2f}")
        self.ax.set_xticks(range(8))
        self.ax.set_yticks(range(8))
        self.ax.set_xticklabels(range(1,9)) # 1 a 8 visualmente
        self.ax.set_yticklabels(range(1,9)) # 1 a 8 visualmente
        plt.draw()
        plt.pause(0.01)

    def search(self):
        it = 0
        # --- Item 5: Critério de Parada (Max iterações ou Custo Ótimo = 0) ---
        while it < self.max_it and self.cost > 0:
            
            # Gera vizinho
            new_board = self.perturb(self.board)
            new_cost = self.calculate_attacks(new_board)
            
            delta = new_cost - self.cost

            # Critério de Aceitação (Metropolis)
            # Se melhorou (delta < 0), aceita sempre.
            # Se piorou, aceita com probabilidade e^(-delta/T)
            if delta < 0 or np.random.uniform(0, 1) < np.exp(-delta / self.T):
                self.board = new_board
                self.cost = new_cost
                
                # Salva o melhor encontrado até agora
                if self.cost < self.min_cost:
                    self.min_cost = self.cost
                    self.best_solution = np.copy(self.board)

                self.update_plot()

            self.historico.append(self.cost)
            
            # --- Item 3: Decaimento da Temperatura ---
            self.T *= self.cooling_rate 
            it += 1
        
        return self.best_solution, self.min_cost

    def plot_convergence(self):
        if plt.fignum_exists(2): plt.close(2)
        plt.figure(2)
        plt.plot(self.historico)
        plt.title("Convergência (Minimização de Conflitos)")
        plt.xlabel("Iterações")
        plt.ylabel("Conflitos (h)")
        plt.grid()
        plt.show()