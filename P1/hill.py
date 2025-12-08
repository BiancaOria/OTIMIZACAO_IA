import numpy as np
import matplotlib.pyplot as plt

class Hill:
    def __init__(self, target_func, max_it, sigma, max_vizinhos, opt='min', lim_inf=-100, lim_sup=100):
        
        self.target_func = target_func 
        self.sigma = sigma
        self.max_it = max_it
        self.max_vizinhos = max_vizinhos
        self.opt = opt
        
        # --- CORREÇÃO DE DIMENSIONALIDADE (CRÍTICO) ---
        # Garante que lim_inf e lim_sup sejam vetores de tamanho 2
        self.lim_inf = np.array(lim_inf)
        self.lim_sup = np.array(lim_sup)
        
        # Se veio como escalar (ex: -8), transforma em vetor [-8, -8]
        if self.lim_inf.ndim == 0:
            self.lim_inf = np.full(2, float(self.lim_inf))
        if self.lim_sup.ndim == 0:
            self.lim_sup = np.full(2, float(self.lim_sup))
        # -----------------------------------------------

        # if not (0 < self.sigma < 1):
        #      print("Aviso: Sigma geralmente deve ser pequeno, mas depende da escala.") 
        
        
        self.x_opt = self.lim_inf.copy() 
        
        self.f_opt = self.target_func(self.x_opt)
        
        self.path = [np.array([self.x_opt[0], self.x_opt[1], self.f_opt])]
        self.historico = [self.f_opt]
        
        # ===== FIGURA =====
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Para o gráfico, precisamos dos valores escalares min/max
        val_min = np.min(self.lim_inf)
        val_max = np.max(self.lim_sup)

        X = np.linspace(val_min, val_max, 200)
        Y = np.linspace(val_min, val_max, 200)
        X, Y = np.meshgrid(X, Y)
        
        Z = self.target_func([X, Y])

        self.ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.3, edgecolor='none')
        self.ax.contour(X, Y, Z, 20, cmap='gray', offset=0)

        self.point_plot = self.ax.scatter(
            self.x_opt[0], self.x_opt[1], self.f_opt,
            color='red', s=80
        )
        P = np.array(self.path)
        self.path_line, = self.ax.plot(
            P[:,0], P[:,1], P[:,2],
            'r-', linewidth=2  
        )

        self.ax.set_title(f"Hill Climbing ({self.opt})")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("f(x)")

    def update_plot(self):
        P = np.array(self.path)
        self.point_plot._offsets3d = (
            np.array([self.x_opt[0]]),
            np.array([self.x_opt[1]]),
            np.array([self.f_opt])
        )
        self.path_line.set_data(P[:,0], P[:,1])
        self.path_line.set_3d_properties(P[:,2])
        # plt.pause(0.01) 

    def perturb(self):
        low_bound = self.x_opt - self.sigma
        high_bound = self.x_opt + self.sigma
        
        # size=2 não é estritamente necessário se x_opt for vetor, mas mal não faz
        x_cand = np.random.uniform(low=low_bound, high=high_bound)
        x_cand = np.clip(x_cand, self.lim_inf, self.lim_sup)
        return x_cand

    def search(self):
        it = 0
        melhoria = True
        
        while it < self.max_it and melhoria:
            
            melhoria = False
            
            for j in range(self.max_vizinhos):
                x_cand = self.perturb()
                f_cand = self.target_func(x_cand)

                # Lógica Min/Max
                if self.opt == 'min':
                    if f_cand < self.f_opt:
                        self.x_opt = x_cand
                        self.f_opt = f_cand
                        self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                        melhoria = True
                        self.update_plot()
                        break
                else: # max
                    if f_cand > self.f_opt:
                        self.x_opt = x_cand
                        self.f_opt = f_cand
                        self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                        melhoria = True
                        self.update_plot()
                        break
            
            self.historico.append(self.f_opt)
            it += 1
        
            
        plt.figure()
        plt.title(f"Convergência do Hill ({self.opt})")
        plt.plot(self.historico)
        plt.xlabel("Iterações (Moves)")
        plt.ylabel("f(x_best)")
        plt.grid()
        # plt.show()
        plt.close(self.fig)  # FECHA FIGURA 3D
        plt.close()          # FECHA FIGURA DO GRÁFICO
        return self.f_opt