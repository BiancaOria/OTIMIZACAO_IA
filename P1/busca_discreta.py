import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class GlobalRandomSearch:
    def __init__(self, target_func, max_it, lim_inf=-100, lim_sup=100, opt='min'):
        self.target_func = target_func  
        self.max_it = max_it
        self.opt = opt
        
        # --- CORREÇÃO DE DIMENSIONALIDADE ---
        self.lim_inf = np.array(lim_inf)
        self.lim_sup = np.array(lim_sup)
        
        if self.lim_inf.ndim == 0: self.lim_inf = np.full(2, float(self.lim_inf))
        if self.lim_sup.ndim == 0: self.lim_sup = np.full(2, float(self.lim_sup))
        # ------------------------------------

        # size=2 removido pois lim_inf/sup já definem o shape (2,)
        self.x_opt = np.random.uniform(self.lim_inf, self.lim_sup)
        self.f_opt = self.target_func(self.x_opt)
        
        self.path = [np.array([self.x_opt[0], self.x_opt[1], self.f_opt])]
        self.historico = [self.f_opt]
    
        # ===== FIGURA =====
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        val_min = np.min(self.lim_inf)
        val_max = np.max(self.lim_sup)
        X = np.linspace(val_min, val_max, 200)
        Y = np.linspace(val_min, val_max, 200)
        X, Y = np.meshgrid(X, Y)
        
        Z = self.target_func([X, Y])

        self.ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.3, edgecolor='none')
        self.ax.contour(X, Y, Z, 20, cmap='gray', offset=0)

        self.point_plot = self.ax.scatter(
            self.x_opt[0], self.x_opt[1], self.f_opt, color='red', s=80
        )
        P = np.array(self.path)
        self.path_line, = self.ax.plot(P[:,0], P[:,1], P[:,2], 'r-', linewidth=2)
        
        self.ax.set_title(f"Global Random Search ({self.opt})")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("f(x)")
        
    def update_plot(self):
        P = np.array(self.path)
        self.point_plot._offsets3d = (
            np.array([self.x_opt[0]]), np.array([self.x_opt[1]]), np.array([self.f_opt])
        )
        self.path_line.set_data(P[:,0], P[:,1])
        self.path_line.set_3d_properties(P[:,2])
        # plt.pause(0.05)
        
    def perturb(self):
        # Usa os limites vetoriais
        x_cand = np.random.uniform(self.lim_inf, self.lim_sup)
        x_cand = np.clip(x_cand, self.lim_inf, self.lim_sup)
        return x_cand
    
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.target_func(x_cand)
            self.historico.append(self.f_opt)
            
            if self.opt == 'min':
                if f_cand < self.f_opt:
                    self.x_opt = x_cand
                    self.f_opt = f_cand
                    # plt.pause(.05) 
                    self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                    self.update_plot()
            else:
                if f_cand > self.f_opt:
                    self.x_opt = x_cand
                    self.f_opt = f_cand
                    # plt.pause(.05) 
                    self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                    self.update_plot()
                
            it += 1
            
        plt.figure()
        plt.title(f"Convergência do GRS ({self.opt})")
        plt.plot(self.historico); plt.xlabel("Iterações"); plt.ylabel("f(x_best)"); plt.grid(); 
        # plt.show()
        plt.close(self.fig)  # FECHA FIGURA 3D
        plt.close()          # FECHA FIGURA DO GRÁFICO
        return self.f_opt


class LocalRandomSearch:
    def __init__(self, target_func, max_it, sigma, lim_inf=-100, lim_sup=100, opt='min'):
        self.target_func = target_func  
        self.sigma = sigma
        self.max_it = max_it
        self.opt = opt
        
        # --- CORREÇÃO DE DIMENSIONALIDADE ---
        self.lim_inf = np.array(lim_inf)
        self.lim_sup = np.array(lim_sup)
        
        if self.lim_inf.ndim == 0: self.lim_inf = np.full(2, float(self.lim_inf))
        if self.lim_sup.ndim == 0: self.lim_sup = np.full(2, float(self.lim_sup))
        # ------------------------------------
        
        if not (0 < self.sigma < 1):
            raise ValueError("O valor de sigma deve estar no intervalo (0, 1).")
        
        self.x_opt = np.random.uniform(self.lim_inf, self.lim_sup)
        self.f_opt = self.target_func(self.x_opt)
        
        self.path = [np.array([self.x_opt[0], self.x_opt[1], self.f_opt])]
        self.historico = [self.f_opt]
        
        # ===== FIGURA =====
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        val_min = np.min(self.lim_inf)
        val_max = np.max(self.lim_sup)
        X = np.linspace(val_min, val_max, 200)
        Y = np.linspace(val_min, val_max, 200)
        X, Y = np.meshgrid(X, Y)
        
        Z = self.target_func([X, Y])

        self.ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.3, edgecolor='none')
        self.ax.contour(X, Y, Z, 20, cmap='gray', offset=0)

        self.point_plot = self.ax.scatter(
            self.x_opt[0], self.x_opt[1], self.f_opt, color='red', s=80
        )
        P = np.array(self.path)
        self.path_line, = self.ax.plot(P[:,0], P[:,1], P[:,2], 'r-', linewidth=2)

        self.ax.set_title(f"Local Random Search ({self.opt})")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("f(x)")
        
    def update_plot(self):
        P = np.array(self.path)
        self.point_plot._offsets3d = (
            np.array([self.x_opt[0]]), np.array([self.x_opt[1]]), np.array([self.f_opt])
        )
        self.path_line.set_data(P[:,0], P[:,1])
        self.path_line.set_3d_properties(P[:,2])
        # plt.pause(0.05)
    
    def perturb(self):
        # Usa o shape de x_opt para definir o tamanho do ruído
        noise = np.random.normal(loc=0, scale=self.sigma, size=self.x_opt.shape)
        x_cand = self.x_opt + noise
        x_cand = np.clip(x_cand, self.lim_inf, self.lim_sup)
        return x_cand

    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.target_func(x_cand)

            if self.opt == 'min':
                if f_cand < self.f_opt:
                    self.x_opt = x_cand
                    self.f_opt = f_cand
                    # plt.pause(.05)
                    self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                    self.update_plot()
            else:
                if f_cand > self.f_opt:
                    self.x_opt = x_cand
                    self.f_opt = f_cand
                    # plt.pause(.05)
                    self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                    self.update_plot()
            
            it += 1
            self.historico.append(self.f_opt)
            
        plt.figure()
        plt.title(f"Convergência do LRS ({self.opt})")
        plt.plot(self.historico); plt.xlabel("Iterações"); plt.ylabel("f(x_best)"); plt.grid(); 
        plt.show()
        plt.close(self.fig)  # FECHA FIGURA 3D
        plt.close()          # FECHA FIGURA DO GRÁFICO
        return self.f_opt