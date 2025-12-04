import numpy as np
import matplotlib.pyplot as plt

class GlobalRandomSearch:
    def __init__(self,max_it,epsilon, lim_inf = -100,lim_sup = 100):
        self.max_it = max_it
        self.lim_inf = lim_inf
        self.lim_sup = lim_sup
        self.epsilon = epsilon
        
        #ótimo inicial:
        self.x_opt = np.random.uniform(lim_inf, lim_sup, size=2)
        self.f_opt = self.f(self.x_opt)
        
        self.path = [np.array([self.x_opt[0], self.x_opt[1], self.f_opt])]
        self.historico = [self.f_opt]
    
        # ===== FIGURA =====
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # superfície pré-computada
        X = np.linspace(lim_inf, lim_sup, 200)
        Y = np.linspace(lim_inf, lim_sup, 200)
        X, Y = np.meshgrid(X, Y)
        Z = X**2 + Y**2

        self.ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.3, edgecolor='none')
        self.ax.contour(X, Y, Z, 20, cmap='gray', offset=0)

        # ponto que se moverá
        self.point_plot = self.ax.scatter(
            self.x_opt[0], self.x_opt[1], self.f_opt,
            color='red', s=80
        )
        P = np.array(self.path)
        self.path_line, = self.ax.plot(
            P[:,0], P[:,1], P[:,2],
            'r-', linewidth=2  
        )
        
        self.ax.set_title("Local Random Search")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("f(x)")
        
    def update_plot(self):
        z = self.f_opt
        self.point_plot._offsets3d = (
            np.array([self.x_opt[0]]),
            np.array([self.x_opt[1]]),
            np.array([self.f_opt])
        )
        
        P = np.array(self.path)
        self.path_line.set_data(P[:,0], P[:,1])
        self.path_line.set_3d_properties(P[:,2])
        
        plt.pause(0.05)
        
    def f(self,x):
        return (x[0]**2+x[1]**2)
    
    def perturb(self):
        
        return np.random.uniform(self.lim_inf, self.lim_sup, size=2)
    
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            self.historico.append(self.f_opt)
            if f_cand < self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                plt.pause(.5)
                self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                self.update_plot()
            
            it+=1
            
        plt.figure()
        plt.title("Convergência do GRS")
        plt.plot(self.historico)
        plt.xlabel("Iterações")
        plt.ylabel("f(x_best)")
        plt.grid()
        plt.show()

        

class LocalRandomSearch:
    def __init__(self,max_it,sigma, lim_inf = -100,lim_sup = 100):
        
        self.sigma = sigma
        self.max_it = max_it
        self.lim_sup =lim_sup
        self.lim_inf =lim_inf
        
        # ---- Verificação exigida pela questão ----
        if not (0 < self.sigma < 1):
            raise ValueError("O valor de sigma deve estar no intervalo (0, 1).")
        # -------------------------------------------
        
        #ótimo inicial:
        self.x_opt = np.random.uniform(lim_inf, lim_sup, size=2)
        self.f_opt = self.f(self.x_opt)
        
        self.path = [np.array([self.x_opt[0], self.x_opt[1], self.f_opt])]
        self.historico = [self.f_opt]
        
        # ===== FIGURA =====
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # superfície pré-computada
        X = np.linspace(lim_inf, lim_sup, 200)
        Y = np.linspace(lim_inf, lim_sup, 200)
        X, Y = np.meshgrid(X, Y)
        Z = X**2 + Y**2

        self.ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.3, edgecolor='none')
        self.ax.contour(X, Y, Z, 20, cmap='gray', offset=0)

        # ponto que se moverá
        self.point_plot = self.ax.scatter(
            self.x_opt[0], self.x_opt[1], self.f_opt,
            color='red', s=80
        )
        P = np.array(self.path)
        self.path_line, = self.ax.plot(
            P[:,0], P[:,1], P[:,2],
            'r-', linewidth=2  
        )

        self.ax.set_title("Local Random Search")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("f(x)")
        
    

        
    def update_plot(self):
        z = self.f_opt
        self.point_plot._offsets3d = (
            np.array([self.x_opt[0]]),
            np.array([self.x_opt[1]]),
            np.array([self.f_opt])
        )
        
        P = np.array(self.path)
        self.path_line.set_data(P[:,0], P[:,1])
        self.path_line.set_3d_properties(P[:,2])
        
        plt.pause(0.05)

    def f(self,x):
        return (x[0]**2+x[1]**2)
    
    def perturb(self):
        
        x_cand = np.random.normal(loc=0,scale=self.sigma, size=2)+ self.x_opt
        x_cand = np.clip(x_cand, self.lim_inf, self.lim_sup)

        return x_cand
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            
            if f_cand < self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
            
                plt.pause(.05)
                self.path.append(np.array([self.x_opt[0], self.x_opt[1], self.f_opt]))
                self.update_plot()
            it+=1
            
            self.historico.append(self.f_opt)
            
        
        plt.figure()
        plt.title("Convergência do LRS")
        plt.plot(self.historico)
        plt.xlabel("Iterações")
        plt.ylabel("f(x_best)")
        plt.grid()
        plt.show()
    
        