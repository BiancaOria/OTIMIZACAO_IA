import random
import numpy as np

class GA:
    def __init__(self, coordenadas, tam_populacao=50, max_geracoes=1000, taxa_mutacao=0.01, elitismo=True):
        self.coordenadas = coordenadas
        self.num_pontos = len(coordenadas)
        self.tam_populacao = tam_populacao
        self.max_geracoes = max_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.elitismo = elitismo
        self.num_elite = 2 if elitismo else 0 
        self.populacao = [self.criar_individuo() for _ in range(tam_populacao)]

    def criar_individuo(self):
        individuo = list(range(self.num_pontos))
        random.shuffle(individuo)
        return individuo

    def calcular_custo(self, individuo):
        distancia = 0
        for i in range(self.num_pontos - 1):
            p1 = self.coordenadas[individuo[i]]
            p2 = self.coordenadas[individuo[i+1]]
            distancia += np.linalg.norm(p1 - p2)
        # Fechar o ciclo
        distancia += np.linalg.norm(self.coordenadas[individuo[-1]] - self.coordenadas[individuo[0]])
        return distancia

    def selecao_torneio(self, k=3):
        competidores = random.sample(self.populacao, k)
        competidores.sort(key=lambda x: self.calcular_custo(x))
        return competidores[0] 

    def recombinacao_dois_pontos_ordenada(self, pai1, pai2):
        tamanho = self.num_pontos
        filho = [None] * tamanho
        
        p1, p2 = sorted(random.sample(range(tamanho), 2))
        
        filho[p1:p2] = pai1[p1:p2]
        
        genes_no_filho = set(filho[p1:p2])
        pos_atual = 0
        for gene in pai2:
            if gene not in genes_no_filho:
                while pos_atual < tamanho and filho[pos_atual] is not None:
                    pos_atual += 1
                if pos_atual < tamanho:
                    filho[pos_atual] = gene
        return filho

    def mutacao_swap(self, individuo):
        """Operador de mutação por troca (1%)."""
        if random.random() < self.taxa_mutacao:
            idx1, idx2 = random.sample(range(self.num_pontos), 2)
            individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
        return individuo

    def executar(self):
        """Executa o GA e retorna o melhor resultado e o histórico."""
        melhor_rota = None
        melhor_custo = float('inf')
        historico = []
        
        sem_melhoria = 0
        limite_estagnacao = 50 
        geracao_parada = self.max_geracoes

        for t in range(self.max_geracoes):
            populacao_avaliada = [(ind, self.calcular_custo(ind)) for ind in self.populacao]
            populacao_avaliada.sort(key=lambda x: x[1]) 
            
            melhor_da_geracao, custo_da_geracao = populacao_avaliada[0]
            
            if custo_da_geracao < melhor_custo:
                melhor_custo = custo_da_geracao
                melhor_rota = list(melhor_da_geracao)
                sem_melhoria = 0
                geracao_parada = t
            else:
                sem_melhoria += 1
            
            historico.append(melhor_custo)
            
            if sem_melhoria >= limite_estagnacao:
                break
            
            nova_populacao = []
            
            if self.elitismo:
                nova_populacao.extend([ind for ind, c in populacao_avaliada[:self.num_elite]])
            
            while len(nova_populacao) < self.tam_populacao:
                pai1 = self.selecao_torneio()
                pai2 = self.selecao_torneio()
                
                filho = self.recombinacao_dois_pontos_ordenada(pai1, pai2)
                filho = self.mutacao_swap(filho)
                nova_populacao.append(filho)
            
            self.populacao = nova_populacao

        return melhor_rota, melhor_custo, historico, geracao_parada