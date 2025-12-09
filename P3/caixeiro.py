import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
from GA_Caixeiro import GA  


NUM_PONTOS_MAX = 40 
TAM_POPULACAO = 100 #
MAX_GERACOES = 1000 #


def carregar_dados():
    arquivo = 'CaixeiroGruposGA.csv'
    try:
        dados = np.loadtxt(arquivo, delimiter=',')
        
       
        if len(dados) > NUM_PONTOS_MAX:
            dados = dados[:NUM_PONTOS_MAX]
            
        return dados
        
    except OSError:
        print(f"ERRO: '{arquivo}' não encontrado. Gerando dados de teste (Clusters)...")
        

def plot_convergencia(historico, titulo="Convergência do GA"):
    plt.figure(figsize=(10, 6))
    plt.plot(historico, linewidth=2, color='#0984e3', label='Melhor Custo')
    plt.title(titulo)
    plt.xlabel("Gerações")
    plt.ylabel("Custo (Distância)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def plot_rota_3d(coordenadas, rota, custo, titulo_extra=""):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = [coordenadas[i][0] for i in rota] + [coordenadas[rota[0]][0]]
    ys = [coordenadas[i][1] for i in rota] + [coordenadas[rota[0]][1]]
    zs = [coordenadas[i][2] for i in rota] + [coordenadas[rota[0]][2]]

    ax.scatter(coordenadas[:,0], coordenadas[:,1], coordenadas[:,2], c='#FE6EC3', s=30, label='Cidades')
    
    ax.plot(xs, ys, zs, c='#74F4FE', linewidth=1.5, label='Rota')
    
    ax.set_title(f"Melhor Rota {titulo_extra} (Custo: {custo:.2f})")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def realizar_comparacao_elitismo(coordenadas, n_rodadas=10):
    print(f"\n=== Iniciando Análise Comparativa - {n_rodadas} Rodadas ===")
    
    custos_com_elitismo = []
    geracoes_com_elitismo = []
    custos_sem_elitismo = []

    print(f"\n Executando COM Elitismo (Ne=2)...")
    for i in range(n_rodadas):
        ga = GA(coordenadas, TAM_POPULACAO, MAX_GERACOES, taxa_mutacao=0.01, elitismo=True)
        _, custo, _, geracao = ga.executar()
        custos_com_elitismo.append(custo)
        geracoes_com_elitismo.append(geracao)
        print(f"   Rodada {i+1}: Convergiu na Geração {geracao} | Custo: {custo:.2f}")

    print(f"\n Executando SEM Elitismo...")
    for i in range(n_rodadas):
        ga = GA(coordenadas, TAM_POPULACAO, MAX_GERACOES, taxa_mutacao=0.01, elitismo=False)
        _, custo, _, _ = ga.executar()
        custos_sem_elitismo.append(custo)
        print(f"   Rodada {i+1}: Custo: {custo:.2f}")

    try:
        moda_geracoes = statistics.mode(geracoes_com_elitismo)
    except statistics.StatisticsError:
        moda_geracoes = "Indefinida (Múltiplas modas)"
        
    media_com = statistics.mean(custos_com_elitismo)
    media_sem = statistics.mean(custos_sem_elitismo)
    
    print("\n" + "="*50)
    print("RELATÓRIO FINAL")
    print("="*50)
    print(f"1. Moda de gerações (Solução Aceitável): {moda_geracoes}")
    print(f"2. Custo Médio COM Elitismo: {media_com:.2f}")
    print(f"3. Custo Médio SEM Elitismo: {media_sem:.2f}")
    
    if media_com < media_sem:
        ganho = ((media_sem - media_com) / media_sem) * 100
        print(f"\nCONCLUSÃO: O operador de Elitismo MELHOROU o resultado em {ganho:.1f}%.")
    else:
        print("\nCONCLUSÃO: O operador de Elitismo não apresentou ganho significativo.")
    print("="*50)


pontos_coordenadas = carregar_dados()

realizar_comparacao_elitismo(pontos_coordenadas, n_rodadas=10)

print("\n>>> Gerando Gráficos da Melhor Solução (COM ELITISMO)...")
ga_com = GA(
    coordenadas=pontos_coordenadas, 
    tam_populacao=TAM_POPULACAO, 
    max_geracoes=MAX_GERACOES, 
    taxa_mutacao=0.01, 
    elitismo=True
)
rota_com, custo_com, historico_com, _ = ga_com.executar()
print(f"Melhor Custo (Com Elitismo): {custo_com:.2f}")

plot_convergencia(historico_com, titulo="Convergência do GA (Com Elitismo)")
plot_rota_3d(pontos_coordenadas, rota_com, custo_com, titulo_extra="(Com Elitismo)")

print("\n>>> Gerando Gráficos da Solução (SEM ELITISMO) para comparação...")
ga_sem = GA(
    coordenadas=pontos_coordenadas, 
    tam_populacao=TAM_POPULACAO, 
    max_geracoes=MAX_GERACOES, 
    taxa_mutacao=0.01, 
    elitismo=False
)
rota_sem, custo_sem, historico_sem, _ = ga_sem.executar()
print(f"Melhor Custo (Sem Elitismo): {custo_sem:.2f}")

plot_convergencia(historico_sem, titulo="Convergência do GA (Sem Elitismo)")
plot_rota_3d(pontos_coordenadas, rota_sem, custo_sem, titulo_extra="(Sem Elitismo)")