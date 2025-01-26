import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
import itertools
import psutil  # Biblioteca para medir uso de CPU e memória
import os

class Grafo:
    def __init__(self):
        # Inicializa o dicionário para armazenar o grafo
        self.grafo_ch = {}
        self.grafo_tsp = {}

    def add_vertice(self, vertice, grafo):
        # Adiciona um vértice ao grafo
        if vertice not in grafo:
            grafo[vertice] = {}

    def add_aresta(self, vertice1, vertice2, peso, grafo):
        # Adiciona uma aresta entre os vértices com um peso (distância)
        if vertice1 in grafo and vertice2 in grafo:
            if vertice2 != vertice1:
                grafo[vertice1][vertice2] = peso
                grafo[vertice2][vertice1] = peso

    def exibir_grafo(self, grafo):
        # Exibe o grafo com os pesos das arestas
        for vertice in grafo:
            print(f"{vertice}: {grafo[vertice]}")
    
    def neighbor(self, vertice, grafo): 
        """Retorna todos os vizinhos do vértice dado.""" 
        if vertice in grafo: 
            return list(grafo[vertice].keys()) 
        else: 
            return []

    def remove_aresta(self, vertice1, vertice2, grafo):
        # Remove uma aresta entre dois vértices
        if vertice1 in grafo and vertice2 in grafo:
            if vertice2 in grafo[vertice1]:
                del grafo[vertice1][vertice2]
                del grafo[vertice2][vertice1]
                print(f"Aresta entre {vertice1} e {vertice2} removida")
            else:
                print(f"Aresta entre {vertice1} e {vertice2} não encontrada")
        else:
            print(f"Vértices não encontrados")

    def contar_arestas_por_vertice(self, grafo):
        contagem_arestas = {}
        for vertice in grafo:
            contagem_arestas[vertice] = len(grafo[vertice])
        return contagem_arestas
    
    def dividir_lista(self, lista, m):
        """Divide uma lista em sublistas de tamanho m."""
        return [lista[i:i + m] for i in range(0, len(lista), m)]

    def gerar_automaticamente_arestas(self, num_vertices, grafo):
        """Gera um grafo convexo com um caminho hamiltoniano."""
        # Programação Dinâmica: Cache dos vértices criados
        vertices = [f"v{i}" for i in range(1, num_vertices + 1)]
        
        # Adiciona vértices ao grafo
        for vertice in vertices:
            self.add_vertice(vertice, grafo)

        # Cria um caminho hamiltoniano
        for i in range(num_vertices - 1):
            peso = random.randint(40, 160)
            self.add_aresta(vertices[i], vertices[i + 1], peso, grafo)
        # Conecta o último vértice ao primeiro para formar um ciclo
        peso = random.randint(40, 160)
        self.add_aresta(vertices[-1], vertices[0], peso, grafo)

        # Adiciona arestas adicionais para garantir a conectividade
        tentativas_max = 100
        for _ in range(num_vertices):
            tentativas = 0
            while tentativas < tentativas_max:
                vertice1 = random.choice(vertices)
                vertice2 = random.choice(vertices)
                if vertice1 != vertice2 and vertice2 not in grafo[vertice1]:
                    peso = random.randint(40, 160)
                    self.add_aresta(vertice1, vertice2, peso, grafo)
                    break
                tentativas += 1
            else:
                print(f"Não foi possível encontrar uma conexão adicional para {vertice1}")

        self.exibir_grafo(grafo)

    def verificar_conectividade(self, grafo):
        """Verifica se o grafo é conexo."""
        visitados = set()
        def dfs(v):
            if v not in visitados:
                visitados.add(v)
                for vizinho in grafo[v]:
                    dfs(vizinho)
        # Inicie a busca a partir do primeiro vértice
        dfs(next(iter(grafo)))
        return len(visitados) == len(grafo)

    def conectar_componente(self, grafo):
        """Conecta componentes desconexos do grafo."""
        vertices = list(grafo.keys())
        visitados = set()

        def dfs(v):
            if v not in visitados:
                visitados.add(v)
                for vizinho in grafo[v]:
                    dfs(vizinho)

        # Inicie a busca a partir do primeiro vértice
        for vertice in vertices:
            if vertice not in visitados:
                # Conectar o vértice não visitado ao grafo
                for u in visitados:
                    peso = random.randint(50, 100)
                    self.add_aresta(u, vertice, peso, grafo)
                    break
            dfs(vertice)

    def atualizar_grafo(self, vertices, heigth_x, grafo):
        # Divide os vértices em camadas de tamanho `heigth_x`
        camadas = self.dividir_lista(vertices, heigth_x)
        
        # Conectar vértices dentro de cada camada
        if heigth_x > 1:
            for i in range(len(camadas[1]) - 1):
                peso = random.randint(40, 160)
                self.atualizar_peso_aresta(camadas[1][i], camadas[1][i + 1], peso, grafo)
            
    def atualizar_peso_aresta(self, vertice1, vertice2, novo_peso, grafo):
        """ Atualiza o peso de uma aresta e ajusta os caminhos afetados dinamicamente """
        if vertice1 in grafo and vertice2 in grafo:
            if vertice2 in grafo[vertice1]:
                grafo[vertice1][vertice2] = novo_peso
                grafo[vertice2][vertice1] = novo_peso
                print(f"Peso da aresta ({vertice1}, {vertice2}) atualizado para {novo_peso}")
            else:
                print(f"Aresta ({vertice1}, {vertice2}) não existe")
        else:
            print(f"Vértices {vertice1} ou {vertice2} não encontrados")
        
    def gerar_matriz_adjacencia(self, grafo): 
        # Obter a lista de vértices 
        vertices = list(grafo.keys()) 
        n = len(vertices) 
        # Criar um dicionário para mapear os vértices para índices 
        indice = {vertices[i]: i for i in range(n)} 
        
        # Inicializar a matriz de adjacências com infinito (ausência de aresta) 
        matriz_adjacencia = np.full((n, n), float('inf')) 
        
        # Preencher a matriz de adjacências com os pesos das arestas 
        for v1 in grafo: 
            for v2 in grafo[v1]: 
                i, j = indice[v1], indice[v2] 
                matriz_adjacencia[i][j] = grafo[v1][v2] 
                
        # Substituir os elementos diagonais (vértice para ele mesmo) por zero 
        np.fill_diagonal(matriz_adjacencia, 0) 
        
        return matriz_adjacencia
    def is_hamiltonian_path(self, caminho):
        n = len(self.grafo_ch)
        if len(caminho) == n:
            return True
        return False
    
    def buscar_caminho_hamiltoniano(self, caminho):
        if self.is_hamiltonian_path(caminho):
            return caminho

        ultimo_vertice = caminho[-1]
        for vizinho in self.grafo_ch[ultimo_vertice]:
            if vizinho not in caminho:
                novo_caminho = self.buscar_caminho_hamiltoniano(caminho + [vizinho])
                if novo_caminho:
                    return novo_caminho
        return None

    def encontrar_caminho_hamiltoniano(self):
        for vertice in self.grafo_ch:
            caminho = self.buscar_caminho_hamiltoniano([vertice])
            if caminho:
                return caminho
        return None

    def transformar_em_tsp(self):
        caminho_hamiltoniano = self.encontrar_caminho_hamiltoniano()
        if not caminho_hamiltoniano:
            return None  # Nenhum caminho hamiltoniano encontrado

        grafo = self.grafo_tsp
        vertices = list(self.grafo_ch.keys())
        valor_maximo = 500  # Valor arbitrariamente alto, mas finito

        for v1 in vertices:
            grafo[v1] = {}
            for v2 in vertices:
                if v1 == v2:
                    continue
                if v2 in self.grafo_ch[v1]:
                    peso = self.grafo_ch[v1][v2]
                    self.add_aresta(v1, v2, peso, grafo)
                elif v1 in caminho_hamiltoniano and v2 in caminho_hamiltoniano:
                    self.add_aresta(v1, v2, valor_maximo, grafo)
        # self.visualizar_grafo(grafo)

    def visualizar_grafo(self, grafo):
        # # Ativar o modo interativo
        G = nx.Graph()

        # Adicionando vértices e arestas com peso
        for vertice, vizinhos in grafo.items():
            for vizinho, peso in vizinhos.items():
                G.add_edge(vertice, vizinho, weight=peso)

        num_nos = len(G.nodes)  # Número total de nós

        # Definir o layout dinamicamente
        if num_nos <= 10:
            pos = nx.shell_layout(G)
        elif num_nos <= 50:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=3 / num_nos)  # Ajuste dinâmico

        # Ajuste do tamanho da figura baseado no número de nós
        plt.figure(figsize=(max(12, num_nos // 5), max(8, num_nos // 6)))

        # Ajuste do tamanho dos nós e fontes conforme o tamanho do grafo
        node_size = max(800, 5000 // num_nos)
        font_size = max(6, 16 - num_nos // 10)
        
        # Desenhando o grafo
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=node_size, font_size=font_size, font_weight='bold', 
                edge_color='gray')

        # Destacar as arestas com peso 500
        edges = G.edges(data=True)
        edge_colors = ['red' if edge[2]['weight'] == 500 else 'gray' for edge in edges]
        edge_styles = ['dotted' if edge[2]['weight'] == 500 else 'solid' for edge in edges]

        # Desenhar nós e arestas
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size)
        for style in set(edge_styles):
            edges_style = [edge[:2] for edge, es in zip(edges, edge_styles) if es == style]
            edge_c = [color for color, es in zip(edge_colors, edge_styles) if es == style]
            nx.draw_networkx_edges(G, pos, edgelist=edges_style, edge_color=edge_c, style=style, width=2)

        # Adicionando rótulos nas arestas
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=max(6, font_size - 2))

        plt.title("Visualização Dinâmica do Grafo", fontsize=16)
        # plt.show()

    def caminho_hamiltoniano(self, inicio, grafo):
        # Criar mapeamento entre vértices e índices numéricos
        vertices = list(grafo.keys())  # Lista de vértices (podem ser strings ou números)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}  # v1 -> 0, v2 -> 1, ...

        if inicio not in vertex_to_index:
            raise ValueError(f"Vértice inicial '{inicio}' não encontrado no grafo.")

        vertices.remove(inicio)
        fim = vertices[-1]  # Último vértice da lista será o vértice final

        melhor_caminho = None
        menor_custo = float('inf')
        melhor_caminho_str = ""

        for perm in itertools.permutations(vertices[:-1]):  # Exceto o último vértice
            caminho = [inicio] + list(perm) + [fim]

            valido = True
            for i in range(len(caminho) - 1):
                if caminho[i+1] not in grafo[caminho[i]]:
                    valido = False
                    break

            if valido:
                # Calcular o custo do caminho e montar a string para exibição detalhada
                caminho_str = ""
                custo_total = 0
                for i in range(len(caminho) - 1):
                    v1, v2 = caminho[i], caminho[i + 1]
                    peso = grafo[v1][v2]
                    custo_total += peso
                    caminho_str += f"{v1}: {peso} -> "
                caminho_str += caminho[-1]

                if custo_total < menor_custo:
                    menor_custo = custo_total
                    melhor_caminho = caminho
                    melhor_caminho_str = caminho_str

        if melhor_caminho is not None:
            # Montar string para exibição do caminho e pesos
            caminho_str = ""
            for i in range(len(melhor_caminho) - 1):
                v_atual = melhor_caminho[i]
                v_proximo = melhor_caminho[i + 1]
                peso = grafo[v_atual][v_proximo]
                caminho_str += f"{v_atual}: {peso} -> "
            caminho_str += melhor_caminho[-1]

            # Incluir o custo do último vértice para o primeiro
            peso_retorno = grafo[caminho[-1]].get(caminho[0], float('inf'))
            menor_custo += peso_retorno
            caminho_str += f" : {peso_retorno} -> {caminho[0]}"
            melhor_caminho.append(caminho[0])
            # Exibir o resultado 
            print(f"Caminho CH: {caminho_str}")

            # Retornar caminho mínimo encontrado e métricas
            return melhor_caminho, menor_custo
        else:
            print("Nenhum caminho hamiltoniano válido foi encontrado.")
            return None, float('inf')

    def caminho_hamiltoniano_dp(self, inicio, grafo):
        vertices = list(grafo.keys())
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        index_to_vertex = {i: v for v, i in vertex_to_index.items()}
        
        n = len(vertices)
        if inicio not in vertex_to_index:
            raise ValueError(f"Vértice inicial '{inicio}' não encontrado no grafo.")
        
        inicio_idx = vertex_to_index[inicio]
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        dp[1 << inicio_idx][inicio_idx] = 0
        
        for mask in range(1 << n):
            for u in range(n):
                if (mask & (1 << u)) == 0:
                    continue
                
                for v in range(n):
                    if (mask & (1 << v)) or grafo[vertices[u]].get(vertices[v], 0) == 0:
                        continue
                    
                    novo_mask = mask | (1 << v)
                    novo_custo = dp[mask][u] + grafo[vertices[u]][vertices[v]]
                    
                    if novo_custo < dp[novo_mask][v]:
                        dp[novo_mask][v] = novo_custo
                        parent[novo_mask][v] = u
        
        ultimo_no = -1
        menor_custo = float('inf')
        estado_final = (1 << n) - 1
        
        for v in range(n):
            if dp[estado_final][v] + grafo[vertices[v]].get(inicio, float('inf')) < menor_custo:
                menor_custo = dp[estado_final][v] + grafo[vertices[v]].get(inicio, float('inf'))
                ultimo_no = v
        
        if ultimo_no == -1:
            raise ValueError("Não foi possível encontrar um caminho válido.")
        
        melhor_caminho = []
        mask = estado_final
        while ultimo_no != -1:
            melhor_caminho.append(index_to_vertex[ultimo_no])
            proximo_no = parent[mask][ultimo_no]
            mask &= ~(1 << ultimo_no)
            ultimo_no = proximo_no
        
        melhor_caminho.reverse()
        melhor_caminho.append(inicio)  # Adicionar o início ao final do caminho
        
        caminho_str = ""
        for i in range(len(melhor_caminho) - 1):
            v_atual = melhor_caminho[i]
            v_proximo = melhor_caminho[i + 1]
            peso = grafo[v_atual][v_proximo]
            caminho_str += f"{v_atual}: {peso} -> "
        caminho_str += f"{melhor_caminho[-1]}"
        
        print(f"Caminho CHD: {caminho_str}")
        return melhor_caminho, menor_custo


    def calcular_custo_caminho(self, caminho, grafo):
        custo = 0
        for i in range(len(caminho) - 1):
            v1, v2 = caminho[i], caminho[i + 1]
            if v2 in grafo[v1]:
                custo += grafo[v1][v2]
            else:
                return float('inf')
        return custo

    def calcular_tsp_forcando_ch(self, origem, grafo, caminho_ch):
        permutacoes = itertools.permutations(caminho_ch[1:-1])
        melhor_caminho = None
        menor_custo = float('inf')
        
        for perm in permutacoes:
            caminho_tsp = [origem] + list(perm) + [origem]
            custo = self.calcular_custo_caminho(caminho_tsp, grafo)
            
            if custo < menor_custo:
                menor_custo = custo
                melhor_caminho = caminho_tsp
        
        if melhor_caminho is None:
            print("Erro: Nenhum caminho encontrado!")
            return None, float('inf')
        
        caminho_str = ""
        for i in range(len(melhor_caminho) - 1):
            v_atual = melhor_caminho[i]
            v_proximo = melhor_caminho[i + 1]
            peso = grafo[v_atual][v_proximo]
            caminho_str += f"{v_atual}: {peso} -> "
        caminho_str += melhor_caminho[-1]

        print(f"Caminho TSP: {caminho_str}")
        return melhor_caminho, menor_custo

    def held_karp(self, grafo, caminho_ch):
        """
        Implementa o algoritmo de Held-Karp para encontrar o caminho mínimo no grafo H.
        Compara o custo obtido com o caminho Hamiltoniano do grafo G.
        
        :param grafo: Representação do grafo como um dicionário de adjacências.
        :param caminho_ch: Lista com o caminho Hamiltoniano encontrado pelo CH no grafo G.
        :return: O caminho encontrado pelo TSP e seu custo.
        """
        vertices = list(grafo.keys())
        n = len(vertices)

        # Tabela DP para armazenar o custo mínimo para visitar todos os vértices na máscara terminando em i
        DP = [[float('inf')] * n for _ in range(1 << n)]
        DP[1][0] = 0  # Começa do vértice 0
        predecessor = [[-1] * n for _ in range(1 << n)]

        # Preenchimento da tabela DP
        for mask in range(1 << n):
            for u in range(n):
                if mask & (1 << u):
                    for v in range(n):
                        if mask & (1 << v) == 0:  # Se v ainda não foi visitado
                            mask_next = mask | (1 << v)
                            novo_custo = DP[mask][u] + grafo[vertices[u]].get(vertices[v], float('inf'))
                            if DP[mask_next][v] > novo_custo:
                                DP[mask_next][v] = novo_custo
                                predecessor[mask_next][v] = u

        mask_final = (1 << n) - 1
        menor_custo = float('inf')
        ultimo_vertice = -1

        # Encontrar o custo mínimo e o último vértice
        for i in range(1, n):
            custo_ciclo = DP[mask_final][i] + grafo[vertices[i]].get(vertices[0], float('inf'))
            if menor_custo > custo_ciclo:
                menor_custo = custo_ciclo
                ultimo_vertice = i

        # Reconstruir o melhor caminho encontrado
        melhor_caminho = []
        mask = mask_final
        while mask != 1 and ultimo_vertice != -1:
            melhor_caminho.append(vertices[ultimo_vertice])
            novo_ultimo_vertice = predecessor[mask][ultimo_vertice]
            mask ^= (1 << ultimo_vertice)
            ultimo_vertice = novo_ultimo_vertice

        melhor_caminho.append(vertices[0])
        melhor_caminho.reverse()
        
        # Montar string para exibição do caminho e pesos
        caminho_str = ""
        for i in range(len(melhor_caminho) - 1):
            v_atual = melhor_caminho[i]
            v_proximo = melhor_caminho[i + 1]
            peso = grafo[v_atual][v_proximo]
            caminho_str += f"{v_atual}: {peso} -> "
        caminho_str += melhor_caminho[-1]

        print(f"Caminho TSPD: {caminho_str}")

        # Calcular o custo do caminho Hamiltoniano no grafo G
        custo_ch = self.calcular_custo_caminho(caminho_ch, grafo)

        return melhor_caminho, menor_custo

    
# Função para medir o tempo de execução e o uso de CPU
def medir_tempo_uso(func, *args):
    tempo_inicio = time.time()
    cpu_inicio = psutil.Process(os.getpid()).cpu_times().user
    
    resultado = func(*args)
    
    tempo_fim = time.time()
    cpu_fim = psutil.Process(os.getpid()).cpu_times().user
    cpu_uso = cpu_fim - cpu_inicio
    tempo_total = tempo_fim - tempo_inicio

    horas, resto = divmod(tempo_total, 3600)
    minutos, segundos = divmod(resto, 60)
    
    print(f"Tempo Total: {int(horas)} horas, {int(minutos)} minutos, {segundos:.2f} segundos")
    print(f"Uso de CPU: {cpu_uso} segundos")
    print("Custo Total: ", resultado[1])
    return resultado, tempo_total

def generate_graph_with_edges(grafo, num_vertices, especifico_grafo):
    grafo.gerar_automaticamente_arestas(num_vertices=num_vertices, grafo = especifico_grafo)

    # # Gerando a matriz de adjacência
    # matriz_adjacencia = grafo.gerar_matriz_adjacencia(especifico_grafo) #!
    # print("\nMatriz de Adjacência:")
    # print(np.array(matriz_adjacencia))

    # Exibir grafo
    # grafo.exibir_grafo(especifico_grafo)
    # grafo.visualizar_grafo(especifico_grafo)


def create_graph():
    grafo = Grafo()
    times_ch = []
    times_ch_dp = []
    times_tsp = []
    times_tsp_dp = []
    tempo_total1, tempo_total2, tempo_total3, tempo_total4 = 0, 0, 0, 0
    i = 5
    while i > 0:
        num_vertices = i
        # Generate a graph of the given length
        grafo = Grafo()
        if i == 6:
            i = i 
        # Adicionando arestas com pesos (distâncias)
        generate_graph_with_edges(grafo, num_vertices, especifico_grafo = grafo.grafo_ch)
        grafo.transformar_em_tsp()
        g_list = list(grafo.grafo_ch.keys())
        
        if tempo_total1 < 180:
            print("-------------------------------------------------")
            try:
                caminho_ch, tempo_total1 = medir_tempo_uso(grafo.caminho_hamiltoniano, g_list[0], grafo.grafo_ch)
                times_ch.append([tempo_total1, num_vertices])
                val1 = True
            except:
                times_ch.append([])
                val1 = False
        else:
            times_ch.append([None, num_vertices])
            val1 = False

        if tempo_total2 < 180:
            print("-------------------------------------------------")
            try:
                caminho_tsp, tempo_total2 = medir_tempo_uso(grafo.calcular_tsp_forcando_ch, g_list[0], grafo.grafo_tsp, caminho_ch[0])
                times_tsp.append([tempo_total2, num_vertices])
                val2 = True
            except:
                times_tsp.append([None, num_vertices])
                val2 = False
        else:
            times_tsp.append([None, num_vertices])
            val2 = False

        if tempo_total3 < 180:
            print("-------------------------------------------------")
            try:
                caminho_ch_dp, tempo_total3 = medir_tempo_uso(grafo.caminho_hamiltoniano_dp, g_list[0], grafo.grafo_ch)
                times_ch_dp.append([tempo_total3, num_vertices])
                val3 = True
            except:
                times_ch_dp.append([None, num_vertices])
                val3 = False
        else:
            times_ch_dp.append([None, num_vertices])
            val3 = False
        
        if tempo_total4 < 180:
            print("-------------------------------------------------")
            try:
                caminho_tsp, tempo_total4 = medir_tempo_uso(grafo.held_karp, grafo.grafo_tsp, caminho_ch_dp[0])
                times_tsp_dp.append([tempo_total4, num_vertices])
                val4 = True
            except:
                times_tsp_dp.append([None, num_vertices])
                val4 = False
            print("-------------------------------------------------")
        else:
            times_tsp_dp.append([None, num_vertices])
            val4 = False
        i+=1
        if not val1 and not val2 and not val3 and not val4:
            break
        
    # Separar os valores de tamanho e tempo para as quatro listas
    y_ch, x_ch = zip(*times_ch)
    y_tsp, x_tsp, = zip(*times_tsp)
    y_ch_dp, x_ch_dp  = zip(*times_ch_dp)
    y_tsp_dp, x_tsp_dp  = zip(*times_tsp_dp)

    # Criar o gráfico
    plt.figure(figsize=(10, 6))

    plt.plot(x_ch, y_ch, label='CH - FB', marker='o')
    plt.plot(x_tsp, y_tsp, label='TSP - FB', marker='o')
    plt.plot(x_ch_dp, y_ch_dp, label='CH - PD', marker='o')
    plt.plot(x_tsp_dp, y_tsp_dp, label='TSP - PD', marker='o')

    plt.xlabel('Tamanho da entrada (n)')
    plt.ylabel('Tempo (segundos)')
    plt.title('Tempo de Execução de Diferentes Algoritmos')
    plt.legend()
    plt.grid(True)

    plt.show()


# Call the function to create the graph
create_graph()
