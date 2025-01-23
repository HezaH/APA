import random
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
import itertools
import psutil  # Biblioteca para medir uso de CPU e memória
import os
import math

class Grafo:
    def __init__(self):
        # Inicializa o dicionário para armazenar o grafo
        self.grafo_hcp = {}
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
    
    def dividir_lista(self, lista, m):
        """Divide uma lista em sublistas de tamanho m."""
        return [lista[i:i + m] for i in range(0, len(lista), m)]

    # def gerar_automaticamente_arestas(self, num_vertices, heigth_x, grafo):
    #     """Gera vértices e arestas mantendo um padrão de conexão."""
    #     # Programação Dinâmica: Cache dos vértices criados
    #     vertices = [f"v{i}" for i in range(1, num_vertices + 1)]
        
    #     # Adiciona vértices ao grafo
    #     for vertice in vertices:
    #         self.add_vertice(vertice, grafo)

    #     # Divide os vértices em camadas de tamanho `heigth_x`
    #     camadas = self.dividir_lista(vertices, heigth_x)

    #     # Conectar vértices dentro de cada camada
    #     for camada in camadas:
    #         for i in range(len(camada) - 1):
    #             peso = random.randint(40, 160)
    #             self.add_aresta(camada[i], camada[i + 1], peso, grafo)

    #     # Conectar camadas adjacentes
    #     for i in range(len(camadas) - 1):
    #         min_length = min(len(camadas[i]), len(camadas[i + 1]))  # Evita IndexError
    #         for j in range(min_length):
    #             peso = random.randint(50, 100)
    #             self.add_aresta(camadas[i][j], camadas[i + 1][j], peso, grafo)

    def gerar_automaticamente_arestas(self, num_vertices, heigth_x, grafo):
        """Gera vértices e arestas mantendo um padrão de conexão e assegura que vértices soltos tenham conexões adicionais."""
        # Programação Dinâmica: Cache dos vértices criados
        vertices = [f"v{i}" for i in range(1, num_vertices + 1)]
        
        # Adiciona vértices ao grafo
        for vertice in vertices:
            self.add_vertice(vertice, grafo)

        # Divide os vértices em camadas de tamanho `heigth_x`
        camadas = self.dividir_lista(vertices, heigth_x)

        # Conectar vértices dentro de cada camada
        for camada in camadas:
            for i in range(len(camada) - 1):
                peso = random.randint(40, 160)
                self.add_aresta(camada[i], camada[i + 1], peso, grafo)

        # Conectar camadas adjacentes
        for i in range(len(camadas) - 1):
            min_length = min(len(camadas[i]), len(camadas[i + 1]))  # Evita IndexError
            for j in range(min_length):
                peso = random.randint(50, 100)
                self.add_aresta(camadas[i][j], camadas[i + 1][j], peso, grafo)

        # Adicionar conexões adicionais para vértices soltos
        for vertice in vertices:
            if len(grafo[vertice]) == 1:  # Verifica se o vértice está solto (apenas uma conexão)
                conexao = list(grafo[vertice].keys())[0]
                neighbor = sorted(self.neighbor(conexao, grafo))
                neighbor.remove(vertice)
                
                peso = random.randint(50, 100)
                self.add_aresta(vertice, neighbor[-1] , peso, grafo)


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
    
    def is_hamiltonian_path(self, caminho):
        n = len(self.grafo_hcp)
        if len(caminho) == n:
            return True
        return False

    def buscar_caminho_hamiltoniano(self, caminho):
        if self.is_hamiltonian_path(caminho):
            return caminho

        ultimo_vertice = caminho[-1]
        for vizinho in self.grafo_hcp[ultimo_vertice]:
            if vizinho not in caminho:
                novo_caminho = self.buscar_caminho_hamiltoniano(caminho + [vizinho])
                if novo_caminho:
                    return novo_caminho
        return None

    def encontrar_caminho_hamiltoniano(self):
        for vertice in self.grafo_hcp:
            caminho = self.buscar_caminho_hamiltoniano([vertice])
            if caminho:
                return caminho
        return None

    def transformar_em_tsp(self):
        caminho_hamiltoniano = self.encontrar_caminho_hamiltoniano()
        if not caminho_hamiltoniano:
            return None  # Nenhum caminho hamiltoniano encontrado

        grafo = self.grafo_tsp
        vertices = list(self.grafo_hcp.keys())
        valor_maximo = 500  # Valor arbitrariamente alto, mas finito

        for v1 in vertices:
            grafo[v1] = {}
            for v2 in vertices:
                if v1 == v2:
                    continue
                if v2 in self.grafo_hcp[v1]:
                    peso = self.grafo_hcp[v1][v2]
                    self.add_aresta(v1, v2, peso, grafo)
                elif v1 in caminho_hamiltoniano and v2 in caminho_hamiltoniano:
                    self.add_aresta(v1, v2, valor_maximo, grafo)
    
    def visualizar_grafo(self, grafo):
        # # Ativar o modo interativo
        # plt.ion()
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

        # # Adicionando rótulos nas arestas
        # edge_labels = nx.get_edge_attributes(G, 'weight')
        
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=max(6, font_size - 2))

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
        plt.show()

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
        
        # Montar string para exibição do caminho e pesos
        caminho_str = ""
        for i in range(len(melhor_caminho) - 1):
            v_atual = melhor_caminho[i]
            v_proximo = melhor_caminho[i + 1]
            peso = grafo[v_atual][v_proximo]
            caminho_str += f"{v_atual}: {peso} -> "
        caminho_str += melhor_caminho[-1]

        # Exibir o resultado 
        print(f"Caminho: {melhor_caminho_str}")

        # Retornar caminho mínimo encontrado e métricas
        return melhor_caminho, menor_custo

    def caminho_hamiltoniano_dp(self, inicio, grafo):
        # Criar mapeamento entre vértices e índices numéricos
        vertices = list(grafo.keys())  # Lista de vértices (podem ser strings ou números)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}  # v1 -> 0, v2 -> 1, ...
        index_to_vertex = {i: v for v, i in vertex_to_index.items()}  # 0 -> v1, 1 -> v2, ...

        n = len(vertices)

        if inicio not in vertex_to_index:
            raise ValueError(f"Vértice inicial '{inicio}' não encontrado no grafo.")

        inicio_idx = vertex_to_index[inicio]  # Converte 'v1' para índice numérico

        # Criar tabela DP [2^n][n] preenchida com infinito
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Criar matriz para armazenar o caminho
        parent = [[-1] * n for _ in range(1 << n)]

        # O custo para começar no nó 'inicio' e já tê-lo visitado é 0
        dp[1 << inicio_idx][inicio_idx] = 0  

        # Percorrer todos os subconjuntos de nós visitados (máscaras)
        for mask in range(1 << n):  
            for u in range(n):  
                if (mask & (1 << u)) == 0:  # Se 'u' NÃO foi visitado
                    continue  

                # Tentar adicionar cada nó 'v' ao subconjunto
                for v in range(n):  
                    if (mask & (1 << v)) or grafo[vertices[u]].get(vertices[v], 0) == 0:  # Se 'v' já foi visitado OU não há aresta
                        continue  

                    novo_mask = mask | (1 << v)  # Novo estado com 'v' incluído
                    novo_custo = dp[mask][u] + grafo[vertices[u]][vertices[v]]  # Obter custo da aresta

                    # Atualizar custo mínimo e armazenar o nó anterior
                    if novo_custo < dp[novo_mask][v]:
                        dp[novo_mask][v] = novo_custo
                        parent[novo_mask][v] = u  # Registrar quem veio antes de 'v'

        # Encontrar o menor custo possível ao final do percurso
        ultimo_no = -1
        menor_custo = float('inf')
        estado_final = (1 << n) - 1  # Todos os nós foram visitados

        for v in range(n):
            if dp[estado_final][v] < menor_custo:
                menor_custo = dp[estado_final][v]
                ultimo_no = v  # Último nó do melhor caminho

        if ultimo_no == -1:
            raise ValueError("Não foi possível encontrar um caminho válido.")

        # Reconstruir o melhor caminho e calcular os pesos das arestas
        melhor_caminho = []
        pesos_arestas = []
        mask = estado_final
        while ultimo_no != -1:
            melhor_caminho.append(index_to_vertex[ultimo_no])  # Converter índice para nome original
            proximo_no = parent[mask][ultimo_no]
            if proximo_no != -1:
                peso = grafo[index_to_vertex[proximo_no]][index_to_vertex[ultimo_no]]
                pesos_arestas.append((index_to_vertex[proximo_no], peso))
            mask &= ~(1 << ultimo_no)  # Remover último nó do estado
            ultimo_no = proximo_no

        melhor_caminho.reverse()  # Corrigir ordem (começa do final)

        # Montar string para exibição do caminho e pesos
        caminho_str = ""
        for i in range(len(melhor_caminho) - 1):
            v_atual = melhor_caminho[i]
            v_proximo = melhor_caminho[i + 1]
            peso = grafo[v_atual][v_proximo]
            caminho_str += f"{v_atual}: {peso} -> "
        caminho_str += melhor_caminho[-1]

        print(f"Caminho: {caminho_str}")

        # Retornar caminho mínimo encontrado e métricas
        return melhor_caminho, menor_custo

    def calcular_custo_caminho(self, caminho, grafo):
        """Calcula o custo total de um caminho dado, somando os pesos das arestas e exibe o caminho detalhado"""
        custo = 0
        caminho_str = ""
        for i in range(len(caminho) - 1):
            v1, v2 = caminho[i], caminho[i + 1]
            if v2 in grafo[v1]:
                peso = grafo[v1][v2]
                custo += peso
                caminho_str += f"{v1}: {peso} -> "
            else:
                # Se não houver aresta entre v1 e v2, retorna um custo muito alto
                return float('inf'), ""
        caminho_str += caminho[-1]
        return custo, caminho_str

    def calcular_tsp(self, origem, grafo):    
        # Gerar todas as permutações dos vértices (exceto a origem)
        vertices = list(grafo.keys())
        vertices.remove(origem)
        permutacoes = itertools.permutations(vertices)
        
        melhor_caminho = None
        menor_custo = float('inf')
        melhor_caminho_str = ""

        # Verificar todas as permutações possíveis
        for caminho in permutacoes:
            custo, caminho_str = self.calcular_custo_caminho([origem] + list(caminho) + [origem], grafo)
            if custo < menor_custo:
                menor_custo = custo
                melhor_caminho = [origem] + list(caminho) + [origem]
                melhor_caminho_str = caminho_str
        
        # Exibir o resultado 
        print(f"Caminho: {melhor_caminho_str}")
        
        # Retornar caminho mínimo encontrado e métricas
        return melhor_caminho, menor_custo

    def held_karp(self, grafo):
        vertices = list(grafo.keys())
        n = len(vertices)

        # DP[mascara][i] armazena o custo mínimo para visitar todos os vértices na máscara terminando em i
        DP = [[float('inf')] * n for _ in range(1 << n)]
        DP[1][0] = 0

        # Predecessor array to reconstruct the path
        predecessor = [[-1] * n for _ in range(1 << n)]

        for mask in range(1 << n):
            for u in range(n):
                if mask & (1 << u):
                    for v in range(n):
                        if mask & (1 << v) == 0:
                            mask_next = mask | (1 << v)
                            if DP[mask_next][v] > DP[mask][u] + grafo[vertices[u]].get(vertices[v], float('inf')):
                                DP[mask_next][v] = DP[mask][u] + grafo[vertices[u]].get(vertices[v], float('inf'))
                                predecessor[mask_next][v] = u

        mask_final = (1 << n) - 1
        menor_custo = float('inf')
        ultimo_vertice = -1

        for i in range(1, n):
            if menor_custo > DP[mask_final][i] + grafo[vertices[i]].get(vertices[0], float('inf')):
                menor_custo = DP[mask_final][i] + grafo[vertices[i]].get(vertices[0], float('inf'))
                ultimo_vertice = i

        # if ultimo_vertice == -1:
        #     raise ValueError("Não foi possível encontrar um caminho válido.")

        # Reconstruir o melhor caminho e calcular os pesos das arestas
        melhor_caminho = []
        pesos_arestas = []
        mask = mask_final
        while mask != 1 and ultimo_vertice != -1:
            novo_ultimo_vertice = predecessor[mask][ultimo_vertice]
            if novo_ultimo_vertice != -1:
                peso = grafo[vertices[novo_ultimo_vertice]][vertices[ultimo_vertice]]
                pesos_arestas.append((vertices[novo_ultimo_vertice], peso))
            melhor_caminho.append(vertices[ultimo_vertice])
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

        print(f"Caminho: {caminho_str}")

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

    return resultado

# Criar a instância do grafo dinâmico
grafo = Grafo()

# Adicionando arestas com pesos (distâncias)
#!para calcular_tsp não exceder 13 vértices  
num_vertices = 22 #? Da erro quando é numero impar
heigth_x = num_vertices//4
grafo.gerar_automaticamente_arestas(num_vertices=num_vertices, heigth_x=heigth_x, grafo = grafo.grafo_hcp)

# Gerando a matriz de adjacência
matriz_adjacencia = grafo.gerar_matriz_adjacencia(grafo.grafo_hcp) #!
print("\nMatriz de Adjacência:")
print(np.array(matriz_adjacencia))

# Exibir grafo
grafo.exibir_grafo(grafo.grafo_hcp)
grafo.visualizar_grafo(grafo.grafo_hcp)

grafo.transformar_em_tsp()
grafo.visualizar_grafo(grafo.grafo_tsp)
# Exibir grafo
# grafo.exibir_grafo(grafo.grafo_tsp)


g_list = list(grafo.grafo_hcp.keys())

# print("-------------------------------------")
# # Melhor Caminho usando o Cálculo do caminho Hamiltoniano
# print("Executando caminho_hamiltoniano:")
# melhor_caminho1, menor_custo1 = medir_tempo_uso(grafo.caminho_hamiltoniano, g_list[0], grafo.grafo_hcp)
# print("Melhor Caminho usando o Cálculo do caminho Hamiltoniano:", melhor_caminho1)
# print("Menor Custo:", menor_custo1)
# print("-------------------------------------")

# Melhor Caminho usando o Cálculo do caminho Hamiltoniano (Programação Dinâmica)
print("\nExecutando caminho_hamiltoniano_dp: grafo HCP")
melhor_caminho2, menor_custo2 = medir_tempo_uso(grafo.caminho_hamiltoniano_dp, g_list[0], grafo.grafo_hcp)
print("Melhor Caminho usando o Cálculo do caminho Hamiltoniano:", melhor_caminho2)
print("Menor Custo:", menor_custo2)
print("-------------------------------------")

# # Calcular o TSP a partir de um vértice de origem (por exemplo, 'v1')
# print("Calcular o TSP a partir de um vértice de origem (por exemplo, 'v1')")
# melhor_caminho3, menor_custo3 = medir_tempo_uso(grafo.calcular_tsp, g_list[0],  grafo.grafo_hcp)
# print("Melhor Caminho usando o cálculo do TSP (Caixeiro Viajante):", melhor_caminho3)
# print("Menor Custo:", menor_custo3)
# print("-------------------------------------")

# Calcular o TSP usando o algoritmo de Held-Karp e obter métricas de desempenho
print("Calcular o grafo HCP usando o algoritmo de Held-Karp e obter métricas de desempenho")
melhor_caminho4, menor_custo4 = medir_tempo_uso(grafo.held_karp, grafo.grafo_hcp)
print("Melhor Caminho usando Held-Karp:", melhor_caminho4)
print("Menor Custo:", menor_custo4)
print("-------------------------------------")





# print("-------------------------------------")
# # Melhor Caminho usando o Cálculo do caminho Hamiltoniano
# print("Executando caminho_hamiltoniano:")
# melhor_caminho1, menor_custo1 = medir_tempo_uso(grafo.caminho_hamiltoniano, g_list[0], grafo.grafo_tsp)
# print("Melhor Caminho usando o Cálculo do caminho Hamiltoniano:", melhor_caminho1)
# print("Menor Custo:", menor_custo1)
# print("-------------------------------------")

# Melhor Caminho usando o Cálculo do caminho Hamiltoniano (Programação Dinâmica)
print("\nExecutando caminho_hamiltoniano_dp: grafo TSP")
melhor_caminho2, menor_custo2 = medir_tempo_uso(grafo.caminho_hamiltoniano_dp, g_list[0], grafo.grafo_tsp)
print("Melhor Caminho usando o Cálculo do caminho Hamiltoniano:", melhor_caminho2)
print("Menor Custo:", menor_custo2)
print("-------------------------------------")


# # Calcular o TSP a partir de um vértice de origem (por exemplo, 'v1')
# print("Calcular o TSP a partir de um vértice de origem (por exemplo, 'v1')")
# melhor_caminho3, menor_custo3 = medir_tempo_uso(grafo.calcular_tsp, g_list[0],  grafo.grafo_tsp)
# print("Melhor Caminho usando o cálculo do TSP (Caixeiro Viajante):", melhor_caminho3)
# print("Menor Custo:", menor_custo3)
# print("-------------------------------------")

# Calcular o TSP usando o algoritmo de Held-Karp e obter métricas de desempenho
print("Calcular o grafo TSP usando o algoritmo de Held-Karp e obter métricas de desempenho")
melhor_caminho4, menor_custo4 = medir_tempo_uso(grafo.held_karp, grafo.grafo_tsp)
print("Melhor Caminho usando Held-Karp:", melhor_caminho4)
print("Menor Custo:", menor_custo4)
print("-------------------------------------")

# Atualizar o grafo dinamicamente
# grafo.atualizar_grafo(g_list, heigth_x, grafo.grafo_hcp)

# Exibir grafo
grafo.visualizar_grafo(grafo.grafo_hcp)



v1 = random.choice(g_list)
v2 = random.choice(g_list)



grafo.grafo_hcp = {
    'v1': {'v2': 137, 'v6': 60},
    'v2': {'v1': 137, 'v3': 136, 'v7': 66},
    'v3': {'v2': 136, 'v4': 106, 'v8': 78},
    'v4': {'v3': 106, 'v5': 89, 'v9': 76},
    'v5': {'v4': 89, 'v10': 58},
    'v6': {'v7': 143, 'v1': 60, 'v11': 68},
    'v7': {'v6': 143, 'v8': 155, 'v2': 66, 'v12': 85},
    'v8': {'v7': 155, 'v9': 78, 'v3': 78, 'v13': 75},
    'v9': {'v8': 78, 'v10': 97, 'v4': 76, 'v14': 98},
    'v10': {'v9': 97, 'v5': 58, 'v15': 54},
    'v11': {'v12': 119, 'v6': 68, 'v16': 55},
    'v12': {'v11': 119, 'v13': 100, 'v7': 85, 'v17': 53},
    'v13': {'v12': 100, 'v14': 40, 'v8': 75, 'v18': 61},
    'v14': {'v13': 40, 'v15': 112, 'v9': 98, 'v19': 63},
    'v15': {'v14': 112, 'v10': 54, 'v20': 78},
    'v16': {'v17': 144, 'v11': 55, 'v21': 78},
    'v17': {'v16': 144, 'v18': 124, 'v12': 53, 'v21': 95},
    'v18': {'v17': 124, 'v19': 147, 'v13': 61}
}