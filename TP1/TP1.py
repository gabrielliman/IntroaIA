import sys
import time
import heapq
import numpy as np
from collections import deque

def verificar_objetivo(matriz):
    for linha in matriz:
        if len(np.unique(linha)) != 9 or 0 in linha:
            return False

    for coluna in range(9):
        if len(np.unique(matriz[:, coluna])) != 9:
            return False

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            quadrado = matriz[i:i+3, j:j+3].flatten()
            if len(np.unique(quadrado)) != 9:
                return False

    return True

def gerar_sucessores(matriz):
    sucessores = []

    for i in range(9):
        for j in range(9):
            if matriz[i][j] == 0:
                for num in range(1, 10):
                    if validar_atribuicao(matriz, i, j, num):
                        novo_estado = np.copy(matriz)
                        novo_estado[i][j] = num
                        sucessores.append(novo_estado)
                return sucessores
    return sucessores

def validar_atribuicao(matriz, linha, coluna, num):
    if num in matriz[linha]:
        return False

    if num in matriz[:, coluna]:
        return False

    inicio_linha, inicio_coluna = 3 * (linha // 3), 3 * (coluna // 3)
    quadrado = matriz[inicio_linha:inicio_linha+3, inicio_coluna:inicio_coluna+3]
    if num in quadrado:
        return False

    return True

def bfs(sudoku_inicial):
    fila = deque()
    num_visitados = 0

    fila.append(sudoku_inicial)

    while fila:
        estado = fila.popleft()
        num_visitados += 1

        if verificar_objetivo(estado):
            return estado, num_visitados

        sucessores = gerar_sucessores(estado)

        for suc in sucessores:
            fila.append(suc)
    return None, num_visitados



def dfs_limitado(sudoku, profundidade_maxima):
    def dfs_recursivo(estado, profundidade):
        if profundidade > profundidade_maxima:
            return None, 0

        if verificar_objetivo(estado):
            return estado, 1

        sucessores = gerar_sucessores(estado)
        num_visitados = 1

        for suc in sucessores:
            solucao, visitados = dfs_recursivo(suc, profundidade + 1)
            num_visitados += visitados
            if solucao is not None:
                return solucao, num_visitados

        return None, num_visitados


    return dfs_recursivo(sudoku, 0)

def ids(sudoku_inicial):
    profundidade_maxima = 1
    num_visitados=0

    while True:
        solucao, num_visitados_lim = dfs_limitado(sudoku_inicial, profundidade_maxima)
        num_visitados+=num_visitados_lim
        if solucao is not None:
            return solucao, num_visitados
        else:
            profundidade_maxima += 1

def uniform_cost_search(sudoku_inicial):
    fila = [(0, sudoku_inicial.tobytes())]
    visitados = 0

    while fila:
        custo, estado = heapq.heappop(fila)
        estado = np.frombuffer(estado, dtype=int).reshape((9, 9)) 
        visitados += 1

        if verificar_objetivo(estado):
            return estado, visitados

        sucessores = gerar_sucessores(estado)

        for suc in sucessores:
            estado_str = suc.tobytes()
            heapq.heappush(fila, (np.count_nonzero(suc == 0), estado_str))

    return None, visitados


def calcular_heuristica(matriz):
    heuristica = np.zeros_like(matriz, dtype=int)  

    for i in range(9):
        for j in range(9):
            if matriz[i][j] == 0:
                valores_possiveis = set(range(1, 10)) - set(matriz[i]) - set(matriz[:, j]) - set(matriz[i//3*3:i//3*3+3, j//3*3:j//3*3+3].flatten())
                heuristica[i][j] = len(valores_possiveis)

    return heuristica

def buscar_proxima_jogada(matriz, heuristica):
    menor_heuristica = float('inf')
    melhor_jogada = None

    for i in range(9):
        for j in range(9):
            if matriz[i][j] == 0 and heuristica[i][j] < menor_heuristica:
                menor_heuristica = heuristica[i][j]
                melhor_jogada = (i, j)

    return melhor_jogada

def greedy(sudoku):
    visitados = 0
    heap = []

    heapq.heappush(heap, (0, sudoku.tobytes(), calcular_heuristica(sudoku)))

    while heap:
        _, sudoku, heuristica = heapq.heappop(heap)
        sudoku = np.frombuffer(sudoku, dtype=int).reshape((9, 9))
        visitados += 1

        proxima_jogada = buscar_proxima_jogada(sudoku, heuristica)
        
        if proxima_jogada is None:
            return sudoku, visitados
        else:
            i, j = proxima_jogada
            valores_possiveis = set(range(1, 10)) - set(sudoku[i]) - set(sudoku[:, j]) - set(sudoku[i//3*3:i//3*3+3, j//3*3:j//3*3+3].flatten())

            if len(valores_possiveis) == 0:
                continue
            else:
                for valor in valores_possiveis:
                    novo_sudoku = sudoku.copy()
                    novo_sudoku[i][j] = valor
                    novo_sudoku_str=novo_sudoku.tobytes()
                    heapq.heappush(heap, (np.count_nonzero(novo_sudoku == 0), novo_sudoku_str, calcular_heuristica(novo_sudoku)))  # Atualiza a fila de prioridade com a nova configuração

    return sudoku, visitados



def heuristicaAstar(matriz):
    heuristica = np.zeros_like(matriz, dtype=int)
    for i in range(9):
        for j in range(9):
            if matriz[i][j] == 0:
                valores_possiveis = set(range(1, 10)) - set(matriz[i]) - set(matriz[:, j]) - set(matriz[i//3*3:i//3*3+3, j//3*3:j//3*3+3].flatten())
                heuristica[i][j] = len(valores_possiveis) + manhattan_distancia(i, j)
            else:
                heuristica[i][j] = 0
    return heuristica

def manhattan_distancia(i, j):
    centro_linha, centro_coluna = 4, 4
    return abs(centro_linha - i) + abs(centro_coluna - j)

def buscar_proxima_jogada_Astar(matriz, heuristica):
    menor_custo_total = float('inf')
    melhor_jogada = None

    for i in range(9):
        for j in range(9):
            if matriz[i][j] == 0 and heuristica[i][j] < menor_custo_total:
                menor_custo_total = heuristica[i][j]
                melhor_jogada = (i, j)

    return melhor_jogada

def Astar(sudoku):
    visitados = 0
    heap = []  

    heapq.heappush(heap, (0, 0, sudoku.tobytes(), heuristicaAstar(sudoku)))

    while heap:
        _, custo_acumulado, sudoku, heuristica = heapq.heappop(heap)  
        sudoku = np.frombuffer(sudoku, dtype=int).reshape((9, 9))
        visitados += 1

        proxima_jogada = buscar_proxima_jogada_Astar(sudoku, heuristica)
        
        if proxima_jogada is None:
            return sudoku, visitados
        else:
            i, j = proxima_jogada
            valores_possiveis = set(range(1, 10)) - set(sudoku[i]) - set(sudoku[:, j]) - set(sudoku[i//3*3:i//3*3+3, j//3*3:j//3*3+3].flatten())

            if len(valores_possiveis) == 0:
                continue
            else:
                for valor in valores_possiveis:
                    novo_sudoku = sudoku.copy()
                    novo_sudoku[i][j] = valor
                    novo_custo_acumulado = custo_acumulado
                    novo_sudoku_str = novo_sudoku.tobytes()
                    heapq.heappush(heap, (novo_custo_acumulado + np.count_nonzero(novo_sudoku == 0), novo_custo_acumulado, novo_sudoku_str, heuristicaAstar(novo_sudoku)))  

    return sudoku, visitados





def main():
    if len(sys.argv) < 3:
        print("Uso: TP1 algoritmo entrada")
        return

    algoritmo = sys.argv[1]
    entrada = sys.argv[2:]

    grid = [[int(entrada[i][j]) for j in range(9)] for i in range(9)]
    sudoku=np.array(grid)
    start_time = time.time()
    estados_expandidos=0
    if algoritmo == 'B':
        solucao, estados_expandidos = bfs(sudoku)
    elif algoritmo == 'I':
        solucao, estados_expandidos = ids(sudoku)
    elif algoritmo == 'U':
        solucao, estados_expandidos = uniform_cost_search(sudoku)
    elif algoritmo == 'A':
        solucao, estados_expandidos = Astar(sudoku)
    elif algoritmo == 'G':
        solucao, estados_expandidos = greedy(sudoku)
    else:
        print("Algoritmo não reconhecido")
        return

    end_time = time.time()
    tempo_execucao = int((end_time - start_time) * 1000)  # Tempo em milissegundos

    print("Estados expandidos:", estados_expandidos)
    print("Tempo de execução (ms):", tempo_execucao)

    print("Solução:")
    for linha in solucao:
        print("".join(map(str, linha)))


if __name__ == "__main__":
    main()
