from __future__ import print_function
import matplotlib, sys
from matplotlib import pyplot as plt
import numpy as np

def prever(ent, pes): #Função de previsão, com entradas e pesos
    ativ = 0.0
    for e,p in zip(ent, pes):
        ativ += e * p
    return 1.0 if ativ >= 0.0 else 0.0 

def plotar(matriz, pesos=None, title="Matriz de previsão"): #Função para plotar o
    #gráfico da matriz
    if len(matriz[0]) == 3: #Se for uma entrada 1D (ponto somente):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("i1")
        ax.set_ylabel("Classificações")

        if(pesos != None): #Se houverem pesos...
            y_min = -0.1 # Organizamos as medidas para
            y_max = 1.1 # plotar o gráfico.
            x_min = 0.0
            x_max = 1.1
            y_res = 0.001
            x_res = 0.001
            ys = np.arange(y_min, y_max, y_res)
            xs = np.arange(x_min, x_max, x_res)
            zs = []
            for cur_y in np.arange(y_min, y_max, y_res):
                for cur_x in np.arange(x_min, x_max, x_res):
                    zs.append(prever([1.0, cur_x], pesos))
            xs, ys = np.meshgrid(xs, ys) # Criamos nossa grade...
            zs = np.array(zs)
            zs = zs.reshape(xs.shape)
            cp = plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('b', 'r'), alpha=0.1)
            # ...e plotamos os contornos.

        c1_data = [[],[]]
        c0_data = [[],[]]
        for i in range(len(matriz)): # Para cada elemento da matriz...
            atual_i1 = matriz[i][1] # distribuímos na classe
            atual_y = matriz[i][-1]
            if atual_y == 1:
                c1_data[0].append(atual_i1)
                c1_data[1].append(1.0)
            else:
                c0_data[0].append(atual_i1)
                c0_data[1].append(0.0)
        
        plt.xticks(np.arange(x_min, x_max, 0.1)) # Organizamos...
        plt.yticks(np.arange(y_min, y_max, 0.1))
        plt.xlim(0, 1.05) # ..definimos o limite...
        plt.ylim(-0.05, 1.05)
        # ...e espalhamos no gráfico. Classe -1 em pontos vermelhos, e 1 em azuis.
        c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label="Classe -1")
        c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label="Classe 1")
        
        plt.legend(fontsize=10, loc=1)
        plt.show()
        return

    if len(matriz[0]) == 4: #Se for uma entrada 2D (reta), repetimos o processo,
        # com algumas alterações:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("i1")
        ax.set_ylabel("i2")

        if(pesos != None): #Se houverem pesos...
            mapa_min = 0.0 # Organizamos as medidas para
            mapa_max = 1.1 # plotar o gráfico.
            y_res = 0.001
            x_res = 0.001
            ys = np.arange(mapa_min, mapa_max, y_res)
            xs = np.arange(mapa_min, mapa_max, x_res)
            zs = []
            for cur_y in np.arange(mapa_min, mapa_max, y_res):
                for cur_x in np.arange(mapa_min, mapa_max, x_res):
                    zs.append(prever([1.0, cur_x, cur_y], pesos))
            xs, ys = np.meshgrid(xs, ys) # Criamos nossa grade...
            zs = np.array(zs)
            zs = zs.reshape(xs.shape)
            cp = plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('b', 'r'), alpha=0.1)
            # ...e plotamos os contornos.

        c1_data = [[],[]]
        c0_data = [[],[]]
        for i in range(len(matriz)): # Para cada elemento da matriz...
            atual_i1 = matriz[i][1] # distribuímos em duas classes:
            atual_i2 = matriz[i][2] # -1 e 1.
            atual_y = matriz[i][-1]
            if atual_y == 1:
                c1_data[0].append(atual_i1)
                c1_data[1].append(atual_i2)
            else:
                c0_data[0].append(atual_i1)
                c0_data[1].append(atual_i2)
        
        plt.xticks(np.arange(0.0, 1.1, 0.1)) # Organizamos...
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlim(0, 1.05) # ..definimos o limite...
        plt.ylim(0, 1.05)
        # ...e espalhamos no gráfico. Classe -1 em pontos vermelhos, e 1 em azuis.
        c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label="Classe -1")
        c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label="Classe 1")
        
        plt.legend(fontsize=10, loc=1)
        plt.show() # Aqui mostramos o gráfico numa janela.
        return
    print("Dimensões da matriz não cobertas.")

def acuracia(matriz, pesos): # Que tal definir a acurácia do algoritmo?
    num_acertos = 0.0
    prevs = [] # Aqui vêm as previsões.
    for i in range(len(matriz)):
        prev = prever(matriz[i][:-1], pesos)
        prevs.append(prev)
        if prev == matriz[i][-1]:
            num_acertos += 1.0
    print("Previsões: ", prevs)
    return(num_acertos/float(len(matriz)))

def treinar_pesos(matriz, pesos, num_epocas=10, taxa_l=1.00, plota=False, parar_antes=True, verbose=True):
    # Aqui treinamos os pesos. Nada de água com músculo, aqui a gente constroi fibra!
    for epoca in range(num_epocas): # PAra cada época...
        acu_atu = acuracia(matriz, pesos) # ...calculamos a acurácia atual...
        print("\nÉpoca: %d\nPesos: "%epoca, pesos)
        print("Acurácia: ", acu_atu)

        if acu_atu == 1.0 and parar_antes:
            break
        if plota:
            plotar(matriz, pesos, title="Época %d"%epoca)
        
        for i in range(len(matriz)): # ...e para cada elemento da matriz,
            previsao = prever(matriz[i][:-1], pesos) # pegamos a previsão...
            erro = matriz[i][-1] - previsao # ...descontamos o erro...
            if verbose:
                sys.stdout.write("Treinando com dados do índice %d...\n"%(i))
            for j in range(len(pesos)): #...calculamos novos pesos para cada nó...
                if verbose:
                    sys.stdout.write("\tPeso [%d]: %0.5f -->"%(j, pesos[j]))
                pesos[j] = pesos[j]+(taxa_l * erro * matriz[i][j])
                if verbose:
                    sys.stdout.write("%05.f\n"%(pesos[j]))
            
    plotar(matriz, pesos, title="Época Final") # ...e plotamos o gráfico final.
    return pesos

def main():
    num_epocas = 10
    taxa_l = 1.0
    plotar_cada_epoca = False
    parar_antes = True

    parte_A = True

    if parte_A: # 3 entradas (incluindo bias único), 3 pesos
        matriz = [[1.00, 0.08, 0.72, 1.0],[1.00, 0.10, 1.00, 0.0],[1.00, 0.26, 0.58, 1.0],[1.00, 0.35, 0.95, 0.0],[1.00, 0.45, 0.15, 1.0],[1.00, 0.60, 0.30, 1.0],[1.00, 0.70, 0.65, 0.0],[1.00, 0.92, 0.45, 0.0]]
        pesos = [0.20, 1.00, -1.00]

    else: # 2 entradas (incluindo bias único), 2 pesos
        num_epocas = 1000
        matriz = [[1.00, 0.08, 1.0],[1.00, 0.10, 0.0],[1.00, 0.26, 1.0],[1.00, 0.35, 0.0],[1.00, 0.45, 1.0],[1.00, 0.60, 1.0],[1.00, 0.70, 0.0],[1.00, 0.92, 0.0]]
        pesos = [0.20, 1.00]
        
    treinar_pesos(matriz,pesos=pesos,num_epocas=num_epocas,taxa_l=taxa_l,plota=plotar_cada_epoca,parar_antes=parar_antes)

if __name__ == "__main__":
    main()