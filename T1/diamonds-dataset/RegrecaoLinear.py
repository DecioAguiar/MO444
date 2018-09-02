import pandas as pd
import numpy as np

#treino = matriz elementos x features
#label = target, vetor com tamanho = elementos

#adicona coluna x0 = 1
def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

def RegrecaoLinearEstocastica(treino, label, interacoes, learnRate):
    treino = inicializacaoTreino(treino)
    linhas, colunas = treino.shape
    thetas = np.zeros((colunas,1))
    
    for interacao in range(interacoes):
        for i in range(linhas):
            elem_index = np.random.randint(linhas)
            xi = treino[elem_index:elem_index+1]
            yi = label[elem_index:elem_index+1].values
            #print('xi=',xi)
            #print('thetas=',thetas)
            #print('yi=',yi)
            #print('dot=',xi.dot(thetas) )
            gradientes = xi * (xi.dot(thetas) - yi)
            thetas = thetas - (learnRate * gradientes)
    return thetas
            
def RegrecaoLinearBatch(treino, label, interacoes, learnRate):
    linhas, colunas = inicializacaoTreino(treino).shape
    thetas = np.zeros(colunas)
    
    for interacao in range(interacoes):
        for l in range(linhas):
            for l in range(linhas):
                xi = treino[l:l+1]
                yi = label[l:l+1]
                gradientes += xi * (xi.dot(thetas) - yi)
            thetas = thetas - ( (learnRate/linhas) * gradientes)
            
def RegrecaoLinearMiniBatch(treino, label, interacoes, learnRate, batchSize):
    linhas, colunas = inicializacaoTreino(treino).shape
    thetas = np.zeros(colunas)
    
    for interacao in range(interacoes):
        for l in range(0, linhas, bathSize):
            for l in range(i, i+bathSize):
                xi = treino[l:l+1]
                yi = label[l:l+1]
                gradientes += xi * (xi.dot(thetas) - yi)
            thetas = thetas - ( (learnRate/linhas) * gradientes)
