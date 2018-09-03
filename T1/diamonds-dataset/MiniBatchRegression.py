import numpy as np
import pandas as pd
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error

def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

def inicializacaoLabels(labels):
	return labels.reshape(labels.shape[0], 1)

class MiniBatchRegression:
    def __init__(self, max_iter=1000, eta0=0.01, batchSize=10):
        self.interacoes = max_iter
        self.eta0 = eta0
        self.coef = []
        self.batchSize = batchSize

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        linhas, colunas = treino.shape
        thetas = np.random.randn(colunas,1)
        labels = inicializacaoLabels(labels)
        erroHistorico = []
    
        for interacao in range(self.interacoes):
            elem_index = np.random.randint(linhas)
            elemFinal = elem_index+self.batchSize if elem_index+self.batchSize < linhas else linhas
            xi = treino[elem_index:elemFinal]
            yi = labels[elem_index:elemFinal]
            gradientes = xi.T.dot(xi.dot(thetas) - yi) / self.batchSize
            thetas = thetas - (self.eta0 * gradientes)
            self.coef = thetas
            hxs = safe_sparse_dot(treino, self.coef)
            erroHistorico.append(mean_squared_error(hxs, labels))

        return erroHistorico

    def predict(self, Data):
        return coef * Data