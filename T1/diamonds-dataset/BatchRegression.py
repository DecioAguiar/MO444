import numpy as np
import pandas as pd

def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

def inicializacaoLabels(labels):
	return labels.reshape(labels.shape[0], 1)

class BatchRegression:
    def __init__(self, max_iter=1000, eta0=0.01):
        self.interacoes = max_iter
        self.eta0 = eta0
        self.coef = []

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        labels = inicializacaoLabels(labels)
        linhas, colunas = treino.shape
        thetas = np.random.randn(colunas,1)

        for interacao in range(self.interacoes):
            gradientes =  treino.T.dot( treino.dot(thetas) - labels) / linhas
            thetas = thetas - self.eta0 * gradientes
            self.coef = thetas

    def predict(self, Data):
        return coef * Data