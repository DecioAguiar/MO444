import numpy as np
import pandas as pd
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error

def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

class SGDRegression:
    def __init__(self, max_iter=1000, eta0=0.01):
        self.interacoes = max_iter
        self.eta0 = eta0
        self.coef = []
        self.erroHistorico = []

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        linhas, colunas = treino.shape
        thetas = np.random.randn(colunas,1)
    
        for interacao in range(self.interacoes):
            for i in range(linhas):
                elem_index = np.random.randint(linhas)
                xi = treino[elem_index:elem_index+1]
                yi = labels[elem_index:elem_index+1]
                gradientes = xi.T.dot(xi.dot(thetas) - yi)
                thetas = thetas - (self.eta0 * gradientes)
            hxs = safe_sparse_dot(treino, thetas)
            self.erroHistorico.append(mean_squared_error(hxs, labels))

        self.coef = thetas


    def predict(self, Data):
        return coef * Data