import numpy as np
import pandas as pd

def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

class SGDRegression:
    def __init__(self, max_iter=1000, eta0=0.01):
        self.interacoes = max_iter
        self.eta0 = eta0
        self.coef = []

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        linhas, colunas = treino.shape
        thetas = np.zeros((colunas))
    
        for interacao in range(self.interacoes):
            for i in range(linhas):
                elem_index = np.random.randint(linhas)
                xi = treino[elem_index,0:]
                yi = labels[elem_index]
                gradientes = xi * (thetas.T.dot(xi) - yi)
                thetas = thetas - (self.eta0 * gradientes)
                self.coef = thetas

    def predict(self, Data):
        return coef * Data