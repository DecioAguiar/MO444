import numpy as np
import pandas as pd

def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

class BatchRegression:
    def __init__(self, max_iter=1000, eta0=0.01):
        self.interacoes = max_iter
        self.eta0 = eta0
        self.coef = []

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        linhas, colunas = treino.shape
        thetas = np.zeros((colunas))
        
        for interacao in range(self.interacoes):
            for l in range(linhas):
                gradientes = None
                for l in range(linhas):
                    xi = treino[l, 0:]
                    yi = labels[l]
                    if gradientes is None:
                        gradientes = xi * (xi.dot(thetas) - yi)
                    else:
                        gradientes += xi * (xi.dot(thetas) - yi)
                thetas = thetas - ( (self.eta0/linhas) * gradientes)
                self.coef = thetas

    def predict(self, Data):
        return coef * Data