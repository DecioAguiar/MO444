import numpy as np
import pandas as pd
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error

def inicializacaoTreino(treino):
    coluna_theta = pd.DataFrame(np.ones(treino.shape[0]), columns=['theta'])
    treino = pd.concat([coluna_theta, treino], axis=1)
    return treino.values

class NormalRegression:
    def __init__(self):
        self.coef = []
        self.erroHistorico = []

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        linhas, colunas = treino.shape
        thetas = np.random.randn(colunas,1)
        thetas = np.linalg.inv(treino.T.dot(treino)).dot(treino.T).dot(labels)
        hxs = safe_sparse_dot(treino, thetas)
        self.erroHistorico.append(mean_squared_error(hxs, labels))
        self.coef = thetas


    def predict(self, Data):
        return coef * Data