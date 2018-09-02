class MiniBatchRegression:
    def __init__(self, max_iter=1000, eta0=0.01, batchSize=10):
        self.interacoes = max_iter
        self.eta0 = eta0
        self.coef = []
        self.batchSize = batchSize

    def fit(self, Data, labels):
        treino = inicializacaoTreino(Data)
        linhas, colunas = treino.shape
        thetas = np.zeros((colunas))
    
        for interacao in range(interacoes):
            for l in range(0, linhas, bathSize):
                for l in range(i, i+bathSize):
                    xi = treino[l:l+1]
                    yi = label[l:l+1]
                    gradientes += xi * (xi.dot(thetas) - yi)
                thetas = thetas - ( (learnRate/linhas) * gradientes)

    def predict(self, Data):
        return coef * Data