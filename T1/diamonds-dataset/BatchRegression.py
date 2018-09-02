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
            for l in range(self.linhas):
                for l in range(self.linhas):
                    xi = treino[l:l+1]
                    yi = label[l:l+1]
                    gradientes += xi * (xi.dot(thetas) - yi)
                thetas = thetas - ( (learnRate/linhas) * gradientes)

    def predict(self, Data):
        return coef * Data