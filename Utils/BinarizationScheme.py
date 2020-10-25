from Problema.SCP import SCP


class BinarizationScheme:
    
    def TwoSteps(self, matrixCont, matrixBin, transferFunction, discretizationOperator):

        self.transferFunction = transferFunction
        self.binarizationOperator = binarizationOperator

        self.matrixCont = np.ndarray(matrixCont.shape, dtype=float, buffer=matrixCont)
        self.matrixBin = np.ndarray(matrixBin.shape, dtype=float, buffer=matrixBin)
        self.SolutionRanking = SolutionRanking
        self.bestRow = SolutionRanking[0]

        #output
        self.matrixProbT = np.zeros(self.matrixCont.shape)
        self.matrixBinOut = np.zeros(self.matrixBin.shape)

        #TransferFunction

        #Discretization

        return matrixBinOut





