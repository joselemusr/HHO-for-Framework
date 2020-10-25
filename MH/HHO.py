from MH.Metaheuristica import Metaheuristica
import numpy as np
from DTO.IndicadoresMH import IndicadoresMH
from DTO import TipoIndicadoresMH
import math

class HHO(Metaheuristica):
    def __init__(self):
        self.problema = None
        self.soluciones = None
        self.parametros = {}
        self.idxMejorSolucion = None
        self.mejoraPorSol = None
        self.mejoraPorSolAcumulada = None
        self.mejorSolHistorica = None
        self.mejorFitHistorica = None
        self.fitnessAnterior = None
        self.IteracionActual = None
        print(f"Mh HHO creada")
        

    
    def setIteracionActual(self, IteracionActual):
        self.IteracionActual = IteracionActual

    def getIteracionActual(self):
        return self.IteracionActual

    def setProblema(self, problema):
        self.problema = problema

    def getProblema(self):
        return self.problema

    def generarPoblacion(self, numero):
        self.soluciones = self.problema.generarSoluciones(numero)
    
    def setParametros(self, parametros):
        for parametro in parametros:
            self.parametros[parametro] = parametros[parametro]
    
    def getParametros(self):
        return self.parametros

    def realizarBusqueda(self):
        self._perturbarSoluciones()
        fitness = self.problema.evaluarFitness(self.problema.decode(self.soluciones))
        assert self.soluciones.shape[0] == fitness.shape[0], "El numero de fitness es diferente al numero de soluciones"
        if self.fitnessAnterior is None: self.fitnessAnterior = fitness
        self.idxMejorSolucion = self.problema.getMejorIdx(fitness)
        self.mejoraPorSol = self.problema.getIndsMejora(self.fitnessAnterior, fitness)
        assert self.soluciones.shape[0] == self.mejoraPorSol.shape[0], "El numero de indices de mejora es diferente al numero de soluciones"
        if self.mejoraPorSolAcumulada is None: self.mejoraPorSolAcumulada = np.zeros((self.soluciones.shape[0]))
        self.mejoraPorSolAcumulada += self.mejoraPorSol
        if self.mejorSolHistorica is None: self.mejorSolHistorica = self.soluciones
        if self.mejorFitHistorica is None: self.mejorFitHistorica = fitness

        mejorIdx = self.problema.getIndsMejora(self.mejorFitHistorica, fitness) > 0
        self.mejorSolHistorica[mejorIdx] = self.soluciones[mejorIdx]
        self.mejorFitHistorica[mejorIdx] = fitness[mejorIdx]


        self.fitnessAnterior = fitness

        
        #minFitness = np.min(fitness)
        #indicadorFitness = IndicadoresMH()
        #indicadorFitness.setNombre(TipoIndicadoresMH.FITNESS)
        #indicadorFitness.setValor(minFitness)
        #indicadorMejora = IndicadoresMH()
        #indicadorMejora.setNombre(TipoIndicadoresMH.INDICE_MEJORA)
        #indicadorMejora.setValor(self.problema.getIndiceMejora())
        #self.indicadores = [indicadorFitness, indicadorMejora]
        self.indicadores = {
            TipoIndicadoresMH.INDICE_MEJORA:self.problema.getIndiceMejora()
            ,TipoIndicadoresMH.FITNESS_MEJOR_GLOBAL:self.problema.getMejorEvaluacion()
            ,TipoIndicadoresMH.FITNESS_MEJOR_ITERACION:fitness[self.idxMejorSolucion]
            ,TipoIndicadoresMH.FITNESS_PROMEDIO:np.mean(fitness)
        }

    def _perturbarSoluciones(self):
        #Evaluar si "normizar" las soluciones entre valores min y max (-10,10)
        #X[i,:]=numpy.clip(X[i,:], lb, ub)

        E0 = np.random.uniform(low=-1.0,high=1.0,size=self.soluciones.shape[0]) #vector de tam Pob
        print(f'self.IteracionActual: {self.IteracionActual}')
        print(f'self.getParametros()[HHO.NUM_ITER]: {self.getParametros()[HHO.NUM_ITER]}')
        E = 2 * E0 * (1-(self.IteracionActual/self.getParametros()[HHO.NUM_ITER])) #vector de tam Pob
        Eabs = np.abs(E)
        
        q = np.random.uniform(low=0.0,high=1.0,size=self.soluciones.shape[0]) #vector de tam Pob
        r = np.random.uniform(low=0.0,high=1.0,size=self.soluciones.shape[0]) #vector de tam Pob
        
        LB = -10 #Limite inferior de los valores continuos
        UB = 10 #Limite superior de los valores continuos

        Xm = np.mean(self.soluciones,axis=0)

        beta=1.5 #Escalar según paper
        sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) #Escalar
        if self.idxMejorSolucion is None:
            self.idxMejorSolucion = 0


        if np.min(Eabs) >= 1:
            if np.min(q) >= 0.5: # ecu 1.1
                indexCond11 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.1
                Xrand = self.soluciones[np.random.randint(low=0, high=self.soluciones.shape[0], size=indexCond11.shape[0])] #Me entrega un conjunto de soluciones rand de tam indexCond11.shape[0] (osea los que cumplen la cond11)
                self.soluciones[indexCond11] = Xrand - np.multiply(np.random.uniform(low= 0.0, high=1.0, size=indexCond11.shape[0]),np.abs(Xrand-(2*np.multiply(np.random.uniform(low= 0.0, high=1.0, size = indexCond11.shape[0]),self.soluciones[indexCond11])))) #Aplico la ecu 1.1 solamente a las que cumplen las condiciones np.argwhere(Eabs>=1),np.argwhere(q>=0.5)

            else: # ecu 1.2
                indexCond12 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q<0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.2
                
                self.soluciones[indexCond12] = (self.soluciones[self.idxMejorSolucion] - Xm)- np.multiply( np.random.uniform(low= 0.0, high=1.0, size = indexCond12.shape[0]), (LB + np.random.uniform(low= 0.0, high=1.0, size = indexCond12.shape[0]) * (UB-LB)) )
        else:
            if np.min(Eabs) >= 0.5:
                if np.min(r) >= 0.5: # ecu 4
                    indexCond4 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 4
                    self.soluciones[indexCond4] = (self.soluciones[self.idxMejorSolucion] - self.soluciones[indexCond4]) - np.multiply( E[indexCond4], np.abs( np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond4.shape[0])), self.soluciones[self.idxMejorSolucion] )- self.soluciones[indexCond4] ) )                
                else: #ecu 10
                    indexCond10 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10
                    #ecu 7
                    y10 = self.soluciones
                    y10[indexCond10] = y10[self.idxMejorSolucion]- np.multiply( E[indexCond10], np.abs( np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond10.shape[0])), y10[self.idxMejorSolucion] )- y10[indexCond10] ) )  

                    #ecu 8
                    z10 = y10
                    S = np.random.uniform(low= 0.0, high=1.0, size=(y10.shape))
                    LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y10.shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y10.shape))),(1/beta)))
                    z10[indexCond10] = y10[indexCond10] + S[indexCond10]*LF

                    #evaluar fitness de ecu 7 y 8
                    Fy10 = self.fitnessAnterior
                    Fy10[indexCond10] = self.problema.evaluarFitness(self.problema.decode(y10[indexCond10]))
                    Fz10 = self.fitnessAnterior
                    Fz10[indexCond10] = self.problema.evaluarFitness(self.problema.decode(z10[indexCond10]))                    
                    
                    #ecu 10.1
                    indexCond101 = np.intersect1d(indexCond10, np.argwhere(Fy10 < self.fitnessAnterior)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.1
                    self.soluciones[indexCond101] = y10[indexCond101]

                    #ecu 10.2
                    indexCond102 = np.intersect1d(indexCond10, np.argwhere(Fz10 < self.fitnessAnterior)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.2
                    self.soluciones[indexCond102] = z10[indexCond102]
            else:
                if np.min(r) >= 0.5: # ecu 6
                    indexCond6 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 6
                    self.soluciones[indexCond6] = self.soluciones[self.idxMejorSolucion]- np.multiply(E[indexCond6], np.abs(self.soluciones[self.idxMejorSolucion] - self.soluciones[indexCond6] ) )

                else: #ecu 11
                    indexCond11 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11
                    #ecu 12
                    y11 = self.soluciones
                    array_Xm = np.zeros(self.soluciones[indexCond11].shape)
                    array_Xm = array_Xm + Xm

                    array_E = np.zeros(self.soluciones[indexCond11].shape)
                    array_E = array_E + E[indexCond11]
                    print(f'y11[self.idxMejorSolucion].shape: {y11[self.idxMejorSolucion].shape}')
                    print(f'E[indexCond11].shape: {E[indexCond11].shape}')
                    print(f'np.random.uniform(low= 0.0, high=1.0, size=indexCond11.shape[0]).shape: {np.random.uniform(low= 0.0, high=1.0, size=indexCond11.shape[0]).shape}')
                    print(f'y11[self.idxMejorSolucion].shape: {y11[self.idxMejorSolucion].shape}')
                    print(f'array_Xm.shape: {array_Xm.shape}')

                    y11[indexCond11] = y11[self.idxMejorSolucion]-  np.multiply(  array_E,  np.abs(  np.multiply(  2*(1-np.random.uniform(low= 0.0, high=1.0, size=self.soluciones[indexCond11].shape)),  y11[self.idxMejorSolucion]  )- array_Xm ) )

                    #ecu 13
                    z11 = y11
                    S = np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))
                    LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y11.shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))),(1/beta)))
                    z11[indexCond11] = y11[indexCond11] + S[indexCond11]*LF

                    #evaluar fitness de ecu 12 y 13
                    Fy11 = self.fitnessAnterior
                    Fy11[indexCond11] = self.problema.evaluarFitness(self.problema.decode(y11[indexCond11]))
                    Fz11 = self.fitnessAnterior
                    Fz11[indexCond11] = self.problema.evaluarFitness(self.problema.decode(z11[indexCond11]))                    
                    
                    #ecu 11.1
                    indexCond111 = np.intersect1d(indexCond11, np.argwhere(Fy11 < self.fitnessAnterior)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.1
                    self.soluciones[indexCond111] = y11[indexCond111]

                    #ecu 11.2
                    indexCond112 = np.intersect1d(indexCond10, np.argwhere(Fz11 < self.fitnessAnterior)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.2
                    self.soluciones[indexCond112] = z11[indexCond112]

    def getIndicadores(self):
        return self.indicadores
        