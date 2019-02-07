from poids import Poids
import itertools
import numpy as np

class Neurone(object):

    def __init__(self, numEntrees, uniteSortie=False, sortieDesire=0, eta = 0.1, fctAct = "sigmoid"):
        self.__entrees = []
        self.__sortie = 0

        self.__poids = list(np.random.rand(numEntrees))
        self.seuil = 0
        self.deltaPoids = 0
        self.uniteSortie = uniteSortie
        if uniteSortie:
            self.sortieDesire = sortieDesire
        self.fctAct = fctAct
        self.i = 0

        self.tauxApprentissage = eta

    def getPoids(self, numEntree=None):
        #Obtenir le poids d'une entree du neurone
        if numEntree is not None:            
            return self.__poids[numEntree]
        return self.__poids
    
    def setPoids(self, numEntree, poids):
         self.__poids[numEntree] = poids
    

    def setEntrees(self, val, numEntree=None):
        #Passer une valeur a une entree du neurone
        if numEntree is not None:   
            self.__entrees[numEntree] =  val
            return
        self.__entrees =  val


    def getEntree(self, numEntree=None):
        #Obtenir une valeur de une entree du neurone
        if numEntree is not None:
            return self.__entrees[numEntree] 
        return self.__entrees


    def setSortie(self, val):
        #Forcer une valeur a une sortie du neurone
        self.__sortie = val
    
    def getSortie(self):
        #Obtenir la sortie calcule d'un neurone
        return  self.__sortie

    def setSeuil(self, val):
        self.seuil = val

    def calculActivation(self):
        for (j,entree) in enumerate(self.__entrees) :
            self.i = self.i + (self.__poids[j] * self.__entrees[j])
        self.i = self.i + self.seuil
        #print("Un i "+str(self.i))
        self.__sortie = self.fonctionActivation(self.i, self.fctAct) 

    def calculSignalErreur(self, poidsNext=None, deltasNext=None):
        if self.uniteSortie:
            self.delta = (self.sortieDesire - self.__sortie)*self.fonctionActivation(self.i, self.fctAct, derive = True)
            return
        sum = 0
        for i in range(len(poidsNext)):
            sum = sum + (poidsNext[i] * deltasNext[i])
        self.delta = sum * self.fonctionActivation(self.i, self.fctAct, derive = True)


    def fonctionActivation(self, i, fonction = "sigmoid" ,derive=False):
        if fonction.lower() == "sigmoid":
            if derive:
                return (1/(1+np.e**(-i)))*(1 - (1/(1+np.e**(-i))))
            return 1/(1+np.e**(-i))

        if fonction.lower() is "tanh":
            pass


    def correction(self):
        self.deltaPoids = []
        for i in range(len(self.__poids)):
             self.deltaPoids.append(self.tauxApprentissage * self.delta * self.__entrees[i])

    def actualisation(self):
        for (i,poids) in enumerate(self.__poids):
            self.__poids[i] = self.__poids[i] +  self.deltaPoids[i]



if __name__ == "__main__":

    n = Neurone(3)
    print("Act: "+ n.fonctionActivation())
    n.setToutesValeursEntree([1,2,3])

    n.calculActivation()
    print(n.getPoids())

    print(n.getToutesValeursSortie())
    


            
            




    


