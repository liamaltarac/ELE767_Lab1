from poids import Poids
import itertools
import numpy as np

class Neurone(object):

    def __init__(self, numEntrees, uniteDeEntree=False, uniteDeSortie=False, sortieDesire=0):
        if uniteDeEntree and uniteDeSortie:
            raise("Erreur: Neurone d'entree et de sortie")
        self.__entrees = []
        self.__sortie = 0

        self.__poidsDeEntree = list(np.random.rand(numEntrees))
        self.__poidsDeSortie = []
        self.seuil = 0
        self.deltaPoids = 0
        self.uniteDeEntree = uniteDeEntree
        self.uniteDeSortie = uniteDeSortie
        if uniteDeSortie:
            self.sortieDesire = sortieDesire
        self.SIGMOID = 1
        self.i = 0

        self.tauxApprentissage = 0.1

    def getPoidsDeEntree(self, numEntree=None):
        #Obtenir le poids d'une entree du neurone
        if numEntree is not None:            
            return self.__poidsDeEntree[numEntree]
        return self.__poidsDeEntree
    
    def setPoidsDeEntree(self, numEntree, poids):
         self.__poidsDeEntree[numEntree] = poids
    
    def getToutPoidsDeEntree(self):
        return self.__poidsDeEntree
    
    def setToutPoidsDeEntree(self, poids):
        self.__poidsDeEntree = poids

    def getPoidsDeSortie(self):
        return self.__poidsDeSorties

    def setPoidsDeSortie(self, poids):
        return self.__poidsDeSortie

    def setValeurEntree(self, val, numEntree=None):
        #Passer une valeur a une entree du neurone
        if numEntree:   
            self.__entrees[numEntree] =  val
            return
        self.__entrees =  val


    
    def getValeurEntree(self, numEntree=None):
        #Obtenir une valeur de une entree du neurone
        if numEntree:
            return self.__entrees[numEntree] 
        return self.__entrees

    def setToutesValeursEntree(self, entrees): #Array
        #Passer toute les valeurs a toutes les entrees du neurone d'un coups
        self.__entrees =  entrees
    
    def getToutesValeursEntree(self, numEntree): #Array
        #Obtenir toute les valeurs des entrees du neurone d'un coups
        return self.__entrees

    def setValeurSortie(self, val):
        #Forcer une valeur a une sortie du neurone
        self.__sortie = val
    
    def getValeurSortie(self):
        #Obtenir la sortie calcule d'un
        return  self.__sortie

    def setSeuil(self, val):
        self.seuil = val

    def calculSortieActivation(self):
        for (j,entree) in enumerate(self.__entrees) :
            self.i = self.i + (self.__poidsDeEntree[j] * self.__entrees[j])
        self.i = self.i + self.seuil
        self.__sortie = self.fonctionActivation(self.i, self.SIGMOID) 
    
    def calculSignalErreur(self, poidsNext=None, deltasNext=None):
        if self.uniteDeSortie:
            print(self.uniteDeSortie)
            self.delta = (self.sortieDesire - self.__sortie)*self.fonctionActivation(self.i, self.SIGMOID, derive = True)
            return
        sum = 0
        for i in range(len(poidsNext)):
            sum = sum + (poidsNext[i] * deltasNext[i])

        print("sum = "+str(sum))
        self.delta = sum * self.fonctionActivation(self.i, self.SIGMOID, derive = True)


    def fonctionActivation(self, i, fonction = 1 ,derive=False):
        if fonction is 1 and not derive:
            return 1/(1+np.e**(-i))
        if fonction is 1 and derive:
            return (1/(1+np.e**(-i)))*(1 - (1/(1+np.e**(-i))))

    def correction(self):
        self.deltaPoids = []
        for i in range(len(self.__poidsDeEntree)):
             self.deltaPoids.append(self.tauxApprentissage * self.delta * self.__entrees[i])

    def actualisation(self):
        for (i,poids) in enumerate(self.__poidsDeEntree):
            self.__poidsDeEntree[i] = self.__poidsDeEntree[i] +  self.deltaPoids[i]



if __name__ == "__main__":

    n = Neurone(3)
    print("Act: "+ n.fonctionActivation())
    n.setToutesValeursEntree([1,2,3])

    n.calculSortieActivation()
    print(n.getPoids())



    print(n.getToutesValeursSortie())
    


            
            




    


