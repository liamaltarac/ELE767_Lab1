from mlp_math import FonctionActivation
import numpy as np
#from profilestats import profile

#######################################################
##  Fichier : ccouhe.py
##  Auteurs : G. Cavero , L. Frija, S. Mohammed
##  Date de creation : 25 Fev 2019          
##  Description : Ce fichier contient toutes les
##                fonctions qui sont perminant aux couches
##                Tel que : activation, calculSignauxErreur, 
##               actualisation, correction
##
#######################################################

class Couche(object):

    def __init__(self, numEntrees, numNeurones, coucheSortie = False, eta = 0.1,
                 fctAct = "sigmoid", poids = None):

        self.entrees = np.zeros(numEntrees)
        self.numEntrees = numEntrees
        self.sorties = np.zeros(numNeurones)
        self.coucheSortie = coucheSortie
        self.numEntrees = numEntrees
        self.poids = np.array(poids)
        self.numNeurones = numNeurones

        self.i = np.zeros(numNeurones)
        self.delta = np.zeros(numNeurones)

        self.seuils = np.zeros(numNeurones)

        self.fctAct = fctAct

        self.tauxApprentissage = eta

        if poids == None:
            self.poids = np.random.uniform(-0.1,  0.1, [self.numEntrees,self.numNeurones])

    def setEntrees(self, valeurs):
        self.entrees = valeurs

    def getSortie(self):
        return self.sorties 

    def getPoids(self):
        return self.poids

    def setSeuil(self, seuils):
        self.seuils = seuils

    def setSortiesDesire(self, sortiesDesire):

        self.sortiesDesire = sortiesDesire


    def calculSorties(self):
        #Forward propagation

        i = np.zeros(self.numNeurones)

        for neurone in range(self.numNeurones):
            if self.numNeurones > 1:

                poid = self.poids[:, neurone]
            else:
                poid = self.poids
            i[neurone] =  np.sum(self.entrees * poid)              

        i += self.seuils
        self.sorties = FonctionActivation(i, self.fctAct)

    def activerNeurons(self):
        #Fonction d'activation
        self.i = np.zeros(self.numNeurones)

        for neurone in range(self.numNeurones):
            if self.numNeurones > 1:
                poid = self.poids[:, neurone]
            else:
                poid = self.poids
            self.i[neurone] =  np.sum(self.entrees * poid)              
        self.i += self.seuils
        self.sorties = FonctionActivation(self.i, self.fctAct)


    def calculSignauxErreur(self, prochaineCouche = None): #La prochaine couche doit etre specifie pour connaitre ses poids
        
        if self.coucheSortie:
            self.delta = (self.sortiesDesire - self.sorties)*FonctionActivation(self.i, self.fctAct, derive = True)
            return
       
        self.delta = np.zeros(self.numNeurones)
        somme = 0
        for neurone in range(self.numNeurones):
            somme = 0
            for neuroneNextCouche in range(prochaineCouche.numNeurones):  #Chaque neurone contien un delta
                if(prochaineCouche.numNeurones <= 1):
                    poidsNextCouche = prochaineCouche.getPoids()[neurone]
                else:
                    poidsNextCouche = prochaineCouche.getPoids()[neurone,neuroneNextCouche]

                somme += poidsNextCouche * prochaineCouche.delta[neuroneNextCouche]
            self.delta[neurone] = somme * FonctionActivation(self.i[neurone], self.fctAct, derive = True)

    def correction(self):

        self.deltaPoids =  np.empty((0, self.numEntrees), float)

        if self.numNeurones <= 1:
            self.deltaPoids = self.tauxApprentissage * self.delta * self.entrees
        else:
            self.deltaPoids = np.tile(self.entrees, (self.numNeurones,1)).T * self.delta * self.tauxApprentissage


    def actualisation(self):
        self.poids = self.poids + self.deltaPoids 


    def fonctionActivation(self, i , fonction = "sigmoid", derive = False):
        if fonction.lower() == "sigmoid":
            if derive:
                return (1/(1+np.e**(-i)))*(1 - (1/(1+np.e**(-i))))
            return 1/(1+np.e**(-i))

    def softmax(self, X, prob=False):
        softmax_prob  = (np.e**X)/np.sum(np.e**X) 
        softmax_bin = np.zeros(len(X))
        if prob:
            return softmax_prob
        softmax_bin[np.argmax(softmax_prob)] = 1
        return softmax_bin


    



if __name__ == "__main__":

    #Exemple du cours d'un NN (P. 60 PDF CHAP 2 NN)

    #Setup du RN
    inputLayer = Couche(numEntrees = 2, numNeurones = 2, fctAct = "sigmoid")
    inputLayer.setEntrees([1,0])
    inputLayer.neurones[0].setPoids(0,3)
    inputLayer.neurones[0].setPoids(1,6)
    inputLayer.neurones[1].setPoids(0,4)
    inputLayer.neurones[1].setPoids(1,5)
    inputLayer.neurones[0].setSeuil(1)
    inputLayer.neurones[1].setSeuil(0)

    outputLayer = Couche(numEntrees = 2, numNeurones = 1, coucheSortie=True, fctAct = "sigmoid")
    outputLayer.setSortiesDesire(sortiesDesire=[1])
    outputLayer.neurones[0].setPoids(0,2)
    outputLayer.neurones[0].setPoids(1,4)
    outputLayer.neurones[0].setSeuil(-3.92)

    #Etape 1 : Activation des Neurons
    inputLayer.activerNeurons()
    outputLayer.setEntrees(inputLayer.sorties)
    outputLayer.activerNeurons()

    #Etape 2 : Calcule des sigs d'erreurs
    outputLayer.calculSignauxErreur()
    print(outputLayer.neurones[0].delta)

    inputLayer.calculSignauxErreur(prochaineCouche=outputLayer)
    print(inputLayer.neurones[0].uniteSortie)
    print(inputLayer.neurones[0].delta)

    #Etape 3: Correction et Actualisation
    inputLayer.correction()
    inputLayer.actualisation()

    outputLayer.correction()
    print(outputLayer.neurones[0].getPoids())

    #print(outputLayer.neurones[0].deltaPoids)
    