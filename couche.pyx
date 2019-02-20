from mlp_math import FonctionActivation
import numpy as np


class Couche(object):

    def __init__(self, numEntrees, numNeurones, coucheSortie = False, eta = 0.1, fctAct = "sigmoid", poids = None):

        self.entrees = np.zeros(numEntrees)
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
            self.poids = np.random.uniform(-0.05,  0.05, [self.numEntrees,self.numNeurones])

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
        i = np.zeros(self.numNeurones)
        #print("p ", self.poids)
        #("num neuron", self.numNeurones)
        for neurone in range(self.numNeurones):
            #print("n ", neurone)
            for num_entree,entree in enumerate(self.entrees):
                #print("i ", i)
                if self.numNeurones > 1:
                    #print("p", self.poids)
                    poid = self.poids[num_entree, neurone]
                else:
                    poid = self.poids[num_entree]
                i[neurone] += (entree * poid )
        i += self.seuils


        self.sorties = FonctionActivation(i, self.fctAct)
        #print("activation out ", self.sorties)
        #return self.sorties

    def activerNeurons(self):
        self.i = np.zeros(self.numNeurones)
        #print("p ", self.poids)
        #("num neuron", self.numNeurones)
        #print("poids de cettec couche", self.poids)

        for neurone in range(self.numNeurones):
            for num_entree, entree in enumerate(self.entrees):
                #print("i ", i)
                if self.numNeurones > 1:
                    #print("p", self.poids)
                    poid = self.poids[num_entree, neurone]
                    
                else:
                    #print("onluy one layer")
                    poid = self.poids[num_entree]
                    #print("this poid is", poid)
                self.i[neurone] += (entree * poid )
                #print(entree ,poid)
        #print(self.seuils)
        #print("i pre", self.i)
        self.i += self.seuils
        #print("i calced", self.i)



        self.sorties = FonctionActivation(self.i, self.fctAct)
        #print("activation out ", self.sorties)
        #return self.sorties

    def calculSignauxErreur(self, prochaineCouche = None):
        
        if self.coucheSortie:
            #print("unite sortie")
            self.delta = (self.sortiesDesire - self.sorties)*FonctionActivation(self.i, self.fctAct, derive = True)
            #("output layer , num neurones ", self.numNeurones)
            return
       
        self.delta = np.zeros(self.numNeurones)
        somme = 0
        for neurone in range(self.numNeurones):
            somme = 0
            for neuroneNextCouche in range(prochaineCouche.numNeurones):  #Chaque neurone contien un delta
                if(prochaineCouche.numNeurones <= 1):
                    poidsNextCouche = prochaineCouche.getPoids()[neurone]
                    #print("Here 11")
                else:
                    #print("poids", prochaineCouche.getPoids())
                    poidsNextCouche = prochaineCouche.getPoids()[neurone,neuroneNextCouche]
                    #print(prochaineCouche.delta)

                somme += poidsNextCouche * prochaineCouche.delta[neuroneNextCouche]
            self.delta[neurone] = somme * FonctionActivation(self.i[neurone], self.fctAct, derive = True)


        '''if self.coucheSortie:
            for neurone in self.neurones:
                neurone.calculSignalErreur()
            return
        for num, neurone in enumerate(self.neurones):

            try:
                listePoids = [n.getPoids(numEntree = num) for n in prochaineCouche.neurones]
                listeDeltas = [n.delta for n in prochaineCouche.neurones]
                neurone.calculSignalErreur(poidsNext = listePoids, deltasNext = listeDeltas)
            except:
                print("nb neurones", len(self.neurones))
                print("nb neurones prochaine couche", len(prochaineCouche.neurones))

                raise("")'''

    
    def correction(self):

        self.deltaPoids = np.zeros(self.poids.shape)
        #print("Shape poids : ", self.poids.shape)
        for entree in range(self.numEntrees):
            #try:
            if self.numNeurones <= 1:
                #print("delat size", self.deltaPoids)
                self.deltaPoids[entree] = self.tauxApprentissage * self.delta * self.entrees[entree]
                
            else:
                #print("delat calculee", self.tauxApprentissage * self.delta * self.entrees[entree])
                self.deltaPoids[entree:,] = self.tauxApprentissage * self.delta * self.entrees[entree]
            #except Exception as e:
                #print(e)
                #print(len(self.__entrees))
                #print(self.tauxApprentissage * self.delta * self.__entrees[i])

        '''for neurone in self.neurones:
            neurone.correction()'''
    
    def actualisation(self):
        '''for neurone in self.neurones:
            neurone.actualisation()   '''
        #print("delta poids ", self.deltaPoids)
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
    