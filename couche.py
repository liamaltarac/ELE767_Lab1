from poids import Poids
from neurone import Neurone

class Couche(object):

    def __init__(self, numEntrees, numNeurones, coucheSortie = False, eta = 0.1, fctAct = "sigmoid"):
        self.neurones = [Neurone(numEntrees,
                                uniteSortie = coucheSortie, 
                                eta = eta,
                                fctAct=fctAct) for i in range(numNeurones)]
        self.entrees = []
        self.sorties = []
        self.coucheSortie = coucheSortie
        self.numEntrees = numEntrees

    def setEntrees(self, valeurs):
        self.entrees = valeurs
        for neuron in self.neurones:
            neuron.setEntrees(self.entrees)

    def getSortie(self):
        return self.sorties 
    
    def setSortiesDesire(self, sortiesDesire):
        self.sortiesDesire = sortiesDesire
        if self.coucheSortie:
            for (i,sortieDesire) in enumerate(sortiesDesire):
                self.neurones[i].sortieDesire = sortieDesire

    def calculSorties(self):
        for neurone in self.neurones:
            neurone.calculSortie()
        self.updateSortie()  

    def activerNeurons(self):
        for neurone in self.neurones:
            neurone.calculActivation()
        self.updateSortie()

    def calculSignauxErreur(self, prochaineCouche = None):
        self.updateSortie()
        if self.coucheSortie:
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

                raise("")

    
    def correction(self):
        for neurone in self.neurones:
            neurone.correction()
    
    def actualisation(self):
        for neurone in self.neurones:
            neurone.actualisation()    

    def updateSortie(self):
        self.sorties = [neuron.getSortie() for neuron in self.neurones]
    

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
    