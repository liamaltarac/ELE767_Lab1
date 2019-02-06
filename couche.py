from poids import Poids
from neurone import Neurone

class Couche(object):

    def __init__(self, numEntrees, numDeNeurones, coucheEntree = False, coucheSortie = False, sortiesDesire=None):
        self.neurones = [Neurone(numEntrees,
                                uniteDeEntree = coucheEntree,
                                uniteDeSortie = coucheSortie) for i in range(numDeNeurones)]
        self.entrees = []
        self.sorties = []
        self.coucheEntree = coucheEntree
        self.coucheSortie = coucheSortie
        self.sortiesDesire = sortiesDesire

        if coucheSortie:
            for (i,sortieDesire) in enumerate(sortiesDesire):
                self.neurones[i].sortieDesire = sortieDesire

    def setEntrees(self, valeurs):
        self.entrees = valeurs
        for neuron in self.neurones:
            neuron.setValeurEntree(self.entrees)

    def getSortie(self):
        return self.sorties 

    def activerNeurons(self):
        for neurone in self.neurones:
            neurone.calculSortieActivation()
        self.updateSortie()

    def calculSignauxErreur(self, coucheProchaine = None):
        self.updateSortie()
        if self.coucheSortie:
            for neurone in self.neurones:
                neurone.calculSignalErreur()
            return
        for num, neurone in enumerate(self.neurones):

            listePoids = [n.getPoidsDeEntree(numEntree = num) for n in coucheProchaine.neurones]
            listeDeltas = [n.delta for n in coucheProchaine.neurones]
            neurone.calculSignalErreur(poidsNext = listePoids, deltasNext = listeDeltas)
    
    def correction(self):
        for neurone in self.neurones:
            neurone.correction()
    
    def actualisation(self):
        for neurone in self.neurones:
            neurone.actualisation()    

    def updateSortie(self):
        self.sorties = [neuron.getValeurSortie() for neuron in self.neurones]
    

if __name__ == "__main__":

    #Exemple du cours

    #Setup du RN
    inputLayer = Couche(numEntrees = 2, numDeNeurones = 2, coucheEntree = True)
    inputLayer.setEntrees([1,0])
    inputLayer.neurones[0].setPoidsDeEntree(0,3)
    inputLayer.neurones[0].setPoidsDeEntree(1,6)
    inputLayer.neurones[1].setPoidsDeEntree(0,4)
    inputLayer.neurones[1].setPoidsDeEntree(1,5)
    inputLayer.neurones[0].setSeuil(1)
    inputLayer.neurones[1].setSeuil(0)

    outputLayer = Couche(numEntrees = 2, numDeNeurones = 1, coucheSortie=True, sortiesDesire=[1])
    outputLayer.neurones[0].setPoidsDeEntree(0,2)
    outputLayer.neurones[0].setPoidsDeEntree(1,4)
    outputLayer.neurones[0].setSeuil(-3.92)

    #Etape 1 : Activation des Neurons
    inputLayer.activerNeurons()
    outputLayer.setEntrees(inputLayer.sorties)
    outputLayer.activerNeurons()

    #Etape 2 : Calcule des sigs d'erreurs
    outputLayer.calculSignauxErreur()
    print(outputLayer.neurones[0].delta)

    inputLayer.calculSignauxErreur(coucheProchaine=outputLayer)
    print(inputLayer.neurones[0].uniteDeSortie)
    print(inputLayer.neurones[0].delta)

    #Etape 3: Correction
    inputLayer.correction()
    outputLayer.correction()
    print(outputLayer.neurones[0].deltaPoids)

    #Etape 4: Actualisation
    inputLayer.actualisation()
    outputLayer.actualisation()
    print(outputLayer.neurones[0].getPoidsDeEntree())







    
   


