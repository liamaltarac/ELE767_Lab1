from couche import Couche
import logging, sys
debug = True

class ReseauNeuronal(object):
    
    def __init__(self, numEntrees, numSorties, fonctionActivation = "sigmoid", neuronesParCC = [20], eta = 0.1):        
        self.numEntrees = numEntrees
        self.numSorties = numSorties
        self.fonctionActivation = fonctionActivation
        self.numCC = len(neuronesParCC)
        self.neuronesParCC = neuronesParCC
        self.eta = eta

        self.couches = [] #Creer une liste de toute les couches du MLP, commencant par les couches cachees,
                          #et terminant par la couche de sortie.

        for i in range(self.numCC):
            coucheCachee = Couche(numEntrees = numEntrees,
                        numNeurones = self.neuronesParCC[i],
                        eta = self.eta,
                        fctAct=fonctionActivation)
            self.couches.append(coucheCachee)
        
        coucheSortie = Couche(numEntrees = numEntrees,
                                numNeurones = self.numSorties,
                                coucheSortie=True,
                                eta = self.eta,
                                fctAct=fonctionActivation)
        self.couches.append(coucheSortie)

    def entraine(self, entree, sortieDesire):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux
        if type(entree[0]) is not list:
            entree = [entree]
        if type(sortieDesire[0]) is not list:
            sortieDesire = [sortieDesire]
        if len(entree) is not len(sortieDesire):
            raise("len(entree) != len(sortieDesire) : Chaque entree doit avoir une sortie desire conrespondante ")
        for i in range(len(entree)):
            if self.numEntrees != len(entree[i]) or self.numSorties != len(sortieDesire[i]):
                raise("self.numEntrees != len(entree) or self.numSorties != len(sortieDesire)")

            #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
            #Etape 1 : Activation des Neurons
            for (j,couche) in enumerate(self.couches):
                couche.setEntrees(entree[i])
                couche.activerNeurons()
                entree[i] = couche.sorties
            self.couches[-1].setSortiesDesire(sortieDesire[i])
            
            #Etape 2 : Calcule des signaux d'erreurs commencant par la couche de sortie qui n'a pas de prochaine couche
            self.couches[-1].calculSignauxErreur()
            for j in reversed(range(len(self.couches[0:-1]))):
                self.couches[j].calculSignauxErreur(prochaineCouche = self.couches[j-1])  

            #Etape 3: Correction et Actualisation
            for couche in self.couches:
                couche.correction()
                couche.actualisation()

            for (n,couche) in enumerate(self.couches):
                print("Couche ", n)
                for (m,neurone) in enumerate(couche.neurones):
                    print("Neurone " + str(m) + " : " + str(neurone.getPoids()))

    def setPoids(self, couche, neurones, poids):

        '''
        setPoids(couche, neurones, poids)

        Specifier le poids des neurones d'un RN.

        Parametres
        ----------
        couche: int
        neurones : list
                Tout les neurones que tu veut modifier.
                len(neurones) = len(poids)
        poids : list
                Liste contenant une liste de poids pour chaque entrees de chaque neurones
                Ex.: poids = [[1,2,3], [4,5,6], [7,8,9]]  --> 3 neurones avec 3 entrees chaques
                Ex.: poids = [1, 3, 4, 5, 7.9] --> 1 neurone avec 4 entrees
                Ex.: poids = [1]  --> 1 neurone avec 1 entree

        '''
        if type(poids) is not list:
            if type(poids) is int:
                poids = [poids]
            else:
                raise("forcerPoids(couche, neurones, poids). Poids doit etre une liste")
        if type(poids[1]) is not list:
            poids = [poids] 

        for i in neurones:
            poidsDeUnNeurone = poids[i]
            for (j,poid) in enumerate(poidsDeUnNeurone):
                self.couches[couche].neurones[i].setPoids(j,poid)

        def setBias(self, couche, neurones, bias):
            pass


    def sauvegarder(self):
        pass
    
    def ouvrirReseauExistant(self, fichier):
        pass
    
    def demarrerEntrainement(self):
        pass

if __name__ == "__main__":
    
    mlp = ReseauNeuronal(numEntrees=2, numSorties=1, neuronesParCC = [2])
    mlp.setPoids(0, [0,1], [[3,6],[4,5]])
    mlp.setPoids(1, [0], [2,4])

    mlp.couches[0].neurones[0].setSeuil(1)
    mlp.couches[0].neurones[1].setSeuil(0)
    mlp.couches[1].neurones[0].setSeuil(-3.92)

    mlp.entraine([1,0], [1])

