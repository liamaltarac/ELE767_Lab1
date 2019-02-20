from couche2 import Couche
import logging, sys
import numpy as np
debug = True

class MLP(object):
    
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
            numEntrees = neuronesParCC[i]
            self.couches.append(coucheCachee)
        
        coucheSortie = Couche(numEntrees = neuronesParCC[-1],
                                numNeurones = self.numSorties,
                                coucheSortie=True,
                                eta = self.eta,
                                fctAct=fonctionActivation)
        self.couches.append(coucheSortie)

        for i in range(self.numCC):
            print("CC ", i)        
            print("Nb neurones ", len(self.couches[i].neurones))
            print("Nb entrees ", self.couches[i].numEntrees)
            print("Nb entrees par neurone ", self.couches[i].neurones[0].numEntrees)
            print("Nb poids par neurone ", len(self.couches[i].neurones[0].getPoids()))

        print("Csortie ")
        print("Nb neurones ", len(self.couches[-1].neurones))
        print("Nb entrees ", self.couches[-1].numEntrees)
        print("Nb entrees par neurone ", self.couches[-1].neurones[0].numEntrees)
        print("Nb poids par neurone ", len(self.couches[-1].neurones[0].getPoids()))

    def entraine(self, entree, sortieDesire):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux
        tailleEntree = len(entree)
        tailleSortie = len(sortieDesire)

        if type(entree[0]) is not list and type(entree[0]) is not np.ndarray:
            entree = [entree]
            tailleEntree = len(entree)

        if type(sortieDesire[0]) is not list and type(sortieDesire[0]) is not np.ndarray:
            sortieDesire = [sortieDesire]
            tailleSortie = len(sortieDesire)

        if tailleEntree != tailleSortie:
            raise("len(entree) != len(sortieDesire) : Chaque entree doit avoir une sortie desire conrespondante ")
        for i in range(len(entree)):
            if self.numEntrees != len(entree[i]) or self.numSorties != len(sortieDesire[i]):
                raise("self.numEntrees != len(entree) or self.numSorties != len(sortieDesire)")
            print("Input Data ", i+1)
            #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
            #Etape 1 : Activation des Neurons
            #print("Activation de neurones")
            x = entree[i]
            for (j,couche) in enumerate(self.couches):
                couche.setEntrees(x)
                couche.activerNeurons()
                x = np.array(couche.sorties)
            self.couches[-1].setSortiesDesire(sortieDesire[i])

            
            #Etape 2 : Calcule des signaux d'erreurs commencant par la couche de sortie qui n'a pas de prochaine couche
            #print("Sig Erreur")
            self.couches[-1].calculSignauxErreur()
            for j in reversed(range(len(self.couches[0:-1]))):
                self.couches[j].calculSignauxErreur(prochaineCouche = self.couches[j+1])  
                #print("j", j)

            #Etape 3: Correction et Actualisation
            #print("Correction et act couche ",i )
            for couche in self.couches:
                couche.correction()
                couche.actualisation()

            for (n,couche) in enumerate(self.couches):
                #print("Couche ", n)
                for (m,neurone) in enumerate(couche.neurones):
                    pass
                    #print("Neurone " + str(m) + " : " + str(neurone.getPoids()))

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

    def setSeuils(self, couche, seuils):

        '''
        setSeuils(couche, seuils)

        Specifier les seuils d'une couche d'un RN.

        Parametres
        ----------
        couche: int

        seuils : list
                Liste contenant les seuils de chaque neurones de la couche
                Ex.: seuils = [1,2,3,4,5]  --> 5 neurones avec 5 seuils

        '''

        if type(seuils) is not list:
            seuils = [seuils]
        for neurone,seuil in enumerate(seuils):
            self.couches[couche].neurones[neurone].setSeuil(seuil)


    def sauvegarder(self):
        pass
    
    def ouvrirReseauExistant(self, fichier):
        pass

    def test(self, entree):
        #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
        #Etape 1 : Activation des Neurons
        x = entree
        for (j,couche) in enumerate(self.couches):
            couche.setEntrees(x)
            couche.calculSorties()
            x = np.array(couche.sorties)
            couche.updateSortie()
        return self.couches[-1].sorties
    

if __name__ == "__main__":
    
    #Exemple du cours
    '''mlp = MLP(numEntrees=2, numSorties=1, neuronesParCC = [2])
    mlp.setPoids(0, [0,1], [[3,6],[4,5]])
    mlp.setPoids(1, [0], [2,4])

    mlp.setSeuils(0, [1,0])
    mlp.setSeuils(1, -3.92)

    mlp.entraine([1,0], [1])
    print("My out ",mlp.test([1,0]))'''



    '''from random import randint
    import numpy as np

    sortiesDesire ={'o': [0,0,0,0,0,0,0,0,0],
                    '1': [1,0,0,0,0,0,0,0,0],
                    '2': [0,1,0,0,0,0,0,0,0],
                    '3': [0,0,1,0,0,0,0,0,0],
                    '4': [0,0,0,1,0,0,0,0,0],
                    '5': [0,0,0,0,1,0,0,0,0],
                    '6': [0,0,0,0,0,1,0,0,0],
                    '7': [0,0,0,0,0,0,1,0,0],
                    '8': [0,0,0,0,0,0,0,1,0],
                    '9': [0,0,0,0,0,0,0,0,1]}


    mlp = MLP(numEntrees = 1560, numSorties = 9, neuronesParCC = [50,50, 50] )

    f = open("data/data_train.txt", 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    sorties = [sortiesDesire[datas[i].split(":")[0]] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)
    print(len(entrees[5]))

    print(len(entrees))
    print(len(sorties))
   
    
    #entrees = [float(entrees[i]) for i in range(len(entrees[j])) for j in range(len(entrees))]
    print("nb entrees = ", len(entrees))
    
    mlp.entraine(entrees[0:5], sorties[0:5])

    print("Training DIne")

    f = open("data/data_test.txt", 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    sorties = [sortiesDesire[datas[i].split(":")[0]] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)
    for entree in entrees[0:5]:
        print(mlp.test(entree))'''
        
    
    
    '''import sys
    print(sys.path)
    from sklearn.neural_network import MLPClassifier

    sortiesDesire ={'o': [0,0,0,0,0,0,0,0,0],
                    '1': [1,0,0,0,0,0,0,0,0],
                    '2': [0,1,0,0,0,0,0,0,0],
                    '3': [0,0,1,0,0,0,0,0,0],
                    '4': [0,0,0,1,0,0,0,0,0],
                    '5': [0,0,0,0,1,0,0,0,0],
                    '6': [0,0,0,0,0,1,0,0,0],
                    '7': [0,0,0,0,0,0,1,0,0],
                    '8': [0,0,0,0,0,0,0,1,0],
                    '9': [0,0,0,0,0,0,0,0,1]}



    mlp = MLPClassifier(hidden_layer_sizes=(50,50,50,50), max_iter=1000, activation='tanh')  
    f = open("data/data_train.txt", 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    sorties = [sortiesDesire[datas[i].split(":")[0]] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)
    print(len(entrees[5]))

    mlp.fit(entrees, sorties) 

    f = open("data/data_test.txt", 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    sorties = [sortiesDesire[datas[i].split(":")[0]] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)
    print(mlp.predict(entrees[0:8]))'''
   





