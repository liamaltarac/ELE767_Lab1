#from couche2 import Couche
from couche import Couche
import logging, sys
import numpy as np
import random
debug = True

class MLP(object):
    
    def __init__(self, numEntrees, numSorties, fonctionActivation = "sigmoid", neuronesParCC = [20], eta = 0.1, sortiePotentielle = None, epoche = 1):        
        self.numEntrees = numEntrees
        self.numSorties = numSorties
        self.fonctionActivation = fonctionActivation
        self.numCC = len(neuronesParCC)
        self.neuronesParCC = neuronesParCC
        self.eta = eta
        self.sortiesPotentielle = sortiePotentielle
        self.epoche = epoche
        self.performance  = np.zeros(epoche)

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
            print("Nb neurones ", self.couches[i].numNeurones)
            print("Nb entrees ", self.couches[i].numEntrees)
            print("Nb entrees par neurone ", self.couches[i].numEntrees)
            print("Nb poids par neurone ", (self.couches[i].getPoids()))

        print("Csortie ")
        print("Nb neurones ", (self.couches[-1].numNeurones))
        print("Nb entrees ", self.couches[-1].numEntrees)
        print("Nb entrees par neurone ", self.couches[-1].numEntrees)
        print("Nb poids par neurone ", (self.couches[-1].getPoids()))

    def entraine(self, entree, sortieDesire):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux

        permanantEntree = entree
        permanantSortieDesire = sortieDesire

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
        for numEpoche , epoche in enumerate(range(self.epoche)): 
            print("epoche " ,numEpoche)
            entree = permanantEntree 
            sortieDesire = permanantSortieDesire 

            
            for i in range(len(entree)):
                
                index = random.randint(0,len(list(entree))-1)
                _entree = entree[index]
                _sortieDesire = sortieDesire[index]
                np.delete(entree, index)
                np.delete(sortieDesire,index)

                if self.numEntrees != len(_entree) or self.numSorties != len(_sortieDesire):
                    print( len(_entree))
                    raise("self.numEntrees != len(entree) or self.numSorties != len(sortieDesire)")
                #print("Input Data ", i+1)
                #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
                #Etape 1 : Activation des Neurons
                #print("Activation de neurones")
                x = entree[i]


                for (j,couche) in enumerate(self.couches):
                    couche.setEntrees(x)
                    couche.activerNeurons()
                    x = np.array(couche.sorties)
                    #print(couche.getPoids())
                
                if (self.softmax(self.couches[-1].getSortie()) == _sortieDesire).all():
                    self.performance[numEpoche] += 1 
                    print("performance ", self.performance)
                    continue
                if self.performance[numEpoche] >= 7 :
                    return

                self.couches[-1].setSortiesDesire(_sortieDesire)
                #print("smax: ",self.softmax(self.couches[-1].getSortie()))
                #print("sort des: ",sortieDesire[i])
                #print("i", i)

                #print(self.couches[-1].getSortie())

                
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

                #for (n,couche) in enumerate(self.couches):
                    #pass
                    #print("Couche ", n)
                    '''for (m,neurone) in enumerate(couche.neurones):
                        pass'''
                        #print("Neurone " + str(m) + " : " + str(neurone.getPoids()))




    def setPoids(self, couche, poids):

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
        '''if type(poids) is not np.ndarray :
            if type(poids) is int:
                poids = np.array(poids)
                print("cond 1")
            else:
                raise("forcerPoids(couche, neurones, poids). Poids doit etre une liste")
        if type(poids[1]) is not np.ndarray:
            poids = np.array(poids)
            print("cond 2")'''

        self.couches[couche].poids = poids
        print("setting couche : " + str(couche) + " to "+  str(self.couches[couche].poids))

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
        self.couches[couche].setSeuil(seuils)


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
        return self.softmax(self.couches[-1].getSortie())

    def getMeilleurSortie(self, sortie):
        meilleurSortie = 0
        minDifference = 0
        for sortieDecimal, sortieEncode in self.sortiesPotentielle.items():
            difference = np.abs(sortie - sortieEncode)
            if np.sum(minDifference) < np.sum(difference):
                minDifference = difference
                meilleurSortie = sortieDecimal

        return meilleurSortie

    def softmax(self, X, prob=False):
        softmax_prob  = (np.e**X)/np.sum(np.e**X) 
        softmax_bin = np.zeros(len(X))
        if prob:
            return softmax_prob
        softmax_bin[np.argmax(softmax_prob)] = 1
        return softmax_bin
        
        



    

if __name__ == "__main__":
    
    #Exemple du cours
    '''mlp = MLP(numEntrees=2, numSorties=1, neuronesParCC = [2])
    mlp.setPoids(0, np.array([[3,4],
                              [6,5]]))
    mlp.setPoids(1, np.array([2,4]))

    mlp.setSeuils(0, [1,0])
    mlp.setSeuils(1, [-3.92])

    mlp.entraine([1,0], [1])

    print("My out ",mlp.test([1,0]))
    print("MLP Poids couche 0  ", mlp.couches[0].getPoids())
    print("MLP Poids couch 1  ", mlp.couches[1].getPoids())'''

    
    
    #todo: choisir l'entree de l'entainement d'une facon aleatore
    #todo: test le mlp avec data train
    #todo: varie le eta
    #todo: une époche c'est toute les donnée une fois

    from random import randint
    import numpy as np
    
    sortiesDesire ={'o': [1,0,0,0,0,0,0,0,0,0],
                    '1': [0,1,0,0,0,0,0,0,0,0],
                    '2': [0,0,1,0,0,0,0,0,0,0],
                    '3': [0,0,0,1,0,0,0,0,0,0],
                    '4': [0,0,0,0,1,0,0,0,0,0],
                    '5': [0,0,0,0,0,1,0,0,0,0],
                    '6': [0,0,0,0,0,0,1,0,0,0],
                    '7': [0,0,0,0,0,0,0,1,0,0],
                    '8': [0,0,0,0,0,0,0,0,1,0],
                    '9': [0,0,0,0,0,0,0,0,0,1]}

    '''sortiesDesire ={'o': [0,0,0,0],
                    '1': [0,0,0,1],
                    '2': [0,0,1,0],
                    '3': [0,0,1,1],
                    '4': [0,1,0,0],
                    '5': [0,1,0,1],
                    '6': [0,1,1,0],
                    '7': [0,1,1,1],
                    '8': [1,0,0,0],
                    '9': [1,0,0,1]}'''

    fct = "tanh"
    mlp = MLP(numEntrees = 1560, numSorties = 10, neuronesParCC = [100,20],
              sortiePotentielle=sortiesDesire, fonctionActivation=fct , epoche= 30, eta=0.025)

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
    
    mlp.entraine(entrees[0:10], sorties[0:10])

    print("Training Done")

    print("Performace ", mlp.performance )

    f = open("data/data_train.txt", 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    sorties = [sortiesDesire[datas[i].split(":")[0]] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)
    for entree in entrees[0:10]:
        print(mlp.test(entree)) 
        
    
    
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



    mlp = MLPClassifier(hidden_layer_sizes=(50,50,50,50), max_iter=1000, activation='logistic')  
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
    print(mlp.predict(entrees[0:8])) '''
   





