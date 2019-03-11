#######################################################
##  Fichier : ELE767_mlp_lib.py
##  Auteurs : G. Cavero , L. Frija, S. Mohammed
##  Date de creation : 25 Fev 2019          
##  Description : Ce fichier contient toutes les
##                fonctions qui servent a entrainner  
##                un RN MLP comme on a vu en classe
##
##
#######################################################


#from couche2 import Couche
from couche import Couche
import logging, sys
import numpy as np
import random
debug = True


class MLP(object):
    
    def __init__(self, numEntrees, numSorties, fonctionActivation = "sigmoid", 
                neuronesParCC = [20], eta = 0.1, sortiePotentielle = None, 
                epoche = 1, tempsMax = None, adaptive = False, perf_VC = 0.75, 
                VCin = None, VCout = None, condtionArret = None):    

        self.numEntrees = numEntrees
        print("# entrees ", self.numEntrees)
        self.numSorties = numSorties
        print("# Sorties ", self.numSorties)

        self.fonctionActivation = fonctionActivation
        self.numCC = len(neuronesParCC)
        self.neuronesParCC = neuronesParCC
        self.eta = eta
        self.sortiesPotentielle = sortiePotentielle
        self.epoche = epoche
        self.performance  = np.zeros(epoche)
        self.tempsMax = tempsMax     #Temps max specifie le temps maximale alloue a l'entrainement du MLP.
                                     #Si ceci est None, le MLP peut s'entrainer sans cesse. 
        self.adaptive = adaptive     #Si Vrai, le systeme ignore le nombre d'epoches et va jouer avec le nombres de 
                                     #couches et le taux d'aprentissage (eta) pour essayer d'optimiser la performance

        self.couches = [] #Creer une liste de toute les couches du MLP, commencant par les couches cachees,
                          #et terminant par la couche de sortie.

        self.perf_VC = 0
        self.perf_ENT = 0

        self.VCin = VCin
        self.VCout = VCout

        self.conditionArret = condtionArret
        

        if self.adaptive:
            self.numCC = np.mean(self.numCC,  dtype=np.int16)

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

        self.perf_VC_desire = perf_VC
        if self.perf_VC_desire > 1:
            self.perf_VC_desire = 1

        # for i in range(self.numCC):
        #     print("CC ", i)        
        #     print("Nb neurones ", self.couches[i].numNeurones)
        #     print("Nb entrees ", self.couches[i].numEntrees)
        #     print("Nb entrees par neurone ", self.couches[i].numEntrees)
        #     print("Nb poids par neurone ", (self.couches[i].getPoids()))

        # print("Csortie ")
        # print("Nb neurones ", (self.couches[-1].numNeurones))
        # print("Nb entrees ", self.couches[-1].numEntrees)
        # print("Nb entrees par neurone ", self.couches[-1].numEntrees)
        # print("Nb poids par neurone ", (self.couches[-1].getPoids()))
    def entraine(self, entree, sortieDesire):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux

        if type(entree[0]) is not list and type(entree[0]) is not np.ndarray:
            entree = [entree]
            tailleEntree = len(entree)

        if type(sortieDesire[0]) is not list and type(sortieDesire[0]) is not np.ndarray:
            sortieDesire = [sortieDesire]
            tailleSortie = len(sortieDesire)


        permanantEntree = entree
        permanantSortieDesire = sortieDesire

        tailleEntree = len(entree)
        tailleSortie = len(sortieDesire)

        if tailleEntree != tailleSortie:
            raise("len(entree) != len(sortieDesire) : Chaque entree doit avoir une sortie desire conrespondante ")
        for numEpoche , epoche in enumerate(range(self.epoche)): 
            print("epoche " ,numEpoche)

            #print("ALL OUT: ", sortieDesire)
            entree = permanantEntree 
            
            sortieDesire = permanantSortieDesire 

            for i in range(len(entree)):
                
                index = random.randint(0,len(list(entree))-1)
                _entree = entree[index]
                _sortieDesire = sortieDesire[index]
                entree = np.delete(entree, index,0)
                sortieDesire = np.delete(sortieDesire,index,0)
                #print("Sortie desirees", sortieDesire)



                #print("Input Data ", i+1)
                #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
                #Etape 1 : Activation des Neurons
                #print("Activation de neurones")
                x = _entree
                #print("Etape 1 : Activation")
                for (j,couche) in enumerate(self.couches):
                    #print("Couche ", j)
                    couche.setEntrees(x)
                    couche.activerNeurons()
                    #print("Couche num neurones ", couche.numEntrees )
                    x = np.array(couche.sorties)
                    #print(x)
                    #print(couche.getPoids())
                #print(len(self.softmax(self.couches[-1].getSortie())))
                #print(_sortieDesire)
                if (self.softmax(self.couches[-1].getSortie()) == _sortieDesire).all():
                    self.performance[numEpoche] += 1 
                    print("performance ", self.performance)
                    continue
                #if self.performance[numEpoche] >= 7 :
                #    return

                self.couches[-1].setSortiesDesire(_sortieDesire)
                #print("smax: ",self.softmax(self.couches[-1].getSortie()))
                #print("sort des: ",sortieDesire[i])

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

        #self.eta *= 0.9
        '''if self.VCin is not None and self.VCout is not None: #On check si notre classe a des donees 
                                                             #de validation croise. Ceci nous indique si on est en mesure de 
                                                             #faire une apprentisage incrementale.
            _, self.perfVC = self.test(self.VCin , self.VCout)
            if self.adaptive:
                if self.perf_VC < self.perf_VC_desire:
                    self.ajouteCoucheCachee([self.couches[0].numNeurones])'''



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

    def ajouteCoucheCachee(self, nb_neurones_par_cc):
        #Enleve la couche sortie pour inserer une couche cachee
        coucheSortie =  self.couches[-1]
        del self.couches[-1]
        numEntrees = self.couches[-1].numNeurones
        for nb_neurones in nb_neurones_par_cc:
            coucheCachee = Couche(numEntrees = numEntrees,
                        numNeurones = nb_neurones,
                        eta = self.eta,
                        fctAct=self.fonctionActivation)
            numEntrees = nb_neurones
            self.couches.append(coucheCachee) 
        self.couches.append(coucheSortie)

    def sauvegarder(self):
        pass
    
    def ouvrirReseauExistant(self, fichier):
        pass

    def test(self, entrees, sortieDesire = None):
        #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
        #Etape 1 : Activation des Neurons
        if type(entrees[0]) is not list and type(entrees[0]) is not np.ndarray:
            entrees = [entrees]
            tailleEntree = len(entrees)

        if type(sortieDesire[0]) is not list and type(sortieDesire[0]) is not np.ndarray:
            sortieDesire = [sortieDesire]
            tailleSortie = len(sortieDesire)
        resultats = []
        performance = 0
        for i, entree in enumerate(entrees):
            print("Test Sortie Desire: " + str(i)+ " : "+ str(sortieDesire[i]))
            x = entree
            for (j,couche) in enumerate(self.couches):
                couche.setEntrees(x)
                couche.calculSorties()
                x = np.array(couche.sorties)

            MLP_out = self.softmax(self.couches[-1].getSortie()) 
            resultats.append(MLP_out)
            if sortieDesire != None:
                if (MLP_out == sortieDesire[i]).all():
                    performance += 1
        return MLP_out, performance

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
        

def getES(fichier, sortiesDesire):
    f = open(fichier, 'r')
    data = f.read()
    datas = data.split("\n")
    nb_data = len(datas)
    print(sortiesDesire)
    sorties = [sortiesDesire[datas[i].split(":")[0]] for i in range(nb_data)]
    entrees = [(datas[i].split(":")[1]).split(" ") for i in range(nb_data)]
    entrees = [list(filter(None, entree)) for entree in entrees]
    entrees = np.array(entrees)
    entrees = entrees.astype(float)

    return entrees, sorties


if __name__ == "__main__":

    #TODO : Multiple Layers, interface, 
    
    #Exemple du cours
    '''mlp = MLP(numEntrees=2, numSorties=1, neuronesParCC = [2],epoche=1)
    mlp.setPoids(0, np.array([[3,4],
                              [6,5]]))
    mlp.setPoids(1, np.array([2,4]))

    mlp.setSeuils(0, [1,0])
    mlp.setSeuils(1, [-3.92])

    mlp.entraine([1,0], [1])
    print("My out ",mlp.test([1,0]))
    print("MLP Poids couche 0  ", mlp.couches[0].getPoids())
    print("MLP Poids couch 1  ", mlp.couches[1].getPoids())

    print("MLP test sortie  ", mlp.test([1,0]))'''

    #Exemple du cours #2
    '''mlp = MLP(numEntrees=3, numSorties=2, neuronesParCC = [3],epoche=2)
    mlp.setPoids(0, np.array([[0.1,0.2, 0.1],
                              [0.1,0.3,0.5],
                              [0.2, 0.1, 0.05]]))
    mlp.setPoids(1, np.array([[0.3, 0.1],
                             [0.4, 0.3],
                             [0.1, 0.2]]))

    mlp.setSeuils(0, [1,0,1])
    mlp.setSeuils(1, [1,0])

    mlp.entraine([1,2,3], [3,2])
    #mlp.entraine([1,2,3], [3,2])



    print("My out ",mlp.test([1,2,3]))
    print("MLP Poids couche 0  ", mlp.couches[0].getPoids())
    print("MLP Poids couch 1  ", mlp.couches[1].getPoids())

    print("MLP test sortie  ", mlp.test([1,0]))'''

    #Exemple du cours #3
    '''mlp = MLP(numEntrees=3, numSorties=4, neuronesParCC = [3,3],epoche=5, fonctionActivation="sinus")
    mlp.setPoids(0, np.array([[2, 3, 4],
                              [3 ,4, 5],
                              [4, 5, 6]]))
    mlp.setPoids(1, np.array([[2, 3, 4],
                              [3 ,4, 5],
                              [4, 5, 6]]))

    mlp.setPoids(2, np.array([[2, 3, 4, 5],
                              [3 ,4, 5, 6],
                              [4, 5, 6, 7]]))

    mlp.setSeuils(0, [1,2,3])
    mlp.setSeuils(1, [4,5,6])
    mlp.setSeuils(2, [7,8,9,10])

    mlp.entraine([1,2,3], [1,2,3,4])
    #mlp.entraine([1,2,3], [3,2])



    #print("My out ",mlp.test([1,2,3]))
    print("MLP Poids couche 0  ", mlp.couches[0].getPoids())
    print("MLP Poids couch 1  ", mlp.couches[1].getPoids())
    print("MLP Poids couch 2  ", mlp.couches[2].getPoids())'''

    #print("MLP test sortie  ", mlp.test([1,0]))
    
    #todo: choisir l'entree de l'entainement d'une facon aleatore
    #todo: test le mlp avec data train
    #todo: varie le eta
    #todo: une epoche c'est toute les donnee une fois

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


    VCin, VCout = getES("data/data_vc.txt", sortiesDesire)

    fct = "sigmoid"
    
    mlp = MLP(numEntrees = 1560, numSorties = 10, neuronesParCC = [50],
              sortiePotentielle=sortiesDesire, fonctionActivation=fct , 
              epoche= 4, eta=0.1, VCin = VCin, VCout = VCout, adaptive=False)


    entrees, sorties = getES("data/data_train.txt", sortiesDesire)
   
    #entrees = [float(entrees[i]) for i in range(len(entrees[j])) for j in range(len(entrees))]
    print("nb entrees = ", len(entrees))
    
    mlp.entraine(entrees, sorties)

    print("Training Done")

    print("Performace ", mlp.performance )


    entrees, sorties = getES("data/data_test.txt", sortiesDesire)

    
    print(mlp.test(entrees, sortieDesire = sorties)[1]) 
        
    
    
    '''import sys
    print(sys.path)
    from sklearn.neural_network import MLPClassifier

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



    mlp = MLPClassifier(hidden_layer_sizes=(50), max_iter=2000, activation='logistic', learning_rate_init=0.1)  
    entrees, sorties = getES("data/data_train.txt", sortiesDesire)


    mlp.fit(entrees, sorties) 

    entrees, sorties = getES("data/data_test.txt", sortiesDesire)

    mlpOut = mlp.predict(entrees)
    print(mlpOut)
    performance = 0
    for i, sortie in enumerate(mlpOut):
        if (sortie == sorties[i]).all():
            performance+=1
    print(performance)'''
    
   






