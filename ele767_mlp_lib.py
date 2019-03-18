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
import re
import os
debug = True


class MLP(object):
    
    def __init__(self, numEntrees = None, numSorties = None, fonctionActivation = "sigmoid", 
                neuronesParCC = [20], eta = 0.1, sortiePotentielle = None, 
                epoche = 1, etaAdaptif = False, perf_VC = 0.75, 
                VCin = None, VCout = None, fichier_mlp = None):   

        self.numEntrees = numEntrees
        print("# entrees ", self.numEntrees)
        self.numSorties = numSorties
        print("# Sorties ", self.numSorties)

        self.fonctionActivation = fonctionActivation
        self.numCC = len(neuronesParCC)
        self.neuronesParCC = neuronesParCC
        self.eta = eta
        self.etaInit = eta
        self.sortiesPotentielle = sortiePotentielle
        self.epoche = epoche
        self.performance  = np.array([])
        self.performanceVC = np.array([])

        self.etaAdaptif = etaAdaptif   



        self.couches = [] #Creer une liste de toute les couches du MLP, commencant par les couches cachees,
                          #et terminant par la couche de sortie.

        self.perf_VC = 0
        self.perf_ENT = 0

        self.VCin = VCin
        self.VCout = VCout
        
        self.totalNumEpoche = 0


        if fichier_mlp is not None:
            seuilsArray = []
            with open(fichier_mlp,'r') as f:
                data = f.read().replace(" ", "").lower()  #On enleve tout les espaces pour eviter d'avoir une erreur

            for line in data.split("\n"):
                if len(line) > 1:
                    print("Part 1 ", line)

                    key = line.split("=")[0]

                    value = line.split("=")[1]
                    print(key)
                    if key == "nb_neurones_par_cc":
                        
                        self.neuronesParCC = eval(value)
                        self.numCC = len(self.neuronesParCC)

                    elif key == "nb_entrees":
                        numEntrees = int(value)
                        self.numEntrees = numEntrees
                    elif key == "nb_sorties":
                        self.numSorties = int(value)
                    elif key == "sortiespotentielles": 
                        self.sortiesPotentielle = eval(value)
        #creation des couches cachees 
        for i in range(self.numCC):
            print("numEntrees ", self.numEntrees)
            coucheCachee = Couche(numEntrees = numEntrees,
                        numNeurones = self.neuronesParCC[i],
                        eta = self.eta,
                        fctAct=fonctionActivation)
            numEntrees = self.neuronesParCC[i]
            self.couches.append(coucheCachee)
        #creation de la couche sortie
        print("Couche Sortie")
        coucheSortie = Couche(numEntrees = self.neuronesParCC[-1],
                                numNeurones = self.numSorties,
                                coucheSortie=True,
                                eta = self.eta,
                                fctAct=fonctionActivation)
        self.couches.append(coucheSortie)

        if fichier_mlp is not None:
            seuilsArray = []
            with open(fichier_mlp,'r') as f:
                data = f.read().replace(" ", "").lower()  #On enleve tout les espaces pour eviter d'avoir une erreur

            for line in data.split("\n"):
                print("Part 2 ", line)
                if len(line) > 1:
                    key = line.split("=")[0]
                    value = line.split("=")[1]

                    if key[0] == "s" and key != "sortiespotentielles":      #Seuil
                        print(key)
                        print(re.findall("\((.*?)\)", key))
                        couche = int(re.findall( "\((.*?)\)", key)[0])
                        self.couches[couche].setSeuil(eval(value))
                    elif key[0] == "w":
                        info = re.findall("\((.*?)\)", key)[0].split(",")
                        couche = int(info[0])
                        src = int(info[1])
                        dst = int(info[2])
                        print(self.couches[couche].poids.shape)
                        self.couches[couche].poids[src, dst] = np.float64(value)

        self.perf_VC_desire = perf_VC
        if self.perf_VC_desire > 1:
            self.perf_VC_desire = 1

        print(self.sortiesPotentielle)
       
    def entraine(self, entree, sortieDesire, ajoutDeDonnees = False, varierEta = False):
        #Premieremnt, nous allons tester si les tableaux entree et sortieDesire contienent des sous tableaux
        #sinon, on les force dans un tableau
        if type(entree[0]) is not list and type(entree[0]) is not np.ndarray: 
            entree = [entree]
            tailleEntree = len(entree)

        if type(sortieDesire[0]) is not list and type(sortieDesire[0]) is not np.ndarray:
            sortieDesire = [sortieDesire]
            tailleSortie = len(sortieDesire)

        if ajoutDeDonnees:
            entreeNoisy = self.ajoutDeBruit(entree)
            entree = np.concatenate((entree, entreeNoisy))
            sortieDesire = np.concatenate((sortieDesire, sortieDesire))
            print(entree.shape)
            print(len(sortieDesire))

        permanantEntree = entree
        permanantSortieDesire = sortieDesire

        tailleEntree = len(entree)
        tailleSortie = len(sortieDesire)

        if tailleEntree != tailleSortie:
            raise("len(entree) != len(sortieDesire) : Chaque entree doit avoir une sortie desire conrespondante ")
        for numEpoche , epoche in enumerate(range(self.epoche)): 
            self.performance = np.append(self.performance, [0])
            print("epoche " ,self.totalNumEpoche)
            entree = permanantEntree 
            
            sortieDesire = permanantSortieDesire 

            for i in range(len(entree)):    
                print("entree ", i)
                
                index = random.randint(0,len(list(entree))-1)
                _entree = entree[index][0:self.numEntrees]
                _sortieDesire = sortieDesire[index]
                entree = np.delete(entree, index,0)
                sortieDesire = np.delete(sortieDesire,index,0)
                
                #Etape 1: Activation
                x = _entree
                for (j,couche) in enumerate(self.couches):
                    couche.setEntrees(x)
                    couche.activerNeurons()
                    x = np.array(couche.sorties)

                #Si apres l'activation la sortie du MLP = sortie desiree, on skip l'entrainement
                if (self.softmax(self.couches[-1].getSortie()) == _sortieDesire).all():
                    self.performance[self.totalNumEpoche] += 1
                    print("performance ", self.performance)
                    continue

                #Etape 2: Signale Erreur
                self.couches[-1].setSortiesDesire(_sortieDesire)
                self.couches[-1].calculSignauxErreur()
                for j in reversed(range(len(self.couches[0:-1]))):
                    self.couches[j].calculSignauxErreur(prochaineCouche = self.couches[j+1])  

                #Etapes 3 et 4 : correction et actualisation
                for couche in self.couches:
                    couche.correction()
                    couche.actualisation()

            #Fin de l'epoche

            #Calucl de la performance avec les donnees de test et avec la validation croisee.
            self.performance[self.totalNumEpoche] = self.performance[self.totalNumEpoche]/tailleEntree
            if self.VCin is not None and self.VCout is not None:
                perfVC = self.test(self.VCin, self.VCout)
                self.performanceVC = np.append(self.performanceVC, [perfVC])
            if self.etaAdaptif:
                 self.eta = self.etaInit * 0.2 ** (self.totalNumEpoche)
                 print("Eta changed")
            self.totalNumEpoche += 1

        print(self.performance[-1])


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

    def exporterMLP(self, fichier):
        repertoire = os.path.dirname(fichier)
        if not os.path.exists(repertoire):
            os.makedirs(repertoire)
        f=open(fichier, "w+")
        f.write("Nb_neurones_par_CC= %s\n" % (str(self.neuronesParCC)))
        f.write("Nb_entrees= %d\n" % (self.numEntrees))
        f.write("Nb_sorties= %d\n" % (self.numSorties))
        f.write("sortiesPotentielles= %s\n" % (str(self.sortiesPotentielle)))
    

        for i, couche in enumerate(self.couches):
            #f.write("couche #%d \n" % (i))
            f.write("S(%d) = %s \n" % (i, str(list(couche.seuils))))
            #f.write("Neurone#%d \n" % (neurone))
            for src in range(np.shape(couche.poids)[0]):
                for dst in range(np.shape(couche.poids)[1]):
                    f.write("W(%d,%d,%d) =%s \n" % (i,src,dst,str(couche.poids[src ,dst])))
        f.close()
    



    def test(self, entrees, sortieDesire = None):

        '''
        test(entrees, sortieDesire = None)

        Tester le MLP et predire le resultat d'une entree

        '''
        #Les neurones de la premiere couche cache vont prendre les entree du MLP comme entrees 
        #Etape 1 : Activation des Neurons
        MLP_out = np.empty((0,self.numSorties), int)
        if type(entrees[0]) is not list and type(entrees[0]) is not np.ndarray:
            entrees = [entrees]
            tailleEntree = len(entrees)

        if type(sortieDesire[0]) is not list and type(sortieDesire[0]) is not np.ndarray:
            sortieDesire = [sortieDesire]
            tailleSortie = len(sortieDesire)
        resultats = []
        performance = 0
        for i, entree in enumerate(entrees):
            #print("Test Sortie Desire: " + str(i)+ " : "+ str(sortieDesire[i]))
            x = entree[0:self.numEntrees]
            for (j,couche) in enumerate(self.couches):
                couche.setEntrees(x)
                couche.calculSorties()
                x = np.array(couche.sorties)
            #print(MLP_out)
            nn_output = self.softmax(self.couches[-1].getSortie())
            print(MLP_out.size)
            print(nn_output.size)

            MLP_out = np.append(MLP_out,[nn_output], axis=0) 
            print(MLP_out[-1].size)

            #resultats.append(MLP_out)
            if sortieDesire != None:
                if (MLP_out[-1] == sortieDesire[i]).all():
                    performance += 1
        return MLP_out, performance/len(entrees)

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
    
    def ajoutDeBruit(self, entree):
        #http://mms.etsmtl.ca/ELE778/Synthese/8-OptimizationforTrainingDeepModels.pdf
        print("Ajout de bruit")
        noisyArray =  np.random.uniform(0.8,  1.2, [len(entree), entree[0].size])
        print("Noisy Array generated")
        return  np.multiply(noisyArray, entree)

        

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
    #TODO: arret aniticipe, ajout de bruit
    
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
    
    mlp = MLP(numEntrees = 1560, numSorties = 10, neuronesParCC = [50,50],
              sortiePotentielle=sortiesDesire, fonctionActivation=fct , 
              epoche= 2, eta=0.1, VCin = VCin, VCout = VCout, etaAdaptif=False)
    print("num cc , ", mlp.numCC)
    #mlp = MLP(fichier_mlp= "weightOut.txt")


    entrees, sorties = getES("data/data_train.txt", sortiesDesire)
   
    #entrees = [float(entrees[i]) for i in range(len(entrees[j])) for j in range(len(entrees))]
    #print("nb entrees = ", len(entrees))
    
    mlp.entraine(entrees, sorties)

    print("Training Done")

    print("Performace ", mlp.test(entrees, sortieDesire=sorties) )


    #mlp.exporterMLP("weightOut.txt")

    #entrees, sorties = getES("data/data_test.txt", sortiesDesire)

    
    #print(mlp.test(entrees, sortieDesire = sorties)[1]) 
        
    
    
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
    
   






