import numpy as np

class MFCC(object):

    def __init__(self, array, tailleTrame):

        self.statique = []
        self.dynamique = []
        self.energieStatique = []
        self.energieDynamique = []
        self.array = array

        while len(array) > 0:
            self.statique.append(array[:tailleTrame])
            self.energieStatique.append(array[tailleTrame])
            del array[:tailleTrame+1]
            self.dynamique.append(array[:tailleTrame])
            self.energieDynamique.append(array[tailleTrame])            
            del array[:tailleTrame+1]


    def getStatique(self):
        return self.statique

    def getDynamique(self):
        return self.dynamique
    
    def getEnergieStatique(self):
        return self.energieStatique

    def getEnergieDynamique(self):
        return self.energieDynamique
    
