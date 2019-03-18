
import numpy as np
from profilestats import profile

class FonctionsActivation(object):

    #https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

    @staticmethod
    def sigmoid(X, derive):
        if derive:
            return (1/(1+np.e**(-X)))*(1 - (1/(1+np.e**(-X))))
        return 1/(1+np.e**(-X))
    
    @staticmethod
    def tanh(X, derive):
        if derive:
            return 1-(np.tanh(X)**2)
        return np.tanh(X)
   
    @staticmethod
    def softplus(X, derive):
        if derive:
            return 1/(1+np.e**(-X))
        return np.log(1 + np.e**X)
    
    @staticmethod
    def relu(X, derive):
        if derive:
            return np.where(X < 0, 0, 1)
        return np.where(X < 0, 0, X)

    @staticmethod
    def leakyrelu(X, derive, a=0.01):
        if derive:
            return np.where(X > 0, 1, a)
        return np.where(X > 0, X, a*X)      

    @staticmethod
    def  sinus(X, derive):
        if derive:
            return np.cos(X * np.pi / 180., dtype=np.double)
        return np.sin(X* np.pi / 180., dtype=np.double)




fonctions = {"sigmoid": FonctionsActivation.sigmoid,
             "tanh": FonctionsActivation.tanh,
              "softplus": FonctionsActivation.softplus,
              "relu": FonctionsActivation.relu,
              "leakyrelu":FonctionsActivation.leakyrelu,
              "sinus": FonctionsActivation.sinus}

class FonctionActivation(object):
    #@profile(print_stats = 10)
    def __new__(self, X, fct, derive=False):
        return fonctions[fct](X, derive)

if __name__ == "__main__":
    print(type(FonctionActivation(np.array([-4,3,2,1]), "relu" )))



