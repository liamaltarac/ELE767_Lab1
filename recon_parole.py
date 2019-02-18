from reseau import ReseauNeuronal
import numpy as np

class ReconParole(object):

    def __init__(self, numEntrees, numSorties,
                entreesPossible, sortiesDesires = None, 
                neuronesParCC = [100,100,100], eta = 0.1):

        rn = ReseauNeuronal(numEntrees=numEntrees, numSorties = numSorties, neuronesParCC = neuronesParCC, eta = eta)

    def entraine(self):
        pass

    def reconaitre(self, input):
        pass

    def getAccuracy(self):
        pass


if __name__ == "__main__":
    f = open("data/data_train.txt", 'r')
    data = f.read()
    datas = data.split("\n")
    
    test_data = [(datas[i].split(":")[1]).split(" ") for  i in range(len(datas))]
 
    print(len(test_data[2]))
        

    

