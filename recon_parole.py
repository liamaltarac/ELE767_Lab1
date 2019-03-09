from ele767_mlp_lib import MLP
import numpy as np

class ReconnaissanceParole(object):

    def __init__(self, methode = "MLP"):
        self.methode = methode
        self.rn = MLP()


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
        

    

