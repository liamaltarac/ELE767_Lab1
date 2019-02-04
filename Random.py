import random

def RanNumGen(MaxNum):
    random.seed(0)
    Number_List = list(range(1,MaxNum))
    random.shuffle(Number_List)
    return Number_List
