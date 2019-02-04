class Poids(object):

    def __init__(self):
        self.poid = {}

    def ajoutePoids(self, source, destination, weight):
        print(source)
        source = str(source)
        destination = str(destination)
        coord = source + "," + destination
        poid.weight[coord] = weight
        print(weight)

        
if __name__ == "__main__":
    print("starting")
    w = Poids(1)
    w.ajoutePoids(1,2,11)
    print(w.weight)
