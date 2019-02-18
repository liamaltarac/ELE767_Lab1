import os


def getData(filename):

    script_dir = os.path.dirname(__file__)  # Repertoire absolue du fichier
    rel_path = "data/" + filename  # Repertoire relative du fichier voulu
    abs_file_path = os.path.join(script_dir, rel_path)
    with open(abs_file_path) as f:
        lines = f.read().splitlines()

    i = 0
    datalist = []
    for line in lines:
        #j = 0
        dataline = line.split()
        datalist.append([])
        for word in dataline:
        #for j in xrange(2):
            datalist[i].append(word)
            #j += 1
        #liste[i].append(data[0])
        i += 1
    return datalist

def __main__():
    liste = getData("data_train.txt")


__main__()