from flask import Flask,  Response
from flask import request
from flask import render_template
import json
from werkzeug.utils import secure_filename
import os
import numpy as np

from ele767_mlp_lib import MLP


app = Flask(__name__)
app.mlp = None

UPLOAD_FOLDER = 'UPLOAD_FOLDER'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

'''
'o': [0,0,0,0,0,0,0,0,0,1],
'1': [0,0,0,0,0,0,0,0,1,0],
'2': [0,0,0,0,0,0,0,1,0,0],
'3': [0,0,0,0,0,0,1,0,0,0],
'4': [0,0,0,0,0,1,0,0,0,0],
'5': [0,0,0,0,1,0,0,0,0,0],
'6': [0,0,0,1,0,0,0,0,0,0],
'7': [0,0,1,0,0,0,0,0,0,0],
'8': [0,1,0,0,0,0,0,0,0,0],
'9': [1,0,0,0,0,0,0,0,0,0]
'''

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


@app.route('/')
def test():

    if request.method == "GET":
        return render_template("index.html")
    else:
        return "GOT POST REQUETST"

@app.route('/start_training',methods=['POST'])
def startTraining():
    trainingParams = {}
    if request.method == "POST":
        print(request.form)
        #Nous allons premierement prendre les fichiers d'entrainement et de validation croisé. 
        if 'dataTrain' in request.files:
            file = request.files["dataTrain"]
            trainingParams["data_entrain"] = file
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dataTrainFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        else:
            dataTrainFile = None
        if 'dataVC' in request.files:
            file = request.files['dataVC']
            trainingParams["data_vc"] = file
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dataVCFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        else:
            dataVCFile = None
        print(request.form['eta'])
        eta =  float(request.form['eta'])
        fct = request.form["fct"]
        n_p_cc =  eval("[" + request.form['n_p_cc'] + "]")
        db = int(request.form['db'])
        nb_epoche = int(request.form['nb_epoche'])
        sorties_desire = request.form['sortiesDes']
        ajoutBruit = request.form['ajoutBruit']
        etaAdaptif = request.form['etaAdaptif']

        sorties_desire = eval("{" + sorties_desire + "}")

        nb_sorties = len(list(sorties_desire.values())[0])
        nb_entrees = db * 26
        
        if app.mlp == None:
            print("NEW MLP")
            if etaAdaptif == "True":
                etaAdaptif = True
            else:
                etaAdaptif = False

            app.mlp = MLP(nb_entrees,nb_sorties, neuronesParCC = n_p_cc, eta = eta, sortiePotentielle = sorties_desire, 
                        epoche = 1, etaAdaptif=etaAdaptif)
        trainInput, trainOutput = getES(dataTrainFile, sortiesDesire=sorties_desire)

        boolAjoutBruit = False
        if ajoutBruit == "True":
            boolAjoutBruit = True
        status = {}
        app.mlp.entraine(trainInput, trainOutput, boolAjoutBruit)
        
        if dataVCFile is not None:
            vcIn, vcOut = getES(dataVCFile, sortiesDesire=sorties_desire)
            _, vcPerf = app.mlp.test(vcIn, sortieDesire =  vcOut)
            status["vcDataPerf"] = vcPerf

        status["status"] = "OK"
        status["trainDataPerf"] = app.mlp.performance[-1]
        status["eta"] = app.mlp.eta


        return json.dumps(status)


if __name__ == "__main__":
    app.run(debug=True,  threaded=False)
    print("starting")

