from flask import Flask,  Response
from flask import request
from flask import render_template
import json
from werkzeug.utils import secure_filename
import os
import numpy as np

from ele767_mlp_lib import MLP
import webbrowser






def create_app():
    app = Flask(__name__)
    app.mlp = None
    def run_on_start(*args, **argv):
        url = "http://localhost:5000"
        webbrowser.open(url, new = 0, autoraise=True)
        print("opening the webbrowser")
    run_on_start()
    return app
app = create_app()
 
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
        app.mlp = None
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

@app.route('/open_mlp',methods=['POST'])
def openMLP():
    if request.method == "POST":
        print(request.files)
        status = {}
        status["status"] = "OK"

        try:
            if 'mlpFile' in request.files:
                file = request.files["mlpFile"]
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                mlpFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                app.mlp = MLP(fichier_mlp=mlpFile)
                sortiesPotentielle = ""
                for key, val in app.mlp.sortiesPotentielle.items():
                    print(key + ":" + str(val) + "\n")
                    sortiesPotentielle += (key + ":" + str(val) + "\n")
                status["sortiesPotentielle"] = sortiesPotentielle
                print("numEntrees ", app.mlp.numEntrees )
                status["db"] = app.mlp.numEntrees / 26
                status["eta"] = app.mlp.eta
                status["neuronesParCC"] = app.mlp.neuronesParCC
            
            print("Done" + sortiesPotentielle)
        except Exception as e:
            status["status"] = "ERREUR"
            print(e)

        return json.dumps(status)

@app.route('/save_mlp',methods=['POST'])
def saveMLP():
    if request.method == "POST":
        print(request.form)
        fileOut = os.path.join("mlps_sauvgarde", request.form["outputFile"] + ".txt")
        status = {}

        if app.mlp is None:
            status["status"] = "Aucun MLP ouvert"
            return json.dumps(status)

        try:
            app.mlp.exporterMLP(fileOut)
            status["status"] = "Fini : \n\tSauvgardé sous le nom: \n\t "+fileOut

        except Exception as e:
            print(e)    
            status["status"] = "ERREUR"
            
        return json.dumps(status)


@app.route('/test_mlp',methods=['POST'])
def testMLP():
    if request.method == "POST":
        print(request.files)
        status = {}

        #Nous allons premierement prendre les fichiers d'entrainement et de validation croisé. 
        if 'mlpTestFile' in request.files:
            file = request.files["mlpTestFile"]
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            dataTestFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if app.mlp is None:
                status["status"] = "Aucun MLP ouvert"
                return json.dumps(status)
            try:
                
                print(dataTestFile)
                testOut  = np.empty((0,1), int)
                testIn , testOutDes = getES(dataTestFile, sortiesDesire=app.mlp.sortiesPotentielle)
                #print("Test OUT:")
                #print(testOutDes)
                mlp_out, perf = app.mlp.test(testIn, sortieDesire = testOutDes)
                print("MLP_OUT")
                print(mlp_out)
                #print(app.mlp.sortiesPotentielle)
                mlp_out = mlp_out.astype(int)
                sortPotInv = dict([[str(v),k] for k,v in app.mlp.sortiesPotentielle.items()])
                #print(sortPotInv)
                print(len(mlp_out))
                for i in range(len(mlp_out)):
                    print("OUTPUT", sortPotInv[str(list(mlp_out[i]))])
                    testOut = np.append(testOut, ["Echantillon " + str(i) + " : " + sortPotInv[str(list(mlp_out[i]))]])
                #print(testOut)

                status["status"] = "Fin du test"
                status["mlp_out"] = str(np.vstack(testOut))
                print(perf)
                status["perf"] = perf
            except Exception as e:
                status["status"] = "ERREUR"
                print(e)

 
    return json.dumps(status)

if __name__ == "__main__":
    app.run(debug=True,  threaded=False)
    print("starting")

