Lab 1: MLP (README)

Pour utiliser ce programme, vous aurez besoin de ces dépendances :

    -Python 3
    -Numpy (pip install numpy)
    -Flask (pip install flask)
    -Une connection internet pour l'interface.
##################################################################################################
Pour démarer l'application, écrivez cette commande dans le terminal:
    -python main.py
    

Une page web contenant l'interface devrait ouvrir. Ceci vous permettra d'interagir avec le MLP.
L'interface devrait contenir des paramètres chargés par défaut. 
##################################################################################################
Voici une listes de tout les parametres dans l'interface:
    -Data Entraine: Fichier txt contenant les données à utiliser pour l'apprentissage
    -Data VC: Fichier txt contenant les données à utiliser pour la validation croisée.
    -n (eta): Taux d'apprentissage
    -Neurones/CC: Nombre de neurones par couche cachée. 
                  Ceci vous permettra aussi de spécifier le nombre de couches cachées de votre MLP.
                  Par exemple pour spécifier un MLP avec 2 couches cachées avec 50 neurones dans la 
                  premiere couche et 20 neurones dans la deuxième, vous devriez écrire : 50, 20
    -Fct d'activation: La fonction d'activation. Par defaut c'est la sigmoid
    -Base de donnée: Nombre de fenetres a prendre. Ceci specifie aussi le nombre d'entrées du MLP (Base de données * 26)
    -Nb d'époches
    -Sorties Potentiels: Comment encoder la sortie du MLP. Eg.: 'o': [0,0,0,0,0,0,0,0,0,1],
                                                                '1': [0,0,0,0,0,0,0,0,1,0],
                                                                '2': [0,0,0,0,0,0,0,1,0,0],
                                                                '3': [0,0,0,0,0,0,1,0,0,0],
                                                                '4': [0,0,0,0,0,1,0,0,0,0],
                                                                '5': [0,0,0,0,1,0,0,0,0,0],
                                                                '6': [0,0,0,1,0,0,0,0,0,0],
                                                                '7': [0,0,1,0,0,0,0,0,0,0],
                                                                '8': [0,1,0,0,0,0,0,0,0,0],
                                                                '9': [1,0,0,0,0,0,0,0,0,0]
    -n adaptif: Est ce qu'on entraine avec un taux d'apprentissage adaptif
    -Ajout de bruit: On double le nombre de données d'entrainement en ajoutant 
                     du bruit au données qui nous on était fournit.
    -Ouvrir (l'envlope):  Ouvrir un MLP existant 
    -Sauvgarder (Disquette) : Sauvgarder un MLP existant
    -Data à tester: Fichier txt contenant les données à utiliser pour tester la performance du MLP

    ** Pour creer un nouveau MLP, vous n'avez qu'a rafraichir la page ***

##################################################################################################
Voici une liste de tout les fichier et repertoires qui sont important pour l'utilisation du projet:
    -mlps_sauvgarde : Les mlp qui ont étaient sauvgardé sont mis la par defaut
    -default_config.ini: Fichier de configuration
    -main.py : Fichier du programme
    -templates/index.html: interface
    
