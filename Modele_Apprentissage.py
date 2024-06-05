import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt #uniquement utilisé pour la question4
from sklearn.decomposition import PCA #uniquement utilisé pour la question4

######################################################################################################
##################################### I. Préparation des données ##################################### 
######################################################################################################

#Importer les données 
data = pd.read_csv("synthetic.csv")

# Séparer les attributs (X) des étiquettes (y)
X = data.drop(columns=['Class'])  # Attributs: toutes les colonnes sauf 'class'
Y = data['Class']  # Étiquettes: la colonne 'class'

# Transformer X et Y en numpy array
X = X.to_numpy()
Y = Y.to_numpy()

# Appliquer l'encodage one-hot aux étiquettes y
y_encoded = pd.get_dummies(Y)
#print(y_encoded.head())

# Diviser les données en ensembles d'entraînement et de test (20% pour le test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

# Appliquer l'encodage one-hot aux étiquettes y
Y_test = pd.get_dummies(Y_test)

# 1. Combien d’attributs comportent ces données ?
nombre_attributs = data.shape[1]
#print("Nombre d'attributs :", nombre_attributs) #15 attributs dont une classe

#2. En combien de classes différentes les instances sont-elles catégorisées ?
nombre_classes = data['Class'].nunique()
#print("Nombre de classes différentes :", nombre_classes)

#3. Combien d’instances chaque classe compte-elle ?
instances_par_classe = data['Class'].value_counts()
#print("Nombre d'instances par classe :\n", instances_par_classe)

#4. Les données sont-elles linéairement séparables ?
"""
les données ne sont pas linéairement séparables. Selon la figure 1, il est impossible
de tracer une droite qui sépare ces données à cause de leur forte dispersion.
En ayant effectué une analyse en composantes principales, on voit clairement que les données ne sont pas linéairement
séparable puisqu'on ne peut pas tracer de frontière linéaire pour séparer les differentes classes.
"""
x = data.drop('Class', axis = 1)
y = data['Class']

#réduire la dimension
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title('Projection PCA des données')
plt.colorbar(label='Classe')
#plt.show()


#5. L'utilisation de l'encodage one-hot est elle nécessaire ?
"""
L'utilisation de l'encodage one hot est nécessaire pour ce modèle puisque les étiquettes des données sont numérotées :
“le problème de cet encodage en apprentissage automatique, c'est qu'il peut entraîner un biais lors de l'apprentissage du modèle à 
cause de la relation d'ordre" (cf. TD). La normalisation des données est nécessaire pour éviter l’écart entre certaines valeurs.
"""

#6. Rappelez l’intérêt de séparer les données en un jeu d’entrainement et un jeu de test.
"""
On divise les données en 2 groupes : un ensemble d'entraînement pour entrainer le modèle
et un ensemble de test pour s'assurer que le modèle n'a pas simplement appris les données d'entraînement
par coeur et qu'il sait généraliser avec ces données de tests.
"""

##########################################################################################################
##################################### II. Mise en oeuvre des modèles #####################################
########################################################################################################## 

# ---------------------------------- 1. Arbre de décision -----------------------------------------------

def discretiser_avec_quartiles(data, attribut):
    sorted_data = data.sort_values(by=attribut)
    quartiles = sorted_data[attribut].quantile([0.25, 0.5, 0.75])
    """
    for i, quartile in enumerate(quartiles):
        print(f"Quartile {i+1} de {attribut} : {quartile}")
    """
    return quartiles


def creation_partitions(data, attribut, quartile):
    partition1 = data[data[attribut] < quartile]
    partition2 = data[data[attribut] >= quartile]
    return partition1, partition2


def entropie(dataframe, cible):
    nbr_total_instances = len(dataframe)
    occu_classe = dataframe[cible].value_counts()
    entropie = 0
    for i in occu_classe:
        proba = i / nbr_total_instances
        entropie -= proba * math.log2(proba)
    return entropie


def calcul_gain(data, attribut, quartile):
    e = entropie(data, attribut)
    gain = 0
    partitions = creation_partitions(data, attribut, quartile)

    entropy0 = entropie(partitions[0], attribut)
    entropy1 = entropie(partitions[1], attribut)

    if len(partitions[0]) == 0 or len(partitions[1]) == 0:
        return 0

    nbr_instances = len(data)
    gain = e - ((len(partitions[0]) / nbr_instances) * entropy0 + (len(partitions[1]) / nbr_instances) * entropy1)

    return gain


def trouver_meilleur_quartile(data, attribut):
    meilleur_quartile = None
    meilleur_gain = 0

    quartiles = discretiser_avec_quartiles(data, attribut)
    
    for quartile in quartiles :
        if quartile is not None:
            gain = calcul_gain(data, attribut, quartile)
            if gain > meilleur_gain:
                meilleur_gain = gain
                meilleur_quartile = quartile

    return meilleur_quartile

#mq = trouver_meilleur_quartile(data, 'Attr_N')
#print("le meilleur quartile est : ", mq)


def choisir_meilleur_attribut(data, attributs):
    meilleur_gain = 0
    meilleur_attribut = None
    for attribut in attributs:
        gain = calcul_gain(data, attribut, trouver_meilleur_quartile(data, attribut))
        if gain > meilleur_gain:
            meilleur_gain = gain
            meilleur_attribut = attribut
    return meilleur_attribut


class Node:
    def __init__(self, attribute, split_value, right = None, left = None, is_leaf = False, prediction = None):
        self.attribute = attribute
        self.split_value = split_value
        self.right = right
        self.left = left
        self.is_leaf = is_leaf
        self.prediction = prediction #partie de dataframe qui se retrouvent dans le noeuds
        #le 1er noeud contient dans prediction toutes les données

    def afficher_noeud(self, spacing=''):
        if isinstance(self.prediction, pd.Series):
            s = ''
            for v in range(len(self.prediction.values)):
                s += 'Classe ' + str(self.prediction.index[v]) + ' Nombre : ' + str(self.prediction.values[v]) + '\n' + spacing
            return s
        else:
            return f'Classe : {self.prediction}\n{spacing}'


def construire_arbre(data, attributs, profondeur, profondeur_max):

    attributs_restants = attributs[:]

    if len(attributs_restants) == 1 or len(data) == 0 or profondeur == profondeur_max:
        classe_majoritaire = data[data.columns[-1]].mode()[0]
        return Node(attribute = attributs_restants[0], split_value=None, prediction = classe_majoritaire, is_leaf = True)
        
    meilleur_attribut = choisir_meilleur_attribut(data, attributs_restants)
    print(f"Le meilleur attribut de {attributs_restants} : {meilleur_attribut}")
    attributs_restants.remove(meilleur_attribut)
    meilleur_quartile = trouver_meilleur_quartile(data, meilleur_attribut)
    print(f"Le meilleur quartile de {meilleur_attribut} : {meilleur_quartile}")

    partition_gauche, partition_droite = creation_partitions(data, meilleur_attribut, meilleur_quartile)
    gauche = construire_arbre(partition_gauche, attributs_restants, profondeur + 1, profondeur_max)
    droite = construire_arbre(partition_droite, attributs_restants, profondeur + 1, profondeur_max)

    return Node(meilleur_attribut, meilleur_quartile, left = gauche, right = droite)

def afficher_arbre(node, spacing=''):
    if node is None:
        return
    if node.is_leaf:
        print(spacing + node.afficher_noeud(spacing))
        return
    print('{}[Attribut: {} Split value: {}]'.format(spacing, node.attribute, node.split_value))

    print(spacing + '> True')
    afficher_arbre(node.left, spacing + '  ')

    print(spacing + '> False')
    afficher_arbre(node.right, spacing + '  ')

#attributs = data.columns[:-1].toliste()
#arbre = construire_arbre(data, attributs, 0, 4)
#afficher_arbre(arbre)

def test_arbre_decision():
    """
    Fonction qui teste les fonctionnalités de l'arbre de décision
    """

    # Test de la détermination d'un meilleur partitionnement
    meilleur_quartile = trouver_meilleur_quartile(data, 'Attr_A')
    print("Meilleur quartile pour l'attribut A :", meilleur_quartile)

    # Test du calcul du gain
    gain_resultat = calcul_gain(data, 'Attr_A', meilleur_quartile)
    print("Gain de la partition :", gain_resultat)    

#test_arbre_decision()

# ---------------------------------- 2. Réseaux de neurones artificiels -----------------------------------------------

# Diviser les données en ensembles d'entraînement et de validation (15% pour la validation)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

# Appliquer l'encodage one-hot aux étiquettes y
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

class NeuralNet:
    """
    Constructeur de la classe Neural Net
    Args:
        X_train : Les données d'entraînement
        y_train : Les étiquettes d'entraînement
        X_test  : Les données de test
        y_test  : Les étiquettes de test
        hidden_layer_sizes :  La taille de chaque couche cachée. Par défaut, (4,).
        activation : La fonction d'activation à utiliser. Par défaut, 'tanh' 
        learning_rate : Le taux d'apprentissage. Par défaut, 0.01.
        epochs : Le nombre d'époques d'entraînement. Par défaut, 200.
    """
    def __init__(self, X_train=None, Y_train=None, X_test=None, Y_test=None, hidden_layer_sizes = (4,), activation='tanh', learning_rate=0.01, epochs=200):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.n_layers = len(hidden_layer_sizes) +1   # Nombre de couches, y compris la couche de sortie
        self.weights = [None] * (self.n_layers)  # Initialisation de la liste des matrices de poids
        self.biases = [None] * (self.n_layers)   # Initialisation de la liste des vecteurs de biais
        # Initialisation des poids et des biais
        self.__weights_initialization(X_train.shape[1], len(np.unique(Y_train)))
    
    # Initialization des matrices de paramètres (poids et biais)
    def __weights_initialization(self, n_attributes, n_classes):
            # Initialisation des poids et des biais pour chaque couche cachée
            for l in range(self.n_layers - 1):
                # Nombre d'unités dans la couche actuelle
                units_current_layer = self.hidden_layer_sizes[l]
                # Nombre d'unités dans la couche précédente (ou nombre d'attributs pour la première couche cachée)
                units_previous_layer = n_attributes if l == 0 else self.hidden_layer_sizes[l - 1]
                # Initialisation des poids de manière aléatoire dans [-1, 1]
                self.weights[l] = np.random.uniform(low=-1.0, high=1.0, size=(units_current_layer, units_previous_layer))          
                # Initialisation des biais de manière aléatoire dans [-1, 1]
                self.biases[l] = np.random.uniform(low=-1.0, high=1.0, size=(units_current_layer, 1))
            # Initialisation des poids et des biais pour la couche de sortie
            self.weights[-1] = np.random.uniform(low=-1.0, high=1.0, size=(n_classes, self.hidden_layer_sizes[-1]))
            self.biases[-1] = np.random.uniform(low=-1.0, high=1.0, size=(n_classes, 1))
        
    def forward(self, X):
        """
        Effectue la propagation avant à travers le réseau de neurone
        Args:
            X : Les données d'entrée.
        Returns:
            output : Les prédictions du réseau neuronal
        Notes:  
            -"tanh" renvoie  des valeurs entre -1 et 1 
            -"relu"  renvoie des valeurs entre 0 et l'entrée x
        """
        # fonctions d'activation sur les quelles entrainer le modèle
        activation_functions = {
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        }
        # Choisir quelle fonction d'ativation utiliser dans la liste
        activation_function = activation_functions[self.activation]
        #Propagation avant sur la premiere couche        
        Z = np.dot(self.weights[1], X[1].T) + self.biases[1]
        A = activation_function[self.activation](Z)
        # Propagation avant à travers les couches du réseau
        for l in range(2, self.n_layers + 1):
            # Calcul de la sortie de la couche en utilisant les poids et les biais
            Z = np.dot(self.weights[l], A) + self.biases[l]
            # Application de la fonction d'activation à la sortie
            if l < self.n_layers :
                A = activation_function[self.activation](Z)
            else:
                # Application de la fonction softmax pour la derniere couche de sortie
                A = activation_function['softmax'](Z)
            output = A
        # Renvoie des prédictions
        return output
    
    
    def calculate_loss(self, predictions, targets):
        """
        Evalue la différence entre les distributions de probabilité prédites par le modèle et les distributions de probabilité réelles des étiquettes 
        Args:
            predictions : Les prédictions fréalisées par le modèle
            targets : Les valeurs réelles
        Returns:
            float: La valeur de perte (loss)
        Formule entropie: loss=-1/N*∑(i=1àN)yilog(pi)
        """
        print("Taille de predictions :", predictions.shape)
        print("Taille de targets :", targets.shape)
        # Perte de l'entropie croisée (cross-entropy loss)
        # Ajout d'une petite valeur (1e-9) pour éviter les erreurs de division par zéro 
        n  = predictions.shape[1]
        loss = - np.mean (targets * np.log(predictions + 1e-9)) / n 
        return loss


    def backward(self, X, predictions, targets):
        """
        Effectue la rétropropagation du gradient pour mettre à jour les poids et les biais du réseau neuronal
        Args:
            X : Les données d'entrée
            predictions : Les prédictions du modèle
            targets : Les valeurs réelles
        Returns: "erreur" Le gradient pour la couche précédente
        """
        # Calcul de la différence entre les prédictions et les valeurs reeles (E = y'-y)
        function_loss = predictions - targets
        
        # Boucle de rétropropagation à travers les couches du réseau (de la dernière à la première)
        for i in range(self.n_layers - 1, 0, -1):
            # Calcul du gradient par rapport aux poids et biais
            derivate_weights = np.dot(X.T, function_loss)
            derivate_biases = np.sum(function_loss, axis=0)
            
            # Mise à jour des poids et des biais en fontion du taux d'apprentissage et le gradient
            # Formule: w = w - n * dE/dw
            self.weights[i - 1] -= self.learning_rate * derivate_weights
            self.biases[i - 1] -= self.learning_rate * derivate_biases
            
            # Calcul du gradient pour la couche précédente
            function_loss = np.dot(function_loss, self.weights[i - 1].T)
            
            # Backpropagation de la fonction d'activation
            #Formule: tanh: 1−tanh2(x) .... relu: 1 si x>0, 0 sinon
            if self.activation == 'tanh':
                function_loss = function_loss * (1 - np.tanh(np.dot(X, self.weights[i - 1]) + self.biases[i - 1]) ** 2)
            elif self.activation == 'relu':
                function_loss = function_loss * np.where(np.dot(X, self.weights[i - 1]) + self.biases[i - 1] > 0, 1, 0)
        return function_loss


    def train(self):
        """
        Entraîne le modèle et effectue la propagation avant
        calcule la perte, et effectue la rétropropagation du gradient à chaque époque

        Returns:
            None
        """
        # Entraînement du modèle pour chaque époque
        for epoch in range(self.epochs):
            # Propagation avant
            predictions = self.forward(self.X_train)
            # Calcul de la perte
            loss = self.calculate_loss(predictions, self.Y_train)
            
            # Rétropropagation
            self.backward(self.X_train, predictions, self.Y_train)
            
            # Affichage de la perte à chaque epoch
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss}")



#Test d'utilisation
# hidden_layers_tanh = [(10, 8, 6), (10, 8, 4), (6, 4)]
# for architecture in hidden_layers_tanh:
#     model_tanh = NeuralNet(X_train, Y_train, X_test, Y_test, hidden_layer_sizes=architecture, activation='tanh')
#     model_tanh.train()

# # Exemple d'architecture avec activation relu
# hidden_layers_relu = [(10, 8, 6), (10, 8, 4), (6, 4)]
# for architecture in hidden_layers_relu:
#     model_relu = NeuralNet(X_train, Y_train, X_test, Y_test, hidden_layer_sizes=architecture, activation='relu')
#     model_relu.train()

####################################################################################################
##################################### III. Analyse des modèles ##################################### 
####################################################################################################

"""
Etant donné que les modèles ci-dessus n'ont pas donné de prédictions, on a utilisé les prédictions fournies.
Pour cela, on a d'abord calculé pour chaque architecture les métriques globales sur le modèle afin de choisir 
la meilleure architecture. Et ensuite on a calculé les métriques pour chaque classe possible du modèle choisi.
"""

y_test = pd.read_csv('Predictions/y_test.csv', header = None)

# --------------------------------------------------------------------------------------------------
# ---------------------------------- Analyse des Arbres de Décisions -------------------------------
# --------------------------------------------------------------------------------------------------

y_pred_DT4= pd.read_csv('Predictions/y_pred_DT4.csv', header = None)
y_pred_DT5 = pd.read_csv('Predictions/y_pred_DT5.csv', header = None)
y_pred_DT6 = pd.read_csv('Predictions/y_pred_DT6.csv', header = None)

#---------------- évaluation des métriques de chaque modèle  ---------------------------------------

def calculer_metriques_DT(y_true, y_pred):

    exactitude_liste = []
    precision_liste = []
    rappel_liste = []
    f1_liste = []

    for i in range(y_true.shape[1]):
        vrai_positif = 0
        faux_positif = 0
        faux_negatif = 0

        #calcul du nombre de vrais positifs, faux positifs et faux négatifs pour chaque classe
        for true_value, pred_value in zip(y_true.iloc[:, i], y_pred.iloc[:, i]):
            if true_value == pred_value:
                vrai_positif += 1
            elif true_value == i and pred_value != i:
                faux_negatif += 1
            elif true_value != i and pred_value == i:
                faux_positif += 1

        exactitude = vrai_positif / len(y_true)
        precision = vrai_positif / (vrai_positif + faux_positif) if (vrai_positif + faux_positif) != 0 else 0
        rappel = vrai_positif / (vrai_positif + faux_negatif) if (vrai_positif + faux_negatif) != 0 else 0
        f1 = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) != 0 else 0

        exactitude_liste.append(exactitude)
        precision_liste.append(precision)
        rappel_liste.append(rappel)
        f1_liste.append(f1)

    return exactitude_liste, precision_liste, rappel_liste, f1_liste

metrique_DT4 = calculer_metriques_DT(y_test, y_pred_DT4)
metrique_DT5 = calculer_metriques_DT(y_test, y_pred_DT5)
metrique_DT6 = calculer_metriques_DT(y_test, y_pred_DT6)
metrique_liste = [(metrique_DT4, "DT4"), (metrique_DT5, "DT5"), (metrique_DT6, "DT6")]

print("************ Calcul des metriques pour choisir le meilleur model DT ***************** ")
print()
def affichage_metriques_DT():
    for metrique, nom_modele in metrique_liste:
        
        exactitude_liste, precision_liste, rappel_liste, f1_liste = metrique
        
        print("Modèle : ", nom_modele)
        print("Exactitude:", exactitude_liste)
        print("Precision:", precision_liste)
        print("Rappel:", rappel_liste)
        print("F1 Score:", f1_liste)
        print()
affichage_metriques_DT()
"""
On en déduit que DT6 est le plus performant
"""
#---------------------------- évaluation des métriques pour chaque classe --------------------------

def calculer_metriques_par_classe_DT(y_true, y_pred):
    metriques_par_classe = []

    #boucle sur chaque classe
    for classe_label in range(4):
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        #calcul des vrais positifs, faux positifs et faux négatifs pour cette classe
        for true_label, pred_label in zip(y_true[0], y_pred[0]):
            #print("(true_label = ", true_label, ", classe_label = ", classe_label, " ); (pred_label =", pred_label, ", classe_label = ", classe_label, ")")
            if true_label == classe_label and pred_label == classe_label:
                TP += 1
            elif true_label != classe_label and pred_label == classe_label:
                FP += 1
            elif true_label == classe_label and pred_label != classe_label:
                FN += 1
            else :
                TN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + FP + FN + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        metriques_par_classe.append({
            "Classe": classe_label,
            "Exactitude": accuracy,
            "Précision": precision,
            "Rappel": recall,
            "F1 Score": f1
        })

    return metriques_par_classe

def affichage_metriques_par_classe_DT():
    metrics_DT6 = calculer_metriques_par_classe_DT(y_test, y_pred_DT6)
    print("\nMétriques pour chaque classe pour le modèle DT6:")
    for metrics in metrics_DT6:
        print(metrics)
affichage_metriques_par_classe_DT()


def matrice_confusion_DT6():

    #initialisation de la matrice remplie de zéros
    mat_conf = [[0] * 4 for i in range(4)]

    #parcourt des prédictions et des vraies étiquettes pour mettre à jour la matrice
    for true_label, pred_label in zip(y_test.values.flatten(), y_pred_DT6.values.flatten()):
        mat_conf[true_label][pred_label] += 1

    print("Matrice de confusion pour le modèle DT6 :")
    for row in mat_conf:
        print(row)
print()
matrice_confusion_DT6()
print()

# --------------------------------------------------------------------------------------------------
# ---------------------------------- Analyse des Réseaux de Neurones -------------------------------
# --------------------------------------------------------------------------------------------------

# Convertir le DataFrame pandas en tableau numpy, pour faciliter la comparaison
y_test = y_test.to_numpy() 

# Charger les prédictions pour les modèles avec activation tanh
y_pred_tanh_10_8_6 = pd.read_csv('Predictions/y_pred_NN_tanh_10-8-6.csv', header=None)
y_pred_tanh_10_8_4 = pd.read_csv('Predictions/y_pred_NN_tanh_10-8-4.csv', header=None)
y_pred_tanh_6_4 = pd.read_csv('Predictions/y_pred_NN_tanh_6-4.csv', header=None)

# Charger les prédictions pour les modèles avec activation ReLU
y_pred_relu_10_8_6 = pd.read_csv('Predictions/y_pred_NN_relu_10-8-6.csv', header=None)
y_pred_relu_10_8_4 = pd.read_csv('Predictions/y_pred_NN_relu_10-8-4.csv', header=None)
y_pred_relu_6_4 = pd.read_csv('Predictions/y_pred_NN_relu_6-4.csv', header=None)

# Conversion des probabilités en étiquettes de classe prédites
y_pred_tanh_10_8_6 = np.argmax(y_pred_tanh_10_8_6, axis=1)
y_pred_tanh_10_8_4 = np.argmax(y_pred_tanh_10_8_4, axis=1)
y_pred_tanh_6_4 = np.argmax(y_pred_tanh_6_4, axis=1)

y_pred_relu_10_8_6 = np.argmax(y_pred_relu_10_8_6, axis=1)
y_pred_relu_10_8_4 = np.argmax(y_pred_relu_10_8_4, axis=1)
y_pred_relu_6_4 = np.argmax(y_pred_relu_6_4, axis=1)


def calcul_metriques_RNA(y_true, y_pred):
    """
    Calcule les métriques de performance pour chaque modèle.
    Args:
        y_true : Un tableau des étiquettes réelles (vraies valeurs).
        y_pred : Un tableau des étiquettes prédites par le modèle.

    Returns:
        listes : Retourne quatre listes contenant les exactitudes, précisions, rappels et scores F1 
    """
    exactitude_liste = []
    precision_liste = []
    rappel_liste = []
    f1_liste = []
    vrai_positif = 0
    faux_positif = 0
    faux_negatif = 0
    for i in range(y_true.shape[1]):
        #calcul du nombre de vrais positifs, faux positifs et faux négatifs
        for true_value, pred_value in zip(y_true, y_pred):
            if true_value == pred_value:
                vrai_positif += 1
            elif true_value == i and pred_value != i:
                faux_negatif += 1
            elif true_value != i and pred_value == i:
                faux_positif += 1

        exactitude = vrai_positif / len(y_true)
        precision = vrai_positif / (vrai_positif + faux_positif) if (vrai_positif + faux_positif) != 0 else 0
        rappel = vrai_positif / (vrai_positif + faux_negatif) if (vrai_positif + faux_negatif) != 0 else 0
        f1 = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) != 0 else 0

        exactitude_liste.append(exactitude)
        precision_liste.append(precision)
        rappel_liste.append(rappel)
        f1_liste.append(f1)

    return exactitude_liste, precision_liste, rappel_liste, f1_liste

metrique_tanh_10_8_6 = calcul_metriques_RNA(y_test, y_pred_tanh_10_8_6)
metrique_tanh_10_8_4 = calcul_metriques_RNA(y_test, y_pred_tanh_10_8_4)
metrique_tanh_6_4 = calcul_metriques_RNA(y_test, y_pred_tanh_6_4)
metrique_relu_10_8_6 = calcul_metriques_RNA(y_test, y_pred_relu_10_8_6)
metrique_relu_10_8_4 = calcul_metriques_RNA(y_test, y_pred_relu_10_8_4)
metrique_relu_6_4 = calcul_metriques_RNA(y_test, y_pred_relu_6_4)

metrique_liste_tanh = [(metrique_tanh_10_8_6, "10-8-6"), (metrique_tanh_10_8_4, "10-8-4"), (metrique_tanh_6_4, "6-4")]
metrique_liste_relu = [(metrique_relu_10_8_6, "10-8-6"), (metrique_relu_10_8_4, "10-8-4"), (metrique_relu_6_4, "6-4")]
print("************ Calcul des metriques pour choisir le meilleur model RNA ***************** ")
print()

print("************ Réseau de neurones tanh ***************** ")
for metrique, nom_modele in metrique_liste_tanh:
    exactitude_liste, precision_liste, rappel_liste, f1_liste = metrique
    print("Modèle : ", nom_modele)
    print("exactitude:", exactitude_liste)
    print("Precision:", precision_liste)
    print("rappel:", rappel_liste)
    print("F1 Score:", f1_liste)
    print()

print("************ Réseau de neurones relu ***************** ")
for metrique, nom_modele in metrique_liste_relu:
    exactitude_liste, precision_liste, rappel_liste, f1_liste = metrique
    print("Modèle : ", nom_modele)
    print("exactitude:", exactitude_liste)
    print("Precision:", precision_liste)
    print("rappel:", rappel_liste)
    print("F1 Score:", f1_liste)
    print()
"""
Pour la fonction tanh, l'architecture (10 8 6) est la meilleure
Pour la fonction relu, l'architecture (10 8 4) est la meilleure legerement != (10 8 6)

"""

#*********************** Choisir le meilleur modele pour chaque fonction tanh et relu ********************
"""
    Analyse des métriques faite dans le rapport
    tanh => architecture (10, 8, 6)
    relu => architecture (10, 8, 4)
"""

def calculer_metriques_classe_RNA(y_true, y_pred, num_classes):
    """ 
    Calcule et affiche la précision, le rappel, l'exactitude et le score F1 pour chaque classe. 
    Args:
        y_true : Un tableau des étiquettes réelles (vraies valeurs).
        y_pred : Un tableau des étiquettes prédites par le modèle.
        num_classes : Nombre de classe distinct du modele

    Returns:
        listes : Retourne quatre listes contenant les exactitudes, précisions, rappels et scores F1  pour chaque classe
    """
    metrics = {}
    TP, FP, FN, TN = 0, 0, 0, 0
    for k in range(num_classes):
        for yt, yp in zip(y_true, y_pred):
            if yt == k:
                if yp == k:
                    TP += 1
                else:
                    FN += 1
            else:
                if yp == k:
                    FP += 1
                else:
                    TN += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        rappel = TP / (TP + FN) if (TP + FN) > 0 else 0
        exactitude = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        f1_score = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) > 0 else 0

        metrics[k] = {
            'precision': precision,
            'rappel': rappel,
            'exactitude': exactitude,
            'f1_score': f1_score
        }
    return metrics

def print_metrics(metrics):
    """ 
    Affiche les métriques calculées pour chaque classe. 
    """
    for class_id, metrics  in metrics.items():
        print(f"Classe {class_id}: Exactitude  = {metrics['exactitude']:.2f}, Précision = {metrics['precision']:.2f}, Rappel = {metrics['rappel']:.2f}, F1-score = {metrics['f1_score']:.2f}")


num_classes = 4  # Le nombre de classes différentes 

# Calcul des métriques tanh
metrics_tanh = calculer_metriques_classe_RNA(y_test, y_pred_tanh_10_8_6, num_classes)
print("Affichage des métriques calculées pour chaque classe tanh(10,8,6) ")
print_metrics(metrics_tanh)
print()

# Calcul des métriques relu
metrics_relu = calculer_metriques_classe_RNA(y_test, y_pred_relu_10_8_4, num_classes)
print("Affichage des métriques calculées pour chaque classe tanh(10,8,4) ")
print_metrics(metrics_relu)
print()


def matrice_confusion_RNA(y_pred, y_true):
    """
    Calcule et affiche la matrice de confusion pour un modèle de réseau neuronal artificiel.
    Args:
        y_pred : Liste des prédictions faites par le modèle.
        y_true : Liste des vraies étiquettes correspondant aux données.
    Returns:
        None
    """
    #initialisation de la matrice remplie de zéros
    mat_conf = [[0] * 4 for i in range(4)]

    #parcourt des prédictions et des vraies étiquettes pour mettre à jour la matrice
    for true_label, pred_label in zip(y_true, y_pred):
        #mat_conf[int(true_label)][int(pred_label)] += 1
        mat_conf[int(true_label.item())][int(pred_label.item())] += 1

    for row in mat_conf:
        print(row)

print("Matrice de confusion pour tanh(10_8_6) :")
matrice_confusion_RNA(y_pred_tanh_10_8_6, y_test)
print()

print("Matrice de confusion pour relu(10_8_4) :")
matrice_confusion_RNA(y_pred_relu_10_8_4, y_test)