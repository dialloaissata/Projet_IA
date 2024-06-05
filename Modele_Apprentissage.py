import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##################################### I. Préparation des données ##################################### 

#Importer les données 
data = pd.read_csv("synthetic.csv")
#print(data.head())

# Diviser les données en ensembles d'entraînement et de test (20% pour le test)
train_data, test_data = train_test_split(data, test_size=0.2)
#print("Taille de l'ensemble d'entraînement :", len(train_data))
#print("Taille de l'ensemble de test :", len(test_data))

# 1. Combien d’attributs comportent ces données ?
nombre_attributs = data.shape[1]
#print("Nombre d'attributs :", nombre_attributs) #15 attributs dont une classe

#2. En combien de classes différentes les instances sont-elles catégorisées ?
nombre_classes = data['Class'].nunique()
print("Nombre de classes différentes :", nombre_classes)

#3. Combien d’instances chaque classe compte-elle ?
instances_par_classe = data['Class'].value_counts()
#print("Nombre d'instances par classe :\n", instances_par_classe)

#4. Les données sont-elles linéairement séparables ?
"""
les données ne sont pas linéairement séparables. Selon la figure 1, il est impossible
de tracer une droite qui sépare ces données à cause de leur forte dispersion
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


#5. L'utilisation de l'encodagge one hot est elle nécessaire ? + NORMALISATION ?


"""
6. On divise les données en 2 groupes : un ensemble d'entraînement pour entrainer le modèle
et un ensemble de test pour s'assurer que le modèle n'a pas simplement appris les données d'entraînement
par coeur et qu'il sait généraliser avec ces données de tests.
"""


##################################### II. Mise en oeuvre des modèles ##################################### 

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


##################################### III. Analyse des modèles ##################################### 

# ---------------------------------- Analyse des Arbres de Décisions -------------------------------

#---------------- évaluation des métriques de chaque modèle (non demandé dans le sujet) -------------------------
"""
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

def affichage_metriques():
    for metrique, nom_modele in metrique_liste:
        
        exactitude_liste, precision_liste, rappel_liste, f1_liste = metrique
        
        print("Modèle : ", nom_modele)
        print("Exactitude:", exactitude_liste)
        print("Precision:", precision_liste)
        print("Rappel:", rappel_liste)
        print("F1 Score:", f1_liste)
        print()
#affichage_metriques()
#On en déduit que DT6 est le plus performant
"""

#---------------------------- évaluation des métriques pour chaque classe --------------------------

y_test = pd.read_csv('Predictions/y_test.csv', header = None)

y_pred_DT4= pd.read_csv('Predictions/y_pred_DT4.csv', header = None)
y_pred_DT5 = pd.read_csv('Predictions/y_pred_DT5.csv', header = None)
y_pred_DT6 = pd.read_csv('Predictions/y_pred_DT6.csv', header = None)


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
    metrics_DT4 = calculer_metriques_par_classe_DT(y_test, y_pred_DT4)
    metrics_DT5 = calculer_metriques_par_classe_DT(y_test, y_pred_DT5)
    metrics_DT6 = calculer_metriques_par_classe_DT(y_test, y_pred_DT6)

    print("Métriques pour chaque classe pour le modèle DT4:")
    for metrics in metrics_DT4:
        print(metrics)

    print("\nMétriques pour chaque classe pour le modèle DT5:")
    for metrics in metrics_DT5:
        print(metrics)

    print("\nMétriques pour chaque classe pour le modèle DT6:")
    for metrics in metrics_DT6:
        print(metrics)
affichage_metriques_par_classe_DT()

#----------------------------------------------------------------------------------------------------------------


def matrice_confusion_DT6():

    #initialisation de la matrice remplie de zéros
    mat_conf = [[0] * 4 for i in range(4)]

    #parcourt des prédictions et des vraies étiquettes pour mettre à jour la matrice
    for true_label, pred_label in zip(y_test.values.flatten(), y_pred_DT6.values.flatten()):
        mat_conf[true_label][pred_label] += 1

    print("Matrice de confusion pour le modèle DT6 :")
    for row in mat_conf:
        print(row)
#matrice_confusion_DT6()