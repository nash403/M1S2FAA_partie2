#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    FAA Partie 2: TP3 Boosting
    Auteur: Honoré Nintunze et Matthieu Caron
"""

import numpy as np
from sklearn import tree

### LECTURE DATA ###
def parse_dataset(raw_data):
    dataset = np.loadtxt(raw_data, delimiter=",")
    return dataset

def split_dataset(dataset):
    # separate the data from the target attributes
    # X=1ère colonne à l'avant dernière colonne
    # Y= la dernière colonne

    X,Y = dataset[:,0:-1],dataset[:,-1]
    longueur_data = len(dataset)
    learning_setX = X[0:(longueur_data//10)*9,:]
    test_setX = X[(longueur_data//10)*9:,:]

    learning_setY = Y[:(longueur_data//10)*9]
    test_setY = Y[(longueur_data//10)*9:]


    return longueur_data, learning_setX, test_setX, learning_setY, test_setY

spambase = parse_dataset("spambase.data.txt")

print "data set shape", spambase.shape

# Séparation de données de test et d'apprentissage, 10% de la base pour les test (les dernière données)
longueur_data, learning_setX, test_setX, learning_setY, test_setY = split_dataset(spambase)

print "Données", longueur_data, len(learning_setX), len(test_setX), len(learning_setY), len(test_setY)

# lassifieur de profondeur 1 pour AdaBoost
classifieur = tree.DecisionTreeClassifier(max_depth=1)

N = len(learning_setX)
# calcul du vecteur des poids, les poids sont identiques au début et leur somme doit donner 1
w = 1 / N
WS = [w] * N

def adaBoostAlgo(xs, ys, ws,nbFeatures):
    """
    Implémentation de l'algorithme d'AdaBoost
    """
    alphat = 0 # pas d'apprentissage
    LENGTH = len(xs)
    HT = []

    for t in range(nbFeatures): # nombre de classifieur faibles
        # on entraîne le classifieur
        classifieur = tree.DecisionTreeClassifier(max_depth=1)
        classifieur = classifieur.fit(xs,sample_weight=ws)

        # on calcul l'erreur d'apprentissage
        res = classifieur.predict(xs)
        epsilont = 0 # erreur de classifieur
        for i in range(LENGTH):
            if res[i] != ys[i]:
                epsilont += ws[i] # indicatrice

        espilont = epsilont / LENGTH
        # on calcul le pas d'apprentissage
        alphat = 0.5 * np.log((1 - epsilont)/epsilont)

        # màj des poids
        resw = []
        for i in range(LENGTH):
            resw.append(ws[i] * np.exp(- alphat * ys[i] * res[i]))
        resw = map(lambda x: x[i] * np.exp(- alphat * ys[i] * res[i]),ws)
        sumresw = sum(resw)
        wplus = map(lambda x: x / sumresw,resw)
        HT.appen(classifieur,alphat)

    return HT
# TODO tester tout ça, NE PAS OUBLIER DE TRANSFORMER LES O EN -1
