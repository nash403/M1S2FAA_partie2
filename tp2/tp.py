#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    FAA Partie 2: TP2
    Auteur: Honoré Nintunze
"""
__author__ = 'honore-nintunze'

import tensorflow as tf
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

diabetesbase = parse_dataset("pima-indians-diabetes.data.txt")

print "data set shape", diabetesbase.shape

# Séparation de données de test et d'apprentissage, 10% de la base pour les test (les dernière données)
longueur_data, learning_setX, test_setX, learning_setY, test_setY = split_dataset(diabetesbase)

print "Données", longueur_data, len(learning_setX), len(test_setX), len(learning_setY), len(test_setY)

# Tests du taux d'erreur selon la profondeur
for profondeur in range(1,11):
    classifieur = tree.DecisionTreeClassifier(max_depth=profondeur)
    classifieur = classifieur.fit(learning_setX,learning_setY)
    erreur = 0
    res = classifieur.predict(test_setX)
    for i in range(len(test_setX)):
        if test_setY[i] != res[i]:
            erreur += 1
    tauxErreur = float(erreur) /float(len(test_setY))
    print "profondeur %d: %f" % (profondeur, tauxErreur)
