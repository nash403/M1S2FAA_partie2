#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    FAA Partie 2: TP4 Markov
    Auteur: Honoré Nintunze et Matthieu Caron
"""

import numpy as np
import re
import unicodedata

### LECTURE DATA ###
def read_lines(file):
    fichier = open(file,'r')
    lines = fichier.read().split('\n') # on récupère les lignes
    lines = map(lambda line: re.sub("^[0-9]+\t","",line),lines) # on supprime les numéros en début de ligne
    lines =  map(lambda line: normalise(line),lines) # on enlève les acents et autres caractères indésirables
    return lines

def normalise(phrase):
    res = unicode (phrase,'UTF-8') # on définit l'encodage des lignes pour supprimer les accents
    res = unicodedata.normalize("NFD",res).encode('ascii','ignore')
    res = res.lower() # on met chaque ligne en minuscule
    res = re.sub("[,()?:;!\"#*-]"," ",res) # on supprime les caractères indésirables
    return res

def read_words_from_line(line):
    return line.split()

def lines_to_words(lines):
    return map(read_words_from_line,lines)

### CONSTRUCTION DE LA MATRICE DE TRANSITION ###
def build_dico(sentences): # le dictionnaire est ici la matrice de transition non creuse de la chaîne de Markov
    dico = {}
    for sentence in sentences:
        for i in range(0,len(sentence)-1):
            word = sentence[i]
            follower = sentence[i+1]
            if word in dico:
                if follower in dico[word]:
                    dico[word][follower] += 1
                else:
                    dico[word][follower] = 1
            else:
                dico[word] = {}
                dico[word][follower] = 1
    return dico

### FONCTIONS DE PREDICTION ###
def predict_word_with_dico(dico,uncomplete_sentence):
    phrase_normalisee = normalise(uncomplete_sentence)
    words = read_words_from_line(phrase_normalisee)
    last = words[-1]
    if last in dico:
        predictable = ""
        predictable_proba = 0
        for k, v in dico[last].items():
            if v >= predictable_proba:
                predictable_proba = v
                predictable = k

        words.append(predictable)
        return predictable, " ".join(words)
    else:
        return ""," ".join(words)

def predict_word_with_dico_ordre2(dico,uncomplete_sentence):
    phrase_normalisee = normalise(uncomplete_sentence)
    words = read_words_from_line(phrase_normalisee)
    last = words[-1]
    before_last = words[-2]
    # Il faut calculer la probabilité de tirer un mot sachant que le dernier mot est sorti après l'avant dernier mot
    proba_last = 1.
    if before_last in dico:
        if last in dico[before_last]:
            proba_last = float(dico[before_last][last]) / float(sum([v for k,v in dico[before_last].iteritems()]))
    proba_last = proba_last * 1000.
    if last in dico:
        predictable = ""
        predictable_proba = 0.
        itemsLast = dico[last].iteritems()
        total = float(sum([v for k,v in itemsLast]))
        for k, v in dico[last].iteritems():
            proba = (float(v)/total)*1000.
            proba_cond = (float(proba*proba_last)/float(proba))
            if proba_cond > predictable_proba:
                predictable_proba = proba_cond
                predictable = k

        words.append(predictable)
        return predictable, " ".join(words)
    else:
        return ""," ".join(words)

def predict_word_with_dico_ordre3(dico,uncomplete_sentence):
    phrase_normalisee = normalise(uncomplete_sentence)
    words = read_words_from_line(phrase_normalisee)
    last = words[-1]
    before_last = words[-2]
    before_before_last = words[-3]
    proba_before_last = 1.
    proba_last = 1.
    if before_before_last in dico:
        if before_last in dico[before_before_last]:
            proba_before_last = float(dico[before_before_last][before_last]) / float(sum([v for k,v in dico[before_before_last].iteritems()]))
    if before_last in dico:
        if last in dico[before_last]:
            proba_last = float(dico[before_last][last]) / float(sum([v for k,v in dico[before_last].iteritems()]))
    proba_last = proba_last * 1000.
    proba_before_last = proba_before_last * 1000.
    proba_inter = float(proba_before_last * proba_last) / float(proba_before_last)
    if last in dico:
        predictable = ""
        predictable_proba = 0.
        itemsLast = dico[last].iteritems()
        total = float(sum([v for k,v in itemsLast]))
        for k, v in dico[last].iteritems():
            proba = (float(v)/total)*1000.
            proba_cond = ((proba*proba_inter)/proba)
            if proba_cond > predictable_proba:
                predictable_proba = proba_cond
                predictable = k

        words.append(predictable)
        return predictable, " ".join(words)
    else:
        return ""," ".join(words)

### DEROULEMENT DU PROGRAMME ###
# Lecture des données
sentences = read_lines('sentences.txt')
words_in_sentences = lines_to_words(sentences)
# Apprentissage
dictionnaire = build_dico(words_in_sentences)
# print words_in_sentences[40:70]

# Résultat de l'apprentissage
chaine = "Le but de la vie est"
print "######################################################"
print "###### Prédiction avec une chaîne markovienne d'ordre 1 ######"

_,res_phrase = predict_word_with_dico(dictionnaire,chaine)
for i in range(5):
    _,res_phrase = predict_word_with_dico(dictionnaire,res_phrase)

print "Prédiction de:", chaine
print "Résultat 1:\t",res_phrase # le but de la vie est un des annees de la ville

print "######################################################"
print "###### Prédiction avec une chaîne markovienne d'ordre 2 ######"

_,res_phrase = predict_word_with_dico_ordre2(dictionnaire,chaine)
for i in range(10):
    _,res_phrase = predict_word_with_dico_ordre2(dictionnaire,res_phrase)

print "Prédiction de:", chaine
print "Résultat 2:\t",res_phrase # le but de la vie est limitee a l'echelon inferieur.

print "######################################################"
print "###### Prédiction avec une chaîne markovienne d'ordre 3 ######"

_,res_phrase = predict_word_with_dico_ordre3(dictionnaire,chaine)
for i in range(5):
    _,res_phrase = predict_word_with_dico_ordre3(dictionnaire,res_phrase)

print "Prédiction de:", chaine
print "Résultat 3:\t",res_phrase # le but de la vie est limitee a l'echelon inferieur.
