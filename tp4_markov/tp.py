#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    FAA Partie 2: TP4 Markov
    Auteur: Honoré Nintunze
"""
__author__ = 'honore-nintunze'

import numpy as np
import re
import unicodedata

### LECTURE DATA ###
def read_lines(file):
    fichier = open(file,'r')
    lines = fichier.read().split('\n')
    lines = map(lambda line: re.sub("^[0-9]+\t","",line),lines)
    lines = map(lambda l: unicode (l,'UTF-8'),lines)
    lines = map(lambda l: unicodedata.normalize("NFD",l).encode('ascii','ignore'),lines)
    lines = map(lambda l: l.lower(),lines)
    lines = map(lambda l: re.sub("[,()?.:;!\"#*-]","",l),lines)
    return lines

def read_words_from_line(line):
    return line.split()

def lines_to_words(lines):
    return map(read_words_from_line,sentences)

def build_dico(sentences):
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


def create_transition_matrix(dico):
    words = dico.keys()
    matrix = [[0]*10]*len(words)
    for i in range(len(words)):
        for j in range(len(words)):
            matrix[i][j] = 

sentences = read_lines('sentences.txt')
words_in_sentences = lines_to_words(sentences)
dictionnaire = build_dico(words_in_sentences)
print dictionnaire


# for l in sentences:
#     print l

