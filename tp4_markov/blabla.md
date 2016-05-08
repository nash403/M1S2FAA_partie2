## Chaînes de Markov

* Avec Markov on apprend en fonction de ce qu'on a appris auparavant.

* le tirage d'une classe/donnée dépend de ce qui a déjà été tiré avant

  ex: prédire ce qu'un étudiant va manger le jour suivant sachant ce qu'il a mangé les jours précédents.

* Ne pas juste prendre les probabilité issues d'un sondage car on prend pas en comptes les tirages précédents.

  On peut représenter dans une matrice les probabilités, avec autant de dimensions que de nombre de tirages précédents sont pris en compte.
  ex: dimension 2 si on prend en compte le repas de la veille, 3 si on prend en compte les repas des 2 derniers jours.

* On appelle cette matrice la **matrice de transition**

  On peut la représenter comme un diagramme d'état (ou graphe) avec les probabilités sur les transitions et l'item/classe comme état.

* On appelle **vecteur d'état** le vecteur indiquant le tirage au temps t.

* On peut avec la matrice de transition prédire le repas du surlendemain en faisant la simple hypothèse que le repas du jour ne dépend que du repas de la veille.  
