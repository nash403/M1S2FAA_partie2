__author__ = 'julien-perolat'
import numpy as np
import urllib

def import_url(url):
    # download the file
    raw_data = urllib.urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = np.loadtxt(raw_data, delimiter=",")
    return dataset


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
dataset = import_url(url)


print(dataset.shape)
# separate the data from the target attributes
X = dataset[:,0:-1]
Y = dataset[:,-1]