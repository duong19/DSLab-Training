import regex as re
import numpy as np


def extract_data(readfile,writefile):
    with open(readfile,'r') as fread:
        with open(writefile,'w') as fwrite:
            for line in fread:
                if (re.match("(?<=^| )\W+\d+(\.\d+)?(?=$| )", line) or re.match("(?<=^| )\d+\W+(\.\d+)?(?=$| )",line)) and re.match("^[^#]+$",line):
                    fwrite.write(line)

def read_data(filename):
    with open(filename, 'r') as f:
        X = f.readline()
        X = re.sub("\n", "", X)
        X = np.array([float(i) for i in X.split()])
        i = 0
        for line in f:
            line = re.sub("\n", "", line)
            line = np.array([float(i) for i in line.split()])
            X = np.concatenate((X,line),axis=0)
            i = i + 1
        X = X.reshape(i+1, 17)
        X_all = X[:, 1:-1]
        y_all = X[:,-1]
    return X_all, y_all

def normalized_and_add_one(X):
    X_max = np.array([[np.amax(X[:, id]) for id in range(X.shape[1])] for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, id]) for id in range(X.shape[1])] for _ in range(X.shape[0])])

    X_normalized = (X - X_min)/(X_max - X_min)
    X_normalized = np.concatenate((np.ones((X.shape[0], 1)), X_normalized),axis = 1)
    return X_normalized

X_all, y_all = read_data("data.txt")
class RidgeRegression:
    def __init__(self):
        return
    def fit(self, ):
