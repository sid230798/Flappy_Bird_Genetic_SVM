import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing

from NeuralModel import Neural
from settings import SVM_DATA_ONES, SVM_DATA_ZEROS, POPULATION


def get_data_from_file(filename):
    destination = []
    with open(filename, "r") as f:
        for line in f.readlines():
            split_line = [float(x) for x in line.split()]
            destination.append(split_line)

    return np.array(destination)


def load_data():

    ones = get_data_from_file(SVM_DATA_ONES)
    zeros = get_data_from_file(SVM_DATA_ZEROS)

    training_data = np.append(ones, zeros, axis=0)

    return training_data[:, :-1], training_data[:, -1]


def get_model(name="gan"):
    if name.upper() == "GAN":
        neural_list = []
        for _ in range(POPULATION):
            neural_list.append(Neural())

        return neural_list
    elif name.upper() == "SVM":
        X, Y = load_data()
        X_t = preprocessing.scale(X)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        clf = SVC()
        clf.fit(X_t, Y)
        return [clf], mean, std
    elif name.upper() == "LINEARSVM":
        X, Y = load_data()
        X = np.hstack([X, X**2, X**3])
        X_t = preprocessing.scale(X)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        clf = LinearSVC()
        clf.fit(X_t, Y)
        return [clf], mean, std
    elif name.lower() == "gan-collect":
        clf = None
        with open("model/gan.model", "rb") as f:
            clf = pickle.load(f)
        return [clf], 0, 1
    else:
        print("please specify model")
        print("python main.py --model your-choice")
        print("\tchoices")
        print("\t1. svm")
        print("\t2. GAN")
        print("\t3. linearsvm")
        quit()