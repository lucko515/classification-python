import numpy as np
from collections import Counter
from operator import itemgetter

class KNeighborsClassifieR(object):

    def __init__(self):
        pass
    #"training" function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    #predict function, output of this function is lis to
    def predict(self, X_test, k=5):
        distances = self.compute_distances(self.X_train, X_test)
        vote_results = []
        for i in range(len(distances)):
            votesOneSample = []
            for j in range(k):
                votesOneSample.append(distances[i][j][1])
            vote_results.append(Counter(votesOneSample).most_common(1)[0][0])

        return vote_results


    #For each sample and every item in test set algorithm is making tuple in distance list
    #this is how list looks =>> distances = [[[distance, class],[distance, class],[distance, class],[distance, class]]]
    #distances and sort
    def compute_distances(self, X, X_test):
        distances = []
        for i in range(X_test.shape[0]):
            euclidian_distances = np.linalg.norm(np.array(X_test[i])-np.array(X), axis=1)
            euclidian_distances = np.array(euclidian_distances).reshape(X.shape[0], 1)
            classes = np.array(self.y_train)
            classes = classes.reshape(self.y_train.shape[0], 1)
            distances_i = np.append(euclidian_distances, classes, axis=1) 
            
            #Second part of each element is the class for particular element in y_train
            distances.append(sorted(distances_i,key=itemgetter(0)))
        return distances



def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100