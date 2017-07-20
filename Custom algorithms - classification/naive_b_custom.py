import numpy as np

class NaiveBayesClassifier(object):
    
    def __init__(self):
        pass
    
    #Input: X - features of a trainset
    #       y - labels of a trainset
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        self.no_of_classes = int(np.max(self.y_train) + 1)
     
        
    #This is our function to calculate all nodes/samples in our radius    
    def euclidianDistance(self, Xtest, Xtrain):
        return np.sqrt(np.sum(np.power((Xtest - Xtrain), 2)))
    
       
    #our main function is predict
    #All calculation is done by using our test or new samples
    #There are 4 steps to be performed:
    # 1. calculate Prior probability. Ex. P(A) = No_of_elements_of_one_class / total_no_of_samples
    # 2. calculate Margin probability P(X) = No_of_elements_in_radius / total_no_of_samples
    # 3. calculate Likeliyhood (P(X|A) = No_of_elements_of_current_class / total_no_of_samples
    # 4. calculate Posterior probability: P(A|X) = (P(X|A) * P(A)) / P(X)
    # NOTE: Do these steps for all clases in dataset!
    #
    #Inputs: X - test dataset
    #       radius - this parameter is how big circle is going to be around our new datapoint, default = 2
    def predict(self, X, radius=0.4):   
        pred = []
        
        #Creating list of numbers of elements for each class in trainset
        members_of_class = []
        for i in range(self.no_of_classes):
            counter = 0
            for j in range(len(self.y_train)):
                if self.y_train[j] == i:
                    counter += 1
            members_of_class.append(counter)
        
        #Entering the process of prediction
        for t in range(len(X)):
            #Creating empty list for every class probability
            prob_of_classes = []
            #looping through each class in dataset
            for i in range(self.no_of_classes):
                
                #1. step > Prior probability P(class) = no_of_elements_of_that_class/total_no_of_elements
                prior_prob = members_of_class[i]/len(self.y_train)

                #2. step > Margin probability P(X) = no_of_elements_in_radius/total_no_of_elements
                #NOTE: In the same loop collecting infromation for 3. step as well
                
                inRadius_no = 0
                #counter for how many points are from the current class in circle
                inRadius_no_current_class = 0
                
                for j in range(len(self.X_train)):
                    if self.euclidianDistance(X[t], self.X_train[j]) < radius:
                        inRadius_no += 1
                        if self.y_train[j] == i:
                            inRadius_no_current_class += 1
                
                #Computing, margin probability
                margin_prob = inRadius_no/len(self.X_train)
                
                #3. step > Likelihood P(X|current_class) = no_of_elements_in_circle_of_current_class/total_no_of_elements
                likelihood = inRadius_no_current_class/len(self.X_train)
                
                #4. step > Posterial Probability > formula from Bayes theorem: P(current_class | X) = (likelihood*prior_prob)/margin_prob
                post_prob = (likelihood * prior_prob)/margin_prob
                prob_of_classes.append(post_prob)
            
            #Getting index of the biggest element (class with the biggest probability)
            pred.append(np.argmax(prob_of_classes))
                
        return pred



def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100