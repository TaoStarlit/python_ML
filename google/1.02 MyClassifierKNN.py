# coding:utf-8
## part1 datasets, split to train and test
from sklearn import datasets
iris = datasets.load_iris()

X=iris.data
Y=iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=.5)

# part2 define Classifier  my classifier -- implement your classifier, and the interface -- get that pipeline working
#from sklearn.neighbors import KNeighborsClassifier
#my_classifier =KNeighborsClassifier()
import random
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b) #欧几里得

class ScrappKNN(): # K nearest neighnors  distance  E
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test: # each row predict an example
            label = self.closest(row) # implement in the class, because it need the training data
            #label = random.choice(self.y_train)
            predictions.append(label)
        return predictions
    
    def closest(self,row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist<best_dist:
                best_dist=dist
                best_index=i
        return self.y_train[best_index]

my_classifier = ScrappKNN()

# part3 fit the training set,  and predict the result of testing set
my_classifier.fit(X_train,y_train)


predictions=my_classifier.predict(X_test)

# part4 calculat the accuracy score
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)