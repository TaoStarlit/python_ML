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
class ScrappKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test: # each row predict an example
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

my_classifier = ScrappKNN()

# part3 fit the training set,  and predict the result of testing set
my_classifier.fit(X_train,y_train)


predictions=my_classifier.predict(X_test)

# part4 calculat the accuracy score
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)