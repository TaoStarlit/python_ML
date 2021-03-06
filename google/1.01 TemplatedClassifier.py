## part1 datasets, split to train and test
from sklearn import datasets
iris = datasets.load_iris()

X=iris.data
Y=iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=.5)

# part2 define Classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier =KNeighborsClassifier()

# part3 fit the training set,  and predict the result of testing set
my_classifier.fit(X_train,y_train)


predictions=my_classifier.predict(X_test)

# part4 calculat the accuracy score
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)