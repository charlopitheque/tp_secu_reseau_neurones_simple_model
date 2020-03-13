import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# voir : https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(1)
# Etape :
iris_dataset = datasets.load_iris()
print(iris_dataset['DESCR'])
X = iris_dataset.data
y = iris_dataset.target
# Etape :
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# Etape :
model = DecisionTreeClassifier()
# Etape :
model.fit(X_train, y_train)
# Etape :
y_predicted = model.predict(X_test)
# Etape :
print('======= Validation =========')
# https://stats.stackexchange.com/questions/95209/how-can-i-interpret-sklearn-confusion-matrix.p
print('Confusion Matrix:\n', confusion_matrix(y_test, y_predicted))
print('Accuracy:\n', accuracy_score(y_test, y_predicted))
print('Other metrics:\n', classification_report(y_test, y_predicted))