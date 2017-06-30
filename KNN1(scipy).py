from scipy.spatial import distance
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y_ = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y_,test_size = 0.5)
class easyknn():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        predictions = []
        for i in X_test:
            label = self.closest(i)
            predictions.append(label)
        return predictions
    def closest(self, row):
        min_dist = distance.euclidean(row,self.X_train[0])
        min_index = 0
        for i in range(1,len(self.X_train)):
            dist = distance.euclidean(row,self.X_train[i])
            if dist < min_dist:
                min_dist = dist
                min_index = i
        return self.y_train[min_index]

# sklearn or scipy
# knn = KNeighborsClassifier()
knn = easyknn()

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test,predictions))
