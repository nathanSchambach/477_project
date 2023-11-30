'''
Modified code for KNN and SVM
I just edited the set up for new_data
Also test_size=.4 now
If you play with the random_state= and the n_neighbors you will most likely get
different accuracies and maybe can get higher if you had time to play with it.
'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Upload dataset
data = pd.read_csv("dataset.csv")
#print(data)

# Seperate out features and target variable
new_data = data[['Target', 'Curricular units 2nd sem (approved)',
                 'Curricular units 2nd sem (grade)',
                 'Curricular units 1st sem (approved)',
                 'Curricular units 1st sem (grade)', 'Tuition fees up to date',
                 'Scholarship holder', 'Age at enrollment']]
#print(new_data)

# Label Target's as Dropout=0, Enrolled=1, Graduate=2
new_data['Target'] = data['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})
#print(new_data["Target"])

#Data training
X = new_data.drop('Target', axis=1)
y = new_data['Target']

# Split the dataset into 60% training and 40% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Display the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", y_train.shape)
print("Y_test shape:", y_test.shape)

# Run KNN and SVM
knn = KNeighborsClassifier(n_neighbors=3)
svm = svm.SVC(kernel='linear',probability=True)
knn.fit(X_train,y_train)
svm.fit(X_train, y_train)

# KNN and SVM accuracy
y_pred = knn.predict(X_test)
print("KNN Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
y_pred = svm.predict(X_test)
print("SVM Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
