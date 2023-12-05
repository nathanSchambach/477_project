import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns

data = pd.read_csv("dataset.csv")
data.head()

data['Target'] = data['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})

print(data["Target"].unique())

data.info()

data.corr()['Target']
print(data.corr()['Target'])

# Seperate out features and target variable
new_data = data[['Target', 'Curricular units 2nd sem (approved)',
                 'Curricular units 2nd sem (grade)',
                 'Curricular units 1st sem (approved)',
                 'Curricular units 1st sem (grade)', 'Tuition fees up to date',
                 'Scholarship holder', 'Age at enrollment']]
#print(new_data)

#I'm not sure why you have this line so I didn't want to erase it      
new_data['Target'].value_counts()


correlations = data.corr()['Target']
top_7_features = correlations.abs().nlargest(8).index
top_7_corr_values = correlations[top_7_features]

print(top_7_features)

plt.figure(figsize=(8, 11))
plt.bar(top_7_features, top_7_corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Top 7 Features with Highest Correlation to Target')
plt.xticks(rotation=45)
plt.show()


#Scatter plots of relationships to the target

#Age vs. Target
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age at enrollment', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age at enrollment')
plt.title('Relationship between Age at enrollment and Target')
plt.show()

#Curricular units 2nd semester (approved)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 2nd sem (approved)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 2nd sem (approved)')
plt.title('Relationship between Curricular units 2nd sem (approved) and Target')
plt.show()

#Curricular units 2nd semester (grade)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 2nd sem (grade)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 2nd sem (grade)')
plt.title('Relationship between Curricular units 2nd sem (grade) and Target')
plt.show()

#Curricular units 1st semester (approved)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 1st sem (approved)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 1st sem (approved)')
plt.title('Relationship between Curricular units 1st sem (approved) and Target')
plt.show()

#Curricular units 1st semester (grade)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Curricular units 1st sem (grade)', data=new_data)
plt.xlabel('Target')
plt.ylabel('Curricular units 1st sem (grade)')
plt.title('Relationship between Curricular units 1st sem (grade) and Target')
plt.show()

#Tuition fees up to date
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Tuition fees up to date', data=new_data)
plt.xlabel('Target')
plt.ylabel('Tuition fees up to date')
plt.title('Relationship between Tuition fees up to date and Target')
plt.show()

#Scholarship holder
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Scholarship holder', data=new_data)
plt.xlabel('Target')
plt.ylabel('Scholarship holder')
plt.title('Relationship between Scholarship holder and Target')
plt.show()


# This is my code from here down, it should work now

# Data training
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

# KNN confusion matrix
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix for KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SVM confusion matrix
y_pred_svm = svm.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Confusion Matrix for SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



