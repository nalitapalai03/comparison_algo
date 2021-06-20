
# Load libraries
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.svm import SVC
import conf as cm

names = ['mcv', 'aap', 'sala', 'sapa', 'gggt', 'dirnks_per_day', 'class']
dataset = pd.read_csv("bupa.data", names=names)

#print(dataset)
#exit(1)
#for cleaning the data
print(dataset.groupby('class').size())


# #tells about the no of columns and rows
# print(dataset.shape)
# print("\n")
# print(dataset.head(5))
# print("\n")
#
# #shows all aspects of the data
# print(dataset.describe())
# print("\n")


#counts the no of row categorized by class attribute
#print(dataset.groupby('class').size())
#create histograms of all attributes
# dataset.hist()
# plt.show()

#create a graph matrix with all combinations of attributes
# scatter_matrix(dataset)
# plt.show()

arr = dataset.values

# patientAttributes = arr[:, 0:4]
# print(patientAttributes)
patientAttributes = arr[:, 0:6]          #array slicing[x-index start : x-end , y-start : y-end ]
classAttribute = arr[:, 6]


#model training
validationSize = 0.80
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(patientAttributes, classAttribute, train_size=validationSize)

# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# cv_results = model_selection.cross_val_score(SVC(gamma='auto'), X_train, Y_train, cv=kfold, scoring='accuracy')
# print(cv_results.mean() * 100)
# print(cv_results.std())

# making predictions
svcClassifier = SVC(gamma='auto').fit(X_train, Y_train)
predictions = svcClassifier.predict(X_validation)

# checking predictions
length = len(predictions)


print("\n\n")
cm.matrix(predictions, Y_validation)


# preparing confusion matrix
