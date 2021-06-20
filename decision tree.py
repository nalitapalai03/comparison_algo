
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import conf as cm


balance_data = pd.read_csv('bupa.data', header= None)

print("Dataset Lenght:: ", len(balance_data))
print("Dataset Shape:: ", balance_data.shape)

balance_data.head()

X = balance_data.values[:, :6]
Y = balance_data.values[:, 6]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.25, random_state = 100)


# clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
#                                max_depth=3, min_samples_leaf=5)
# clf_gini.fit(X_train, y_train)
#
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
#             max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=100, splitter='best')

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')

# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
#             max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=100, splitter='best')


# y_pred = clf_gini.predict(X_test)
# print(y_pred)


y_pred = clf_entropy.predict(X_test)
print(y_pred)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
cm.matrix(y_pred, y_test)
#
# print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)