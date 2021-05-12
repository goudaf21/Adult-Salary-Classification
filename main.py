# Fady Gouda and Griffin Noe
# CSCI 297a
# Project 6
# 10/24/20

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import heatmap
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error

# Define the column names for the data because it doesn't have a header in the csv
cols = ['age', 'working_class', 'fnlwgt', 'education', 'education_number', 
        'marital_status', 'occupation', 'relationship', 'race', 'sex', 
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Import the data and set the column names to the defined list above
data = pd.read_csv('adult.csv', header=0, names=cols)

# Label encode the income (outcome) 
le = preprocessing.LabelEncoder()
le.fit(data['income'])
data['income'] = le.transform(data['income'])

# Drop all the categorical data for the heatmap
hm_data = data.drop(['working_class','education','marital_status','occupation',
                     'relationship','race','sex','native_country'], axis=1)

# One-hot encode the categorical data for classification
data = pd.get_dummies(data=data,columns=['working_class','education','marital_status',
                                       'occupation','relationship','race','sex','native_country'])

# Create the heatmap with the non-categorical data and headers
hm_cols = hm_data.columns
cm = np.corrcoef(hm_data[hm_cols].values.T)
hm = heatmap(cm, row_names=hm_cols, column_names=hm_cols)
plt.show()

# Split the data into a 70/30 train/test split dropping fnlwgt and setting the income as the outcome
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['fnlwgt'],axis=1),
    data['income'],
    test_size=0.3,
    random_state=1)

# Instantiate the Gaussian Naive Bayes model, fit it, and get the accuracy
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gaussian=gnb.predict(X_test)
print("Accuracy of Gaussian: %.2f" % (100*accuracy_score(y_test, pred_gaussian)), "%")

# Instantiate the Multinomial Naive Bayes model, fit it, and get the accuracy
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
pred_multinominal=mnb.predict(X_test)
print("Accuracy of multinominal: %.2f" % (100*accuracy_score(y_test, pred_multinominal)), "%")

# Instantiate the Complement Naive Bayes model, fit it, and get the accuracy
cnb=ComplementNB()
cnb.fit(X_train,y_train)
pred_complement=cnb.predict(X_test)
print("Accuracy of complement: %.2f" % (100*accuracy_score(y_test, pred_complement)), "%")

# Instantiate the Bernoulli Naive Bayes model, fit it, and get the accuracy
bnb=BernoulliNB()
bnb.fit(X_train,y_train)
pred_bernoulli=bnb.predict(X_test)
print("Accuracy of Bernoulli: %.2f" % (100*accuracy_score(y_test, pred_bernoulli)), "%")

# Plot the confusion matrix for the testing data and the classification report
plot_confusion_matrix(bnb, X_test, y_test)
plt.title("Test Data Confusion Matrix - Bernoulli")
plt.show()
print(classification_report(y_test, pred_bernoulli))

# Instantiate the Categorical Naive Bayes model, fit it, and get the accuracy
catnb=CategoricalNB()
catnb.fit(X_train,y_train)
pred_categorical=catnb.predict(X_test)
print("Accuracy of categorical: %.2f" % (100*accuracy_score(y_test, pred_categorical)), "%")

# Plot the confusion matrix for the testing data and the classification
plot_confusion_matrix(catnb, X_test, y_test)
plt.title("Test Data Confusion Matrix - Categorical")
plt.show()
print(classification_report(y_test, pred_categorical))
