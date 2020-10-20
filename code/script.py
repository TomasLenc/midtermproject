#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import datalad.api as dl
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

dataPath = "input/iris.csv"

dl.get(dataPath)

df = pd.read_csv(dataPath)
attributes = ["sepal_length", "sepal_width", "petal_length","petal_width", "class"]
df.columns = attributes

plot = sns.pairplot(df, hue='class', palette='muted')
plot.savefig('pairwise_relationships.png')


array = df.values
X = array[:,0:4]
Y = array[:,4]
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = test_size, 
                                                                    random_state = seed)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

report = classification_report(Y_test, predictions, output_dict=True)
df_report = pd.DataFrame(report).transpose().to_csv('prediction_report.csv')
