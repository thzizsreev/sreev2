import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

dataset = pd.read_csv('/content/sample_data/international_matches.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

dataset.head()

encoder = LabelEncoder()
dataset["home_team"]=encoder.fit_transform(dataset["home_team"])
dataset["away_team"]=encoder.fit_transform(dataset["away_team"])
dataset["home_team_continent"]=encoder.fit_transform(dataset["home_team_continent"])
dataset["away_team_continent"]=encoder.fit_transform(dataset["away_team_continent"])
dataset["tournament"]=encoder.fit_transform(dataset["tournament"])
dataset["city"]=encoder.fit_transform(dataset["city"])
dataset["country"]=encoder.fit_transform(dataset["country"])
dataset["neutral_location"]=encoder.fit_transform(dataset["neutral_location"])
dataset["shoot_out"]=encoder.fit_transform(dataset["shoot_out"])
dataset["home_team_result"]=encoder.fit_transform(dataset["home_team_result"])

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)