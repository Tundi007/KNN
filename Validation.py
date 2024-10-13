from random import shuffle

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('heart.csv')

X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values

highest = 0

best_fold = 0

best_K = 0

for i in range (2,50):
    for j in range (2,50):

        kf = KFold(n_splits=i, shuffle = True)

        knn = KNeighborsClassifier(n_neighbors=j)

        accuracies = []

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            accuracies.append(accuracy)

        average = sum(accuracies) / len(accuracies) * 100

        if highest < average:

            highest = average

            best_fold = i

            best_K = j

        print(f'folds: {i} | K:{j}\nAverage accuracy: {average:.2f}%\n')

print(f'\nbest:{highest}\nfold:{best_fold} | K:{best_K}')
