from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = read_csv('heart.csv')

X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values

best_test_size = 0

best_accuracy = 0

best_n = 0

for i in range (5,23):

    for j in range (1,20):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i/25,shuffle=True)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=j)

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)*100

        print(f'size:{i/25} | n:{j}\naccuracy:{accuracy}')

        if accuracy > best_accuracy:
            best_test_size = i/25
            best_accuracy = accuracy
            best_n = j

print(f'best accuracy:{best_accuracy}\nsize:{best_test_size} | n:{best_n}')
