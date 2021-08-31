from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

# leggo i dati di training dai csv
train_input = pd.read_csv('./mobile_price_dataset/train_input.csv')
train_output = pd.read_csv('./mobile_price_dataset/train_output.csv')

print('Contenuto dataset:')
print(train_input.describe().T)

# leggo i dati di test dai csv
test_input = pd.read_csv('./mobile_price_dataset/test_input.csv')
test_output = pd.read_csv('./mobile_price_dataset/test_output.csv')

column_length = train_input.shape[1]

X_train = np.array(train_input)
y_train = np.array(train_output)

X_test = np.array(test_input)
y_test = np.array(test_output)

# n input
N_FEATURES = X_test.shape[1]

# adam = stochastic gradient optimizer
mlp = MLPClassifier(
    solver='adam',
    max_iter=100000,
    hidden_layer_sizes=(N_FEATURES*2, 2),
    learning_rate_init=0.0001,
)
classifier = Pipeline([('scaler', StandardScaler()), ('mlp', mlp)])

# alleno la rete
classifier.fit(X_train, y_train)

y_predicted = classifier.predict(X_test)
print('\nOutput costo smartphone: \n')

print('0 - low cost\n1 - medium cost\n2 - high cost\n3 - very high cost\n')

print('Output: ', y_predicted)

accuracy = classifier.score(X_test, y_test)
train_accuracy = classifier.score(X_train, y_train)

# print('Loss: ', classifier.loss_)
print('Accuracy: ', accuracy)

print('Train accuracy: ', train_accuracy)

# funzione per calcolare l'importanza di ciascuna feature
def get_importanza_feature(j, n):
    base_score = accuracy_score(y_test, y_predicted)
    total = 0.0
    for i in range(n):
        permutazione = np.random.permutation(range(X_test.shape[0]))
        X_test_ = X_test.copy()
        X_test_[:, j] = X_test[permutazione, j]
        y_pred_ = classifier.predict(X_test_)
        average_score = accuracy_score(y_test, y_pred_)
        total += average_score
    return base_score - total / n

importance = []

for j in range(N_FEATURES):
  feature = get_importanza_feature(j, 100)
  importance.append(feature)

plt.figure(figsize=(10, 5))
plt.bar(range(N_FEATURES), importance, color="r", alpha=0.7)
plt.xticks(ticks=range(N_FEATURES), labels=test_input.columns.tolist(), rotation='vertical')

plt.subplots_adjust(bottom=0.5)

plt.ylabel("Importanza")
plt.title("Importanza delle features")
plt.show()

