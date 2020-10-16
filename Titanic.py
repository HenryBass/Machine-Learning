import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

Generations = 1000

print("Titanic Survival Predictor\n#################################")

data = pd.read_csv("titanic.csv", sep=",")

data = data[["Survived", "Sex", "Pclass", "Age"]]
data = data.replace("female", 0).replace("male", 1).replace(" ", "")

predict = "Survived"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
best = 0
attempts = np.array([])
score = np.array([])
for i in range(Generations):
    attempts = np.append(attempts, i, axis=None)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Generation: " + str(i) + " Out of: " + str(Generations))
    score = np.append(score,1 - best, axis=None)
    if acc > best:
        best = acc
        with open("titanic.pickle", "wb") as f:
            pickle.dump(linear, f)

print("\nFinal Model Accuracy: " + str(best) + "\n")

pickle_in = open("titanic.pickle", "rb")
linear = pickle.load(pickle_in)
predictions = linear.predict(x_test)


for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print("Key: Survived Prediction(1 = Survived), Sex(1 = Male), Passenger Class, Age, Actual Survived(1 = Survived)")
style.use("ggplot")
plt.plot(attempts, score)
plt.title("Loss Over " + str(Generations) + " Generations:", fontsize = 20)
plt.show()
while True:
    pass