import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
import pandas as pd

data = pd.read_csv("./heart.csv")

x = data['age']
y = data['chol']

x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)

x_train = x[:-20]
x_test = x[-20:]

y_train = y[:-20]
y_test = y[-20:]

plt.scatter(x_test, y_test, color='blue')
plt.title("Heart Test Data")
plt.xlabel("Age Group")
plt.xlabel("Cholestrol Levels")
plt.xticks(())
plt.yticks(())

plt.show()

rgr = linear_model.LinearRegression()
rgr.fit(x_train, y_train)
plt.plot(x_test, rgr.predict(x_test), color='yellow', linewidth=3)