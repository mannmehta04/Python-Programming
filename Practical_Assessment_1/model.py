import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

np.random.seed(42)

num_samples = 1000

p_cores = np.random.randint(1, 9, num_samples)
e_cores = np.random.randint(1, 17, num_samples)
base_speed = np.random.uniform(2.0, 4.0, num_samples)

performance = 50 * p_cores + 20 * e_cores + 10 * base_speed + np.random.normal(0, 5, num_samples)

data = pd.DataFrame({'p_cores': p_cores, 'e_cores': e_cores, 'base_speed': base_speed, 'performance': performance})

print(data.head())

features = data[['p_cores', 'e_cores', 'base_speed']]
target = data['performance']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

plt.scatter(y_test, predictions)
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs. Predicted CPU Performance')
plt.show()