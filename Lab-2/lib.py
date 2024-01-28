# Import necessary libraries
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

# Introduction to Matplotlib for data visualization
x_values = np.linspace(0, 10, 100)
y_values = np.sin(x_values)

plt.plot(x_values, y_values, label='Sin(x)')
plt.title('Matplotlib Example - Sin(x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# Introduction to Loading .csv files using different packages
# Using Standard Library
csv_file_path = 'example.csv'

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)

# Using Pandas
df = pd.read_csv(csv_file_path)
print("\nDataFrame using Pandas:")
print(df)

# Using NumPy
data_array = np.genfromtxt(csv_file_path, delimiter=',', names=True, dtype=None)
print("\nArray using NumPy:")
print(data_array)