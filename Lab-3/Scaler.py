from pandas import read_csv
import pandas as pd
from numpy import set_printoptions
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

file = r'./heart.csv'

name = ['age','sex','bp','chol']

data = read_csv(file, names=name)

data = data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
data = data.dropna()
array = data.values

dataScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataRescaled = dataScaler.fit_transform(array)
set_printoptions(precision=1)

print("MinMax")
print(dataRescaled[0:2])

dataScaler = StandardScaler().fit(array)
rescaled = dataScaler.transform(array)
set_printoptions(precision=2)

print()
print("StdScaler")
print(rescaled[0:2])

normalizer = Normalizer(norm='l1').fit(array)
normalized = normalizer.transform(array)
set_printoptions(precision=2)

print()
print("Normalized")
print(normalized[0:3])

binarizer = Binarizer(threshold=0.5).fit(array)
binarized = binarizer.transform(array)

print()
print("Binarized")
print(binarized[0:3])

onehot_encoder = OneHotEncoder(drop='first')
onehot_encoded = onehot_encoder.fit_transform(data[['age', 'sex']])

print()
print("One hot encoding")
print(onehot_encoded[0:3])

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data['age'])

print()
print("label encoder")
print(encoded_labels[0:5])