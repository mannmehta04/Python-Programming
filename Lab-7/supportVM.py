from sklearn import svm
import numpy as np

X_train = np.array([[1, 2], [1, 4], [1, 0], [4, 0]])
y_train = np.array([0, 0, 1, 1])

def svm_classifier(X_train, y_train):
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

svm_model = svm_classifier(X_train, y_train)

predicted_labels = svm_model.predict(X_train)
print("Predicted labels:", predicted_labels)
