import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X, y)

plt.figure(figsize=(12, 8))
tree.plot_tree(decision_tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

tree_rules = export_text(decision_tree, feature_names=iris.feature_names)
print("Decision Tree Rules:")
print(tree_rules)