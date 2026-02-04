# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset and separate input features (X) and target labels (y).

2.Split the dataset into training and testing sets.

3.Initialize the SGD Classifier with logistic loss.

4.Train the model using the training data.

5.Predict on test data and evaluate the model performance.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: S.TINKU
RegisterNumber:  25006607

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(
    loss="log_loss",
    max_iter=1000,
    learning_rate="optimal",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

*/
```

## Output:


<img width="567" height="339" alt="Screenshot 2026-02-04 194227" src="https://github.com/user-attachments/assets/24afde04-c370-45ca-97ec-25a3e2a0ac7d" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
