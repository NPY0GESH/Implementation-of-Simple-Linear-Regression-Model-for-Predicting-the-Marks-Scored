# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries. 

2.Set variables for assigning dataset values. 3.Import linear regression from sklearn. 

3.Assign the points for representing in the graph. 

4.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given dates. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jeevanantham C
RegisterNumber: 212225240059 
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 1) Load dataset
df = pd.read_csv('exp_2_dataset_student_scores.csv')   # CSV should have two columns, e.g. "Hours","Scores"
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
```
<img width="157" height="271" alt="image" src="https://github.com/user-attachments/assets/b7606661-0a9a-43ca-8926-dfb555f3a1dc" />

```
# 2) Prepare input (X) and output (Y)
# Assume CSV columns: Hours (feature) and Scores (target)
X = df.iloc[:, :-1].values   # all rows, all columns except last -> shape (n_samples, 1)
Y = df.iloc[:, -1].values    # all rows, last column -> shape (n_samples,)
print("X (features):", X.flatten())
print("Y (targets):", Y)
```
<img width="677" height="75" alt="image" src="https://github.com/user-attachments/assets/ab07af16-b631-490f-b0ff-83d2ee321bad" />

```
# 3) Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
```
<img width="326" height="23" alt="image" src="https://github.com/user-attachments/assets/2a82b827-28e8-4e0d-ba23-3a48e932c65f" />

```
# 4) Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)   # fit on training data
```
<img width="212" height="70" alt="image" src="https://github.com/user-attachments/assets/295b365c-fd0b-4af4-9c5a-a522b8efbf75" />

```
# 5) Predict on the test set
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
```
<img width="571" height="40" alt="image" src="https://github.com/user-attachments/assets/cc0ebcdc-2e49-4f08-b9e0-a98e462f3253" />

```
# 6) Plot training results
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
<img width="367" height="253" alt="image" src="https://github.com/user-attachments/assets/37559e33-c85c-4c3e-ba50-d598f0c1ce5f" />


```
# 7) Plot testing results (use X_test sorted for a nicer line)
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```
<img width="366" height="261" alt="image" src="https://github.com/user-attachments/assets/7e7ccfab-f453-4c78-a2e5-29f85ae0a33e" />


```
# 8) Evaluation metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
```
<img width="387" height="62" alt="image" src="https://github.com/user-attachments/assets/47c561b4-c633-43ff-bb89-881e235ceb08" />

```
# 9) Example: predict for new students
new_hours = np.array([[2.5], [8.0]])   # shape must be (n_samples, 1)
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```
<img width="417" height="22" alt="image" src="https://github.com/user-attachments/assets/500a49f7-b153-43d2-81d2-0f1844f98868" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
