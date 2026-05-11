# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function compute Cost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: NAVEEN M
RegisterNumber:212225230197
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    data = pd.read_csv("50_Startups.csv") 
except FileNotFoundError:
    try:
        data = pd.read_csv("Startup.csv")
    except FileNotFoundError:
        print("CSV file not found. Creating sample data...")
        np.random.seed(42)  # For reproducible results
        rd_spend = np.random.uniform(0, 200000, 50)  # R&D Spend values
        profit = rd_spend * 0.8 + np.random.normal(0, 10000, 50)  # Profit with some noise
        data = pd.DataFrame({'R&D Spend': rd_spend, 'Profit': profit})

X = data['R&D Spend'].values
y = data['Profit'].values

X = (X - X.mean()) / X.std()

m = 0
b = 0
learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)
y_pred = m * X + b
plt.scatter(X, y, alpha=0.7, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on Startup Dataset")
plt.legend()
plt.show()
```

## Output:

<img width="850" height="605" alt="Screenshot 2026-05-11 063915" src="https://github.com/user-attachments/assets/b46063ad-4430-46b8-860e-c026f06acd9b" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
