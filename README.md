# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 2.Set variables for assigning dataset values. 3.Import linear regression from sklearn. 4.Assign the points for representing in the graph. 5.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:REXLIN R
Register Number:212222220034

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='gold')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='pink')
plt.title('Hours vs Scores(Test Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![image](https://github.com/rexlinrajan2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119406566/fa128769-a5ee-4ab8-96be-36d9989dfbc1)
![image](https://github.com/rexlinrajan2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119406566/e446e77c-4082-4572-a2f6-ae96cc2fe6fd)
![image](https://github.com/rexlinrajan2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119406566/7f7796b8-044f-4408-9c48-11af284376c4)
![image](https://github.com/rexlinrajan2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119406566/06c589a2-9a2c-4f24-9ca8-6dce58aef77f)
![image](https://github.com/rexlinrajan2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119406566/3cbcd2ac-9381-475e-a219-aaf29b483eb2)
![image](https://github.com/rexlinrajan2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119406566/e00ab984-ff2f-4e17-a6e7-2f340ab4ce27)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
