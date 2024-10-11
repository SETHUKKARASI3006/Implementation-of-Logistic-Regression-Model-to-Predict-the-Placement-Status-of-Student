# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Data Preprocessing: Load dataset, drop unnecessary columns, and label encode categorical features.

2. Feature Selection: Separate features (X) and labels (Y), then split the data into training and testing sets.
 
3. Model Training: Train a Logistic Regression model using the training data.
 
4. Prediction and Evaluation: Predict labels for test data and evaluate model performance using confusion matrix and accuracy score.


## Program and Outputs:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```
<br>

```
# import modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
<br>

```
dataset = pd.read_csv("Placement_Data_Full_Class.csv")
dataset.head()
```
![output1](/1.png)
<br>

```
dataset.tail()
```
<br>

![output2](/2.png)

```
dataset.info()
```
<br>

![output3](/3.png)
<br>

```
# dropping the serial 
dataset = dataset.drop('sl_no',axis=1)
dataset.info()
```
<br>

![output4](/4.png)
<br>

```
#categorizing column for further labelling
dataset['gender'] = dataset['gender'].astype('category')
dataset['ssc_b'] = dataset['ssc_b'].astype('category')
dataset['hsc_b'] = dataset['hsc_b'].astype('category')
dataset['degree_t'] = dataset['degree_t'].astype('category')
dataset['workex'] = dataset['workex'].astype('category')
dataset['specialisation'] = dataset['specialisation'].astype('category')
dataset['status'] = dataset['status'].astype('category')
dataset['hsc_s'] = dataset['hsc_s'].astype('category')
dataset.dtypes
```
<br>

![output5](/5.png)
<br>

```
#Labelling the columns
dataset['gender']=dataset['gender'].cat.codes
dataset['ssc_b']=dataset['ssc_b'].cat.codes
dataset['hsc_b']=dataset['hsc_b'].cat.codes
dataset['degree_t']=dataset['degree_t'].cat.codes
dataset['workex']=dataset['workex'].cat.codes
dataset['specialisation']=dataset['specialisation'].cat.codes
dataset['status']=dataset['status'].cat.codes
dataset['hsc_s']=dataset['hsc_s'].cat.codes

```
<br>

```
# display dataset
dataset.head()
```
<br>

![output6](/6.png)
<br>

```
#Selecting the features and labels
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
```
<br>

```
#Analyse the shape of independent variable
X.shape
```
<br>

![output7](/7.png)
<br>

```
#Analyse the shape of dependent variable.
Y.shape
```
<br>

![output8](/8.png)
<br>

```
# Split the X and Y data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=11)
```
<br>

```
X_train.shape
```
<br>

![output9](/9.png)
<br>

```
X_test.shape
```
<br>

![output10](/10.png)
<br>

```
Y_train.shape
```
<br>

![output11](/11.png)
<br>

```
Y_test.shape
```
<br>

![output12](/12.png)
<br>

```
# Creating a classifier using sklearn.
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
```
<br>

![output13](/13.png)
<br>

```
#Predicting the y values
Y_pred=clf.predict(X_test)
Y_pred
```
<br>

![output14](/14.png)
<br>

```
from sklearn.metrics import confusion_matrix, accuracy_score
cf=confusion_matrix(Y_test,Y_pred)
cf
```
<br>

![output15](/15.png)
<br>

```
accuracy=accuracy_score(Y_test,Y_pred)
accuracy
```
<br>

![output16](/16.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
