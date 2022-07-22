import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics, datasets
from sklearn.metrics import r2_score
#matplotlib inline

#load the boston dataset
boston = datasets.load_boston(return_X_y=False)

print(boston.data.shape)
print(boston.target.shape)
# print feature names of the columns
print(boston.feature_names)
#Print dataset characteristics 
print (boston.DESCR)

#convert the Dataset captured in the boston variable into the pandas DataFrame
boston_df = pd.DataFrame(boston.data)
# create the DF column names from boston feature names
boston_df.columns = boston.feature_names
print(boston_df.head())

# split the data into X and Y
X = boston_df
y = boston.target  # Boston Housing Price

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


#create the instance or object for our Linear Regression model
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)


m = lr.coef_

print('Value for m (slope): \n', m)

c = lr.intercept_

print ('Value of c (Intercept) ', c)

#plot regression model and scatter plot 
plt.scatter(y_test, y_pred, color = "b", marker = "o" ) 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = 'g')
# putting labels 
plt.xlabel('Actual House Price') 
plt.ylabel('Predicted Price') 
# function to show plot 
plt.show() 

print(r2_score(y_test, y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


plt.figure(2)
seabornInstance.distplot(y_test, hist=False, label="Actual Value")
seabornInstance.distplot(y_pred, hist=False, label="Predicted Value")
