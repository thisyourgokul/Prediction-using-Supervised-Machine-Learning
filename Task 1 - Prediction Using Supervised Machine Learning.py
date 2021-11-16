#!/usr/bin/env python
# coding: utf-8

# TASK 1 : PREDICTION USING SUPERVISED MACHINE LEARNING.
#          Predict the Percentage of a Student based on number of Study Hours. 
#          It is a Simple Linear Regression task involving 2 variables.
#          
# LEVEL  : Beginner
# 
# DATA SET USED : http://bit.ly/w-data
# 
# EXAMPLE QUESTION : What will be the Predicted Study Hours if the Student studies for 9.25Hours/Day?
# 
# NAME OF THE AUTHOR : Gokul Raj, Data Science and Business Analytics Intern, The Sparks Foundation.
# 
#          

# In[59]:


#importing all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


#Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[61]:


#plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')


# In[62]:


#Divide the data into attributes and labels
x=s_data.iloc[:, :-1].values
y=s_data.iloc[:, 1].values


# In[63]:


#Split the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[64]:


#Training the algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("Training complete")


# In[65]:


#Plotting the regression line
line=regressor.coef_*x+regressor.intercept_

#plotting for the test data
plt.scatter(x,y)
plt.plot(x, line)
plt.show()


# In[66]:


#Testing data in hours
print(x_test)
#Predicting the scores
y_pred=regressor.predict(x_test)


# In[67]:


#Comparing Actual Value vs Predicted Value
df=pd.DataFrame({'Actual Value':y_test, 'Predicted Value':y_pred})
df




# In[68]:


#Testing with my own Data

hours=[9.25]
own_pred = regressor.predict([hours])
print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[69]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

