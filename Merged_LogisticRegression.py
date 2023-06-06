#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import IterativeImputer
pd.set_option('display.width', 180)
plt.style.use('ggplot')
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer
import math
import scipy.stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[7]:


# import the merged dataset
merged_df = pd.read_csv('merged_dataset.csv')


# In[38]:


# split the encoded features into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('target', axis=1),
    merged_df['target'],
    test_size=0.2,  # adjust the test size as desired (e.g., 0.2 for 20% test data)
    random_state=38  # set a random seed for reproducibility
)

# create an instance of a Random Forest classifier
logreg = LogisticRegression(penalty = 'l2', C = 0.1, random_state=38, max_iter = 5000, solver = 'lbfgs')

# train the model
logreg.fit(X_train, y_train)

# generate predictions
predictions = logreg.predict(X_test)

# calculate the F1 score
f1 = f1_score(y_test, predictions)

# Print the F1 score
print("F1 score:", f1)


# In[30]:


# Create an instance of a Logistic Regression classifier
logreg = LogisticRegression(penalty = 'l2', solver = 'lbfgs', C = 0.1, random_state=38)

# Perform 10-fold cross-validation with F1 score
f1_scores = cross_val_score(logreg, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring="f1", n_jobs=-1)

# Print the mean and standard deviation of the F1 scores
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score
# 
# # Define the parameter grid for the grid search
# param_grid = {
#     'penalty': ['l1', 'l2', None],
#     'C': [0.1, 1.0, 10.0],
#     'solver': ['liblinear', 'saga', 'lbfgs']
# }
# 
# # Create an instance of Logistic Regression
# logreg = LogisticRegression(random_state=38)
# 
# # Perform grid search using 5-fold cross-validation
# grid_search = GridSearchCV(logreg, param_grid, cv=10, scoring='f1', verbose = 2)
# grid_search.fit(merged_df.drop('target', axis=1), merged_df['target'])
# 
# # Print the best parameters and best score
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# Best parameters:  {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}
# Best score:  0.8159145776086516

# In[39]:


# Create an instance of a Logistic Regression classifier
logreg = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.1, random_state=38)

# Perform 10-fold cross-validation with F1 score
f1_scores = cross_val_score(logreg, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring="f1", n_jobs=-1)

# Print the mean and standard deviation of the F1 scores
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


# In[40]:


# split the encoded features into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('target', axis=1),
    merged_df['target'],
    test_size=0.2,  # adjust the test size as desired (e.g., 0.2 for 20% test data)
    random_state=38  # set a random seed for reproducibility
)

# create an instance of a Random Forest classifier
logreg = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.1, random_state=38, max_iter = 5000)

# train the model
logreg.fit(X_train, y_train)

# generate predictions
predictions = logreg.predict(X_test)

# calculate the F1 score
f1 = f1_score(y_test, predictions)

# Print the F1 score
print("F1 score:", f1)

