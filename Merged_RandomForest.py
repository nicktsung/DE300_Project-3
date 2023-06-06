#!/usr/bin/env python
# coding: utf-8

# In[33]:


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
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import QuantileTransformer
import math
import scipy.stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# In[6]:


# import the merged dataset
merged_df = pd.read_csv('merged_dataset.csv')


# In[44]:


# split the encoded features into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('target', axis=1),
    merged_df['target'],
    test_size=0.2,  # adjust the test size as desired (e.g., 0.2 for 20% test data)
    random_state=38  # set a random seed for reproducibility
)

# create an instance of a Random Forest classifier
rfc = RandomForestClassifier(n_estimators = 15, max_depth = 9, random_state=38)

# train the model
rfc.fit(X_train, y_train)

# generate predictions
predictions = rfc.predict(X_test)

# calculate the F1 score
f1 = f1_score(y_test, predictions)

# Print the F1 score
print("F1 score:", f1)


# In[49]:


# create an instance of a Random Forest classifier
rfc = RandomForestClassifier(n_estimators = 15, max_depth = 8, random_state=38)

# perform 10-fold cross validation with F1 score
f1_scores = cross_val_score(rfc, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring="f1", n_jobs = -1)

# print the mean and standard deviation of the F1 scores
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score
# 
# # Define the parameter grid for the grid search
# param_grid = {
#     'n_estimators': [15, 30, 50, 100],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }
# 
# # Create an instance of Logistic Regression
# rfc = RandomForestClassifier(random_state=38)
# 
# # Perform grid search using 5-fold cross-validation
# grid_search = GridSearchCV(rfc, param_grid, cv=10, scoring='f1', verbose = 3)
# grid_search.fit(merged_df.drop('target', axis=1), merged_df['target'])
# 
# # Print the best parameters and best score
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# Best parameters:  {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
# Best score:  0.8176791355530921

# In[54]:


# create an instance of a Random Forest classifier
rfc = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_split = 5, random_state=38)

# perform 10-fold cross validation with F1 score
f1_scores = cross_val_score(rfc, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring="f1", n_jobs = -1)

# print the mean and standard deviation of the F1 scores
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


# In[55]:


# split the encoded features into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('target', axis=1),
    merged_df['target'],
    test_size=0.2,  # adjust the test size as desired (e.g., 0.2 for 20% test data)
    random_state=38  # set a random seed for reproducibility
)

# create an instance of a Random Forest classifier
rfc = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_split = 5, random_state=38)

# train the model
rfc.fit(X_train, y_train)

# generate predictions
predictions = rfc.predict(X_test)

# calculate the F1 score
f1 = f1_score(y_test, predictions)

# Print the F1 score
print("F1 score:", f1)

