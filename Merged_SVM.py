#!/usr/bin/env python
# coding: utf-8

# In[23]:


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
from sklearn.svm import SVC


# In[24]:


# import the merged dataset
merged_df = pd.read_csv('merged_dataset.csv')


# In[25]:


# Split the encoded features into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('target', axis=1),
    merged_df['target'],
    test_size=0.2,  # Adjust the test size as desired (e.g., 0.2 for 20% test data)
    random_state=38  # Set a random seed for reproducibility
)

# Create an instance of an SVM classifier
svm = SVC(C=1.0, kernel='rbf', random_state=38)

# Train the model
svm.fit(X_train, y_train)

# Generate predictions
predictions = svm.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, predictions)

# Print the F1 score
print("F1 score:", f1)


# In[31]:


# Create an instance of an SVM classifier
svm = SVC(C=1.0, kernel='rbf', random_state=38)

# Perform 10-fold cross-validation with F1 score
f1_scores = cross_val_score(svm, merged_df.drop('target', axis=1), merged_df['target'], cv=5, scoring='f1', n_jobs=-1)

# Print the mean and standard deviation of the F1 scores
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score
# 
# # Define the parameter grid for the grid search
# param_grid = {
#     'C': [0.1, 1.0],
#     'kernel': ['linear', 'rbf', 'poly']
# }
# 
# # Create an instance of SVM
# svm = SVC(random_state=38)
# 
# # Perform grid search using 5-fold cross-validation
# grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='f1', verbose=3)
# grid_search.fit(merged_df.drop('target', axis=1), merged_df['target'])
# 
# # Print the best parameters and best score
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# Best Parameters: C=0.1, kernel='linear', random_state=38

# In[38]:


# Create an instance of an SVM classifier
svm = SVC(C=0.1, kernel='linear', random_state=38)

# Perform 10-fold cross-validation with F1 score
f1_scores = cross_val_score(svm, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring='f1', n_jobs=-1)

# Print the mean and standard deviation of the F1 scores
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))


# In[37]:


# Split the encoded features into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    merged_df.drop('target', axis=1),
    merged_df['target'],
    test_size=0.2,  # Adjust the test size as desired (e.g., 0.2 for 20% test data)
    random_state=38  # Set a random seed for reproducibility
)

# Create an instance of an SVM classifier
svm = SVC(C=0.1, kernel='linear', random_state=38)

# Train the model
svm.fit(X_train, y_train)

# Generate predictions
predictions = svm.predict(X_test)

# Calculate the F1 score
f1 = f1_score(y_test, predictions)

# Print the F1 score
print("F1 score:", f1)

