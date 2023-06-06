#!/usr/bin/env python
# coding: utf-8

# In[11]:


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


# In[12]:


# read in the data
heart = pd.read_excel('heart_disease.xlsx').iloc[:899,:]
df_module1 = pd.read_csv('Module1_dataset.csv', index_col = 0)
df_module2 = pd.read_csv('spark_df.csv', index_col = 0)
smoking_rates = pd.read_csv('smoking_rates.csv', index_col = 0)


# In[13]:


print(list(df_module1.columns), '\n')
print(list(df_module2.columns))


# There's a lot of ways that we can merge these two datasets, but considering that the first includes all variables from the second module, except smoke columns and painexer, we'll consider that dataframe the base and add/replace columns from the second module/scraped data
# 
# Due to the limitations of spark, the imputation done between both modules heavily differed. The first module utilized multiple imputation with an xgboost regressor to impute numerical variables while the second module utilized median/minimum/maximum imputation depending on the context. Furthermore, for categorical variables, the first module simply imputed a missing column, while the second module imputed a 'missing' category or the column's most frequent value depending on the context.
# 
# In general, xgboost is much more robust than the other aforementioned forms of imputation, so we'll stick with the imputed numerical columns from the first module. Though, the first module did not address tresbps values under 100 mmHg and oldpeak values outside of the range [0,4]. This is something we can merge between the two.
# 
# For categorical columns, the following was done in the second module:
# 
# fbs, prop, nitr, pro, diuretic: Replace the missing values and values greater than 1
# painloc, painexer, exang, slope: Replace the missing values
# 
# The first module's method of just replacing the missing values with a new missing category (including values greater than 1) is overall better for reducing bias, so we'll stick with those columns. Though, painexer isn't used in the first module, pncaden was used instead to encapsulate more information. We'll stick with that column as during the first round of EDA, that variable appeared more useful.
# 
# As for the smoke columns, these were left out of the first module for the amount of missing values. In the second module, we used data scraped online to impute the missing values. The first source only used the age group, while the second included patient's sex as well. As talked about in the webscraping section of this module, the dataset is from 1998, so the data we scraped in module 2 is long after the time of our dataset. In turn, we scraped some data from the new source given to us in this module to try to scale the smoking rates to more accurately represent what they might have been in that time.
# 
# We don't want 3 entire columns dedicated to imputed smoke values as this is redundant. We'll keep the imputed values from source 2 as they provide more information and not use column from source 1. Then we'll include a new imputed smoke column which uses the new smoking rates we calculated from combining the three sources' information.

# In[14]:


import numpy as np

# replace oldpeak values less than 0 with 0 and values greater than 4 with 4
df_module1['oldpeak'] = np.clip(df_module1['oldpeak'], 0, 4)

# replace trestbps values under 100 with 100
df_module1.loc[df_module1['trestbps'] < 100, 'trestbps'] = 100


# In[15]:


# add the imputed smoke column to the df
df_module1['smoke_imputed_source2'] = df_module2['smoke_imputed_source2']

# add the original smoke column to the df to impute
df_module1['smoke'] = heart['smoke']
df_module1['smoke'] = df_module1['smoke'].fillna(1000)


# In[16]:


smoking_rates


# In[17]:


keys = list(smoking_rates['age_group'])
male_rates = list(smoking_rates['smoking_rate_male_adjusted'])
female_rates = list(smoking_rates['smoking_rate_female_adjusted'])
length = len(keys)

def get_smoking_rate(x):
    
    if x['smoke'] != 1000:
        return x['smoke']
    
    age = x['age'] 
    sex = x['sex']
    
    if sex == 0:
        for i in range(length - 1):
            # extract the lower and upper bounds of the age range
            lower = int(keys[i].split('–')[0])
            upper = int(keys[i].split('–')[1])
            
            # check if the age falls within the range
            if age in range(lower, upper + 1):
                return female_rates[i]
        
        # if age is outside the provided ranges, use the last value
        return female_rates[length - 1]
    
    else:
        for i in range(length - 1):
            # extract the lower and upper bounds of the age range
            lower = int(keys[i].split('–')[0])
            upper = int(keys[i].split('–')[1])
            
            # check if the age falls within the range
            if age in range(lower, upper + 1):
                return male_rates[i]
        
        # if age is outside the provided ranges, use the last value
        return male_rates[length - 1]
        


# In[18]:


# apply to new function to get a new imputed smoke column
df_module1['smoke_new_imputed'] = df_module1[['smoke', 'age', 'sex']].apply(get_smoking_rate, axis = 1)
df_module1 = df_module1.drop('smoke', axis = 1)


# In[19]:


df_module1['smoke_new_imputed']


# Clearly, we kept most columns from the first module and only the imputed column from the second module. This is because in the first module, we were much less strict on removing columns than the given columns were in the second module. Furthermore, pyspark's integration of more complex forms of imputation, such as the multiple imputation with xgboost regressors done in module 1, is not nearly as good as that of sklearn. Because of this, the imputed columns that both modules share are more likely to be less biased from the first module.
# 
# The only columns from module 2 which provided completely new information were the imputed smoke columns, and with a new imputed column needed to included the data from the new web source in this module, it would have been redundant to keep both.

# In[21]:


df_module1.to_csv('merged_dataset.csv')

