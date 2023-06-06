#!/usr/bin/env python
# coding: utf-8

# In[99]:


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math
#%%


# In[3]:


# source 1: Smoking rates by age group
url_source1 = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release"
response_source1 = requests.get(url_source1)
soup_source1 = BeautifulSoup(response_source1.content, "html.parser")

# extract smoking rates by age group
smoking_rates_source1 = {}
table_source1 = soup_source1.find("table", class_="responsive-enabled")
rows_source1 = table_source1.find_all("tr")
for row in rows_source1[1:]:
    age_groups = row.find_all("th")
    columns = row.find_all("td")
    age_group = age_groups[0].text.strip()
    smoking_rate = float(columns[0].text.strip())
    smoking_rates_source1[age_group] = smoking_rate

# send a GET request to the website
url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"
response = requests.get(url)

# create a BeautifulSoup object
soup = BeautifulSoup(response.text, "html.parser")

# find the relevant section containing smoking rates by sex
div_element = soup.find_all('div', class_='card border-0 rounded-0 mb-3')[0]
li_elements = div_element.find_all('li')

# extract smoking rates by sex
percentage_list = []


for li_element in li_elements:
    percentage_text = li_element.text.strip()  # Get the text content and remove leading/trailing spaces
    
    # extract the percentage using regular expression
    percentage_match = re.search(r'(\d+\.\d+)%', percentage_text)
    if percentage_match:
        percentage = round(float(percentage_match.group(1))/100, 4)
        percentage_list.append(percentage)
        
# find the relevant section containing smoking rates by age group
div_element = soup.find_all('div', class_='card border-0 rounded-0 mb-3')[1]
li_elements = div_element.find_all('li')
age_groups2 = []
percentage_list2 = []

for li_element in li_elements:
    percentage_text = li_element.text.strip()  # Get the text content and remove leading/trailing spaces
    
    # extract the percentage using regular expression
    percentage_match = re.search(r'(\d+\.\d+)%', percentage_text)
    if percentage_match:
        percentage = round(float(percentage_match.group(1))/100, 4)
        percentage_list2.append(percentage)
        
    # extract the 
    age_range_match = re.search(r"aged (\d+–\d+)", percentage_text)
    if age_range_match:
        age_range = age_range_match.group(1)
        age_groups2.append(age_range)

print(smoking_rates_source1)

print(percentage_list)
print(age_groups2)
print(percentage_list2)

#%%


# In[4]:


# source 1: Smoking rates by age group
url_source = "https://wayback.archive-it.org/5774/20211119125806/https:/www.healthypeople.gov/2020/data-search/Search-the-Data?nid=5342"
response_source = requests.get(url_source)
soup = BeautifulSoup(response_source.content, "html.parser")


# In[5]:


soup_titles = soup.find_all(class_='ds-inner-poptitle')

titles = []

for soup_title in soup_titles:
    titles.append(soup_title.text)

titles


# In[15]:


soup_dat


# In[18]:


soup_data = soup.find_all(class_='ds-data-point ds-1999')

data = []

for soup_dat in soup_data:
    estimate = soup_dat.find(class_ = 'dp-data-estimate')
    if estimate is not None:
        data.append(estimate.text)

removables = []

for item in removables:
    data.remove(item)
    
data


# In[19]:


indices = [2,3]

panda_dict = {}

for i in indices:
    panda_dict[titles[i]] = data[i]
    
panda_dict


# In[20]:


print(smoking_rates_source1)
print(percentage_list)
print(age_groups2)
print(percentage_list2)
print(panda_dict)


# The goal here is to combine all of the data to merge with the other two datasets.
# 
# Across the three sources, we have smoking rates based on people's sex, age, race, region, state, education, income, mental health status, and health insurance.
# 
# However, the only information we have on each patient from these factors is sex and age. Trying to blindly utilize rates without knowing the proper information would likely introduce a lot of unnecessary bias into the model, so it's best to just stick to age and sex.
# 
# This heart disease dataset is from 1998, but the data from the former 2 of the 3 sources is based on 2018 and 2021. Considering that smoking rates have likely drastically changed since that time, using the rates from 2018 and 2021 is likely inaccurate. The first source contains data dating back to 1999, but it's only data on adolescents.
# 
# Not only are the smoking rates inaccurate just in general, but even the way they are distributed amongst age groups is likely highly inaccurate as well. Not only was smoking more popular in 1998 than today, but it was comparetively much more popular amongst young people then than young people now due to the success of anti-smoking movements stigmatizing smoking cigarettes as gross. This means that we can't just use the adolescent rates to factor up the age group smoking rates by a constant factor.
# 
# Ideally, we could have just originally scraped data from 1999, but part of this project is to be creative in how we address this issue with the sites we were given.
# 
# Proposition:
# First combine the rates by age group from the first two sources, by averaging rates from overlapping age groups. Then, split them by sex, with each sex scaled by (average smoking rate between sexs in 1999)/(average smoking rate between sexs in 2021) for females and that same factor but multiplied by (male smoking rate in 2021)/(female smoking rate in 2021)/n for males. The reason for the extra division by n is that the smoke rates of adolescents being so similar back then points to the possibility that females and males smoked in much closer proportions rather than in the 2021 data. We'll choose a value of n that still keeps the male factor larger than the female factor while reducing the difference.
# 
# Furthermore, we'll multiply the rates by some ratio 1/ln(e + i/r), where i is the index for each age group and r is some chosen parameter. Including e allows the function to be 1 when the index, i, is 0. As i increases, the ratio decreases so that older age groups are not as affected by the upscaling. Including r allows us to toggle how much the ratio decreases as i gets larger; higher values of r make it so that i has less of an effect.
# 
# Again, this isn't a perfect way to do this, but with the lack of the necessary information to have the most unbiased imputation of the smoke column, this isn't a bad solution.

# In[31]:


# average age group rates
smoking_rates = {}
smoking_rates['18–24'] = (smoking_rates_source1['18–24']/100 + percentage_list2[0])/2
smoking_rates['25–34'] = (smoking_rates_source1['25–34']/100 + percentage_list2[1])/2
smoking_rates['35–44'] = (smoking_rates_source1['35–44']/100 + percentage_list2[1])/2
smoking_rates['45–54'] = (smoking_rates_source1['45–54']/100 + percentage_list2[2])/2
smoking_rates['55–64'] = (smoking_rates_source1['55–64']/100 + percentage_list2[2])/2
smoking_rates['65–74'] = (smoking_rates_source1['65–74']/100 + percentage_list2[3])/2
smoking_rates['75+'] = (smoking_rates_source1['75+']/100 + percentage_list2[3])/2

smoking_rates


# In[38]:


# import data into panda dataframe
dict_panda = {
    'age_group' : list(smoking_rates.keys()),
    'smoking_rate' : list(smoking_rates.values())
}

pd_smoke = pd.DataFrame(dict_panda)
pd_smoke


# In[147]:


# calculate the scale factors by sex for data
average_1999 =(float(panda_dict['Male ']) + float(panda_dict['Female ']))/2/100
average_2021 = (percentage_list[0] + percentage_list[1])/2

female_factor = average_1999 / average_2021
male_factor = female_factor * (percentage_list[0] / percentage_list[1] / 1.1)


# In[148]:


female_factor


# In[149]:


male_factor


# In[66]:


# apply the factors
pd_smoke['smoking_rate_male'] = pd_smoke['smoking_rate'].apply(lambda x: x*male_factor)
pd_smoke['smoking_rate_female'] = pd_smoke['smoking_rate'].apply(lambda x: x*female_factor)
pd_smoke


# In[123]:


# calculate the list of factors
factors = []
for i in range(0,len(pd_smoke)):
    factors.append(1/np.log(math.e + i/1.8))

factors


# In[127]:


pd_smoke['smoking_rate_male_adjusted'] = pd_smoke['smoking_rate_male']*factors
pd_smoke['smoking_rate_female_adjusted'] = pd_smoke['smoking_rate_female']*factors
pd_smoke


# In[151]:


pd_smoke[['age_group', 'smoking_rate_male_adjusted', 'smoking_rate_female_adjusted']].to_csv('smoking_rates.csv')

