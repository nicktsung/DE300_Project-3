# DE300_Project-3

## Setup

### S3 Bucket
1. Create an S3 bucket.
2. Upload requirements.txt and heart_disease.csv.
3. Add a /dags folder.  
4. Upload workflow.py to be inside the /dags folder.

### MWAA
Steps from the class tutorial were followed to create Airflow environment.  Just be sure to include the right bucket folder, dags folder, and requirements file path.

## General Approach and Rationale

### Task 1: Web scraping  

The goal here is to combine all of the data to merge with the other two modules' datasets.

Across the three sources, we have smoking rates based on people's sex, age, race, region, state, education, income, mental health status, and health insurance.

However, the only information we have on each patient from these factors is sex and age. Trying to blindly utilize rates without knowing the proper information would likely introduce a lot of unnecessary bias into the model, so it's best to just stick to age and sex.

This heart disease dataset is from 1998, but the data from the 2 past sources is based on 2018 and 2021. Considering that smoking rates have likely drastically changed since that time, using the rates from 2018 and 2021 is probably inaccurate. The first source contains data dating back to 1999, but it's only data on adolescents.

Not only are the smoking rates inaccurate just in general, but even the way they are distributed amongst age groups is likely highly inaccurate as well. Not only was smoking more popular in 1998 than today, but it was comparetively much more popular amongst young people then than young people now due to the success of anti-smoking movements stigmatizing smoking cigarettes as gross in the 2000s. This means that we can't just use difference in the adolescent rates to factor up the smoking rates of all age groups by a constant factor as the distribution will still be highly inaccurate.

Ideally, we could have just originally scraped data from 1998, but part of this project is to be creative in how we address this issue with the sites we were given.

Proposition:
First combine the rates by age group from the first two sources, by averaging rates from overlapping age groups. Then, split them by sex, with each sex scaled by (average smoking rate between sexs of adolescents in 1999)/(average smoking rate between sexs in 2021) for females and that same factor for males, but multiplied by (male smoking rate in 2021)/(female smoking rate in 2021)/n. This allows us to still account for men smoking for often. Also, the reason for the extra division by n is that the smoke rates of adolescents being so similar back then points to the possibility that females and males smoked in much closer proportions rather than in the 2021 data. We'll choose a value of n that still keeps the male factor larger than the female factor while also reducing the difference between the two.

Furthermore, we'll multiply the rates by some ratio 1/ln(e + i/r), where i is the index for each age group (18-24 being index 0) and r is some chosen parameter. Including e allows the function to be 1 when i is 0. As i increases, the ratio decreases so that older age groups are not as affected by the upscaling. Including r allows us to toggle how much the ratio decreases as i gets larger; higher values of r make it so that i has less of an effect.

Again, this isn't a perfect way to do this, but with the lack of the necessary information to have the most unbiased imputation of the smoke column, this isn't a bad solution.

```
| age_group | smoking_rate | smoking_rate_male | smoking_rate_female | smoking_rate_male_adjusted | smoking_rate_female_adjusted |
|-----------|--------------|-------------------|---------------------|----------------------------|------------------------------|
| 18–24     | 0.0680       | 0.264594          | 0.2040              | 0.264594                   | 0.204000                     |
| 25–34     | 0.1160       | 0.451366          | 0.3480              | 0.380591                   | 0.293432                     |
| 35–44     | 0.1215       | 0.472767          | 0.3645              | 0.352100                   | 0.271467                     |
| 45–54     | 0.1425       | 0.554480          | 0.4275              | 0.375111                   | 0.289207                     |
| 55–64     | 0.1430       | 0.556426          | 0.4290              | 0.348317                   | 0.268550                     |
| 65–74     | 0.0860       | 0.334634          | 0.2580              | 0.196378                   | 0.151406                     |
| 75+       | 0.0585       | 0.227629          | 0.1755              | 0.126438                   | 0.097482                     |
```
As we can see, the smoking rates for both sexes heavily increased from the upscaling alone, without the extra adjustment ration of 1/ln(e + i/r). 55% smoking rate for any age group seems ridiculous, which is why the adjustment was included. While even 35% seems high, considering that 37.8% of high schoolers in 1999 had smoked a cigarette in the last 30 days according to the new source, this is more reasonable.

The final panda DataFrame only included the age group and adjusted smoking rates.

### Task 2: Workflow orchestration of EDA and EDA with spark
Converting the previous two modules to be compatible was relatively simple.  All that had to be done was to partition the code into individual tasks (cleaning, feature engineering, and modelling), then have dataframes saved as .csv files in between tasks, and finally make sure everything in the workflow works as intended.

### Task 3: Combining with smoking data

Merging of the datasets:

There's a lot of ways that we can merge these two datasets, but considering that the first includes all variables from the second module, except the smoke columns and painexer, we'll consider that dataframe the base and add/replace columns from the second module/scraped data where we see fit.

Due to the limitations of spark, the imputation done between both modules heavily differed. The first module utilized multiple imputation with an xgboost regressor to impute numerical variables while the second module utilized median/minimum/maximum imputation depending on the context. Furthermore, for categorical variables, the first module simply imputed a missing column for all of them, while the second module imputed a 'missing' category or the column's most frequent value depending on the context.

In general, multiple xgboost imputation is much more robust than the other aforementioned forms of imputation, as the latter imputes only a single value for all missing values. Thus, we'll stick with the imputed numerical columns from the first module. Though, the first module did not address tresbps values under 100 mmHg and oldpeak values outside of the range [0,4]. In the second module, we replaced values not in these ranges with the closest value in the range (i.e. 5 is replaced with 4 for oldpeak). This is something we can merge between the two.

For categorical columns, the following was done in the second module:

fbs, prop, nitr, pro, diuretic: Replace the missing values and values greater than 1 painloc, painexer, exang, slope: Replace the missing values

The first module's method of just replacing the missing values with a new missing category (including values greater than 1) is overall better for reducing bias, so we'll stick with those columns. Though, painexer isn't used in the first module, pncaden was used instead to encapsulate more information. We'll stick with that column as evident in the first round of EDA, that variable appeared more useful.

As for the smoke columns, these were left out of the first module for the amount of missing values. In the second module, we used data scraped online to impute the missing values. The first source only used the age group, while the second included patient's sex as well. As talked about in the webscraping section of this module, the dataset is from 1998, so the data we scraped in module 2 is long after the time of our dataset. In turn, we scraped some data from the new source given to us in this module to try to scale the smoking rates to more accurately represent what they might have been in that time.

We don't want 3 entire columns dedicated to imputed smoke values as this is redundant. We'll keep the imputed values from source 2 as they provide more information and not use column from source 1. Then we'll include a new imputed smoke column which uses the new smoking rates we calculated from combining the three sources' information. See that calculation in task 1.
