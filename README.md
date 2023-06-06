# DE300_Project-3

### Task 1: Web scraping  

The goal here is to combine all of the data to merge with the other two modules' datasets.

Across the three sources, we have smoking rates based on people's sex, age, race, region, state, education, income, mental health status, and health insurance.

However, the only information we have on each patient from these factors is sex and age. Trying to blindly utilize rates without knowing the proper information would likely introduce a lot of unnecessary bias into the model, so it's best to just stick to age and sex.

This heart disease dataset is from 1998, but the data from the 2 past sources is based on 2018 and 2021. Considering that smoking rates have likely drastically changed since that time, using the rates from 2018 and 2021 is probably inaccurate. The first source contains data dating back to 1999, but it's only data on adolescents.

Not only are the smoking rates inaccurate just in general, but even the way they are distributed amongst age groups is likely highly inaccurate as well. Not only was smoking more popular in 1998 than today, but it was comparetively much more popular amongst young people then than young people now due to the success of anti-smoking movements stigmatizing smoking cigarettes as gross in the 2000s. This means that we can't just use difference in the adolescent rates to factor up the smoking rates of all age groups by a constant factor as the distribution will still be highly inaccurate.

Ideally, we could have just originally scraped data from 1999, but part of this project is to be creative in how we address this issue with the sites we were given.

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
