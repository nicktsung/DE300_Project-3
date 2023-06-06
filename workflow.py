from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fancyimpute import IterativeImputer
from IPython.display import display

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import math

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.types import BooleanType, DoubleType
from pyspark.sql.functions import (
    date_trunc, expr, create_map, count, desc, avg, log10, dayofmonth,
    date_format, hour, minute, dayofweek, when, col, udf
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC, LogisticRegression, GBTClassifier

import requests
from bs4 import BeautifulSoup
import re

def load_data():
    hook = S3Hook(aws_conn_id='aws_default')
    bucket_name = 'de300-project4-mwaa'
    file_key = 'heart_disease.csv'

    file_content = hook.read_key(file_key, bucket_name)

    with open('/tmp/heart_disease.csv', 'w') as f:
        f.write(file_content)

def python_manipulation():

    heart = pd.read_csv('/tmp/heart_disease.csv')
    heart.fillna(np.nan)

    heart.iloc[898:,:]
    heart = heart.iloc[:899,:]

    heart.isna().sum()/len(heart)

    missing_percs = list(heart.isna().sum()/len(heart))
    columns = list(heart.columns)
    missing = []
    for i in range(0,len(missing_percs)):
        if missing_percs[i] >= 0.5:
            missing.append(columns[i])

    print(heart[missing].isna().sum()/len(heart))

    heart.head()[['smoke','cigs']]

    missing2 = missing
    missing2.remove('smoke')
    missing2.remove('pncaden')

    def impute_smoke(x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            return None

    heart['smoke'] = heart['cigs'].apply(impute_smoke)

    heart['pncaden'] = heart['painloc'] + heart['painexer'] + heart['relrest']

    print(heart[['smoke', 'pncaden']].isna().sum()/len(heart))

    heart_dropped = heart.drop(missing2, axis = 1)

    categories = ['sex','painloc','painexer','relrest','pncaden','cp','htn','smoke','fbs','famhist',
                'restecg','dig','prop','nitr','pro','diuretic','proto','exang','xhypo', 'target', 'slope']

    nums = ['trestbps','cigs','years','ekgmo','ekgday(day','ekgyr','thaldur','met','thalach',
            'thalrest','tpeakbps','tpeakbpd','oldpeak','dummy','trestbpd','rldv5','rldv5e', 'cmo',
        'cday','cyr']

    for cat in categories:
        heart_dropped[cat] = heart_dropped[cat].astype('category')

    for num in nums:
        heart_dropped[num] = heart_dropped[num].astype('float')

    missing_percs = list(heart_dropped.isna().sum()/len(heart_dropped))
    columns = list(heart_dropped.columns)
    missing = []
    for i in range(0,len(missing_percs)):
        if missing_percs[i] >= 0.2 and missing_percs[i] <= 0.5:
            missing.append(columns[i])

    print(heart_dropped[missing].isna().sum()/len(heart_dropped))

    categoricals = ['painloc', 'painexer','relrest','pncaden','smoke','famhist','slope']

    print(heart_dropped.target.value_counts()/len(heart_dropped),'\n')

    print("Correlation Matrix for Chest Pain Features\n",
        heart_dropped[['painloc', 'painexer','relrest','pncaden']].astype('float').dropna().corr())

    heart_dropped['slope']=heart_dropped['slope'].replace(0,np.nan)

    columns = ['painexer','relrest','smoke', 'famhist', 'cigs', 'years', 'rldv5']
    heart_dropped2 = heart_dropped.drop(columns, axis = 1)

    missing_percs = list(heart_dropped2.isna().sum()/len(heart_dropped2))
    columns = list(heart_dropped2.columns)
    missing = []
    for i in range(0,len(missing_percs)):
        if missing_percs[i] < 0.2 and missing_percs[i] > 0.00001:
            missing.append(columns[i])

    print(heart_dropped2[missing].isna().sum()/len(heart_dropped2), '\n')
    print(f'Out of {len(heart_dropped2.columns)} remaining features, {len(heart_dropped2[missing].isna().sum())} of them that are missing values are missing less than 20% \n')

    df = heart_dropped2[missing]
    categoricals = df.select_dtypes(include=['category']).columns.tolist()

    all_categoricals = heart_dropped2.select_dtypes(include=['category']).columns.tolist()
    all_categoricals.remove('pncaden')

    print(heart_dropped2.target.value_counts()/len(heart_dropped2),'\n')

    average = heart_dropped2[heart_dropped2['chol'] == 0]['target'].astype('float').mean()

    columns = ['proto', 'dummy', 'ekgmo', 'ekgday(day', 'ekgyr', 'cmo', 'cday', 'cyr', 'rldv5e']
    heart_dropped3 = heart_dropped2.drop(columns, axis = 1)

    heart_dropped3['chol']=heart_dropped2['chol'].replace(0,np.nan)

    heart_dropped3['prop']=heart_dropped2['prop'].replace(22,np.nan)

    print(heart_dropped3.isna().sum()/len(heart_dropped3), '\n')

    print(f'Out of the individuals with the erroneous cholesterol of 0, a proportion of {average} of them have target = 1\n')

    copy = heart_dropped2.copy()
    copy['chol_is_0'] = heart_dropped2['chol'].apply(lambda x: x == 0).astype('float').astype('category')
    columns = ['xhypo', 'exang']
    
    heart_dropped3.loc[heart_dropped3['met'] > 40, 'met'] = np.nan

    categoricals = heart_dropped3.select_dtypes(include=['category']).columns.tolist()

    for cat in categoricals:
        heart_dropped3[cat] = heart_dropped3[cat].astype('float')
        heart_dropped3[cat] = heart_dropped3[cat].fillna('missing')
        heart_dropped3[cat] = heart_dropped3[cat].astype('category')

    imputer = IterativeImputer(estimator=RandomForestRegressor(n_jobs = 1), random_state=0)
    heart_dropped4 = heart_dropped3.replace('missing', -999)
    heart_imputed = pd.DataFrame(imputer.fit_transform(heart_dropped4), columns=heart_dropped3.columns)
    heart_imputed = heart_imputed.replace(-999, 'missing')

    imputer = IterativeImputer(estimator=XGBRegressor(n_jobs = 1), random_state=0)
    heart_dropped4 = heart_dropped3.replace('missing', -999)
    heart_imputed = pd.DataFrame(imputer.fit_transform(heart_dropped4), columns=heart_dropped3.columns)
    heart_imputed = heart_imputed.replace(-999, 'missing')

    heart_imputed.isna().sum()/len(heart_imputed)

    heart_imputed['target'] = heart_imputed['target'].astype('category')
    heart_imputed['cp'] = heart_imputed['cp'].astype('category')
    heart_imputed['sex'] = heart_imputed['sex'].astype('category')

    placeholder = heart_imputed.copy()
    placeholder['trestbps'] = placeholder['trestbps'].replace(0, np.nan)
    placeholder['trestbpd'] = placeholder['trestbpd'].replace(0, np.nan)

    imputer = IterativeImputer(estimator=XGBRegressor(n_jobs = 1), random_state=0)
    placeholder = placeholder.replace('missing', -999)
    heart_imputed2 = pd.DataFrame(imputer.fit_transform(placeholder), columns=placeholder.columns)
    heart_imputed2 = heart_imputed2.replace(-999, 'missing')

    heart_imputed2['target'] = heart_imputed2['target'].astype('category')
    heart_imputed2['cp'] = heart_imputed2['cp'].astype('category')
    heart_imputed2['sex'] = heart_imputed2['sex'].astype('category')

    categoricals = heart_imputed2.select_dtypes(include=['object', 'category'])
    categoricals.describe()

    heart_imputed2.to_csv('/tmp/heart_imputed2.csv', index=False) 
    heart_imputed.to_csv('/tmp/heart_imputed.csv', index=False)

def feature_engineering_1():
    
    heart_imputed2 = pd.read_csv('/tmp/heart_imputed2.csv')
    heart_imputed = pd.read_csv('/tmp/heart_imputed.csv')

    heart_imputed2.select_dtypes(include = ['float']).skew()

    heart_trans = heart_imputed2.copy()
    heart_trans['chol'] = np.log(heart_imputed2['chol'])

    oldpeak = heart_imputed['oldpeak'].values.reshape(-1, 1)
    qt = QuantileTransformer(n_quantiles=len(oldpeak), output_distribution='normal')
    oldpeak_transformed = qt.fit_transform(oldpeak)
    heart_trans['oldpeak'] = pd.DataFrame(oldpeak_transformed)

    print('Skew of the transformed features')
    heart_trans.select_dtypes(include = ['float']).skew()[['chol','oldpeak']]

    categoricals = heart_imputed2.iloc[:,:-1].select_dtypes(include=['object', 'category']).columns.tolist()
    encoded_features = pd.get_dummies(heart_trans, columns=categoricals)
    encoded_features = encoded_features.drop(['target'], axis = 1)
    encoded_features['target'] = heart_imputed['target']

    encoded_features.to_csv('/tmp/encoded_features.csv', index=False) 

def svm_1():
    encoded_features = pd.read_csv('/tmp/encoded_features.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features.drop('target', axis=1),
        encoded_features['target'],
        test_size=0.2,  
        random_state=38  
    )

    svm = SVC(C=1.0, kernel='rbf', random_state=38)

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    f1 = f1_score(y_test, predictions)

    print("F1 score:", f1)

    svm = SVC(C=1.0, kernel='rbf', random_state=38)

    f1_scores = cross_val_score(svm, encoded_features.drop('target', axis=1), encoded_features['target'], cv=5, scoring='f1', n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    svm = SVC(C=0.1, kernel='linear', random_state=38)

    f1_scores = cross_val_score(svm, encoded_features.drop('target', axis=1), encoded_features['target'], cv=10, scoring='f1', n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features.drop('target', axis=1),
        encoded_features['target'],
        test_size=0.2,  
        random_state=38  
    )

    svm = SVC(C=0.1, kernel='linear', random_state=38)

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    f1 = f1_score(y_test, predictions)

    return f1, svm.get_params()

def lr_1():
    from sklearn.linear_model import LogisticRegression

    encoded_features = pd.read_csv('/tmp/encoded_features.csv')

    
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features.drop('target', axis=1),
        encoded_features['target'],
        test_size=0.2,  
        random_state=38  
    )

    logreg = LogisticRegression(solver = 'lbfgs')

    logreg.fit(X_train, y_train)

    predictions = logreg.predict(X_test)

    f1 = f1_score(y_test, predictions)

    print("F1 score:", f1)

    logreg = LogisticRegression(solver = 'lbfgs')

    f1_scores = cross_val_score(logreg, encoded_features.drop('target', axis=1), encoded_features['target'], cv=10, scoring="f1", n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    logreg = LogisticRegression(solver = 'liblinear')

    f1_scores = cross_val_score(logreg, encoded_features.drop('target', axis=1), encoded_features['target'], cv=10, scoring="f1", n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_features.drop('target', axis=1),
        encoded_features['target'],
        test_size=0.2,  
        random_state=38  
    )

    logreg = LogisticRegression(solver = 'liblinear')

    logreg.fit(X_train, y_train)

    predictions = logreg.predict(X_test)

    f1 = f1_score(y_test, predictions)

    return f1, logreg.get_params()

def spark_manipulation():
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.types import BooleanType, DoubleType
    from pyspark.sql.functions import (
        date_trunc, expr, create_map, count, desc, avg, log10, dayofmonth,
        date_format, hour, minute, dayofweek, when, col, udf
    )
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.classification import LinearSVC, LogisticRegression, GBTClassifier


    spark = SparkSession.builder.config("spark.driver.memory", "10g").getOrCreate()

    df = spark.read.csv('/tmp/heart_disease.csv', header=True, inferSchema=True).limit(899)
    df = df.withColumn('age', col('age').cast('integer'))

    selected_columns = [
        "age", "sex", "painloc", "painexer", "cp", "trestbps", "smoke", "fbs",
        "prop", "nitr", "pro", "diuretic", "thaldur", "thalach", "exang", "oldpeak", "slope", "target"
    ]

    df_filtered = df.select(selected_columns)


    df_filtered = df_filtered.withColumn("painloc", when(col("painloc").isNull(), 2).otherwise(col("painloc")))
    df_filtered = df_filtered.withColumn("painexer", when(col("painexer").isNull(), 2).otherwise(col("painexer")))
    df_filtered = df_filtered.withColumn("trestbps", when((col("trestbps").isNull()) | (col("trestbps") < 100), df_filtered.approxQuantile("trestbps", [0.5], 0.001)[0]).otherwise(col("trestbps")))
    df_filtered = df_filtered.withColumn("oldpeak", when(col("oldpeak") < 0, 0).otherwise(col("oldpeak")))
    df_filtered = df_filtered.withColumn("oldpeak", when(col("oldpeak") > 4, 4).otherwise(col("oldpeak")))
    df_filtered = df_filtered.withColumn("oldpeak", when(col("oldpeak").isNull(), df_filtered.approxQuantile("oldpeak", [0.5], 0.001)[0]).otherwise(col("oldpeak")))
    df_filtered = df_filtered.withColumn("thaldur", when(col("thaldur").isNull(), df_filtered.approxQuantile("thaldur", [0.5], 0.001)[0]).otherwise(col("thaldur")))
    df_filtered = df_filtered.withColumn("thalach", when(col("thalach").isNull(), df_filtered.approxQuantile("thalach", [0.5], 0.001)[0]).otherwise(col("thalach")))
    df_filtered = df_filtered.withColumn("fbs", when((col("fbs").isNull()) | (col("fbs") > 1), df_filtered.approxQuantile("fbs", [0.5], 0.001)[0]).otherwise(col("fbs")))
    df_filtered = df_filtered.withColumn("prop", when((col("prop").isNull()) | (col("prop") > 1), df_filtered.approxQuantile("prop", [0.5], 0.001)[0]).otherwise(col("prop")))
    df_filtered = df_filtered.withColumn("nitr", when((col("nitr").isNull()) | (col("nitr") > 1), df_filtered.approxQuantile("nitr", [0.5], 0.001)[0]).otherwise(col("nitr")))
    df_filtered = df_filtered.withColumn("pro", when((col("pro").isNull()) | (col("pro") > 1), df_filtered.approxQuantile("pro", [0.5], 0.001)[0]).otherwise(col("pro")))
    df_filtered = df_filtered.withColumn("diuretic", when((col("diuretic").isNull()) | (col("diuretic") > 1), df_filtered.approxQuantile("diuretic", [0.5], 0.001)[0]).otherwise(col("diuretic")))
    df_filtered = df_filtered.withColumn("exang", when(col("exang").isNull(), df_filtered.approxQuantile("exang", [0.5], 0.001)[0]).otherwise(col("exang")))
    df_filtered = df_filtered.withColumn("slope", when(col("slope").isNull(), 4).otherwise(col("slope")))

    df_filtered.show()

    url_source1 = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release"
    response_source1 = requests.get(url_source1)
    soup_source1 = BeautifulSoup(response_source1.content, "html.parser")

    smoking_rates_source1 = {}
    table_source1 = soup_source1.find("table", class_="responsive-enabled")
    rows_source1 = table_source1.find_all("tr")
    for row in rows_source1[1:]:
        age_groups = row.find_all("th")
        columns = row.find_all("td")
        age_group = age_groups[0].text.strip()
        smoking_rate = float(columns[0].text.strip())
        smoking_rates_source1[age_group] = smoking_rate

    url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    div_element = soup.find_all('div', class_='card border-0 rounded-0 mb-3')[0]
    li_elements = div_element.find_all('li')

    percentage_list = []

    for li_element in li_elements:
        percentage_text = li_element.text.strip()  

        percentage_match = re.search(r'(\d+\.\d+)%', percentage_text)
        if percentage_match:
            percentage = round(float(percentage_match.group(1))/100, 4)
            percentage_list.append(percentage)

    div_element = soup.find_all('div', class_='card border-0 rounded-0 mb-3')[1]
    li_elements = div_element.find_all('li')
    age_groups2 = []
    percentage_list2 = []

    for li_element in li_elements:
        percentage_text = li_element.text.strip()  

        percentage_match = re.search(r'(\d+\.\d+)%', percentage_text)
        if percentage_match:
            percentage = round(float(percentage_match.group(1))/100, 4)
            percentage_list2.append(percentage)

        age_range_match = re.search(r"aged (\d+–\d+)", percentage_text)
        if age_range_match:
            age_range = age_range_match.group(1)
            age_groups2.append(age_range)

    print(smoking_rates_source1)

    print(percentage_list)
    print(age_groups2)
    print(percentage_list2)

    length = len(smoking_rates_source1)
    keys = list(smoking_rates_source1.keys())
    values = list(smoking_rates_source1.values())

    def get_smoking_rate(age):
        x = 0
        for i in range(0, length - 1):

            lower = int(keys[i].split('–')[0])
            upper = int(keys[i].split('–')[1])

            if age in range(lower, upper + 1):
                x = 1
                return values[i] / 100

        if x == 0:
            return values[length - 1] / 100

    get_smoking_rate_udf = udf(get_smoking_rate, DoubleType())

    df_imputed = df_filtered.withColumn('smoke_imputed_source1', when(col('smoke').isNull(), get_smoking_rate_udf(col('age'))).otherwise(col('smoke')))
    df_imputed.show()

    from pyspark.sql.functions import udf, col, when
    from pyspark.sql.types import DoubleType

    smoking_rate_source2_female = percentage_list[1]
    smoking_rate_source2_male = percentage_list[0]

    length2 = len(percentage_list2)
    keys2 = list(age_groups2)
    values2 = list(percentage_list2)

    def get_smoking_rate2(age, sex):
        if sex == 0:
            for i in range(length2 - 1):

                lower = int(keys2[i].split('–')[0])
                upper = int(keys2[i].split('–')[1])

                if age in range(lower, upper + 1):
                    return values2[i]

            return values2[length2 - 1]

        else:
            for i in range(length2 - 1):

                lower = int(keys2[i].split('–')[0])
                upper = int(keys2[i].split('–')[1])

                if age in range(lower, upper + 1):
                    return values2[i] * smoking_rate_source2_male / smoking_rate_source2_female

            return values2[length2 - 1] * smoking_rate_source2_male / smoking_rate_source2_female

    get_smoking_rate_udf2 = udf(get_smoking_rate2, DoubleType())

    df_imputed = df_imputed.withColumn('smoke_imputed_source2', when(col('smoke').isNull(), get_smoking_rate_udf2(col('age'), col('sex'))).otherwise(col('smoke')))
    df_imputed.show()

    df_dropped = df_imputed.drop("smoke")

    string_columns = ["trestbps", "fbs", "prop", "nitr", "pro", "diuretic", "exang"]

    for column in string_columns:
        df_dropped = df_dropped.withColumn(column, col(column).cast('boolean'))

    categorical_columns = ["slope", "painloc", "painexer"]

    for column in categorical_columns:
        df_dropped = df_dropped.withColumn(column, col(column).cast('int'))

    df_dropped.show()

    df_dropped.write.csv('/tmp/df_dropped.csv', header=True, mode='overwrite')

    return categorical_columns

def feature_engineering_2(**kwargs):
    
    spark = SparkSession.builder.config("spark.driver.memory", "10g").getOrCreate()

    ti = kwargs['ti']
    categorical_columns = ti.xcom_pull(task_ids='spark_manipulation_task')
    df_dropped = spark.read.csv('/tmp/df_dropped.csv', header=True, inferSchema=True).limit(899)

    encoder = OneHotEncoder(inputCols=categorical_columns, outputCols=[col + "_encoded" for col in categorical_columns])
    encoder_model = encoder.fit(df_dropped)
    df_encoded = encoder_model.transform(df_dropped)

    df_encoded = df_encoded.drop(*categorical_columns)

    df_encoded = df_encoded.drop(*["slope_encoded", "painloc_encoded", "painexer_encoded"])

    df_encoded.printSchema()

    train_data, test_data = df_encoded.randomSplit([0.9, 0.1], seed=38)

    df_encoded.write.csv('/tmp/df_encoded.csv', header=True, mode='overwrite')
    train_data.write.csv('/tmp/train_data.csv', header=True, mode='overwrite')
    test_data.write.csv('/tmp/test_data.csv', header=True, mode='overwrite')


def svm_2():
    spark = SparkSession.builder.config("spark.driver.memory", "10g").getOrCreate()
    train_data = spark.read.csv('/tmp/train_data.csv', header=True, inferSchema=True)
    test_data = spark.read.csv('/tmp/test_data.csv', header=True, inferSchema=True)

    feature_columns = ["age", "sex", "cp", "trestbps",  "fbs", "prop", "nitr", "pro", 
        "diuretic", "thaldur", "thalach", "exang", "oldpeak", 
        "smoke_imputed_source2", "smoke_imputed_source1"]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    train_data_transformed = assembler.transform(train_data)
    test_data_transformed = assembler.transform(test_data)

    svm = LinearSVC()

    param_grid = ParamGridBuilder().build()
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    cv = CrossValidator(estimator=svm, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=10)


    train_data_transformed = train_data_transformed.withColumn("label", train_data_transformed["target"])
    test_data_transformed = test_data_transformed.withColumn("label", test_data_transformed["target"])

    cv_model = cv.fit(train_data_transformed)

    best_model = cv_model.bestModel

    f1_score = evaluator.evaluate(best_model.transform(test_data_transformed))

    # Get the parameters of the model
    params = {}

    return f1_score, params

def lr_2():

    spark = SparkSession.builder.config("spark.driver.memory", "10g").getOrCreate()
    train_data = spark.read.csv('/tmp/train_data.csv', header=True, inferSchema=True)
    test_data = spark.read.csv('/tmp/test_data.csv', header=True, inferSchema=True)

    feature_columns = ["age", "sex", "cp", "trestbps",  "fbs", "prop", "nitr", "pro", 
        "diuretic", "thaldur", "thalach", "exang", "oldpeak", 
        "smoke_imputed_source2", "smoke_imputed_source1"]

    assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    train_data_transformed = assembler.transform(train_data)
    test_data_transformed = assembler.transform(test_data)
    
    lr = LogisticRegression(featuresCol="features", labelCol="target")

    param_grid = ParamGridBuilder().build()
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=10)

    train_data_transformed = train_data_transformed.withColumn("label", train_data_transformed["target"])
    test_data_transformed = test_data_transformed.withColumn("label", test_data_transformed["target"])

    cv_model = cv.fit(train_data_transformed)

    best_model = cv_model.bestModel

    f1_score = evaluator.evaluate(best_model.transform(test_data_transformed))

    # Get the parameters of the model
    params = {}

    return f1_score, params


def scrape():

    url_source1 = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release"
    response_source1 = requests.get(url_source1)
    soup_source1 = BeautifulSoup(response_source1.content, "html.parser")

    smoking_rates_source1 = {}
    table_source1 = soup_source1.find("table", class_="responsive-enabled")
    rows_source1 = table_source1.find_all("tr")
    for row in rows_source1[1:]:
        age_groups = row.find_all("th")
        columns = row.find_all("td")
        age_group = age_groups[0].text.strip()
        smoking_rate = float(columns[0].text.strip())
        smoking_rates_source1[age_group] = smoking_rate

    url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    div_element = soup.find_all('div', class_='card border-0 rounded-0 mb-3')[0]
    li_elements = div_element.find_all('li')

    percentage_list = []

    for li_element in li_elements:
        percentage_text = li_element.text.strip()  

        percentage_match = re.search(r'(\d+\.\d+)%', percentage_text)
        if percentage_match:
            percentage = round(float(percentage_match.group(1))/100, 4)
            percentage_list.append(percentage)

    div_element = soup.find_all('div', class_='card border-0 rounded-0 mb-3')[1]
    li_elements = div_element.find_all('li')
    age_groups2 = []
    percentage_list2 = []

    for li_element in li_elements:
        percentage_text = li_element.text.strip()  

        percentage_match = re.search(r'(\d+\.\d+)%', percentage_text)
        if percentage_match:
            percentage = round(float(percentage_match.group(1))/100, 4)
            percentage_list2.append(percentage)

        age_range_match = re.search(r"aged (\d+–\d+)", percentage_text)
        if age_range_match:
            age_range = age_range_match.group(1)
            age_groups2.append(age_range)

    print(smoking_rates_source1)

    print(percentage_list)
    print(age_groups2)
    print(percentage_list2)

    url_source = "https://wayback.archive-it.org/5774/20211119125806/https:/www.healthypeople.gov/2020/data-search/Search-the-Data?nid=5342"
    response_source = requests.get(url_source)
    soup = BeautifulSoup(response_source.content, "html.parser")

    soup_titles = soup.find_all(class_='ds-inner-poptitle')

    titles = []

    for soup_title in soup_titles:
        titles.append(soup_title.text)

    titles

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

    indices = [2,3]

    panda_dict = {}

    for i in indices:
        panda_dict[titles[i]] = data[i]

    panda_dict

    print(smoking_rates_source1)
    print(percentage_list)
    print(age_groups2)
    print(percentage_list2)
    print(panda_dict)

    smoking_rates = {}
    smoking_rates['18–24'] = (smoking_rates_source1['18–24']/100 + percentage_list2[0])/2
    smoking_rates['25–34'] = (smoking_rates_source1['25–34']/100 + percentage_list2[1])/2
    smoking_rates['35–44'] = (smoking_rates_source1['35–44']/100 + percentage_list2[1])/2
    smoking_rates['45–54'] = (smoking_rates_source1['45–54']/100 + percentage_list2[2])/2
    smoking_rates['55–64'] = (smoking_rates_source1['55–64']/100 + percentage_list2[2])/2
    smoking_rates['65–74'] = (smoking_rates_source1['65–74']/100 + percentage_list2[3])/2
    smoking_rates['75+'] = (smoking_rates_source1['75+']/100 + percentage_list2[3])/2

    smoking_rates

    dict_panda = {
        'age_group' : list(smoking_rates.keys()),
        'smoking_rate' : list(smoking_rates.values())
    }

    pd_smoke = pd.DataFrame(dict_panda)
    pd_smoke

    average_1999 =(float(panda_dict['Male ']) + float(panda_dict['Female ']))/2/100
    average_2021 = (percentage_list[0] + percentage_list[1])/2

    female_factor = average_1999 / average_2021
    male_factor = female_factor * (percentage_list[0] / percentage_list[1] / 1.1)

    female_factor

    male_factor

    pd_smoke['smoking_rate_male'] = pd_smoke['smoking_rate'].apply(lambda x: x*male_factor)
    pd_smoke['smoking_rate_female'] = pd_smoke['smoking_rate'].apply(lambda x: x*female_factor)
    pd_smoke

    factors = []
    for i in range(0,len(pd_smoke)):
        factors.append(1/np.log(math.e + i/1.8))

    factors

    pd_smoke['smoking_rate_male_adjusted'] = pd_smoke['smoking_rate_male']*factors
    pd_smoke['smoking_rate_female_adjusted'] = pd_smoke['smoking_rate_female']*factors
    pd_smoke

    smoking_rates = pd_smoke[['age_group', 'smoking_rate_male_adjusted', 'smoking_rate_female_adjusted']]
    smoking_rates.to_csv('/tmp/smoking_rates.csv', index=False) 

def merge(**kwargs):

    spark = SparkSession.builder.config("spark.driver.memory", "10g").getOrCreate()

    ti = kwargs['ti']
    heart = pd.read_csv('/tmp/heart_disease.csv')
    df_module1 = pd.read_csv('/tmp/encoded_features.csv')
    df_module2 = spark.read.csv('/tmp/df_encoded.csv', header=True, inferSchema=True).toPandas()
    smoking_rates = pd.read_csv('/tmp/smoking_rates.csv')

    print(list(df_module1.columns), '\n')
    print(list(df_module2.columns))

    df_module1['smoke_imputed_source2'] = df_module2['smoke_imputed_source2']

    df_module1['smoke'] = heart['smoke']
    df_module1['smoke'] = df_module1['smoke'].fillna(1000)

    smoking_rates

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

                lower = int(keys[i].split('–')[0])
                upper = int(keys[i].split('–')[1])

                if age in range(lower, upper + 1):
                    return female_rates[i]

            return female_rates[length - 1]

        else:
            for i in range(length - 1):

                lower = int(keys[i].split('–')[0])
                upper = int(keys[i].split('–')[1])

                if age in range(lower, upper + 1):
                    return male_rates[i]

            return male_rates[length - 1]

    df_module1['smoke_new_imputed'] = df_module1[['smoke', 'age', 'sex']].apply(get_smoking_rate, axis = 1)
    df_module1 = df_module1.drop('smoke', axis = 1)

    df_module1['smoke_new_imputed']

    df_module1.to_csv('/tmp/merged_df.csv', index=False)

def svm_3():

    merged_df = pd.read_csv('/tmp/merged_df.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        merged_df.drop('target', axis=1),
        merged_df['target'],
        test_size=0.2,  
        random_state=38  
    )

    svm = SVC(C=1.0, kernel='rbf', random_state=38)

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    f1 = f1_score(y_test, predictions)

    print("F1 score:", f1)

    svm = SVC(C=1.0, kernel='rbf', random_state=38)

    f1_scores = cross_val_score(svm, merged_df.drop('target', axis=1), merged_df['target'], cv=5, scoring='f1', n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    svm = SVC(C=0.1, kernel='linear', random_state=38)

    f1_scores = cross_val_score(svm, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring='f1', n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    X_train, X_test, y_train, y_test = train_test_split(
        merged_df.drop('target', axis=1),
        merged_df['target'],
        test_size=0.2,  
        random_state=38  
    )

    svm = SVC(C=0.1, kernel='linear', random_state=38)

    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    f1 = f1_score(y_test, predictions)

    return f1, svm.get_params()

def lr_3():
    from sklearn.linear_model import LogisticRegression
    
    merged_df = pd.read_csv('/tmp/merged_df.csv')

    
    X_train, X_test, y_train, y_test = train_test_split(
        merged_df.drop('target', axis=1),
        merged_df['target'],
        test_size=0.2,  
        random_state=38  
    )

    logreg = LogisticRegression(penalty = 'l2', C = 0.1, random_state=38, max_iter = 5000, solver = 'lbfgs')

    logreg.fit(X_train, y_train)

    predictions = logreg.predict(X_test)

    f1 = f1_score(y_test, predictions)

    print("F1 score:", f1)

    logreg = LogisticRegression(penalty = 'l2', solver = 'lbfgs', C = 0.1, random_state=38)

    f1_scores = cross_val_score(logreg, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring="f1", n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    logreg = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.1, random_state=38)

    f1_scores = cross_val_score(logreg, merged_df.drop('target', axis=1), merged_df['target'], cv=10, scoring="f1", n_jobs=-1)

    print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))

    X_train, X_test, y_train, y_test = train_test_split(
        merged_df.drop('target', axis=1),
        merged_df['target'],
        test_size=0.2,  
        random_state=38  
    )

    logreg = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.1, random_state=38, max_iter = 5000)

    logreg.fit(X_train, y_train)

    predictions = logreg.predict(X_test)

    f1 = f1_score(y_test, predictions)

    return f1, logreg.get_params()

def compare_models(**kwargs):
    ti = kwargs['ti']
    svm_f1_score_1, svm_1_params = ti.xcom_pull(task_ids='svm_task_1')
    lr_f1_score_1, lr_1_params = ti.xcom_pull(task_ids='lr_task_1')
    svm_f1_score_2, svm_2_params = ti.xcom_pull(task_ids='svm_task_2')
    lr_f1_score_2, lr_2_params = ti.xcom_pull(task_ids='lr_task_2')
    svm_f1_score_3, svm_3_params = ti.xcom_pull(task_ids='svm_task_3')
    lr_f1_score_3, lr_3_params = ti.xcom_pull(task_ids='lr_task_3')

    if svm_f1_score_1 > lr_f1_score_1 and svm_f1_score_1 > svm_f1_score_2 and svm_f1_score_1 > lr_f1_score_2 and svm_f1_score_1 > svm_f1_score_3 and svm_f1_score_1 > lr_f1_score_3:
        return svm_1_params, 'sklearn svm', 'SVM'
    elif lr_f1_score_1 > svm_f1_score_1 and lr_f1_score_1 > svm_f1_score_2 and lr_f1_score_1 > lr_f1_score_2 and lr_f1_score_1 > svm_f1_score_3 and lr_f1_score_1 > lr_f1_score_3:
        return lr_1_params, 'sklearn lr', 'LR'
    elif svm_f1_score_2 > svm_f1_score_1 and svm_f1_score_2 > lr_f1_score_1 and svm_f1_score_2 > lr_f1_score_2 and svm_f1_score_2 > svm_f1_score_3 and svm_f1_score_2 > lr_f1_score_3:
        return svm_2_params, 'spark svm', 'SVM'
    elif lr_f1_score_2 > svm_f1_score_1 and lr_f1_score_2 > lr_f1_score_1 and lr_f1_score_2 > svm_f1_score_2 and lr_f1_score_2 > svm_f1_score_3 and lr_f1_score_2 > lr_f1_score_3:
        return lr_2_params, 'spark lr', 'LR'
    elif svm_f1_score_3 > svm_f1_score_1 and svm_f1_score_3 > lr_f1_score_1 and svm_f1_score_3 > svm_f1_score_2 and svm_f1_score_3 > lr_f1_score_2 and svm_f1_score_3 > lr_f1_score_3:
        return svm_3_params, 'merged svm', 'SVM'
    else:
        return lr_3_params, 'merged lr', 'LR'

def evaluate_models(**kwargs):
    ti = kwargs['ti']
    best_params, model_string, model = ti.xcom_pull(task_ids='best_model_task')
    merged_df = pd.read_csv('/tmp/merged_df.csv')

    if model == 'LR':
        best_model = LogisticRegression(**best_params)
    else:
        best_model = SVC(**best_params)

    X_train, X_test, y_train, y_test = train_test_split(
        merged_df.drop('target', axis=1),
        merged_df['target'],
        test_size=0.2,  
        random_state=38  
    )

    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    
    print("Best Model:", model_string)
    print("F1 Score:", f1_score(y_test, predictions))
    print("Predictions:", predictions)

default_args = {
    'owner': 'group4',
    'start_date': datetime(2023, 6, 1),
}

dag = DAG('data_science_project_test_27', default_args=default_args, schedule_interval=None)

load_data_task = PythonOperator(task_id='load_data_task', python_callable=load_data, dag=dag)
python_manipulation_task = PythonOperator(task_id='python_manipulation_task', python_callable=python_manipulation, provide_context=True, dag=dag)
spark_manipulation_task = PythonOperator(task_id='spark_manipulation_task', python_callable=spark_manipulation, provide_context=True, dag=dag)
feature_engineering_task_1 = PythonOperator(task_id='feature_engineering_task_1', python_callable=feature_engineering_1, provide_context=True, dag=dag)
feature_engineering_task_2 = PythonOperator(task_id='feature_engineering_task_2', python_callable=feature_engineering_2, provide_context=True, dag=dag)
svm_task_1 = PythonOperator(task_id='svm_task_1', python_callable=svm_1, provide_context=True, dag=dag)
lr_task_1 = PythonOperator(task_id='lr_task_1', python_callable=lr_1, provide_context=True, dag=dag)
svm_task_2 = PythonOperator(task_id='svm_task_2', python_callable=svm_2, provide_context=True, dag=dag)
lr_task_2 = PythonOperator(task_id='lr_task_2', python_callable=lr_2, provide_context=True, dag=dag)
scrape_task = PythonOperator(task_id='scrape_task', python_callable=scrape, dag=dag)
merge_task = PythonOperator(task_id='merge_task', python_callable=merge, provide_context=True, dag=dag)
svm_task_3 = PythonOperator(task_id='svm_task_3', python_callable=svm_3, provide_context=True, dag=dag)
lr_task_3 = PythonOperator(task_id='lr_task_3', python_callable=lr_3, provide_context=True, dag=dag)
best_model_task = PythonOperator(task_id='best_model_task', python_callable=compare_models, provide_context=True, dag=dag)
evaluate_task = PythonOperator(task_id='evaluate_task', python_callable=evaluate_models, provide_context=True, dag=dag)

load_data_task >> [python_manipulation_task, spark_manipulation_task]
python_manipulation_task >> feature_engineering_task_1 >> [svm_task_1, lr_task_1, merge_task]
spark_manipulation_task >> feature_engineering_task_2 >> [svm_task_2, lr_task_2, merge_task]
scrape_task >> merge_task
merge_task >> [svm_task_3, lr_task_3]
[svm_task_1, lr_task_1, svm_task_2, lr_task_2, svm_task_3, lr_task_3] >> best_model_task >> evaluate_task