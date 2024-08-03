from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, mean, stddev, corr, desc
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier, LinearSVC, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import MinMaxScaler
import pandas as pd

from pyspark import SparkConf, SparkContext
import re


# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 0 - Create Spark Context
# +--------------------------------------------------------------------------------------------------------------------------------+
conf = SparkConf().setAppName('WineTrain')
conf.set('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
conf.set('spark.hadoop.fs.s3a.aws.credentials.provider', 'com.amazonaws.auth.InstanceProfileCredentialsProvider')

# Initialize Spark context and session
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 1 - Gather Training Data
# +--------------------------------------------------------------------------------------------------------------------------------+
try:
  train_df = spark.read.csv('TrainingDataset.csv', header=True, inferSchema=True, sep=";")
except:
  train_df = spark.read.csv(r'C:\Users\Christopher\Desktop\NJIT\CS643\Assignment 2\TrainingDataset.csv', header=True, inferSchema=True, sep=";")

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 2 - Training Data Preprocessing
# +--------------------------------------------------------------------------------------------------------------------------------+
for col in train_df.columns:
  new_col = re.sub(r'"', '', col)
  train_df = train_df.withColumnRenamed(col, new_col)

train_df = train_df.dropna()

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 2 - Training Data EDA
# +--------------------------------------------------------------------------------------------------------------------------------+

# Descriptive statistics
train_df.describe().show()

# Correlation matrix
for col_name in train_df.columns:
    if col_name != 'quality':
        print(f"Correlation between quality and {col_name}: {train_df.stat.corr('quality', col_name)}")

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 3 - Training Data Preparation for Modeling
# +--------------------------------------------------------------------------------------------------------------------------------+

feature_columns = train_df.columns
try:
  feature_columns.remove('quality')
except:
  pass

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
train_df_assembled = assembler.transform(train_df)

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(train_df_assembled)
train_df_scaled = scalerModel.transform(train_df_assembled)

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 4 - Modeling the Training Data
# +--------------------------------------------------------------------------------------------------------------------------------+

# Initialize ML models
rf = RandomForestClassifier(labelCol='quality', featuresCol='scaledFeatures')
lr = LogisticRegression(labelCol='quality', featuresCol='scaledFeatures')
nb = NaiveBayes(labelCol='quality', featuresCol='scaledFeatures')
dt = DecisionTreeClassifier(labelCol='quality', featuresCol='scaledFeatures')

# Define parameter grids for hyperparameter tuning
rf_param_grid = ParamGridBuilder() \
    .build()

lr_param_grid = ParamGridBuilder() \
    .build()

dt_param_grid = ParamGridBuilder() \
    .build()

# Initialize CrossValidator for each model
rf_cv = CrossValidator(estimator=rf,
                       estimatorParamMaps=rf_param_grid,
                       evaluator=MulticlassClassificationEvaluator(labelCol='quality', metricName='accuracy'),
                       numFolds=2)

lr_cv = CrossValidator(estimator=lr,
                       estimatorParamMaps=lr_param_grid,
                       evaluator=MulticlassClassificationEvaluator(labelCol='quality', metricName='accuracy'),
                       numFolds=2)


dt_cv = CrossValidator(estimator=dt,
                       estimatorParamMaps=dt_param_grid,
                       evaluator=MulticlassClassificationEvaluator(labelCol='quality', metricName='accuracy'),
                       numFolds=2)


# Fit models
rf_model = rf_cv.fit(train_df_scaled)
lr_model = lr_cv.fit(train_df_scaled)
dt_model = dt_cv.fit(train_df_scaled)

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 5 - Saving the Model
# +--------------------------------------------------------------------------------------------------------------------------------+

rf_model.bestModel.write().overwrite().save(f"random_forest_model")
lr_model.bestModel.write().overwrite().save(f"logistic_regression_model")
dt_model.bestModel.write().overwrite().save(f"decision_tree_model")


try:
  sc.stop()
  spark.stop()
except:
  pass