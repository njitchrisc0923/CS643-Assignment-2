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

from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel, NaiveBayesModel, DecisionTreeClassificationModel


# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 0 - Create Spark Context
# +--------------------------------------------------------------------------------------------------------------------------------+
conf = SparkConf().setAppName('WineTest')
conf.set('spark.hadoop.fs.s3a.impl', 'org.apache.hadoop.fs.s3a.S3AFileSystem')
conf.set('spark.hadoop.fs.s3a.aws.credentials.provider', 'com.amazonaws.auth.InstanceProfileCredentialsProvider')

# Initialize Spark context and session
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 1 - Gather Validation Data
# +--------------------------------------------------------------------------------------------------------------------------------+
try:
    validation_df = spark.read.csv('ValidationDataset.csv', header=True, inferSchema=True, sep=";")
except:
    validation_df = spark.read.csv(r'C:\Users\Christopher\Desktop\NJIT\CS643\Assignment 2\TrainingDataset.csv', header=True, inferSchema=True, sep=";")

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 2 - Model Loading
# +--------------------------------------------------------------------------------------------------------------------------------+

rf_model = RandomForestClassificationModel.load("random_forest_model")
lr_model = LogisticRegressionModel.load("logistic_regression_model")
dt_model = DecisionTreeClassificationModel.load("decision_tree_model")

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 3 - Training Data Preprocessing
# +--------------------------------------------------------------------------------------------------------------------------------+

for col in validation_df.columns:
  new_col = re.sub(r'"', '', col)
  validation_df = validation_df.withColumnRenamed(col, new_col)

validation_df = validation_df.dropna()

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 4 - Training Data Preparation for Modeling
# +--------------------------------------------------------------------------------------------------------------------------------+

feature_columns = validation_df.columns
try:
    feature_columns.remove('quality')
except:
    pass

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
validation_df_assembled = assembler.transform(validation_df)

# Scale features
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(validation_df_assembled)  
validation_df_scaled = scalerModel.transform(validation_df_assembled)

# +--------------------------------------------------------------------------------------------------------------------------------+
# Part 5 - Testing the Model
# +--------------------------------------------------------------------------------------------------------------------------------+

def evaluate_model(model, name):
    predictions = model.transform(validation_df_scaled)
    
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='quality', predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='quality', predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)
    
    print(f"{name} Accuracy on validation dataset = %g" % accuracy)
    print(f"{name} F1 Score on validation dataset = %g" % f1_score)

evaluate_model(rf_model, "Random Forest")
evaluate_model(lr_model, "Logistic Regression")
evaluate_model(dt_model, "Decision Tree")

try:
    sc.stop()
    spark.stop()
except:
    pass
