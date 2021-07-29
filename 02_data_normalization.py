import pandas as pd
import pickle
from pyspark.sql import SparkSession
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Create Spark Session
spark_app_name = "heart_failure_demo_nismaily"
spark = SparkSession.builder.appName(spark_app_name).getOrCreate()

## Load Data
heart_failure_data = spark.read.csv(
    "resources/raw_data/heart_failure_clinical_records_dataset.csv", header=True, mode="DROPMALFORMED",inferSchema=True
)
heart_failure_dataframe = heart_failure_data.toPandas()

## Define Features
numerical_features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", 
                      "serum_creatinine", "serum_sodium"]

categorical_features = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "time", 
                        "DEATH_EVENT"]

numerical_data_frame = pd.DataFrame(heart_failure_dataframe, columns = numerical_features)
categorical_data_frame = pd.DataFrame(heart_failure_dataframe, columns = categorical_features)

## Normalize Data
scaler = MinMaxScaler()
scaler.fit(numerical_data_frame)
scaled_numerical_features  = scaler.transform(numerical_data_frame)
numerical_data_frame = pd.DataFrame(scaled_numerical_features,columns=numerical_features)
dataset_norm = pd.concat([numerical_data_frame,categorical_data_frame],axis=1)

## Create Correlation Matrix
all_features = categorical_features.copy()
all_features.extend(numerical_features)
plt.figure(figsize=(8, 7))
sns.heatmap(dataset_norm[all_features].corr(method='pearson'), vmin=-1, vmax=1, cmap='viridis', annot=True, fmt='.2f');

## Save Normalized Data and Scaler Information
dataset_norm.to_pickle("resources/normalized_data/heart_failure_normalized_dataframe.pkl",compression="gzip")
pickle.dump(scaler, open('resources/scaler_functions/scaler.pkl', 'wb'))