from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import cdsw
import os
import glob
from cmlbootstrap import CMLBootstrap
import time

## CML API Config
HOST = "https://" + os.environ['CDSW_DOMAIN']
USERNAME = "nismaily"
API_KEY = "udy6l809q4ahd0tr907e0qudrwuuirt8"
PROJECT_NAME = "ml-heart-failure"
# Job ID for 09_retrain_model.py
JOB_ID = "84"

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
timestr = os.environ['timestr']

# New Data Path
path = "resources/new_data_extracts"

## Create DataFrame from Original and New Data
li = []
# Read Raw Data
df = pd.read_csv("resources/raw_data/heart_failure_clinical_records_dataset.csv")
li.append(df)
  
# Read New Data Extracts
all_files = os.listdir(path)
for filename in all_files:
  df = pd.read_csv(path + '/'+ filename)
  li.append(df)
    
heart_failure_dataframe = pd.concat(li, axis=0,ignore_index=True)
  
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
  
## Save Normalized Data and Scaler Information
dataset_norm.to_pickle("resources/normalized_data/heart_failure_normalized_dataframe_" + timestr + ".pkl",
                         compression="gzip")
pickle.dump(scaler, open("resources/scaler_functions/scaler_" + timestr + ".pkl", 'wb'))

x = dataset_norm.iloc[:, :-1].values
y = dataset_norm.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 2)

## Get Latest Models
list_of_models = glob.glob('resources/ml_models/*')
latest_model = max(list_of_models, key=os.path.getctime)

## Load Latest ML Model
randF = pickle.load(open(latest_model,"rb"))

## Determine Accuracy of Model with New Data
predictions_rand=randF.predict(x_test)
auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)

print("auroc =",auroc)
print("ap =",ap)

if auroc < 0.99:
  print("model needs retraining")
  ## Add Enviornment Variables
  job_env_params = {"timestr": timestr}
  start_job_params = {"environment": job_env_params}
    
  # Start Model Retrainig job
  job_status = cml.start_job(JOB_ID, start_job_params)
  print("Model Retraining Job started")
else:
    print("Model is Fine")
