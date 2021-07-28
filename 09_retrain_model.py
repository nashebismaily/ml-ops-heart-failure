from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import pickle
import os
from cmlbootstrap import CMLBootstrap
import time

## CML API Config
HOST = "https://" + os.environ['CDSW_DOMAIN']
USERNAME = "nismaily"
API_KEY = "udy6l809q4ahd0tr907e0qudrwuuirt8"
PROJECT_NAME = "ml-heart-failure"
JOB_ID = "85"

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
timestr = os.environ['timestr']

heart_failure_data = pd.read_pickle("resources/normalized_data/heart_failure_normalized_dataframe_" + timestr + ".pkl",
                                    compression="gzip")

# Data Modeling
x = heart_failure_data.iloc[:, :-1].values
y = heart_failure_data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 2)

# Train Random Forest Model
param_numTrees = 200
param_impurity = 'entropy' 

randF=RandomForestClassifier(n_estimators=param_numTrees, 
        criterion = param_impurity,
        random_state=0)
                             

randF.fit(x_train, y_train)


predictions_rand=randF.predict(x_test)
pd.crosstab(y_test, predictions_rand, rownames=['Actual'], colnames=['Prediction'])

auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)
print(auroc, ap)


pickle.dump(randF, open("resources/ml_models/heart_failure_model_" + timestr + ".pkl","wb"))


## Add Enviornment Variables
job_env_params = {"timestr": timestr}
start_job_params = {"environment": job_env_params}
  
## Start Model Deployment Job
job_status = cml.start_job(JOB_ID, start_job_params)
print("Model Deployment Job started")
