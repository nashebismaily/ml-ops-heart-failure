from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import pickle
import glob
import os
import cdsw
import time

## Current Time
timestr = time.strftime("%Y%m%d-%H%M%S")

## Get Latest Normalized Data Frame
list_of_normalized_dataframe= sorted(filter( os.path.isfile,glob.glob("resources/normalized_data/*")))
latest_normalized_dataframe = list_of_normalized_dataframe[-1]

## Load Data
heart_failure_data = pd.read_pickle(latest_normalized_dataframe,compression="gzip")

## Data Modeling
x = heart_failure_data.iloc[:, :-1].values
y = heart_failure_data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 2)

## Train Random Forest Model
param_numTrees = int(sys.argv[1])
param_impurity = sys.argv[2]

randF=RandomForestClassifier(n_estimators=param_numTrees, 
        criterion = param_impurity,
        random_state=0)

randF.fit(x_train, y_train)

## Test Model
predictions_rand=randF.predict(x_test)
pd.crosstab(y_test, predictions_rand, rownames=['Actual'], colnames=['Prediction'])

## Model Performance
accuracy_score(y_test,predictions_rand)
auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)
print(auroc, ap)

## Track Metrics
cdsw.track_metric("auroc", round(auroc,2))
cdsw.track_metric("ap", round(ap,2))

## Save Model
pickle.dump(randF, open("resources/ml_models/heart_failure_model_" + timestr + ".pkl","wb"))
cdsw.track_file("resources/ml_models/heart_failure_model_" + timestr + ".pkl")