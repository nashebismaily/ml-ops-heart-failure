from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import pickle

timestr = time.strftime("%Y%m%d-%H%M%S")

heart_failure_data = pd.read_pickle("resources/normalized_data/heart_failure_normalized_dataframe.pkl",compression="gzip")

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

# Test Model
predictions_rand=randF.predict(x_test)
pd.crosstab(y_test, predictions_rand, rownames=['Actual'], colnames=['Prediction'])

# Model Performance
accuracy_score(y_test,predictions_rand)
auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)
print(auroc, ap)

# Save Model
pickle.dump(randF, open("resources/ml_models/heart_failure_model_" + timestr + ".pkl","wb"))




