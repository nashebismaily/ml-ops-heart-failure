import pickle
import pandas as pd
import glob
import os

## Determing Latest Model and Scalar Function from Timestamp
list_of_models = sorted(filter( os.path.isfile,glob.glob("resources/ml_models/*")))
latest_model = list_of_models[-1]

list_of_scaler_functions = sorted(filter( os.path.isfile,glob.glob("resources/scaler_functions/*")))
latest_scaler_function = list_of_scaler_functions[-1]

print(latest_model)
print(latest_scaler_function)
    
model = pickle.load(open(latest_model, 'rb'))
scaler = pickle.load(open(latest_scaler_function, 'rb'))

def predict(args):
  age = float(args.get('age'))
  anaemia = int(args.get('anaemia'))
  creatinine_phosphokinase = float(args.get('creatinine_phosphokinase'))
  diabetes = int(args.get('diabetes'))
  ejection_fraction = int(args.get('ejection_fraction'))
  high_blood_pressure = int(args.get('high_blood_pressure'))
  platelets = float(args.get('platelets'))
  serum_creatinine = float(args.get('serum_creatinine'))
  serum_sodium = int(args.get('serum_sodium'))
  sex = int(args.get('sex'))
  smoking = int(args.get('smoking'))
  time = int(args.get('time'))

  heart_failure_dataframe =pd.DataFrame({'age': age, 
                                         'anaemia': anaemia, 
                                         'creatinine_phosphokinase': creatinine_phosphokinase, 
                                         'diabetes': diabetes, 
                                         'ejection_fraction': ejection_fraction, 
                                         'high_blood_pressure': high_blood_pressure, 
                                         'platelets': platelets, 
                                         'serum_creatinine': serum_creatinine, 
                                         'serum_sodium': serum_sodium, 
                                         'sex': sex, 
                                         'smoking': smoking, 
                                         'time': time, }, index=[0])

  ## Define Features
  numerical_features = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium"]
  categorical_features = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "time"]
  
  numerical_data_frame = pd.DataFrame(heart_failure_dataframe, columns = numerical_features)
  categorical_data_frame = pd.DataFrame(heart_failure_dataframe, columns = categorical_features)

  scaled_numerical_features  = scaler.transform(numerical_data_frame)
  numerical_data_frame = pd.DataFrame(scaled_numerical_features,columns=numerical_features)
  dataset_norm = pd.concat([numerical_data_frame,categorical_data_frame],axis=1)
  
  result = model.predict(dataset_norm)
  
  if result[0] == 1:
    return "Patient Has Heart Failure Risk!"
  else:
    return "No Heart Failure Risk Detected"