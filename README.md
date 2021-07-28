# Heart Rate Failure Detection


## Introduction

This is a demo project to showcase the features of the Cloudera Machine Learning.

The data used in this project is from [UCI](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

This dataset contains 13 clinical features used to predict heart rate failure.

The project consists of 10 parts:
* Part 1:  Data Exploration                    `01_data_exploration.ipynb` 
* Part 2:  Data Normalization                  `02_data_normalization.py`
* Part 3:  Explore Machine Learning Models     `03_explore_machine_learning_models.ipynb`
* Part 4:  Train Machine Learning Model        `04_train_machine_learning_model.py`
* Part 5:  Model Tuning                        `05_tune_model.py`
* Part 6:  Deploy Model                        `06_deploy_model.py`
* Part 7:  Pipeline - Check for New Data       `07_check_new_data.py`
* Part 8:  Pipeline - Test Model on New Data   `08_test_model_new_data.py`
* Part 9:  Pipeline - Retrain Model            `09_retrain_model.py`
* Part 10: Pipeline - Deploy new Model         `10_push_new_model.py`


## 1: Data Exploration
Explore the dataset using Jupyter Notebooks

## 2: Data Normalization
Scale and Normalize numerical  features using sklearn's preprocessing capabilities

## 3: Explore Machine Learning Models
Train and Test a wide range of machine learning models including:
* Random Forest Classifier
* Decision Tree Classifier
* Logistic Regression
* Support Vector Classification
* K-Nearest Neighbors Classifier
* Naive Bayes Classifier
* Gradient Boosting Classifier
* LGBM Classifier
* Extra Trees Classifier
* AdaBoost Classifier

## 4: Train Machine Learning Model
Select the model with the best performance (Random Forest Classifier). Train the model and save a s pickle file.

## 5: Model Tuning
Adjust the model hyper parameter values using an experiment to optimize  the model.

## 6: Deploy Model
Operationalize the model using CML.

## 7: Pipeline - Check for New Data
Check for new/incoming data

## 8: Pipeline - Test Model on New Data
Add incoming data to original dataset. Normalize the new dataset. Test the accuracy of the existing model on the new normalized dataset.

## 9: Pipeline - Retrain Model
Retrain the model with the new normalized dataset

## 10: Pipeline - Deploy new Model
Operationalize the new model


_Note: For the model deployment, use the following JSON as the example input for Heart Failure Detection:_

```
{
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
}
```
