import cdsw
import os
from cmlbootstrap import CMLBootstrap
import time

HOST = "https://" + os.environ['CDSW_DOMAIN']
USERNAME = "nismaily"
API_KEY = "udy6l809q4ahd0tr907e0qudrwuuirt8"
PROJECT_NAME = "ml-heart-failure"

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)
timestr = os.environ['timestr']

example_model_input = {"age":75,
                       "anaemia":0,
                       "creatinine_phosphokinase":582,
                       "diabetes":0,
                       "ejection_fraction":20,
                       "high_blood_pressure":1,
                       "platelets":265000,
                       "serum_creatinine":1.9,
                       "serum_sodium":130,
                       "sex":1,
                       "smoking":0,
                       "time":4}


# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]

# Get Default Engine Details
default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]

create_model_params = {
    "projectId": project_id,
    "name": "Heart Failure Model " + timestr,
    "description": "Detect Heart Failure",
    "visibility": "private",
    "enableAuth": False,
    "targetFilePath": "06_deploy_model.py",
    "targetFunctionName": "predict",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": {}
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "runtimeId": 14,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}



new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]
model_id = new_model_details["id"]

print("New model created with access key", access_key)

#Wait for the model to deploy.
is_deployed = False
while is_deployed == False:
  model = cml.get_model({"id": str(new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True}) 
  if model["latestModelDeployment"]["status"] == 'deployed':
    print("Model is deployed")
    break
  else:
    print ("Deploying Model.....")
    time.sleep(10)