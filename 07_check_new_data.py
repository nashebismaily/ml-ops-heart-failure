# !pip3 install git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap

import os
from cmlbootstrap import CMLBootstrap
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

## CML API Config
HOST = "https://" + os.environ['CDSW_DOMAIN']
USERNAME = "nismaily"
API_KEY = "udy6l809q4ahd0tr907e0qudrwuuirt8"
PROJECT_NAME = "ml-heart-failure"
JOB_ID = "83"

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

## Add Enviornment Variables
job_env_params = {"timestr": timestr}
start_job_params = {"environment": job_env_params}

## Check for New Data
path = "resources/new_data_extracts"
dir = os.listdir(path)
  
if len(dir) == 0:
  print("No New Data")
else:
  print("New Data Found")
  ## Start Model Testing Job
  job_status = cml.start_job(JOB_ID, start_job_params)
  print("Model Testing Job started")
