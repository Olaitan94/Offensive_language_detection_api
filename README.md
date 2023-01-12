This repo contains code for the Offensive Language Detection Model API, the requirements, tests as well as other set up files.

The API uses the offensive_language_detection_model created by me (link).

Find below the description of the different components & their respective uses:

# app:
This is the main package in this directory and It contains a number of files that is used to develop our API.
  1. Schemas: This folder contains files that are used to define the format of the inputs or outputs of the API's endpoints. There are two schema files:
    - predict.py: contains schema for the format of the inputs & outputs for the predict endpoint
    - health.py: contains schema for the format of the response of the health endpoint
  2. api.py:This file contains code for the API's endpoints: health & predict.
  3. config.py: This contains code for configuring the API like setting the logging level, setting up custome logging etc
  4. main.py: This file contains code for creating & running the API
  5. tests: contains scripts for testing the package
      - conftest.py - contains a fixture fxn which is used to provide input text to the other test.
      - test_api.py - used to test the predict endpoint of the api

# Dockerfile
Contains commands used to build the docker image for this api.
The image can be gotten by running "docker pull olaitanlawal/forecast_api"

# requirements.txt:
Contains dependencies required to use the API
# Procfile:
This file is a requirement from heroku when deploying an app to heroku. It tells heroku how to run the application
# Manifest.in:
Contains instructions for what to include or exclude when building the package
# runtime.txt:
File to tell heroku what version of language(python) and version was used to create the API
# test_requirements.txt:
Contains dependencies required to test the API
# tox.ini:
This file contains the settings for using tox for automated test but I didnt use tox to test this API.

# Testing:
- To test this API, run the command "pytest" on cmd

# How to use this API :
**To run the different files and test the modules, I created a conda environment for this project, installed all the dependencies in the requirements.txt except prophet. and then followed the steps in this link to install prophet. https://stackoverflow.com/questions/53178281/installing-fbprophet-python-on-windows-10
** Install the packages in requirements.txt file
** If testing or using the API file on your computer, you have to add the directory to your PYTHONPATH, So that python can find it. Search for 'andrei' on https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath . If you don't do this, python will not be able to find the package and so will not be able to run the import statements

# Deploying:
This API has been deployed to heroku. (https://cad-usd-forecast-api.herokuapp.com/)
