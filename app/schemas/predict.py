from typing import Any, List, Optional, Dict
import pandas as pd
import numpy as np

from pydantic import BaseModel


#create schema for the results of the forecast
#Note that this has to match the names of the prediction results coming out of your model
#In this case my model gives four parameter listed below

class PredictionResults(BaseModel):

    ACTUAL_TEXT: str
    PREDICTED_CLASS: str
    Probability:  float


#create a class for the input of the model
class ModelInputs(BaseModel):
    #Specify that your model takes in an integer input
    inputs: str

    #This is to allow people manipulate the inputs to the model on the web interface
    #"inputs" must be input
    class Config:
        schema_extra = {
            "example": {
                "inputs": "I had a great day"
                }
                }
