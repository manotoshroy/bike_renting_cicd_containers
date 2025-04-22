import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from model import __version__ as _version
from model.config.core import config
from model.pipeline import bike_renting_pipeline
from model.processing.data_manager import load_pipeline
from model.processing.data_manager import pre_pipeline_preparation
from model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bike_rent_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bike_rent_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    #if not errors:

        #predictions = bike_rent_pipe.predict(validated_data)
        #results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'dteday':["2012-11-05"],'season':['winter'],'hr':['6am'],'holiday':['No'],'weekday':['Mon'],
                'workingday':["Yes"],'weathersit':['Mist'],'temp':[6.1],'atemp':[3.0014000000000003],'hum':[49.0],'windspeed':[19.0012],
                'casual':[4],'registered':[135]}
    
    #print(f"data in = {data_in}")
    
    make_prediction(input_data=data_in)
