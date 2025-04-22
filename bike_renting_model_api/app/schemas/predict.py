from typing import Any, List, Optional, Union
from datetime import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[float]


#dteday,season,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
class DataInputSchema(BaseModel):
    dteday: Optional[Union[str, datetime]]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str]
    weekday: Optional[str]
    workingday: Optional[str]
    weathersit: Optional[str]
    temp: Optional[float]
    atemp:Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    casual: Optional[int]
    registered: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]