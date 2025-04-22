import sys
from pathlib import Path

file = Path(__file__).resolve()  # __file__ is a special variable in Python that holds the filename of the script being executed.
parent, root = file.parent, file.parents[1]
sys.path.append(str(root)) # dynamically add a directory to Python's module search path so that Python can find and import modules from that directory.

from sklearn.pipeline import Pipeline
from model.processing.features import WeekdayImputer
from model.processing.features import WeathersitImputer
from model.processing.features import Mapper
from model.processing.features import OutlierHandler
from model.processing.features import WeekdayOneHotEncoder
from model.processing.features import RemoveUnwantedColumns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from model.config.core import config







bike_renting_pipeline = Pipeline([


    ('weekday_imputation', WeekdayImputer('weekday')),
    ('weathersit_imputation', WeathersitImputer('weathersit')),
    ##==========Mapper======##
    ('map_year', Mapper('yr',config.model_config_.yr_mapping)),
    ('map_month', Mapper('mnth', config.model_config_.mnth_mapping)),
    ('map_season', Mapper('season', config.model_config_.season_mapping)),
    ('map_weather', Mapper('weathersit',config.model_config_.weather_mapping)),
    ('map_holiday', Mapper('holiday', config.model_config_.holiday_mapping)),
    ('map_workingday', Mapper('workingday', config.model_config_.workingday_mapping)),
    ('map_hour', Mapper('hr', config.model_config_.hour_mapping)),

    # removing outlier
    ('outlier_remover', OutlierHandler(config.model_config_.numerical_features)),

    # Weekday OneHotEncoder
    ('weekday_onhotencoder', WeekdayOneHotEncoder('weekday')),

    #Removing unwanted columns
    ('remove_unwanted_columns', RemoveUnwantedColumns(config.model_config_.unused_fields)),

    # scale
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config_.n_estimators, max_depth=15, random_state=42, oob_score=True))

])