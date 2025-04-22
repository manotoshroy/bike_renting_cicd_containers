# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
import model

from typing import List, Dict # This is for type safety

from pydantic import BaseModel
from strictyaml import YAML, load

import logging

file = Path(__file__).resolve()  # __file__ is a special variable in Python that holds the filename of the script being executed.
parent, root = file.parent, file.parents[1]
sys.path.append(str(root)) # dynamically add a directory to Python's module search path so that Python can find and import modules from that directory.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Project Directories
PACKAGE_ROOT = Path(model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"



# BaseModel is from Pydantic, it helps to manage  data validation, type enforcement, automatic validation
class AppConfig(BaseModel):

    training_data_file: str
    pipeline_save_file: str

class ModelConfig(BaseModel):
    target: str
    features: List[str]
    
    unused_fields: List[str]

    numerical_features: List[str]
    categorical_features: List[str]

    yr_mapping:Dict[int, int]
    mnth_mapping:Dict[str, int]
    season_mapping:Dict[str, int]
    weather_mapping:Dict[str, int]
    holiday_mapping:Dict[str, int]
    workingday_mapping:Dict[str, int]
    hour_mapping:Dict[str, int]

    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int
    max_features: int

class Config(BaseModel):

    app_config_:AppConfig
    model_config_:ModelConfig


def find_config_file() ->Path:
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(config_path: Path = None) -> YAML:
    print('fetch_config_from_yaml called ')
    """Parse YAML containing the package configuration."""
    if not config_path:
        try:
            config_path = find_config_file()
        except:
            raise
    
    if config_path:
        with open(config_path, "r") as conf_file:
            # load function from strictyml module
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {config_path}")


def create_and_validate_config(parsed_config: YAML = None):
    print('create_and_validate_config called ')
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()


    _config = Config(
        app_config_=AppConfig(**parsed_config.data), # dictionary unpacking
        model_config_=ModelConfig(**parsed_config.data),
    )
    print('create_and_validate_config done ')
    return _config

config = create_and_validate_config()