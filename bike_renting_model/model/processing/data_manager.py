import sys
from pathlib import Path
import pandas as pd

from model import __version__ as _version
from model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

file = Path(__file__).resolve()  # __file__ is a special variable in Python that holds the filename of the script being executed.
parent, root = file.parent, file.parents[1]
sys.path.append(str(root)) # dynamically add a directory to Python's module search path so that Python can find and import modules from that directory.

import joblib
from sklearn.pipeline import Pipeline
import typing as t

def load_raw_dataset(*, file_name)-> pd.DataFrame:
    dataframe= pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name)-> pd.DataFrame:
    dataframe= pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame = dataframe)
    return transformed


def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:

    df = get_year_and_month(data_frame = data_frame)
    #print(f'pre_pipeline_preparation df {df.columns} ')
    #print(f'pre_pipeline_preparation df {df['yr'].head()} type {type(df['yr'])} ')
    return df


def get_year_and_month(*, data_frame: pd.DataFrame)-> pd.DataFrame:

    df = data_frame.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()

    return df

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    # Ensure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model