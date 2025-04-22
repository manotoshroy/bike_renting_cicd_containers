import sys
from pathlib import Path


file = Path(__file__).resolve()  # __file__ is a special variable in Python that holds the filename of the script being executed.
parent, root = file.parent, file.parents[1]
sys.path.append(str(root)) # dynamically add a directory to Python's module search path so that Python can find and import modules from that directory.

import logging
from sklearn.model_selection import train_test_split
from model.pipeline import bike_renting_pipeline
from model.processing.data_manager import load_dataset, save_pipeline
from model.config.core import config
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error



def run_training()-> None:
    print('run_training started ')
    data = load_dataset(file_name=config.app_config_.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],  # predictors
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state
    )

    # Pipeline fitting
    bike_renting_pipeline.fit(X_train,y_train)
    y_pred = bike_renting_pipeline.predict(X_test)
    print("R2 score:", r2_score(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # persist trained model
    save_pipeline(pipeline_to_persist= bike_renting_pipeline)
    # printing the score

if __name__ == "__main__":
    run_training()