import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import pandas as pd
import pandas as np
from sklearn.model_selection import train_test_split

from model.config.core import config
from model.processing.data_manager import load_dataset


# @pytest.fixture is a decorator in the pytest testing framework used to define a fixture. 
# Fixtures are a way to provide setup and teardown functionality for your test functions or test classes. 
# They allow you to create reusable components that can be shared across multiple test functions, 
# improving code modularity and maintainability.

@pytest.fixture
def sample_input_data():
    data = load_dataset(file_name=config.app_config_.training_data_file)

    X = data.drop(config.model_config_.target, axis=1)       # predictors
    y = data[config.model_config_.target]                    # target

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,  # target
        test_size=config.model_config_.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config_.random_state,
    )

    return X_test, y_test


@pytest.fixture
def sample_weekday_data():
    """Fixture: Basic weekday data for testing"""
    return pd.DataFrame({'weekday': ['Mon', 'Tue', 'Wed', 'Mon', 'Fri']})

@pytest.fixture
def sample_data_with_nan():
    """Fixture: Data with NaN values"""
    return pd.DataFrame({'weekday': ['Mon', None , 'Wed', 'Fri']})

@pytest.fixture
def sample_data_unseen():
    """Fixture: Data containing unseen weekday values"""
    return {
        "train": pd.DataFrame({'weekday': ['Mon', 'Tues', 'Wed']}),
        "test": pd.DataFrame({'weekday': ['Sun']})  # Unseen category
    }

@pytest.fixture
def empty_dataframe():
    """Fixture: An empty DataFrame"""
    return pd.DataFrame(columns=['weekday'])

@pytest.fixture
def invalid_input():
    """Fixture: Invalid input (list instead of DataFrame)"""
    return ['Mon', 'Tue']

# -- Data for RemoveUnwantedColumn --

@pytest.fixture
def sample_remove_column_data():
    input_df = pd.DataFrame({
        "Col_1": [1, 2, 3],
        "Col_2": [4, 5, 6],
        "Col_3": [7, 8, 9],
        "Col_4": [10, 11, 12]
    })
    expected_df = pd.DataFrame({
        "Col_1": [1, 2, 3],  # Assuming column "B" is removed
        "Col_3": [7, 8, 9]
    })
    remove_col_names = ['Col_2', 'Col_4']

    return input_df, remove_col_names, expected_df
