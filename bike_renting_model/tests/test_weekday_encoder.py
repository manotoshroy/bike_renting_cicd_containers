import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest

import numpy as np
import pandas as pd
from model.config.core import config
from model.processing.features import WeekdayOneHotEncoder



def test_weekday_onehotencoder_transform_output(sample_weekday_data):
    """Test one-hot encoding of weekday column's output"""
    transformer = WeekdayOneHotEncoder(col_name='weekday')
    transformed_data = transformer.fit_transform(sample_weekday_data)

    # Expected columns after one-hot encoding
    expected_columns = set(transformer.encoder.get_feature_names_out(["weekday"]))

    # Ensure the original "weekday" column is removed
    assert "weekday" not in transformed_data.columns, "Original column should be dropped"

    # Check if all expected columns exist in the transformed DataFrame
    assert set(transformed_data.columns) == expected_columns, "One-hot encoded columns mismatch"


def test_weekday_onehotencoder_with_nan(sample_data_with_nan):
    """Test handling of NaN values."""
    transformer = WeekdayOneHotEncoder(col_name='weekday')

    with pytest.raises(ValueError):  # OneHotEncoder should fail on NaNs
        transformer.fit_transform(sample_data_with_nan)

def test_weekday_onehotencoder_unseen_value(sample_data_unseen):
    """Test behavior when encountering unseen values in transform."""
    transformer = WeekdayOneHotEncoder(col_name='weekday')
    transformer.fit(sample_data_unseen["train"])

    with pytest.raises(ValueError):  # Should raise an error for unseen category
        transformer.transform(sample_data_unseen["test"])

def test_weekday_onehotencoder_empty_dataframe(empty_dataframe):
    """Ensure the transformer handles an empty DataFrame properly."""
    transformer = WeekdayOneHotEncoder(col_name='weekday')
    transformed_data = transformer.fit_transform(empty_dataframe)

    # Expect empty DataFrame with no additional columns
    assert transformed_data.empty

def test_weekday_onehotencoder_invalid_input(invalid_input):
    """Ensure TypeError is raised when input is not a DataFrame."""
    transformer = WeekdayOneHotEncoder(col_name='weekday')
    with pytest.raises(TypeError):
        transformer.transform(invalid_input)
