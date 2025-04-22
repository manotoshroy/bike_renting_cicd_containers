
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from model.config.core import config
from model.processing.features import WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, WeekdayOneHotEncoder


def _test_weekday_transformer(sample_input_data):
    print('sample_input_data shape ' , sample_input_data[0].loc[5])
    # Given
    transformer = WeekdayImputer(
        variables='weekday'
    )

    assert pd.isna(sample_input_data[0].loc[5, 'weekday'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[5,'weekday'] == 'Sun'


def _test_weathersit_transformer(sample_input_data):
    # Given
    transformer = WeathersitImputer(
        variables='weathersit'
    )

    assert pd.isna(sample_input_data[0].loc[7, 'weathersit'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[7,'weathersit'] == 'Clear'

def test_year_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'yr': [2011, 2012],  
    })

    transformer = Mapper('yr',config.model_config_.yr_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['yr']) == [0,1]


def test_mnth_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'mnth': ['January', 'February', 'December', 'March', 'November', 'April', 'October', 'May', 'September', 'June', 'July', 'August']
    })

    transformer = Mapper('mnth',config.model_config_.mnth_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['mnth']) == [0,1,2,3,4,5,6,7,8,9,10,11]


def test_season_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'season': ['spring', 'winter', 'summer', 'fall']
    })

    transformer = Mapper('season',config.model_config_.season_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['season']) == [0,1,2,3]


def test_weathersit_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'weathersit': ['Heavy Rain', 'Light Rain', 'Mist', 'Clear']
    })

    transformer = Mapper('weathersit',config.model_config_.weather_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['weathersit']) == [0,1,2,3]

def test_holiday_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'holiday': ['Yes', 'No']
    })

    transformer = Mapper('holiday',config.model_config_.holiday_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['holiday']) == [0,1]



def test_workingday_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'workingday': ['Yes', 'No']
    })

    transformer = Mapper('workingday',config.model_config_.workingday_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['workingday']) == [1,0]



def test_hr_mapping_transformer(sample_input_data):
    
    sample_df = pd.DataFrame({
        'hr': ['4am', '3am', '5am', '2am', '1am', '12am',  '6am', '11pm', '10pm', '10am', '9pm', '11am', '7am', '9am', '8pm', '2pm', '1pm', '12pm', '3pm', '4pm', '7pm', '8am', '6pm',  '5pm']
    })

    transformer = Mapper('hr',config.model_config_.hour_mapping)
    # When
    result = transformer.fit(sample_df).transform(sample_df)

    # Then
    assert list(result['hr']) == [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]


# helper function to get the right calculation for test outlier 
def get_transdomed_outlier_df(df, columns):
    """Computes expected DataFrame by capping outliers based on IQR."""
    expected = df.copy()
    
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        expected[col] = expected[col].clip(lower=lower_bound, upper=upper_bound)
    return expected

def test_outlier_remover_transformer(sample_input_data):
    
    sample_data = pd.DataFrame({
        'Feature1': [10, 200, 15, 12, 250],  # 200, 250 are outliers
        'Feature2': [5, 8, 6, 300, -100]     # 300, -100 are outliers
    })

    """Test outlier removal for a sample dataset."""
    transformer = OutlierHandler(col_names=['Feature1', 'Feature2'])
    transformed_data = transformer.fit_transform(sample_data)
    expected_data = get_transdomed_outlier_df(sample_data, ['Feature1', 'Feature2'])
    pd.testing.assert_frame_equal(transformed_data, expected_data)

def test_outlier_handler_no_outliers():
    """Test when no outliers are present."""
    sample_data = pd.DataFrame({'Feature1': [10, 12, 14, 16, 18]})
    transformer = OutlierHandler(col_names=['Feature1'])
    transformed_data = transformer.fit_transform(sample_data)
    pd.testing.assert_frame_equal(transformed_data, sample_data) 

def test_outlier_handler_only_upper_outliers():
    """Test when only upper bound outliers exist."""
    df = pd.DataFrame({'Feature1': [10, 12, 300]})  # 300 is an outlier
    transformer = OutlierHandler(col_names=['Feature1'])
    transformed_data = transformer.fit_transform(df)
    expected_data = get_transdomed_outlier_df(df, ['Feature1'])
    pd.testing.assert_frame_equal(transformed_data, expected_data)

def test_outlier_handler_only_lower_outliers():
    """Test when only lower bound outliers exist."""
    df = pd.DataFrame({'Feature1': [-50, 12, 14]})  # -50 is an outlier
    transformer = OutlierHandler(col_names=['Feature1'])
    transformed_data = transformer.fit_transform(df)
    expected_data = get_transdomed_outlier_df(df, ['Feature1'])
    pd.testing.assert_frame_equal(transformed_data, expected_data)

def test_outlier_handler_with_nan_values():
    """Ensure NaN values remain unchanged."""
    df = pd.DataFrame({'Feature1': [10, np.nan, 300]})  # 300 is an outlier
    transformer = OutlierHandler(col_names=['Feature1'])
    transformed_data = transformer.fit_transform(df)
    expected_data = get_transdomed_outlier_df(df, ['Feature1'])
    pd.testing.assert_frame_equal(transformed_data, expected_data)

def test_outlier_handler_all_same_values():
    """Ensure no changes occur when all values are identical."""
    df = pd.DataFrame({'Feature1': [100, 100, 100, 100]})
    transformer = OutlierHandler(col_names=['Feature1'])
    transformed_data = transformer.fit_transform(df)
    pd.testing.assert_frame_equal(transformed_data, df)  # No changes should happen

