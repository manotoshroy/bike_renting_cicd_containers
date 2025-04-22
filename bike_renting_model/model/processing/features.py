import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """
    
    def __init__(self, variables: str):
        logging.info(f"WeekdayImputer Initialized ")
        print("WeekdayImputer Init called")
        if not isinstance(variables, str):
            raise ValueError("variables should be a string")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            X = X.copy()

            # we need the fit statement to accomodate the sklearn pipeline
            wkday_null_idx = X[X[self.variables].isnull() == True].index
            print('fill wkday_null_idx : ', wkday_null_idx)
            self.fill_value = X.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
            #print('fill value : ', self.fill_value)
            #logging.warning(f"WeekdayImputer Tranform value: {self.fill_value}")


            X[self.variables]=X[self.variables].fillna(self.fill_value)

            return X
        except Exception as e:
            logging.error(f"WeekdayImputer Transform failed: {e}")
            raise


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):
        if not isinstance(variables, str):
                raise ValueError("variables should be a string")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            self.value = X[self.variables].mode()[0]
            return self
        except Exception as e:
            logging.error(f"WeathersitImputer fit failed: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            X = X.copy()
            X[self.variables]=X[self.variables].fillna(self.value)

            return X

        except Exception as e:
             logging.error(f"WeathersitImputer transform failed: {e}")
             raise

class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""
    
    def __init__(self, variable: str, mappings: dict):
        #print(f'Mapper called mappings : {mappings}')
        if not isinstance(variable, str):
            raise ValueError("variables should be a str")

        if not isinstance(mappings, dict):
            raise ValueError("mappings should be a dict")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")
            X = X.copy()
            X[self.variable] = X[self.variable].apply(lambda x:self.mappings[x])

            return X

        except Exception as e:
            logging.error(f"Mapper transform failed while setting :{self.variable} Error: {e}")
            raise


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """
    def __init__(self, col_names: list):
        if not isinstance(col_names, list):
            raise ValueError("col_name should be a list")
        self.col_names = col_names

    def fit(self, X=None, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        '''
        df = X.copy()
        q1 = df.describe()[self.col_name].loc['25%']
        q3 = df.describe()[self.col_name].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        '''
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            df = X.copy()
            for col in self.col_names:
                q1 = df.describe()[col].loc['25%']
                q3 = df.describe()[col].loc['75%']
                iqr = q3 - q1
                self.lower_bound = q1 - (1.5 * iqr)
                self.upper_bound = q3 + (1.5 * iqr)

                for i in df.index:
                    if df.loc[i,col] > self.upper_bound:
                        df.loc[i,col]= self.upper_bound
                    if df.loc[i,col] < self.lower_bound:
                        df.loc[i,col]= self.lower_bound
            return df
        except Exception as e:
            logging.error(f"OutlierHandler transform failed: {e}")
            raise


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, col_name: str):
        if not isinstance(col_name, str):
            raise ValueError("col_name should be a str")
        self.col_name = col_name

    def fit(self, X: pd.DataFrame, y=None):
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")
            
            # Ensure the column exists
            if self.col_name not in X.columns:
                raise ValueError(f"Column '{self.col_name}' not found in DataFrame.")

            if X[self.col_name].isnull().any():
                raise ValueError(f"Column '{self.col_name}' contains NaN values.")

            if X.empty:
                logging.warning("Received an empty DataFrame in fit. Skipping fitting.")
                return self 

            self.encoder = OneHotEncoder(sparse_output=False)
            self.encoder.fit(X[[self.col_name]])
            return self
        except Exception as e:
            logging.error(f"WeekdayOneHotEncoder fit failed: {e}")
            raise

    def transform(self, X: pd.DataFrame) ->pd.DataFrame:
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")
            
            if X.empty:
                logging.warning("Received an empty DataFrame in transform. Returning it unchanged.")
                return X

            X = X.copy()

            encoded_data = self.encoder.transform(X[[self.col_name]])
            enc_wkday_features = self.encoder.get_feature_names_out([self.col_name])
            X[enc_wkday_features] = encoded_data

            # Not sure if we should do this here
            X.drop(labels = self.col_name, axis = 1, inplace = True)

            return X

        except Exception as e:
            logging.error(f"WeekdayOneHotEncoder transform failed: {e}")
            raise


class RemoveUnwantedColumns(BaseEstimator, TransformerMixin):
    """Remove specified unwanted columns from the DataFrame"""

    def __init__(self, col_names: list):
        if not isinstance(col_names, list):
            raise ValueError("col_names should be a list.")
        self.col_names = col_names

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            X = X.copy()  # Just avoiding modifying original DataFrame

            # Check if the mentioned columns are part of the data set.
            missing_cols = [col for col in self.col_names if col not in X.columns]
            if missing_cols:
                logging.warning(f"Columns {missing_cols} not found in DataFrame. Skipping.")

            self.col_names = [item for item in self.col_names if item not in missing_cols]

            # Drop only existing columns
            X.drop(self.col_names, axis=1, inplace=True)

            return X  # Return the modified DataFrame

        except Exception as e:
            logging.error(f"RemoveUnwantedColumns failed: {e}")
            raise