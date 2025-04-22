import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest

import numpy as np
import pandas as pd
from model.config.core import config
from model.processing.features import RemoveUnwantedColumns



def test_remove_existing_columns(sample_remove_column_data):
    """Test that specified columns are removed when they exist in the DataFrame."""
    
    input_df, remove_col_names, expected_df = sample_remove_column_data
    
    transformer = RemoveUnwantedColumns(col_names=remove_col_names)
    transformed_df = transformer.fit_transform(input_df)
    
    # Ensure that columns "B" and "C" are removed
    #assert list(transformed_df.columns) == expected_df.columns, "Columns B and C should be removed"
    pd.testing.assert_frame_equal(transformed_df, expected_df)



