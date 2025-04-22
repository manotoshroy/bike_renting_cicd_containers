"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import r2_score

from model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    #expected_no_predictions = 120.91235638

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    print('isinstance(predictions[0], numpy.float32)', type(predictions[0]))
    #assert isinstance(predictions[0], np.float32)
    assert result.get("errors") is None
    #assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_actual = sample_input_data[1]
    #accuracy = accuracy_score(_predictions, y_true)
    r2score = r2_score(_predictions, y_actual)
    print ("r2score ", r2score)
    assert r2score > 0.86

