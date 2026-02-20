import numpy as np

def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def test_wmape_basic():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    result = wmape(y_true, y_pred)

    assert round(result, 4) == round((10+10+10)/600, 4)


def test_wmape_zero_error():
    y_true = np.array([50, 60, 70])
    y_pred = np.array([50, 60, 70])

    result = wmape(y_true, y_pred)

    assert result == 0.0