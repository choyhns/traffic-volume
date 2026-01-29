from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (%)
    - 교통량이 0에 가까운 값이 있을 수 있어 eps로 분모 안정화
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(mean_absolute_error(y_true, y_pred))


def eval_regression(y_true, y_pred) -> dict:
    """
    반환: {"MAE":..., "RMSE":..., "MAPE":...}
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
