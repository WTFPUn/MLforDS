import numpy as np
from . import CostFunction

class MSE(CostFunction):
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> np.float64:
        return np.mean((y - y_hat) ** 2)