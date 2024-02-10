from abc import ABC, abstractmethod
import numpy as np

class CostFunction(ABC):
  @abstractmethod
  def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> np.float64:
    pass
