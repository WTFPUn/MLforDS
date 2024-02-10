import numpy as np
from Optimizer import Optimizer
from ..Function import Function
from ..CostFunction import CostFunction
from typing import TypedDict, List

History = TypedDict('History', {
  'cost': np.ndarray, 
  'parameters': List[np.ndarray],
  })
 
class GradientDescent(Optimizer):
  def __init__(self, cost_fnc: CostFunction, lr: float = 0.01, max_iter: int = 1000, stop_loss: bool = False ) -> None:
    self.lr = lr
    self.max_iter = max_iter
    self.cost_fnc = cost_fnc
    self.history: History = {'cost': [], 'parameters': []}

  def optimize(self, fnc: Function, features: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_i = np.random.rand(features.shape[1], 1)
    for _ in range(self.max_iter):
      cost = self.cost_fnc(y, fnc.forward(features, x_i))
      grad = fnc.backward(features, x_i)
      x_i -= self.lr * grad
      if self.stop_loss and cost < self.stop_loss:
        break

      self.history['cost'].append(cost)

    return x_i
