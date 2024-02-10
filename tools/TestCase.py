from typing import TypeVar, Generic, List
from abc import ABC, abstractmethod

I = TypeVar('I')
O = TypeVar('O')

class TestCase(Generic[I, O], ABC):
  def __init__(self, inputs: List[I], outputs: List[O]):
    self.input = inputs
    self.output = outputs
    self.result: List[bool] = []


  @abstractmethod
  def func(self, input: I) -> O:
    pass

  def run(self):
    for i in range(len(self.input)):
      if self.func(self.input[i]) != self.output[i]:
        print(f"Test case {i} failed")
        return