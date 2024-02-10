import numpy as np
from abc import ABC, abstractmethod
class Function(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def backward(self, *args, **kwargs):
        pass