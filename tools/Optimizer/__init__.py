import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

LinearFunction = Callable[[np.ndarray], np.ndarray]

class Optimizer(ABC):
    '''
    Optimizer 
    ---------
    Abstract class for optimizer, this class must implement the optimize method and this method must return the optimized parameters

    '''
  
    @abstractmethod
    def optimize(self, f: LinearFunction, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        optimize
        --------
        Abstract method for optimization

        Parameters
        ----------
        f : LinearFunction
            Linear function to be optimized
        x : np.ndarray
            Input data(shape: (n, m))
        y : np.ndarray
            Output data(shape: (n, 1))

        Returns
        -------
        np.ndarray
            Optimized parameters
        '''
        pass
    
  