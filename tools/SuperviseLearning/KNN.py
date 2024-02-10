from tools.SuperviseLearning import SuperviseLearning

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import List, Tuple

MIN_R, MAX_R = int, int

SamplingArg = Tuple[MIN_R, MAX_R]
RandomSampling = Tuple[SamplingArg, ...]

MEAN = int
SD = int
X_M, Y_M  = MEAN, MEAN
D2Mean = Tuple[X_M, Y_M]

X_S, Y_S = SD, SD
D2Std = Tuple[X_S, Y_S]

class KNN(SuperviseLearning):
    
    def __init__(self, df, label):
        super().__init__(df, label)
    
    @staticmethod
    def from_random_sampling(features: RandomSampling, label: SamplingArg, row_num: int):
        df = pd.DataFrame()
        for i, (min_r, max_r) in enumerate(features):
            df[f"feature_{i}"] = np.random.randint(min_r, max_r, row_num)
        df["label"] = np.random.randint(*label, row_num)
        return KNN(df, "label")
    
    @staticmethod
    def from_2d_normal_dist(mean: List[D2Mean], std: List[D2Std], row_per_class: int, label_name: Tuple[str, ...] = None):

        assert len(mean) == len(std), "mean and std must have the same length"

        if label_name is None:
            label_name = [i for i in range(len(mean))]
        else:
            assert len(mean) == len(label_name), "mean/std and label_name must have the same length"
        
        data = {
            "feature_x": [np.random.normal(mean[i][0], std[i][0]) for i in range(len(mean)) for _ in range(row_per_class)],
            "feature_y": [np.random.normal(mean[i][1], std[i][1]) for i in range(len(mean)) for _ in range(row_per_class)],
            "label": [label_name[i] for i in range(len(mean)) for _ in range(row_per_class)]
        }
        df = pd.DataFrame(data)

        return KNN(df, "label")
    

    def __euclidient_dist(self, x1: np.ndarray, x2: np.ndarray):
        '''
        private method
        ------
        euclidient distance between two vectors
        params
        ------
        x1: np.ndarray
        x2: np.ndarray
        ------
        this fucntion is base on equation:
        d(x1, x2) = sqrt(sum((x1 - x2)^2))
        '''
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    

