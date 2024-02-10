from tools.SuperviseLearning import SuperviseLearning
from typing import TypedDict, Dict

import numpy as np

StatDeatils = TypedDict("StatDeatils", {"mean": np.ndarray, "std": np.ndarray, "prob": float})

class NaiveBayes(SuperviseLearning):
    def __init__(self, df, label):
        super().__init__(df, label)
        self.stats: Dict[str, StatDeatils] = {k: {"mean": None, "std": None, "prob": None} for k in self.label.unique()}
        
        self.find_stats()

    
    def find_stats(self):
        '''
        find mean and std for each class
        '''
        for k in self.stats.keys():
            self.stats[k]["mean"] = self.features[self.label == k].mean()
            self.stats[k]["std"] = self.features[self.label == k].std()
            self.stats[k]["prob"] = len(self.features[self.label == k])/len(self.features)

    
    def __gaussian_pdf(self, x, mean, sd):
        '''
        Gaussian probability density function
        '''
        return 1/(np.sqrt(2*np.pi*(sd**2))) * np.exp(-((x - mean)**2)/(2*(sd**2)))

    def predict(self, x: Dict[str, float]):
        if list(x.keys()) != self.stats[0]['mean'].keys().to_list():
            raise ValueError("input must have the same keys as features")
        cls_prop = {}
        for cls_name, cls_stats in self.stats.items():
            prob = 1
            for f, v in x.items():
                prob *= self.__gaussian_pdf(v, cls_stats["mean"][f], cls_stats["std"][f])
            prob *= cls_stats["prob"]
            cls_prop[cls_name] = prob

        return max(cls_prop, key=cls_prop.get)
            
            
        