import pandas as pd
from sklearn.model_selection import train_test_split

class SuperviseLearning:
  def __init__(self, df: pd.DataFrame, label: str):
    self.features = df.drop(label, axis=1)
    self.label = df[label]

  def split_train_test(self, test_size: float):
        '''
        split dataset to train and test set
        '''
        df = self.features
        df["label"] = self.label
        self.train, self.test = train_test_split(df, test_size=test_size)
        return self.train, self.test