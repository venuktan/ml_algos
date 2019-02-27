import os
import pandas as pd
from sklearn.model_selection import train_test_split


class HabermanData:
    def __init__(self):
        self.dataset_base_url = "/Users/venu.tangirala/datasets/haberman"
        self.train_url = os.path.join(self.dataset_base_url, "haberman.data")
        self.df = pd.read_csv(self.train_url)
        X = self.df[["age", "operation_age", "pos_nodes"]].values
        y = self.df[["label"]].values-1
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2)