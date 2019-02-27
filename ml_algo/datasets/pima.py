# wget http://archive.ics.uci.edu/ml/machine-learning-databases/00220/Relation%20Network%20(Directed).data
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class Pima:
    def __init__(self):
        self.dataset_base_url = "/Users/venu.tangirala/datasets/pima_indians_diabetes"
        self.train_url = os.path.join(self.dataset_base_url, "diabetes.csv")
        self.df = pd.read_csv(self.train_url)
        X = self.df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].values
        y = self.df[["Outcome"]].values
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, y, test_size=0.2)



if __name__ == '__main__':
    p = Pima()