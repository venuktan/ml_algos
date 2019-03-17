import numpy as np
from sklearn import datasets


class Iris():
    def __init__(self):
        # Load the data
        # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
        iris = datasets.load_iris()
        self.x_vals = np.array([[x[0], x[3]] for x in iris.data])
        self.y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

        # self.y_vals[self.y_vals == -1] = 0
        # Split data into train/test sets
        train_indices = np.random.choice(len(self.x_vals),
                                         int(round(len(self.x_vals)*0.9)),
                                         replace=False)
        test_indices = np.array(list(set(range(len(self.x_vals))) - set(train_indices)))
        self.train_x = self.x_vals[train_indices]
        self.test_x = self.x_vals[test_indices]
        self.train_y = np.reshape(self.y_vals[train_indices], newshape=[1, -1]).T
        self.test_y = np.reshape(self.y_vals[test_indices], newshape=[1, -1]).T