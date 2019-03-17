from datasets.Iris import Iris
from ml_algo.algorithms.svc import SvcTf
import numpy as np
import matplotlib.pyplot as plt


class Main:
    def __init__(self):
        data = Iris()
        lr = SvcTf(dims=data.train_x.shape[1], learning_rate=0.03, batch_size=135, iter_num=500, seed=None)
        g, loss_trace, train_acc, test_acc = lr.train(data.train_x, data.train_y, data.test_x, data.test_y,
                                                      C=1.5, alpha=.1, epoch_print=10)

        PlotAccLoss(np.array(loss_trace), train_acc, test_acc)
        # lr = LogisticRegressionTf(dims=3, learning_rate=0.09, batch_size=30, iter_num=100, seed=None)
        # g, loss_trace, train_acc, test_acc = lr.train(data.train_x, data.train_y, data.test_x, data.test_y)

        # svc = SVC()
        # svc.fit(data.train_x, data.train_y)
        # acc = svc.score(data.test_x, data.test_y)
        # print("acc ", acc)
        pass


class PlotAccLoss():
    def __init__(self, loss_vec, train_accuracy, test_accuracy):
        # Plot train/test accuracies
        plt.plot(train_accuracy, 'k-', label='Training Accuracy')
        plt.plot(test_accuracy, 'r--', label='Test Accuracy')
        plt.title('Train and Test Set Accuracies')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.show()
        pass


if __name__ == '__main__':
    Main()
    # data = Iris()

    pass
