from algorithms.logistic_regression import LogisticRegressionTf
from ml_algo.algorithms.svc import SvcTf
from ml_algo.datasets.haberman_data import HabermanData
from sklearn.svm  import SVC

class Main:
    def __init__(self):
        data = HabermanData()

        lr = SvcTf(dims=3, learning_rate=0.03, batch_size=7, iter_num=100, seed=None)
        g, loss_trace, train_acc, test_acc = lr.train(data.train_x, data.train_y, data.test_x, data.test_y,
                                                      C=.5, alpha=.1)

        # lr = LogisticRegressionTf(dims=3, learning_rate=0.09, batch_size=30, iter_num=100, seed=None)
        # g, loss_trace, train_acc, test_acc = lr.train(data.train_x, data.train_y, data.test_x, data.test_y)

        # svc = SVC()
        # svc.fit(data.train_x, data.train_y)
        # acc = svc.score(data.test_x, data.test_y)
        # print("acc ", acc)
        pass

if __name__ == '__main__':
    Main()
    pass
