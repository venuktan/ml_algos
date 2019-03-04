from algorithms.logistic_regression import LogisticRegressionTf
from ml_algo.algorithms.svc import SvcTf
from ml_algo.datasets.haberman_data import HabermanData
from sklearn.svm  import SVC

class Main:
    def __init__(self):
        data = HabermanData()
        lr = SvcTf(dims=3, learning_rate=0.07, batch_size=500, iter_num=10, seed=None)
        g, loss_trace, train_acc, test_acc = lr.train(data.train_x, data.train_y, data.test_x, data.test_y,
                                                      C=3.5, alpha=.01)

        # lr = LogisticRegressionTf(dims=3, learning_rate=0.008, batch_size=40, iter_num=100, seed=None)
        # g, loss_trace, train_acc, test_acc = lr.train(data.train_x, data.train_y, data.test_x, data.test_y)

        # svc = SVC()
        # svc.fit(data.train_x, data.train_y)
        # acc = svc.score(data.test_x, data.test_y)
        # print("acc ", acc)
        pass

if __name__ == '__main__':
    Main()
    pass
