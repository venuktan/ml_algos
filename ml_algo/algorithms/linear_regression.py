import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from datasets.pima import Pima


class LinearRegressionTF:
    def __init__(self, dims: int, learning_rate: float, batch_size: int, iter_num: int, seed: int = None):
        """

        :param dims:
        :param learning_rate:
        :param batch_size:
        :param iter_num:
        :param seed:
        """

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iter_num = iter_num
        with tf.Graph().as_default() as self.g:

            self.global_step_tensor = tf.placeholder(tf.int64, name='global_step')
            self.X = tf.placeholder(tf.float32, shape=[None, dims], name="X")
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
            if seed:
                tf.set_random_seed(seed)

            self.w = tf.Variable(tf.random_normal(shape=[dims, 1], mean=0., stddev=0.5, dtype=tf.float32),
                                 name="w")
            self.b = tf.Variable(tf.random_normal(shape=[1, 1], mean=0., stddev=0.5, dtype=tf.float32),
                                 name="b")

            self._model()

    def _model(self):
        self.y_hat = tf.add(tf.matmul(self.X, self.w), self.b, name="y_hat")
        self.cost = tf.square(self.y - self.y_hat)
        self.loss = tf.reduce_mean(self.cost, name="mse")

        # reg = (self.learning_rate/2) * tf.nn.l2_loss(self.w)
        # reg = tf.contrib.slim.l2_regularizer(scale=0.05, scope=None)
        # reg_penalty = tf.contrib.layers.apply_regularization(reg, tf.trainable_variables())
        # self.loss = self.loss + reg_penalty
        # self.optimizer = tf.train.AdagradDAOptimizer(self.learning_rate, l2_regularization_strength=.8,
        #                                              l1_regularization_strength=.2,
        #                                              global_step=tf.get_default_graph().get_tensor_by_name("global_step:0"))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.goal = self.optimizer.minimize(self.loss)

        self.prediction = self.y_hat
        self.mse = self.loss
        self.rmse = tf.sqrt(self.mse, name="rmse")
        self.mae = tf.reduce_mean(tf.abs(self.y - self.y_hat), name="mae")

        return

    def train(self, train_x: np.array, train_y: np.array, test_x: np.array, test_y: np.array):
        """

        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :return:
        """
        loss_trace = []
        train_rmse_trace = []
        test_rmse_trace = []

        with tf.Session(graph=self.g) as sess:
            sess.run([tf.global_variables_initializer()])

            for epoch in range(self.iter_num):
                batch_indexes = np.random.choice(len(train_x), size=self.batch_size)
                batch_train_X = train_x[batch_indexes]
                batch_train_y = np.matrix(train_y[batch_indexes])

                batch_loss = sess.run(self.loss, feed_dict={"X:0": batch_train_X, "y:0": batch_train_y, "global_step:0":epoch})
                train_mse, train_rmse, train_mae = sess.run(["mse:0", "rmse:0", "mae:0"],
                                           feed_dict={"X:0": train_x, "y:0": train_y, "global_step:0":epoch})
                test_mse, test_rmse, test_mae = sess.run(["mse:0", "rmse:0", "mae:0"],
                                          feed_dict={"X:0": test_x, "y:0": test_y, "global_step:0":epoch})

                loss_trace.append(batch_loss)
                train_rmse_trace.append(train_rmse)
                test_rmse_trace.append(test_rmse)

                if (epoch + 1) % 2 == 0:
                    print('epoch: {:4d} loss: {:5f} train_rmse: {:5f} test_rmse: {:5f}'.format(epoch + 1, batch_loss,
                                                                                               train_rmse, test_rmse))
        return self.g, loss_trace, train_rmse, test_rmse

if __name__ == '__main__':
    cal_data = fetch_california_housing()
    dims = len(cal_data.feature_names)
    lr = LinearRegressionTF(dims=dims, learning_rate=0.06, batch_size=10, iter_num=1000)
    train_x, test_x, train_y, test_y = train_test_split(cal_data.data, cal_data.target, test_size=0.2)
    g, loss_trace, train_rmse, test_rmse = lr.train(train_x, np.reshape(train_y, newshape=[-1,1]),
                                                    test_x, np.reshape(test_y, newshape=[-1,1]))
