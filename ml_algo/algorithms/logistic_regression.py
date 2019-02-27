import numpy as np
import tensorflow as tf


class LogisticRegressionTf:
    def __init__(self, dims, learning_rate, batch_size, iter_num, seed=None):

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

            self.model()

    def model(self):
        h = tf.matmul(self.X, self.w) + self.b
        self.y_hat = tf.sigmoid(h)
        self.cost = -1*(self.y * tf.log(self.y_hat) + (1-self.y) * tf.log(1 - self.y_hat))
        self.loss = tf.reduce_mean(self.cost)
        # reg = (self.learning_rate/2) * tf.nn.l2_loss(self.w)
        # reg = tf.contrib.slim.l2_regularizer(scale=0.05, scope=None)
        # reg_penalty = tf.contrib.layers.apply_regularization(reg, tf.trainable_variables())
        # self.loss = self.loss + reg_penalty

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=self.y))

        self.optimizer = tf.train.AdagradDAOptimizer(self.learning_rate, l2_regularization_strength=.8,
                                                     l1_regularization_strength=.2,
                                                     global_step=tf.get_default_graph().get_tensor_by_name("global_step:0"))
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.goal = self.optimizer.minimize(self.loss)

        self.prediction = tf.round(self.y_hat)
        correct = tf.cast(tf.equal(self.prediction, self.y), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(correct)

        return

    def train(self, train_x, train_y, test_x, test_y):
        loss_trace = []
        train_acc = []
        test_acc = []

        with tf.Session(graph=self.g) as sess:
            sess.run([tf.global_variables_initializer()])

            for epoch in range(self.iter_num):
                batch_indexes = np.random.choice(len(train_x), size=self.batch_size)
                batch_train_X = train_x[batch_indexes]
                batch_train_y = np.matrix(train_y[batch_indexes])

                batch_loss = sess.run(self.loss, feed_dict={"X:0": batch_train_X, "y:0": batch_train_y, "global_step:0":epoch })
                batch_train_acc = sess.run(self.accuracy, feed_dict={"X:0": train_x, "y:0": train_y, "global_step:0":epoch})
                batch_test_acc = sess.run(self.accuracy, feed_dict={"X:0": test_x, "y:0": test_y,"global_step:0":epoch})

                loss_trace.append(batch_loss)
                train_acc.append(batch_train_acc)
                test_acc.append(batch_test_acc)

                if (epoch + 1) % 2 == 0:
                    print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, batch_loss, batch_train_acc, batch_test_acc))
                    # print('epoch: {} loss: {} train_acc: {} test_acc: {}'.format(epoch + 1, batch_loss, batch_train_acc, batch_test_acc))
        return self.g, loss_trace, train_acc, test_acc