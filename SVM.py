# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()

#产生数据
x_vals = np.loadtxt(open(r"SVMtestdatasetin.csv", "r"), delimiter=",", skiprows=0)
y_vals = np.loadtxt(open(r"SVMtestdatasetout.csv", "r"), delimiter=",", skiprows=0)

#划分数据为训练集和测试集
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
#批训练中批的大小
batch_size = 100
# 初始化feedin
x_data = tf.placeholder(shape=[None, 256], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 创建变量
W = tf.Variable(tf.random_normal(shape=[256, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
# 定义线性模型
model_output = tf.matmul(x_data, W)+b
# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(W))
#软正则化参数
alpha = tf.constant([0.02])
#定义损失函数
classification_term = tf.reduce_mean(tf.maximum(0., 1.-model_output*y_target))
loss = classification_term+alpha*l2_norm
# classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
# loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
#输出
prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#开始训练
sess.run(tf.global_variables_initializer())
loss_vec = []
train_accuracy = []
test_accuracy = []

for i in range(2000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    train_acc_temp = sess.run(accuracy, feed_dict={
                              x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    test_acc_temp = sess.run(accuracy, feed_dict={
                             x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)
    if (i+1) % 100 == 0:
        print('Step #' + str(i+1) + ' W = ' +
              str(sess.run(W)) + 'b = ' + str(sess.run(b)))
        print('Loss = ' + str(test_acc_temp))  # test_acc_temp
plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['损失', '训练精确度', '测试精确度'])
plt.ylim(0., 1.)
plt.show()
