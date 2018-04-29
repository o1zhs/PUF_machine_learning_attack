import tensorflow as tf
import numpy as np
import random

PUF_num = 64
train_num = 2000  # 训练数据
test_num = 10000  # 测试数据
#仿真PUF数据
data = [194.606, 195.394, 196.01, 195.874, 194.794, 193.683, 192.148, 195.4, 194.362, 193.634, 195.452, 192.461, 194.52,
        194.18, 194.802, 192.773, 194.291, 195.502, 195.698, 193.55, 196.461, 193.983, 196.106, 195.993, 195.995,
        195.289, 194.944, 194.071, 196.365, 194.347, 194.936, 194.951, 196.044, 194.796, 195.777, 193.982, 196.579,
        194.747, 196.359, 194.955, 196.574, 194.404, 194.944, 193.574, 195.21, 195.17, 193.35, 193.762, 193.704, 195.66,
        194.801, 195.984, 195.22, 195.836, 195.341, 195.87, 196.222, 196.488, 194.434, 193.008, 194.658, 196.154,
        194.116, 192.048,
        194.615, 195.435, 196.026, 195.86, 194.767, 193.7, 192.157, 195.408, 194.414, 193.562, 195.477, 192.475,
        194.526, 194.178, 194.732, 192.583, 194.308, 195.519, 195.74, 193.585, 196.455, 194.032, 196.068, 196.012,
        196.055, 195.308, 194.953, 194.048, 196.423, 194.362, 194.977, 194.975, 196.081, 194.846, 195.799, 193.943,
        196.653, 194.844, 196.344, 195.02, 196.567, 194.385, 195.002, 193.54, 195.605, 195.719, 193.845, 193.85,
        193.702, 195.78, 194.8, 195.994, 195.06, 195.775, 195.308, 195.85, 196.388, 196.52, 194.348, 192.991, 194.688,
        196.296, 194.132, 192.139,
        194.643, 195.384, 195.969, 195.894, 194.776, 193.682, 192.129, 195.555, 194.426, 193.58, 195.446, 192.459,
        194.55, 194.2, 194.764, 192.783, 194.282, 195.54, 195.736, 193.621, 196.422, 194.011, 196.165, 195.98, 196.046,
        195.371, 194.909, 194.053, 196.373, 194.323, 194.997, 194.996, 196.169, 194.818, 195.829, 193.953, 196.678,
        194.829, 196.411, 195.006, 196.552, 194.398, 194.999, 193.572, 195.651, 195.73, 193.907, 193.793, 193.698,
        195.06, 194.838, 195.955, 195.053, 195.746, 195.359, 195.877, 196.354, 196.494, 194.449, 192.957, 194.704,
        196.18, 194.108, 192.211,
        194.558, 195.428, 196.014, 195.846, 194.833, 193.671, 192.1, 195.536, 194.341, 193.519, 195.585, 192.485,
        194.518, 194.184, 194.829, 192.84, 194.295, 195.572, 195.657, 193.571, 196.464, 194.05, 196.063, 195.891,
        196.012, 195.263, 194.978, 194.096, 196.328, 194.362, 194.92, 194.847, 196.08, 194.816, 195.736, 194.011,
        196.631, 194.738, 196.345, 194.996, 196.57, 194.413, 194.859, 193.54, 195.33, 195.24, 193.17, 193.733, 193.693,
        195.63, 194.807, 195.945, 195.65, 195.797, 195.355, 195.846, 196.234, 196.473, 194.456, 192.946, 194.744,
        196.188, 194.09, 192.009]


# 4X64建模
def getCRPs_4X64(get_num):
    # 生成激励
    c = []
    for i in range(get_num):
        c.append([random.randint(0, 1) for i in range(PUF_num)])

    # 激励扩展
    cc = [[1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0], [0, -1, 1, 0]]
    ck = []
    for ci in c:
        flag = 0  # 0表示蓝色信号当前在下传播，1表示在上传播
        x = []
        for i in ci:
            x.append(cc[2 * i + flag])
            if i == 1:
                flag = 1 - flag  # i=1蓝色路径翻转
        ck.append(np.array(x).T)

    # 生成输入输出
    input = []
    output = []
    for num in range(get_num):
        sum = 0.0  # 矩阵对应位置和
        ci = ck[num]
        cnew = []
        for i in range(4):
            for j in range(PUF_num):
                sum += ci[i, j] * data[i * PUF_num + j]
                cnew.append(ci[i, j])
        input.append(cnew)
        r = 1 if sum > 0 else 0
        output.append(r)
    input = np.array(input).transpose()
    return np.float32(input), np.float32(output), 4 * PUF_num


# 2X64建模
def getCRPs_2X64(get_num):
    # 生成激励
    c = []
    for i in range(get_num):
        c.append([random.randint(0, 1) for i in range(PUF_num)])

    # 获得每段交叉路径和平行路径的延迟差（上减下）
    delay = []
    for i in range(2 * PUF_num):
        if i < PUF_num:
            delay.append(data[i] - data[3 * PUF_num + i])
        else:
            delay.append(data[i] - data[PUF_num + i])

    # 激励扩展
    cc = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    ck = []
    for ci in c:
        flag = 0  # 0表示路径未翻转，1表示翻转
        x = []
        for i in ci:
            x.append(cc[2 * i + flag])
            if i == 1:
                flag = 1 - flag  # i=1路径翻转
        ck.append(np.array(x).T)

    # 生成输入输出
    input = []
    output = []
    for num in range(get_num):
        sum = 0.0  # 延迟差和
        ci = ck[num]
        cnew = []
        for i in range(2):
            for j in range(PUF_num):
                sum += ci[i, j] * delay[i * PUF_num + j]
                cnew.append(ci[i, j])
        input.append(cnew)
        r = 1 if sum > 0 else 0
        output.append(r)
    input = np.array(input).transpose()
    return np.float32(input), np.float32(output), 2 * PUF_num


# 64参数建模
def getCRPs_64(get_num):
    # 生成激励
    c = []
    for i in range(get_num):
        c.append([random.randint(0, 1) for i in range(PUF_num)])

    # 转化输入
    cc = [[1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0], [0, -1, 1, 0]]
    cz = [1, -1, -1, 1]
    ck = []
    input = []
    for ci in c:
        flag = 0  # 0表示蓝色信号当前在下传播，1表示在上传播
        x = []
        y = []
        for i in ci:
            x.append(cc[2 * i + flag])
            y.append(cz[2 * i + flag])
            if (i == 1):
                flag = 1 - flag  # i=1蓝色路径翻转
        ck.append(np.array(x).T)
        input.append(np.array(y))
    input = np.array(input).transpose()

    # 模拟PUF生成输出
    output = []
    for num in range(get_num):
        sum = 0.0  # 矩阵对应位置和
        ci = ck[num]
        for i in range(4):
            for j in range(PUF_num):
                sum += ci[i, j] * data[i * PUF_num + j]
        r = 1 if sum > 0 else 0
        output.append(r)
    return np.float32(input), np.float32(output), PUF_num


# 逻辑回归
x_data, y_data, w_num = getCRPs_64(train_num)
x_test, y_test, w_num = getCRPs_64(test_num)

W = tf.Variable(tf.zeros([1, w_num]))  # 设置参数个数
b = tf.Variable(tf.zeros(1))
y = tf.nn.sigmoid(tf.matmul(W, x_data) + b)

# 损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.AdamOptimizer(0.1)  # GradientDescentOptimizer,AdamOptimizer,FtrlOptimizer
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 20000):
    sess.run(train)
    if step % 50 == 0:
        print("loss为：", sess.run(loss))
        pred = tf.matmul(W, x_test) + b
        correct_prediction = tf.equal(tf.sign(pred), 2 * y_test - 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("准确率为：", sess.run(accuracy))
