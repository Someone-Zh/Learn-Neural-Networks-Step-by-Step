import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
"""
均方误差为
    e = (y - sigmoid(w*x+b))^2
拆分为
    z = w*x + b
    a = sigmoid(z)
    e = (y - a)^2
    de/da = -2*(y-a)
    da/dz = a*(1-a)
    dz/dw = x
    dz/db = 1
    de/dw = -2*(y-a) * a*(1-a) *  x
    de/db = -2*(y-a) * a*(1-a)
"""
def get_beans(num):
    xs = np.random.rand(num) * 0.8 + 0.1
    xs = np.sort(xs)
    ys  = np.where(xs > 0.4, 1, 0)
    return xs, ys
def Random_drop():

    # 从dataset库中获取100个数据
    m = 100
    xs, ys = get_beans(m)
    # 配置图像
    plt.title("S", fontsize=12) # 设置图像标题
    plt.xlabel("Bean Size") # 设置横坐标名称
    plt.ylabel("Toxicity") # 设置纵坐标名称
    # 绘制散点图
    plt.scatter(xs, ys)
    # 初始化权重
    w = 0.1
    b = 0.1 
    z = w * xs + b
    a = 1/(1+np.exp(-z))
    # 绘制预测线
    plt.plot(xs, a)
    # 显示图像
    plt.show()
    # 学习率
    alpha = 0.05
    # 开始循环
    for _ in range(2000):
        for i in range(m):
            x = xs[i]
            y = ys[i]
            # 前向传播
            z = w * x + b
            a = 1 / (1 + np.exp(-z))  # sigmoid 激活函数
            e = (y - a) ** 2  # 损失函数（均方误差）
            # 反向传播
            deda = -2 * (y - a)
            dadz = a * (1 - a)
            dzdw = x
            dzdb = 1
            dedw = deda * dadz * dzdw
            dedb = deda * dadz * dzdb
            # 参数更新
            w = w - alpha * dedw
            b = b - alpha * dedb
            # plt.clf()  # 清除当前图像
            # plt.scatter(xs, ys)
            # z = w * xs
            # a = 1/(1+np.exp(-z))
            # plt.plot(xs, a)
            # plt.pause(0.001)  # 暂停一小段时间，图像就会更新

    plt.clf()  # 清除当前图像
    plt.scatter(xs, ys)
    z = w * xs + b
    a = 1/(1+np.exp(-z))
    plt.plot(xs, a)
    plt.pause(0.01) 
    # 显示图像
    plt.show()

Random_drop()