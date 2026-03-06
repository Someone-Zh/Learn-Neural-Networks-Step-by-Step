import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
"""

"""
def get_beans(num):
    xs = np.random.rand(num) * 1.9 + 0.1
    xs = np.sort(xs)
    # 定义条件列表
    conditions = [
        xs < 0.4,
        (xs >= 0.4) & (xs <= 1.2),
        xs > 1.2
    ]
    choices = [0, 1, 0]

    # 生成新数组
    ys = np.select(conditions, choices, default=xs)

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
    # 第一层
    w1_1 = np.random.rand()
    b1_1 = np.random.rand()

    w1_2 = np.random.rand()
    b1_2 = np.random.rand()
    # 第二层
    w2_1 = np.random.rand()
    w2_2 = np.random.rand()
    b2_1 = np.random.rand()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def forward_propgation(xs):
        z1_1 = w1_1 * xs + b1_1
        a1_1 = sigmoid(z1_1)
        
        z1_2 = w1_2 * xs + b1_2
        a1_2 = sigmoid(z1_2)

        z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
        a2_1 = sigmoid(z2_1)
        return a1_2,z1_2,a2_1,z2_1,a1_1,z1_1
    
    a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)
    # 绘制预测线
    plt.plot(xs, a2_1)
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
            """
                z1_1 = w1_1 * xs + b1_1
                a1_1 = sigmoid(z1_1)
                
                z1_2 = w1_2 * xs + b1_2
                a1_2 = sigmoid(z1_2)

                z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
                a2_1 = sigmoid(z2_1)
                e = (y-a1_2)^2
            """
            a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(x)

            # 计算损失函数对a2_1的导数
            deda2_1 = -2 * (y - a2_1)
            # 计算a2_1对z2_1的导数
            da2_1dz2_1 = a2_1 * (1 - a2_1)
            # 计算z2_1对w2_1和w2_2的导数
            dz2_1dw2_1 = a1_1
            dz2_1dw2_2 = a1_2
            # 计算损失函数对w2_1和w2_2的导数
            dedw2_1 = deda2_1 * da2_1dz2_1 * dz2_1dw2_1
            dedw2_2 = deda2_1 * da2_1dz2_1 * dz2_1dw2_2

            # 计算z2_1对b2_1的导数
            dz2_1db2_1 = 1
            # 计算损失函数对b1_2的导数
            dedb2_1 = deda2_1 * da2_1dz2_1 * dz2_1db2_1

            # 计算z2_1对a1_1的导数
            dz2_1da1_1 = w2_1
            # 计算a1_1对z1_1的导数
            da1_1dz1_1 = a1_1 * (1 - a1_1)
            # 计算z1_1对w1_1的导数
            dz1_1dw1_1 = x
            # 计算损失函数对w1_1的导数
            dedw1_1 = deda2_1 * da2_1dz2_1 * dz2_1da1_1 * da1_1dz1_1 * dz1_1dw1_1
            # 计算z1_1对b1_1的导数
            dz1_1db1_1 = 1
            # 计算损失函数对b1_1的导数
            dedb1_1 = deda2_1 * da2_1dz2_1 * dz2_1da1_1 * da1_1dz1_1 * dz1_1db1_1

            # 计算z2_1对a1_2的导数
            dz2_1da1_2 = w2_2
            # 计算a1_2对z1_2的导数
            da1_2dz1_2 = a1_2 * (1 - a1_2)
            # 计算z1_2对w1_2的导数
            dz1_2dw1_2 = x
            # 计算损失函数对w1_2的导数
            dedw1_2 = deda2_1 * da1_2dz1_2 * dz2_1da1_2 * da2_1dz2_1 * dz1_2dw1_2
            # 计算z1_1对b1_2的导数
            dz1_1db1_2 = 1
            # 计算损失函数对b1_2的导数
            dedb1_2 = deda2_1 * da1_2dz1_2 * dz2_1da1_1 * da1_1dz1_1 * dz1_1db1_2

            # 学习率
            alpha = 0.03

            # 更新权重
            w1_1 = w1_1 - alpha * dedw1_1
            b1_1 = b1_1 - alpha * dedb1_1

            w1_2 = w1_2 - alpha * dedw1_2
            b1_2 = b1_2 - alpha * dedb1_2

            w2_1 = w2_1 - alpha * dedw2_1
            w2_2 = w2_2 - alpha * dedw2_2
            b2_1 = b2_1 - alpha * dedb2_1
        if _ % 100 ==0:
            plt.clf()  # 清除当前图像
            plt.scatter(xs, ys)
            a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)

            plt.plot(xs, a2_1)
            plt.pause(0.001)  # 暂停一小段时间，图像就会更新

    plt.clf()  # 清除当前图像
    plt.scatter(xs, ys)
    a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)

    plt.plot(xs, a2_1)
    plt.pause(0.01) 
    # 显示图像
    plt.show()

Random_drop()