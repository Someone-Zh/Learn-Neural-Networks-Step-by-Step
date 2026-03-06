import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def get_beans(num):
    xs = np.random.rand(num) * 0.8 + 0.1
    ys = xs * 1.1  + np.random.normal(0, 0.1, num) + 0.6
    return xs, ys
"""
    y = w*x
    误差为  e = (y_r - w*x)^2
            e = y_r^2 + w^2*x^2 - 2*w*x*y_r
    w导数为  2*w*x^2 - 2*x*y_r
    y = w*x+b
    误差为  e = (y_r - (w*x+b))^2
            e = y_r^2 + (w*x)^2 + b^2 + 2*b*w*x - 2*y_r*w*x -2*y_r*b
    w 偏导数为 2*w*x^2 + 2*b*x - 2*x_r*x 
    b 偏导数为 2*b + 2*b*x - 2y_r
"""
def Random_drop():
    plt.ion()  # 开启交互模式
    # 从dataset库中获取100个数据
    xs, ys = get_beans(100)
    # 配置图像
    plt.title("Size-Toxicity Function", fontsize=12) # 设置图像标题
    plt.xlabel("Bean Size") # 设置横坐标名称
    plt.ylabel("Toxicity") # 设置纵坐标名称
    # 绘制散点图
    plt.scatter(xs, ys)
    # 初始化权重
    w = 0.1
    b = 0.1 
    # 计算预测值y_pre
    y_pre = w * xs
    # 绘制预测线
    plt.plot(xs, y_pre)
    # 显示图像
    plt.show()
    # 开始一个for循环，但代码不完整
    for _ in range(100):
        for i in range(100):
            x = xs[i]
            y = ys[i]
            dw = 2 *x**2*w + 2*x*b -2*x*y
            db = 2*b + 2*x*w -2*y
            alpha = 0.1
            w = w - alpha * dw
            b = b - alpha * db
            plt.clf()  # 清除当前图像
            plt.scatter(xs, ys)
            y_pre = w * xs + b
            plt.plot(xs, y_pre)
            plt.pause(0.01)  # 暂停一小段时间，图像就会更新
    plt.clf()  # 清除当前图像
    plt.scatter(xs, ys)
    y_pre = w * xs + b
    plt.plot(xs, y_pre)
    plt.pause(0.01)  # 暂停一小段时间，图像就会更新
    plt.ioff()  # 关闭交互模
    # 显示图像
    plt.show()

def Full_decline():
    # plt.ion()  # 开启交互模式
    # 从dataset库中获取100个数据
    xs, ys = get_beans(100)
    # 配置图像
    plt.title("Size-Toxicity Function", fontsize=12) # 设置图像标题
    plt.xlabel("Bean Size") # 设置横坐标名称
    plt.ylabel("Toxicity") # 设置纵坐标名称
    # 绘制散点图
    plt.scatter(xs, ys)
    # 初始化权重w为0.1
    w = 0.1
    b = 0.1 
    # 计算预测值y_pre
    y_pre = w * xs + b
    # 绘制预测线
    plt.xlim(0,1)
    plt.ylim(0,1.5)
    plt.plot(xs, y_pre)
    for i in range(100):
        #  dw = ((2 *x_1**2*w + 2*x_1*b -2*x_1 *y) + (2 *x_2**2*w + 2*x_2*b -2*x_2 *y)) / 2 
        dw = 2 * np.sum(xs**2)*w + np.sum(2*xs*b) + np.sum(-2 * xs * ys)
        db = np.sum(2 * xs * w + 2*b) - np.sum(2*ys)
        dw_avg = dw/100
        db_avg = db/100
        alpha = 0.1
        w = w - alpha * dw_avg
        b = b - alpha * db_avg
        plt.clf()  # 清除当前图像
        plt.scatter(xs, ys)
        y_pre = w * xs + b
        plt.plot(xs, y_pre)
        plt.pause(0.05)  # 暂停一小段时间，图像就会更新
    # plt.ioff()  # 关闭交互模
    # 显示图像
    plt.clf()  # 清除当前图像
    plt.scatter(xs, ys)
    y_pre = w * xs + b
    plt.plot(xs, y_pre)
    plt.pause(0.01)  # 暂停一小段时间，图像就会更新
    plt.show()
# Full_decline()
# Random_drop()
