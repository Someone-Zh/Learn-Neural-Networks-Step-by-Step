import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from keras.metrics import Accuracy
"""

"""
def get_beans(num):
    xs = np.random.rand(num,2) * 1.9 + 0.1
    # xs = np.sort(xs)
    # 定义条件列表
    func = lambda x1,x2 : -4*x1**2+8*x1-3<x2
    funcf = lambda x1,x2 : -4*x1**2+8*x1-3>=x2
    conditions = [
        func(xs[:, 0],xs[:, 1]),
        funcf(xs[:, 0],xs[:, 1])
    ]
    # 生成新数组
    ys = np.zeros(xs.shape[0])
    ys[conditions[0]] = 0
    ys[conditions[1]] = 1
    return xs, ys


def draw(X,Y,pres=None):
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')  # 关键：指定 projection='3d'
    # 设置中文字体和解决负号显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'KaiTi']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    x = X[:, 0]  # 获取所有点的 x 坐标
    y = X[:, 1]  # 获取所有点的 y 坐标
    z = Y  # 获取所有点的 z 坐标
    # 设置标签和标题
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('三维散点图')
    plt.cla()  # 清除当前图像
    # 绘制三维散点图
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)
    # 添加颜色条
    plt.colorbar(sc, label='Z Value')
    if pres is not None:
        # 使用 linspace 创建规则间隔的数据点用于 meshgrid
        xi = np.linspace(min(x), max(x), 50)
        yi = np.linspace(min(y), max(y), 50)
        X, Y = np.meshgrid(xi, yi)
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # 显示图形
    plt.show()
m=100
def train():
    X,Y = get_beans(m)
    draw(X,Y)
    model = Sequential()
    model.add(Dense(units=1,activation='sigmoid', input_dim=2))
    model.compile(loss=mean_squared_error,optimizer=SGD(),metrics=[Accuracy()])
    model.fit(X,Y,epochs=5000,batch_size=10)
    pres = model.predict(X)
    draw(X,Y,pres)
train()