import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
"""

"""
def get_beans(num):
    xs = np.random.rand(num,2) * 1.9 + 0.1
    # xs = np.sort(xs)
    # 定义条件列表
    conditions = [
        (xs[:, 0]*1.5 + xs[:, 1]*0.6 < 1.8),
        (xs[:, 0]*1.5 + xs[:, 1]*0.6 >= 1.8)
    ]
    # 生成新数组
    ys = np.zeros(xs.shape[0])
    ys[conditions[0]] = 0
    ys[conditions[1]] = 1
    return xs, ys



def draw(xs,ys,func=None):
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')  # 关键：指定 projection='3d'
    # 设置中文字体和解决负号显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei', 'KaiTi']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    x = xs[:, 0]  # 获取所有点的 x 坐标
    y = xs[:, 1]  # 获取所有点的 y 坐标
    z = ys  # 获取所有点的 z 坐标
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
    
    if func is not None:
        # 使用 linspace 创建规则间隔的数据点用于 meshgrid
        xi = np.linspace(min(x), max(x), 50)
        yi = np.linspace(min(y), max(y), 50)
        X, Y = np.meshgrid(xi, yi)
        Z = func(X,Y)
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # 显示图形
    plt.show()

def Random_drop():
    # 从dataset库中获取100个数据
    m = 100
    xs, ys = get_beans(m)
    # 配置图像
    draw(xs,ys)
    # 配置参数
    w1 = np.random.rand()
    w2 = np.random.rand()
    b = np.random.rand()
    x1 = xs[:, 0]  # 获取所有点的 x 坐标
    x2 = xs[:, 1]  # 获取所有点的 y 坐标
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def forward_propgation(x1,x2):
        z = w1*x1 +  w2*x2 +b
        a = sigmoid(z)
        return a
    draw(xs,ys,forward_propgation)
    alpha = 0.01
    for _ in range(500):
        for i in range(m):
            x = xs[i]
            y = ys[i]
            x1 = x[0]
            x2 = x[1]
            a = forward_propgation(x1,x2)
            deda = -2*(y-a)
            dadz = a*(1-a)
            dzdw1 = x1
            dzdw2 = x2
            dzdb = 1
            dedw1 = deda * dadz *dzdw1
            dedw2 = deda * dadz *dzdw2
            dedb = deda * dadz *dzdb
            w1 = w1 - alpha * dedw1
            w2 = w2 - alpha * dedw2
            b = b - alpha * dedb
    draw(xs,ys,forward_propgation)
    pass
Random_drop()
