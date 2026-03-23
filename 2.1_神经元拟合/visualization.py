import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# -------------------------- 1. 解决中文显示问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 原生 Python 实现基础函数
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_beans(num):
    # 生成 0.1 到 2.0 之间的随机数并排序
    xs = sorted([random.uniform(0.1, 2.0) for _ in range(num)])
    ys = []
    for x in xs:
        # 定义条件：0.4 到 1.2 之间有效(1)，否则无效(0)
        if 0.4 <= x <= 1.2:
            ys.append(1)
        else:
            ys.append(0)
    return xs, ys

# 2. 初始化参数 (1-2-1 神经网络)
m = 50  # 样本数
xs, ys = get_beans(m)


history_preds = [] # 存储每一轮的预测结果用于动画

def train_sgd():
     # 第一层参数
    w1_1, b1_1 = random.uniform(-1, 1), random.uniform(-1, 1)
    w1_2, b1_2 = random.uniform(-1, 1), random.uniform(-1, 1)
    # 第二层参数
    w2_1, w2_2, b2_1 = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
    alpha = 0.1  # 学习率
    epochs = 6000
    for epoch in range(epochs):
        current_preds = []
        """
        z1_1 = w1_1 * xs + b1_1
        a1_1 = sigmoid(z1_1)

        z1_2 = w1_2 * xs + b1_2
        a1_2 = sigmoid(z1_2)

        z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
        a2_1 = sigmoid(z2_1)
        e = (y-a1_2)^2
    """
        loss = 0
        for i in range(m):
            x = xs[i]
            y = ys[i]

            # --- 前向传播 ---
            z1_1 = w1_1 * x + b1_1
            a1_1 = sigmoid(z1_1)
            z1_2 = w1_2 * x + b1_2
            a1_2 = sigmoid(z1_2)

            z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
            a2_1 = sigmoid(z2_1)
            
            current_preds.append(a2_1)
            # --- 反向传播 (链式法则手动求导) ---
            # 损失函数 L = (a2_1 - y)^2
            deda2 = 2 * (a2_1 - y)
            da2dz2 = a2_1 * (1 - a2_1)
            
            # 计算第二层梯度（临时变量）
            grad_w2_1 = deda2 * da2dz2 * a1_1
            grad_w2_2 = deda2 * da2dz2 * a1_2
            grad_b2_1 = deda2 * da2dz2 * 1
            
            # 计算第一层梯度（临时变量）
            dL_da1_1 = deda2 * da2dz2 * w2_1
            da1_1_dz1_1 = a1_1 * (1 - a1_1)
            grad_w1_1 = dL_da1_1 * da1_1_dz1_1 * x
            grad_b1_1 = dL_da1_1 * da1_1_dz1_1 * 1
            
            dL_da1_2 = deda2 * da2dz2 * w2_2
            da1_2_dz1_2 = a1_2 * (1 - a1_2)
            grad_w1_2 = dL_da1_2 * da1_2_dz1_2 * x
            grad_b1_2 = dL_da1_2 * da1_2_dz1_2 * 1
            
            # --- 立即用当前样本的梯度更新权重---
            w1_1 -= alpha * grad_w1_1
            b1_1 -= alpha * grad_b1_1
            w1_2 -= alpha * grad_w1_2
            b1_2 -= alpha * grad_b1_2
            w2_1 -= alpha * grad_w2_1
            w2_2 -= alpha * grad_w2_2
            b2_1 -= alpha * grad_b2_1
            loss += (a2_1 - y) ** 2
        # 每 20 代记录一次预测线
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss/m}",end="\r")
            history_preds.append(current_preds)
            
# 3. 迭代梯度下降
def train():
    # 第一层参数
    w1_1, b1_1 = random.uniform(-1, 1), random.uniform(-1, 1)
    w1_2, b1_2 = random.uniform(-1, 1), random.uniform(-1, 1)
    # 第二层参数
    w2_1, w2_2, b2_1 = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)

    alpha = 0.8  # 学习率
    epochs = 20000
    for epoch in range(epochs):
        current_preds = []
        """
        z1_1 = w1_1 * xs + b1_1
        a1_1 = sigmoid(z1_1)

        z1_2 = w1_2 * xs + b1_2
        a1_2 = sigmoid(z1_2)

        z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
        a2_1 = sigmoid(z2_1)
        e = (y-a1_2)^2
    """
        # 模拟批量梯度下降中的累加梯度
        dw1_1, db1_1 = 0, 0
        dw1_2, db1_2 = 0, 0
        dw2_1, dw2_2, db2_1 = 0, 0, 0
        loss = 0
        for i in range(m):
            x = xs[i]
            y = ys[i]

            # --- 前向传播 ---
            z1_1 = w1_1 * x + b1_1
            a1_1 = sigmoid(z1_1)
            z1_2 = w1_2 * x + b1_2
            a1_2 = sigmoid(z1_2)

            z2_1 = w2_1 * a1_1 + w2_2 * a1_2 + b2_1
            a2_1 = sigmoid(z2_1)
            
            current_preds.append(a2_1)
            loss += (a2_1 - y) ** 2 
            # --- 反向传播 (链式法则手动求导) ---
            # 损失函数 L = (a2_1 - y)^2
            deda2 = 2 * (a2_1 - y)
            da2dz2 = a2_1 * (1 - a2_1)
            
            # 第二层梯度
            dw2_1 += deda2 * da2dz2 * a1_1
            dw2_2 += deda2 * da2dz2 * a1_2
            db2_1 += deda2 * da2dz2 * 1
            
            # 第一层梯度 (经过神经元 1_1)
            # 梯度传递：L -> a2 -> z2 -> a1_1 -> z1_1 -> w1_1/b1_1
            dL_da1_1 = deda2 * da2dz2 * w2_1  # ∂L/∂a1_1
            da1_1_dz1_1 = a1_1 * (1 - a1_1)   # ∂a1_1/∂z1_1
            dz1_1_dw1_1 = x                   # ∂z1_1/∂w1_1
            dz1_1_db1_1 = 1                   # ∂z1_1/∂b1_1

            dw1_1 += dL_da1_1 * da1_1_dz1_1 * dz1_1_dw1_1
            db1_1 += dL_da1_1 * da1_1_dz1_1 * dz1_1_db1_1

            # 第一层梯度 (神经元 1_2)
            dL_da1_2 = deda2 * da2dz2 * w2_2  # ∂L/∂a1_2
            da1_2_dz1_2 = a1_2 * (1 - a1_2)   # ∂a1_2/∂z1_2
            dz1_2_dw1_2 = x                   # ∂z1_2/∂w1_2
            dz1_2_db1_2 = 1                   # ∂z1_2/∂b1_2

            dw1_2 += dL_da1_2 * da1_2_dz1_2 * dz1_2_dw1_2
            db1_2 += dL_da1_2 * da1_2_dz1_2 * dz1_2_db1_2
            
            
        loss = loss/m
        # 更新参数 (取平均梯度)
        w1_1 -= alpha * dw1_1 / m
        b1_1 -= alpha * db1_1 / m
        w1_2 -= alpha * dw1_2 / m
        b1_2 -= alpha * db1_2 / m
        w2_1 -= alpha * dw2_1 / m
        w2_2 -= alpha * dw2_2 / m
        b2_1 -= alpha * db2_1 / m

        # 每 20 代记录一次预测线
        if epoch % 20 == 0:
            print(f"loss:{loss}",end="\r")
            history_preds.append(current_preds)
train()
# 4. 创建并显示动画
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(xs, ys, color='blue', alpha=0.3, label='Data (Beans)')
line, = ax.plot(xs, history_preds[0], color='red', lw=2, label='Prediction')
text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

def update(frame):
    line.set_ydata(history_preds[frame])
    text.set_text(f"Training Epoch: {frame * 20}")
    return line, text

ani = animation.FuncAnimation(
    fig, update, frames=len(history_preds), interval=50, blit=True
)

ax.set_title("炼丹")
ax.set_xlabel("弹药大小")
ax.set_ylabel("是否有效")
ax.set_ylim(-0.1, 1.1)
ax.legend()


ani.save("2.1_神经元拟合/manual_bean_bgd.gif", writer='pillow')
print("GIF 已保存为 manual_bean.gif")


plt.show()
