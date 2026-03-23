import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# ===================== 统一的地形生成函数（参数化） =====================
def generate_terrain(terrain_type="double_peak", n_points=None, is_grid=False):
    """
    统一生成地形数据（支持随机点/网格点）
    :param terrain_type: 地形类型（double_peak/mexican_hat/saddle）
    :param n_points: 点数（随机点时为总点数，网格点时为单边点数）
    :param is_grid: True=生成网格点（可视化用），False=生成随机点（训练用）
    :return: X(2, N), z(1, N), X_mesh/Y_mesh（仅is_grid=True时返回）
    """
    if is_grid:
        # 可视化用：生成网格点
        xs = np.linspace(-3, 3, n_points)
        ys = np.linspace(-3, 3, n_points)
        Xm, Ym = np.meshgrid(xs, ys)
        x = Xm.ravel()
        y = Ym.ravel()
    else:
        # 训练用：生成随机点
        x = np.random.uniform(-3, 3, n_points)
        y = np.random.uniform(-3, 3, n_points)
        Xm, Ym = None, None

    # 统一的地形计算逻辑
    if terrain_type == "saddle":
        z = x**2 - y**2
    else:
        z = np.zeros_like(x)

    # 统一的归一化（和训练数据完全一致）
    X = np.vstack((x, y)) / 3.0  # x/y归一化到[-1,1]
    z = z.reshape(1, -1)
    z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-8)  # z归一化到[0,1]

    if is_grid:
        return X, z, Xm, Ym
    else:
        return X, z

# ===================== 最简单可用的 2 层网络 =====================
class SimpleFCN:
    def __init__(self, hidden_size=64):
        self.W1 = np.random.randn(hidden_size, 2) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(1, hidden_size) * 0.1
        self.b2 = np.zeros((1, 1))
        self.loss_history = []

    def forward(self, X):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.Z2  # 无sigmoid，保留地形起伏
        return self.A2

    def backward(self, X, y_true, lr=0.01):
        m = X.shape[1]
        y_pred = self.A2

        dZ2 = (y_pred - y_true) / m
        dW2 = dZ2 @ self.A1.T
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = dZ1 @ X.T
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train_step(self, X, y, lr=0.01):
        y_pred = self.forward(X)
        loss = mse(y_pred, y)
        self.backward(X, y, lr)
        self.loss_history.append(loss)
        return loss