# 优化器实现模块
import math

class Optimizer:
    """优化器基类"""
    def zero_grad(self):
        """将所有参数的梯度清零"""
        for p in self.params:
            if isinstance(p.grad, list):
                if isinstance(p.grad[0], list):
                    # 对二维列表清零梯度
                    p.grad = [[0.0 for _ in row] for row in p.grad]
                else:
                    # 对一维列表清零梯度
                    p.grad = [0.0 for _ in p.grad]
            else:
                # 对标量清零梯度
                p.grad = 0.0

class SGD(Optimizer):
    """随机梯度下降优化器"""
    def __init__(self, params, lr=0.01):
        """
        初始化SGD优化器
        :param params: 待优化的参数
        :param lr: 学习率，默认值为0.01
        """
        self.params = params
        self.lr = lr

    def step(self):
        """执行一步梯度更新"""
        for p in self.params:
            if p.grad is None: continue
            
            if isinstance(p.data, list):
                if isinstance(p.data[0], list):
                    # 更新二维参数矩阵
                    p.data = [[p.data[i][j] - self.lr * p.grad[i][j] for j in range(len(p.data[0]))] for i in range(len(p.data))]
                else:
                    # 更新一维参数向量
                    p.data = [p.data[i] - self.lr * p.grad[i] for i in range(len(p.data))]
            else:
                # 更新标量参数
                p.data = p.data - self.lr * p.grad

class Adam(Optimizer):
    """Adam优化器，自适应矩估计优化算法"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        初始化Adam优化器
        :param params: 待优化的参数
        :param lr: 学习率，默认值为0.001
        :param betas: 用于计算梯度及其平方的运行平均值的系数 (beta1, beta2)，默认值为(0.9, 0.999)
        :param eps: 防止除零错误的小常数，默认值为1e-8
        """
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # 状态存储：为每个参数维护动量和二阶矩估计
        # 由于列表不可哈希，我们使用与self.params对齐的并行列表存储状态
        self.m = []  # 一阶矩估计（梯度的指数加权移动平均）
        self.v = []  # 二阶矩估计（梯度平方的指数加权移动平均）
        
        for p in self.params:
            if isinstance(p.data, list):
                if isinstance(p.data[0], list):
                    # 为二维参数创建对应形状的一阶矩和二阶矩估计
                    m_shape = [[0.0 for _ in row] for row in p.data]
                    v_shape = [[0.0 for _ in row] for row in p.data]
                else:
                    # 为一维参数创建对应形状的一阶矩和二阶矩估计
                    m_shape = [0.0 for _ in p.data]
                    v_shape = [0.0 for _ in p.data]
            else:
                # 为标量参数创建对应形状的一阶矩和二阶矩估计
                m_shape = 0.0
                v_shape = 0.0
            
            self.m.append(m_shape)
            self.v.append(v_shape)

    def _update_val(self, val, grad, m_val, v_val):
        """
        更新单个数值参数
        :param val: 当前参数值
        :param grad: 参数对应的梯度
        :param m_val: 当前一阶矩估计
        :param v_val: 当前二阶矩估计
        :return: 更新后的参数值、一阶矩估计、二阶矩估计
        """
        # 更新一阶矩估计和二阶矩估计
        new_m = self.beta1 * m_val + (1 - self.beta1) * grad
        new_v = self.beta2 * v_val + (1 - self.beta2) * (grad ** 2)
        
        # 偏差校正
        m_hat = new_m / (1 - self.beta1 ** self.t)
        v_hat = new_v / (1 - self.beta2 ** self.t)
        
        # 使用校正后的一阶矩和二阶矩更新参数
        new_val = val - self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        return new_val, new_m, new_v

    def _update_list_1d(self, data, grad, m_list, v_list):
        """
        更新一维列表参数
        :param data: 当前参数值列表
        :param grad: 参数对应的梯度列表
        :param m_list: 当前一阶矩估计列表
        :param v_list: 当前二阶矩估计列表
        :return: 更新后的参数值列表、一阶矩估计列表、二阶矩估计列表
        """
        new_data = []
        new_m = []
        new_v = []
        for i in range(len(data)):
            # 获取当前索引位置的参数值、梯度、一阶矩和二阶矩
            vd, gd, md, vd_old = data[i], grad[i], m_list[i], v_list[i]
            # 调用更新单个值的方法
            nd, nm, nv = self._update_val(vd, gd, md, vd_old)
            new_data.append(nd)
            new_m.append(nm)
            new_v.append(nv)
        return new_data, new_m, new_v

    def _update_list_2d(self, data, grad, m_mat, v_mat):
        """
        更新二维列表参数（矩阵）
        :param data: 当前参数值矩阵
        :param grad: 参数对应的梯度矩阵
        :param m_mat: 当前一阶矩估计矩阵
        :param v_mat: 当前二阶矩估计矩阵
        :return: 更新后的参数值矩阵、一阶矩估计矩阵、二阶矩估计矩阵
        """
        new_data = []
        new_m = []
        new_v = []
        for i in range(len(data)):
            # 获取当前行的参数值、梯度、一阶矩和二阶矩
            row_d, row_g, row_m, row_v = data[i], grad[i], m_mat[i], v_mat[i]
            # 将二维问题降为一维问题处理
            nd_row, nm_row, nv_row = self._update_list_1d(row_d, row_g, row_m, row_v)
            new_data.append(nd_row)
            new_m.append(nm_row)
            new_v.append(nv_row)
        return new_data, new_m, new_v

    def step(self):
        """执行一步梯度更新"""
        self.t += 1  # 更新步数计数器
        for idx, p in enumerate(self.params):
            if p.grad is None: continue
            
            curr_m = self.m[idx]  # 获取当前参数对应的一阶矩估计
            curr_v = self.v[idx]  # 获取当前参数对应的二阶矩估计
            
            if isinstance(p.data, (int, float)):
                # 处理标量参数
                new_d, new_m, new_v = self._update_val(p.data, p.grad, curr_m, curr_v)
                p.data = new_d
                self.m[idx] = new_m
                self.v[idx] = new_v
                
            elif isinstance(p.data, list):
                if isinstance(p.data[0], list):
                    # 处理二维参数矩阵
                    new_d, new_m, new_v = self._update_list_2d(p.data, p.grad, curr_m, curr_v)
                    p.data = new_d
                    self.m[idx] = new_m
                    self.v[idx] = new_v
                else:
                    # 处理一维参数向量
                    new_d, new_m, new_v = self._update_list_1d(p.data, p.grad, curr_m, curr_v)
                    p.data = new_d
                    self.m[idx] = new_m
                    self.v[idx] = new_v