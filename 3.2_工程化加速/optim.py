# 优化器实现模块 - NumPy加速版本
import numpy as np

class Optimizer:
    """优化器基类"""
    def zero_grad(self, release_graph=True):
        """将所有参数的梯度清零，可选择是否释放计算图
        
        参数:
            release_graph: 是否释放计算图节点，默认为 True
        """
        for p in self.params:
            p.grad = np.zeros_like(p.data, dtype=np.float64)
            
            # 释放计算图节点
            if release_graph:
                p._prev = set()
                p._backward = lambda: None
                p._retain_graph = False

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
        self.m = []  # 一阶矩估计（梯度的指数加权移动平均）
        self.v = []  # 二阶矩估计（梯度平方的指数加权移动平均）
        
        for p in self.params:
            # 使用NumPy创建与参数形状相同的零数组
            m_shape = np.zeros_like(p.data, dtype=np.float64)
            v_shape = np.zeros_like(p.data, dtype=np.float64)
            
            self.m.append(m_shape)
            self.v.append(v_shape)

    def step(self):
        """执行一步梯度更新"""
        self.t += 1  # 更新步数计数器
        for idx, p in enumerate(self.params):
            if p.grad is None: continue
            
            curr_m = self.m[idx]  # 获取当前参数对应的一阶矩估计
            curr_v = self.v[idx]  # 获取当前参数对应的二阶矩估计
            
            # 更新一阶矩估计和二阶矩估计
            new_m = self.beta1 * curr_m + (1 - self.beta1) * p.grad
            new_v = self.beta2 * curr_v + (1 - self.beta2) * (p.grad ** 2)
            
            # 偏差校正
            m_hat = new_m / (1 - self.beta1 ** self.t)
            v_hat = new_v / (1 - self.beta2 ** self.t)
            
            # 使用校正后的一阶矩和二阶矩更新参数
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # 更新状态
            self.m[idx] = new_m
            self.v[idx] = new_v