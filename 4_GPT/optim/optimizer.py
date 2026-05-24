# 优化器实现模块 - 支持 NumPy 和 PyTorch GPU 加速
import numpy as np
import torch

# 全局配置：是否使用 PyTorch GPU 加速
USE_PYTORCH_BACKEND = True  # 设置为 True 启用 PyTorch 后端
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Optimizer:
    """优化器基类"""
    def zero_grad(self, release_graph=True):
        """将所有参数的梯度清零，可选择是否释放计算图
        
        参数:
            release_graph: 是否释放计算图节点，默认为 True
        """
        for p in self.params:
            # 支持 PyTorch Tensor
            if hasattr(p, 'torch_tensor') and p.torch_tensor is not None:
                if p.torch_tensor.grad is not None:
                    p.torch_tensor.grad.zero_()
                p.grad = None
            else:
                p.grad = np.zeros_like(p.data, dtype=np.float64)
            
            # 释放计算图节点
            if release_graph and hasattr(p, '_prev'):
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
            
            # 支持 PyTorch Tensor
            if hasattr(p, 'torch_tensor') and p.torch_tensor is not None and USE_PYTORCH_BACKEND:
                # 确保梯度在 PyTorch tensor 上
                if p.torch_tensor.grad is None:
                    # 将 NumPy 梯度转换为 PyTorch
                    p.torch_tensor.grad = torch.from_numpy(p.grad.astype(np.float32)).to(p.torch_tensor.device)
                
                # 使用 PyTorch 进行参数更新
                with torch.no_grad():
                    p.torch_tensor -= self.lr * p.torch_tensor.grad
                # 同步回 NumPy
                p.data = p.torch_tensor.detach().cpu().numpy()
            else:
                # 原有的 NumPy 更新
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
            # 检查是否使用 PyTorch 后端
            if hasattr(p, 'torch_tensor') and p.torch_tensor is not None and USE_PYTORCH_BACKEND:
                # 使用 PyTorch Tensor 存储状态（在 GPU 上）
                device = p.torch_tensor.device
                m_tensor = torch.zeros_like(p.torch_tensor, device=device)
                v_tensor = torch.zeros_like(p.torch_tensor, device=device)
                self.m.append(m_tensor)
                self.v.append(v_tensor)
            else:
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
            
            # 支持 PyTorch Tensor
            if hasattr(p, 'torch_tensor') and p.torch_tensor is not None and USE_PYTORCH_BACKEND:
                # 使用 PyTorch 进行 Adam 更新（全部在 GPU 上）
                grad = p.torch_tensor.grad
                
                # 如果梯度为None，跳过
                if grad is None:
                    continue
                
                with torch.no_grad():
                    # 确保状态变量在同一设备上
                    if isinstance(curr_m, np.ndarray):
                        # 如果状态是 NumPy，转换为 PyTorch
                        curr_m = torch.from_numpy(curr_m).to(p.torch_tensor.device)
                        curr_v = torch.from_numpy(curr_v).to(p.torch_tensor.device)
                        self.m[idx] = curr_m
                        self.v[idx] = curr_v
                    
                    # Adam 更新公式（完全在 GPU 上执行）
                    new_m = self.beta1 * curr_m + (1 - self.beta1) * grad
                    new_v = self.beta2 * curr_v + (1 - self.beta2) * (grad ** 2)
                    
                    m_hat = new_m / (1 - self.beta1 ** self.t)
                    v_hat = new_v / (1 - self.beta2 ** self.t)
                    
                    # 参数更新
                    p.torch_tensor -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                
                # 同步回 NumPy（如果需要）
                p.data = p.torch_tensor.detach().cpu().numpy()
                p.grad = grad.cpu().numpy() if grad is not None else None
                
                # 更新状态（保持为 PyTorch Tensor 在 GPU 上）
                self.m[idx] = new_m
                self.v[idx] = new_v
            else:
                # 原有的 NumPy 实现
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

class LearningRateScheduler:
    """学习率调度器基类"""
    def __init__(self, optimizer, initial_lr):
        """
        初始化学习率调度器
        
        参数:
            optimizer: 优化器
            initial_lr: 初始学习率
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr

class StepLR(LearningRateScheduler):
    """步长学习率调度器"""
    def __init__(self, optimizer, initial_lr, step_size, gamma=0.1):
        """
        初始化步长学习率调度器
        
        参数:
            optimizer: 优化器
            initial_lr: 初始学习率
            step_size: 学习率衰减步长
            gamma: 学习率衰减因子
        """
        super().__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma
        self.current_step = 0
    
    def step(self):
        """执行一步学习率调度"""
        self.current_step += 1
        if self.current_step % self.step_size == 0:
            new_lr = self.initial_lr * (self.gamma ** (self.current_step // self.step_size))
            self.optimizer.lr = new_lr
            print(f"Learning rate updated to: {new_lr}")

class ExponentialLR(LearningRateScheduler):
    """指数学习率调度器"""
    def __init__(self, optimizer, initial_lr, gamma=0.99):
        """
        初始化指数学习率调度器
        
        参数:
            optimizer: 优化器
            initial_lr: 初始学习率
            gamma: 学习率衰减因子
        """
        super().__init__(optimizer, initial_lr)
        self.gamma = gamma
        self.current_step = 0
    
    def step(self):
        """执行一步学习率调度"""
        self.current_step += 1
        new_lr = self.initial_lr * (self.gamma ** self.current_step)
        self.optimizer.lr = new_lr
        if self.current_step % 100 == 0:
            print(f"Learning rate updated to: {new_lr}")


# ========================================
# PyTorch GPU 加速配置函数
# ========================================

def enable_pytorch_backend(device=None):
    """
    启用 PyTorch GPU 加速后端
    
    参数:
        device: 设备名称，'cuda' 或 'cpu'，默认为自动检测
    """
    global USE_PYTORCH_BACKEND, DEVICE
    USE_PYTORCH_BACKEND = True
    if device is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = device
    print(f"Optimizer PyTorch backend enabled. Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


def disable_pytorch_backend():
    """禁用 PyTorch 后端，使用 NumPy"""
    global USE_PYTORCH_BACKEND
    USE_PYTORCH_BACKEND = False
    print("Optimizer PyTorch backend disabled. Using NumPy.")


def get_backend_info():
    """获取当前后端信息"""
    info = {
        'backend': 'PyTorch' if USE_PYTORCH_BACKEND else 'NumPy',
        'device': DEVICE if USE_PYTORCH_BACKEND else 'CPU',
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return info