# 超参数配置管理

class Config:
    """超参数配置类"""
    def __init__(self, **kwargs):
        """
        初始化配置
        
        参数:
            **kwargs: 超参数键值对
        """
        self.vocab_size = kwargs.get('vocab_size', 10000)
        self.d_model = kwargs.get('d_model', 128)
        self.num_heads = kwargs.get('num_heads', 4)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.num_layers = kwargs.get('num_layers', 2)
        self.dropout = kwargs.get('dropout', 0.1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.seq_len = kwargs.get('seq_len', 10)
        self.epochs = kwargs.get('epochs', 10)
        self.lr = kwargs.get('lr', 0.001)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.eps = kwargs.get('eps', 1e-8)
        self.step_size = kwargs.get('step_size', 1000)
        self.gamma = kwargs.get('gamma', 0.1)
    
    def to_dict(self):
        """
        将配置转换为字典
        
        返回:
            配置字典
        """
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'epochs': self.epochs,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'step_size': self.step_size,
            'gamma': self.gamma
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        从字典创建配置
        
        参数:
            config_dict: 配置字典
            
        返回:
            配置实例
        """
        return cls(**config_dict)

# 默认配置
default_config = Config()