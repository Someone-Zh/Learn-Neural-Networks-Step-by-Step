import numpy as np

class TensorBackward:
    def backward(self, retain_graph=False):
        """反向传播，可选择是否保留计算图
        
        参数:
            retain_graph: 是否保留计算图用于多次反向传播，默认为 False
        """
        # 拓扑排序列表，用于按正确顺序执行反向传播
        topo = []
        # 已访问节点集合，避免重复访问
        visited = set()
        
        # 构建拓扑排序
        def build(v):
            if v not in visited:
                visited.add(v)
                # 递归访问前驱节点
                for child in v._prev:
                    build(child)
                # 添加当前节点到拓扑排序列表
                topo.append(v)
        build(self)
        
        # 初始化最终梯度为1.0
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # 按逆序执行反向传播
        for node in reversed(topo):
            node._backward()
            # 梯度检查
            try:
                node._validate_grad_shape()
            except ValueError as e:
                raise ValueError(
                    f"反向传播时梯度形状验证失败 (节点: {node._op}, 标签: {node.label}):\n{e}"
                )
        
        # 释放计算图（除非明确要求保留）
        if not retain_graph:
            for node in topo:
                node._prev = set()
                node._backward = lambda: None
                node._retain_graph = False