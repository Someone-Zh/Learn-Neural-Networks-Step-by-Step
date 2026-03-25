import gzip
import struct
import json
import time
from tensor import Tensor
from model import Layer, SimpleFCN
from optim import SGD, Adam
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 读取文件头：魔数(4字节)、图像数(4字节)、行数(4字节)、列数(4字节)
        # >IIII 表示大端模式(Big-endian)的4个无符号整数
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # 读取剩下的所有像素数据
        raw_data = f.read()
        
        # 将字节流转换为 0-1 之间的浮点数列表
        # 每个像素是一个字节 (0-255)
        # 结果是一个嵌套列表：[[784个像素], [784个像素], ...]
        images = []
        pixels_per_image = rows * cols # 784
        
        for i in range(num_images):
            # 提取单张图片的字节段
            start = i * pixels_per_image
            end = start + pixels_per_image
            image_bytes = raw_data[start:end]
            image_floats = [pixel for pixel in image_bytes]
            images.append(image_floats)
        return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 读取文件头：魔数(4字节)、标签数(4字节)
        magic, num_items = struct.unpack(">II", f.read(8))
        
        # 读取剩下的所有标签字节
        label_bytes = f.read()
        
        # 转换为整数列表
        return list(label_bytes)


class MnistModel():
    def __init__(self):
        self.layer1 = Layer(28 * 28, 128)
        self.layer2 = Layer(128, 128)
        self.layer3 = Layer(128, 10)  # 输出10个类别（数字0-9）
        
        # 收集所有可学习参数
        self.params = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            self.params.extend(layer.params)
            
        # 用于记录训练过程中的损失值
        self.loss_history = []
        
    def forward(self, x_tensor, apply_softmax=False):
        """前向传播计算
        
        参数:
            x_tensor: 输入张量，形状为 (Batch, 784)
            apply_softmax: 是否在输出层应用 softmax（默认 False，保持向后兼容）
            
        返回:
            输出张量，形状为 (Batch, 10)
        """
        # 第一层前向传播
        h1 = self.layer1(x_tensor)
        
        # 应用ReLU激活函数
        h1 = h1.relu()
        
        # 第二层前向传播
        h2 = self.layer2(h1)
        # 应用ReLU激活函数
        h2 = h2.relu()
        
        # 输出层前向传播
        out = self.layer3(h2)
        
        # 应用 softmax 激活函数
        if apply_softmax:
            out = out.softmax()
        
        return out

    def train_step(self, X_data, z_data, lr, optim_class):
        """单步训练
        
        参数:
            X_data: 输入数据，形状为 (Batch, 784)，每张图片展平为784个像素
            z_data: 标签数据，形状为 (Batch,)，值为0-9的整数
            lr: 学习率
            optim_class: 优化器类
            
        返回:
            当前步的损失值
        """
        # 将数据转换为Tensor
        X = Tensor(X_data)  # 输入数据，形状为 (Batch, 784)
        # 将标签转换为one-hot编码，形状为 (Batch, 10)
        z_true_data = []
        for label in z_data:
            one_hot = [0.0] * 10
            one_hot[int(label)] = 1.0
            z_true_data.append(one_hot)
        z_true = Tensor(z_true_data)
        
        # 前向传播（输出层 logits，不应用 softmax）
        z_logits = self.forward(X)
        
        # 计算损失：使用交叉熵损失（内部自动应用 softmax）
        loss = z_logits.cross_entropy_loss(z_true)
        
        # 清零梯度
        for p in self.params:
            if isinstance(p.grad, list):
                if isinstance(p.grad[0], list):
                    # 清零二维梯度
                    p.grad = [[0.0 for _ in row] for row in p.grad]
                else:
                    # 清零一维梯度
                    p.grad = [0.0 for _ in p.grad]
            else:
                # 清零标量梯度
                p.grad = 0.0
                
        # 反向传播
        loss.backward()
        
        # 优化器更新参数
        # 注意：这里为了演示方便，每次训练步都创建一个新的优化器实例
        # 在实际代码中，应该保持优化器实例
        opt = optim_class(self.params, lr=lr)
        opt.step()
        
        # 记录损失值
        self.loss_history.append(loss.data)
        return loss.data
    
    def predict(self, x_data):
        """预测
        
        参数:
            x_data: 输入数据，形状为 (Batch, 784)
            
        返回:
            预测的类别列表
        """
        X = Tensor(x_data)
        out = self.forward(X)
        # 取每行最大值的索引作为预测类别
        predictions = []
        for row in out.data:
            max_idx = row.index(max(row))
            predictions.append(max_idx)
        return predictions
    
    def get_params(self):
        """获取模型参数（用于多进程数据并行）
        
        返回:
            包含所有层权重和偏置的字典
        """
        return {
            'layer1': {
                'W': self.layer1.W.data,
                'b': self.layer1.b.data
            },
            'layer2': {
                'W': self.layer2.W.data,
                'b': self.layer2.b.data
            },
            'layer3': {
                'W': self.layer3.W.data,
                'b': self.layer3.b.data
            }
        }
    
    def load_params(self, params):
        """加载模型参数（用于多进程数据并行）
        
        参数:
            params: 包含权重和偏置的字典
        """
        self.layer1.W.data = params['layer1']['W']
        self.layer1.b.data = params['layer1']['b']
        self.layer2.W.data = params['layer2']['W']
        self.layer2.b.data = params['layer2']['b']
        self.layer3.W.data = params['layer3']['W']
        self.layer3.b.data = params['layer3']['b']
    
    def get_grads(self):
        """获取模型梯度（用于多进程数据并行）
        
        返回:
            包含所有层权重和偏置梯度的字典
        """
        return {
            'layer1': {
                'W': [row[:] for row in self.layer1.W.grad],
                'b': self.layer1.b.grad[:]
            },
            'layer2': {
                'W': [row[:] for row in self.layer2.W.grad],
                'b': self.layer2.b.grad[:]
            },
            'layer3': {
                'W': [row[:] for row in self.layer3.W.grad],
                'b': self.layer3.b.grad[:]
            }
        }
    
    def save(self, filepath):
        """保存模型参数到文件
        
        参数:
            filepath: 保存路径
        """
        # 收集所有层的权重和偏置
        params = {
            'layer1': {
                'W': self.layer1.W.data,
                'b': self.layer1.b.data
            },
            'layer2': {
                'W': self.layer2.W.data,
                'b': self.layer2.b.data
            },
            'layer3': {
                'W': self.layer3.W.data,
                'b': self.layer3.b.data
            },
            'loss_history': self.loss_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f)
        print(f"模型参数已保存到: {filepath}")
    
    def load(self, filepath):
        """从文件加载模型参数
        
        参数:
            filepath: 参数文件路径
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        # 加载各层的权重和偏置
        self.layer1.W.data = params['layer1']['W']
        self.layer1.b.data = params['layer1']['b']
        self.layer2.W.data = params['layer2']['W']
        self.layer2.b.data = params['layer2']['b']
        self.layer3.W.data = params['layer3']['W']
        self.layer3.b.data = params['layer3']['b']
        self.loss_history = params.get('loss_history', [])
        
        print(f"模型参数已从 {filepath} 加载")


def train():
    """训练MNIST模型"""
    print("加载MNIST数据集...")
    x_train = load_mnist_images('data/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
    
    print(f"训练集大小: {len(x_train)}")
    print(f"测试集大小: {len(x_test)}")
    
    # 归一化数据到 0-1 范围
    x_train = [[pixel / 255.0 for pixel in img] for img in x_train]
    x_test = [[pixel / 255.0 for pixel in img] for img in x_test]
    
    # 创建模型
    model = MnistModel()
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.01
    epochs = 5
    
    # 创建优化器
    optimizer = Adam
    
    print("开始训练...")
    total_start_time = time.time()
    num_batches_per_epoch = (len(x_train) + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # 打乱训练数据
        indices = list(range(len(x_train)))
        import random
        random.shuffle(indices)
        
        total_loss = 0
        num_batches = 0
        
        # 按批次训练，使用 tqdm 显示进度
        pbar = tqdm(range(0, len(x_train), batch_size), 
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    total=num_batches_per_epoch,
                    unit="batch")
        
        for i in pbar:
            batch_indices = indices[i:i + batch_size]
            x_batch = [x_train[idx] for idx in batch_indices]
            y_batch = [y_train[idx] for idx in batch_indices]
            
            # 执行一步训练
            loss = model.train_step(x_batch, y_batch, learning_rate, optimizer)
            total_loss += loss
            num_batches += 1
            
            # 更新进度条显示当前损失
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        
        # 在测试集上评估
        correct = 0
        test_batch_size = 100
        for i in range(0, len(x_test), test_batch_size):
            x_batch = x_test[i:i + test_batch_size]
            y_batch = y_test[i:i + test_batch_size]
            predictions = model.predict(x_batch)
            correct += sum(1 for p, y in zip(predictions, y_batch) if p == y)
        
        accuracy = correct / len(x_test) * 100
        elapsed_time = time.time() - total_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (epochs - epoch - 1)
        print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.1f}s - "
              f"Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, "
              f"ETA: {remaining_time/60:.1f}min")
    
    total_time = time.time() - total_start_time
    print(f"训练完成! 总耗时: {total_time/60:.1f}分钟")
    
    # 保存模型参数
    model.save('mnist_model.json')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(model.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.show()
    
    return model


def _train_batch_worker(args):
    """并行训练工作函数（用于多进程）
    
    修正思路：
    1. 子进程接收模型参数，创建本地模型副本
    2. 子进程独立执行前向传播和反向传播计算梯度
    3. 返回梯度数据（而非计算图）给主进程
    
    参数:
        args: 元组 (params, x_batch, y_batch, lr, optim_class_name)
    
    返回:
        (grads, loss) 梯度和损失值
    """
    params, x_batch, y_batch, lr, optim_class_name = args
    
    # 在子进程中创建模型
    model = MnistModel()
    model.load_params(params)
    
    # 根据名称获取优化器类（仅用于确定优化器类型，子进程不执行优化步骤）
    if optim_class_name == 'Adam':
        from optim import Adam as optim_class
    elif optim_class_name == 'SGD':
        from optim import SGD as optim_class
    else:
        raise ValueError(f"未知优化器: {optim_class_name}")
    
    # === 子进程执行前向传播 ===
    X = Tensor(x_batch)
    z_true_data = []
    for label in y_batch:
        one_hot = [0.0] * 10
        one_hot[int(label)] = 1.0
        z_true_data.append(one_hot)
    z_true = Tensor(z_true_data)
    
    # 前向传播
    z_logits = model.forward(X)
    
    # 计算损失
    loss = z_logits.cross_entropy_loss(z_true)
    
    # === 清零梯度（不释放计算图，因为需要反向传播）===
    for p in model.params:
        if isinstance(p.grad, list):
            if isinstance(p.grad[0], list):
                p.grad = [[0.0 for _ in row] for row in p.grad]
            else:
                p.grad = [0.0 for _ in p.grad]
        else:
            p.grad = 0.0
    
    # === 反向传播（使用梯度检查，释放计算图）===
    loss.backward(retain_graph=False)
    
    print(f"loss:{loss.data}", end="\r")
    
    # === 返回梯度数据给主进程 ===
    return model.get_grads(), loss.data


def train_parallel(num_workers=4, batch_size=32, learning_rate=0.01, epochs=3):
    """并行训练MNIST模型（数据并行）
    
    参数:
        num_workers: 并行进程数
        batch_size: 每个进程的批次大小
        learning_rate: 学习率
        epochs: 训练轮数
    """
    from multiprocessing import Pool
    import random
    
    print("加载MNIST数据集...")
    x_train = load_mnist_images('data/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
    
    print(f"训练集大小: {len(x_train)}")
    print(f"测试集大小: {len(x_test)}")
    
    # 归一化数据到 0-1 范围
    x_train = [[pixel / 255.0 for pixel in img] for img in x_train]
    x_test = [[pixel / 255.0 for pixel in img] for img in x_test]
    
    # 创建主模型
    model = MnistModel()
    
    # 创建优化器 交叉
    optimizer = Adam
    # optimizer = SGD
    
    print(f"开始并行训练 (workers={num_workers})...")
    total_start_time = time.time()
    step_size = batch_size * num_workers
    num_steps_per_epoch = (len(x_train) + step_size - 1) // step_size
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # 打乱训练数据
        indices = list(range(len(x_train)))
        random.shuffle(indices)
        if epoch == 3:
          optimizer = SGD  
        total_loss = 0
        num_batches = 0
        
        # 每次处理 num_workers 个批次（数据并行），使用 tqdm 显示进度
        pbar = tqdm(range(0, len(x_train), step_size),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    total=num_steps_per_epoch,
                    unit="step")
        
        for i in pbar:
            # 准备多个批次的数据
            batch_args = []
            for w in range(num_workers):
                start_idx = i + w * batch_size
                if start_idx >= len(x_train):
                    break
                batch_indices = indices[start_idx:start_idx + batch_size]
                x_batch = [x_train[idx] for idx in batch_indices]
                y_batch = [y_train[idx] for idx in batch_indices]
                batch_args.append((model.get_params(), x_batch, y_batch, learning_rate, 'Adam'))
            
            if not batch_args:
                continue
            
            # 并行执行训练
            with Pool(num_workers) as pool:
                results = pool.map(_train_batch_worker, batch_args)
            
            # 聚合梯度并更新参数
            # 计算平均梯度
            avg_grads = None
            total_batch_loss = 0
            for grads, loss in results:
                total_batch_loss += loss
                if avg_grads is None:
                    avg_grads = grads
                else:
                    # 累加梯度
                    for layer_name in ['layer1', 'layer2', 'layer3']:
                        for i in range(len(avg_grads[layer_name]['W'])):
                            for j in range(len(avg_grads[layer_name]['W'][0])):
                                avg_grads[layer_name]['W'][i][j] += grads[layer_name]['W'][i][j]
                        for i in range(len(avg_grads[layer_name]['b'])):
                            avg_grads[layer_name]['b'][i] += grads[layer_name]['b'][i]
            
            # 平均梯度
            num_results = len(results)
            if num_results > 0:
                for layer_name in ['layer1', 'layer2', 'layer3']:
                    for i in range(len(avg_grads[layer_name]['W'])):
                        for j in range(len(avg_grads[layer_name]['W'][0])):
                            avg_grads[layer_name]['W'][i][j] /= num_results
                    for i in range(len(avg_grads[layer_name]['b'])):
                        avg_grads[layer_name]['b'][i] /= num_results
            
            # === 主模型设置聚合梯度并更新参数 ===
            # 直接设置聚合后的梯度（无需清零，因为是覆盖）
            for layer_name in ['layer1', 'layer2', 'layer3']:
                layer = getattr(model, layer_name)
                layer.W.grad = avg_grads[layer_name]['W']
                layer.b.grad = avg_grads[layer_name]['b']
            
            # 使用优化器更新参数
            opt = optimizer(model.params, lr=learning_rate)
            opt.step()
            
            avg_loss = total_batch_loss / num_results
            total_loss += avg_loss
            num_batches += 1
            
            # 更新进度条显示当前损失
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # 在测试集上评估
        correct = 0
        test_batch_size = 100
        for i in range(0, len(x_test), test_batch_size):
            x_batch = x_test[i:i + test_batch_size]
            y_batch = y_test[i:i + test_batch_size]
            predictions = model.predict(x_batch)
            correct += sum(1 for p, y in zip(predictions, y_batch) if p == y)
        
        accuracy = correct / len(x_test) * 100
        elapsed_time = time.time() - total_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (epochs - epoch - 1)
        print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.1f}s - "
              f"Avg Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, "
              f"ETA: {remaining_time/60:.1f}min")
    
    total_time = time.time() - total_start_time
    print(f"训练完成! 总耗时: {total_time/60:.1f}分钟")
    
    # 保存模型参数
    model.save('mnist_model_parallel.json')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(model.loss_history)
    plt.title('Training Loss (Parallel)')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig('training_loss_parallel.png')
    plt.show()
    
    return model


def test(model_path='mnist_model.json'):
    """测试MNIST模型
    
    参数:
        model_path: 模型参数文件路径
    """
    print("加载测试数据集...")
    x_test = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
    
    print(f"测试集大小: {len(x_test)}")
    
    # 归一化数据到 0-1 范围
    x_test = [[pixel / 255.0 for pixel in img] for img in x_test]
    
    # 创建模型并加载参数
    model = MnistModel()
    model.load(model_path)
    
    # 测试
    print("开始测试...")
    correct = 0
    total = len(x_test)
    batch_size = 100
    
    # 统计每个类别的准确率
    class_correct = [0] * 10
    class_total = [0] * 10
    
    for i in range(0, total, batch_size):
        x_batch = x_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]
        predictions = model.predict(x_batch)
        
        for pred, label in zip(predictions, y_batch):
            class_total[label] += 1
            if pred == label:
                correct += 1
                class_correct[label] += 1
    
    accuracy = correct / total * 100
    print(f"\n总体准确率: {accuracy:.2f}% ({correct}/{total})")
    
    # 打印每个类别的准确率
    print("\n各类别准确率:")
    for i in range(10):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i] * 100
            print(f"  数字 {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return accuracy


def visualize_predictions(model_path='mnist_model.json', num_samples=10):
    """可视化预测结果
    
    参数:
        model_path: 模型参数文件路径
        num_samples: 要可视化的样本数量
    """
    print("加载测试数据...")
    x_test = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
    
    # 创建模型并加载参数
    model = MnistModel()
    model.load(model_path)
    
    # 归一化
    x_test_normalized = [[pixel / 255.0 for pixel in img] for img in x_test]
    
    # 随机选择样本
    import random
    indices = random.sample(range(len(x_test)), num_samples)
    
    # 预测
    x_samples = [x_test_normalized[i] for i in indices]
    y_true = [y_test[i] for i in indices]
    predictions = model.predict(x_samples)
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]
        matrix = [x_test[idx][j:j+28] for j in range(0, 784, 28)]
        ax.imshow(matrix, cmap='gray')
        ax.set_title(f"真实: {y_true[i]}, 预测: {predictions[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
    print("预测结果已保存到 predictions.png")


if __name__ == '__main__':
    import sys
    sys.argv.append("train_parallel")
    sys.argv.append("6")
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'train':
            train()
        elif mode == 'train_parallel':
            # 并行训练模式
            num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            train_parallel(num_workers=num_workers)
        elif mode == 'test':
            model_path = sys.argv[2] if len(sys.argv) > 2 else 'mnist_model.json'
            test(model_path)
        elif mode == 'visualize':
            model_path = sys.argv[2] if len(sys.argv) > 2 else 'mnist_model.json'
            visualize_predictions(model_path)
        else:
            print("用法:")
            print("  python mnist.py train              # 串行训练模型")
            print("  python mnist.py train_parallel [N] # 并行训练模型 (N为进程数，默认4)")
            print("  python mnist.py test [model]       # 测试模型")
            print("  python mnist.py visualize [model]  # 可视化预测结果")
    else:
        # 默认执行训练
        train()
