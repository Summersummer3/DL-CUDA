#!/usr/bin/env python3
"""
简单神经网络示例
演示GPU加速的深度学习训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    """简单神经网络模型"""
    
    def __init__(self, input_size=784, hidden_size=512, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    """卷积神经网络模型"""
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 卷积层
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def generate_synthetic_data(num_samples=10000, input_size=784, num_classes=10):
    """生成合成数据"""
    print(f"生成 {num_samples} 个样本...")
    
    # 生成随机特征
    X = torch.randn(num_samples, input_size)
    
    # 生成标签（基于特征的线性组合）
    weights = torch.randn(input_size, num_classes)
    logits = torch.mm(X, weights)
    y = torch.argmax(logits, dim=1)
    
    # 分割训练和测试集
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test

def generate_image_data(num_samples=10000, image_size=28, num_classes=10):
    """生成图像数据"""
    print(f"生成 {num_samples} 个图像样本...")
    
    # 生成随机图像数据
    X = torch.randn(num_samples, 1, image_size, image_size)
    
    # 生成标签
    y = torch.randint(0, num_classes, (num_samples,))
    
    # 分割训练和测试集
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test

def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    """训练模型"""
    print(f"开始训练，设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 记录训练过程
    train_losses = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} 完成, '
              f'平均损失: {avg_loss:.4f}, '
              f'测试准确率: {accuracy:.2f}%, '
              f'时间: {epoch_time:.2f}秒')
        
        scheduler.step()
    
    return train_losses, test_accuracies

def benchmark_training_speed():
    """训练速度基准测试"""
    print("=== 训练速度基准测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成数据
    X_train, X_test, y_train, y_test = generate_synthetic_data(5000)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 测试不同设备
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for dev in devices:
        print(f"\n在 {dev} 上训练...")
        device = torch.device(dev)
        
        # 创建模型
        model = SimpleNN().to(device)
        
        # 训练并计时
        start_time = time.time()
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, device, epochs=5
        )
        total_time = time.time() - start_time
        
        results[dev] = {
            'time': total_time,
            'final_accuracy': test_accuracies[-1],
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
        
        print(f"{dev} 训练完成，总时间: {total_time:.2f}秒")
    
    # 显示结果
    if len(results) > 1:
        cpu_time = results['cpu']['time']
        gpu_time = results['cuda']['time']
        speedup = cpu_time / gpu_time
        print(f"\n加速比: {speedup:.2f}x")
    
    return results

def visualize_training(results):
    """可视化训练结果"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    for device, result in results.items():
        plt.plot(result['train_losses'], label=f'{device} 训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    for device, result in results.items():
        plt.plot(result['test_accuracies'], label=f'{device} 测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('测试准确率')
    plt.legend()
    plt.grid(True)
    
    # 绘制性能对比
    plt.subplot(1, 3, 3)
    devices = list(results.keys())
    times = [results[dev]['time'] for dev in devices]
    accuracies = [results[dev]['final_accuracy'] for dev in devices]
    
    x = np.arange(len(devices))
    width = 0.35
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, times, width, label='训练时间 (秒)', color='skyblue')
    bars2 = ax2.bar(x + width/2, accuracies, width, label='最终准确率 (%)', color='lightcoral')
    
    ax1.set_xlabel('设备')
    ax1.set_ylabel('训练时间 (秒)')
    ax2.set_ylabel('准确率 (%)')
    ax1.set_title('性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(devices)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("PyTorch 神经网络示例")
    print("=" * 50)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # 1. 简单神经网络训练
    print("=== 简单神经网络训练 ===")
    X_train, X_test, y_train, y_test = generate_synthetic_data(5000)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = SimpleNN().to(device)
    train_losses, test_accuracies = train_model(model, train_loader, test_loader, device)
    
    print(f"最终测试准确率: {test_accuracies[-1]:.2f}%")
    print()
    
    # 2. CNN训练
    print("=== CNN训练 ===")
    X_train, X_test, y_train, y_test = generate_image_data(3000)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    cnn_model = CNN().to(device)
    cnn_train_losses, cnn_test_accuracies = train_model(
        cnn_model, train_loader, test_loader, device, epochs=8
    )
    
    print(f"CNN最终测试准确率: {cnn_test_accuracies[-1]:.2f}%")
    print()
    
    # 3. 性能基准测试
    results = benchmark_training_speed()
    
    # 4. 可视化结果
    visualize_training(results)
    
    print("=" * 50)
    print("神经网络示例完成！")

if __name__ == "__main__":
    main() 