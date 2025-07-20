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
import matplotlib.font_manager as fm
import platform
import time
from torch.utils.data import DataLoader, TensorDataset

# 设置中文字体
def setup_chinese_font():
    """设置中文字体支持"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == "Darwin":  # macOS
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 尝试设置字体
    font_found = False
    for font_name in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            # 测试字体是否可用
            test_fig, test_ax = plt.subplots()
            test_ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(test_fig)
            font_found = True
            print(f"✓ 成功设置中文字体: {font_name}")
            break
        except:
            continue
    
    if not font_found:
        print("⚠ 未找到合适的中文字体，图表中的中文可能无法正确显示")
        # 使用默认字体，但禁用中文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()

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
    """训练速度基准测试 - 重新设计以突出GPU优势"""
    print("=== 训练速度基准测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成更大规模的数据
    print("生成大规模训练数据...")
    X_train, X_test, y_train, y_test = generate_synthetic_data(50000)  # 增加到5万样本
    
    # 创建数据加载器 - 使用更大的批次大小
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # 增大批次
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 测试不同设备
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for dev in devices:
        print(f"\n在 {dev} 上训练...")
        device = torch.device(dev)
        
        # 创建更复杂的模型
        class LargeNN(nn.Module):
            def __init__(self, input_size=784, hidden_size=2048, output_size=10):
                super(LargeNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.fc5 = nn.Linear(hidden_size // 4, output_size)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.dropout(self.relu(self.fc3(x)))
                x = self.dropout(self.relu(self.fc4(x)))
                x = self.fc5(x)
                return x
        
        # 创建大型模型
        model = LargeNN().to(device)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练并计时 - 增加训练轮数
        start_time = time.time()
        train_losses, test_accuracies = train_model(
            model, train_loader, test_loader, device, epochs=15  # 增加训练轮数
        )
        total_time = time.time() - start_time
        
        results[dev] = {
            'time': total_time,
            'final_accuracy': test_accuracies[-1],
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
        print(f"{dev} 训练完成，总时间: {total_time:.2f}秒")
    
    # 显示结果
    if len(results) > 1:
        cpu_time = results['cpu']['time']
        gpu_time = results['cuda']['time']
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU加速比: {speedup:.2f}x")
        print(f"CPU训练时间: {cpu_time:.2f}秒")
        print(f"GPU训练时间: {gpu_time:.2f}秒")
    
    return results

def visualize_training(results):
    """可视化训练结果 - 重新设计以突出GPU优势"""
    # 重新设置中文字体，确保在绘图时生效
    setup_chinese_font()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 训练损失曲线
    ax1 = axes[0, 0]
    colors = {'cpu': 'red', 'cuda': 'blue'}
    for device, result in results.items():
        ax1.plot(result['train_losses'], label=f'{device.upper()} 训练损失', 
                color=colors.get(device, 'gray'), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失对比', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 测试准确率曲线
    ax2 = axes[0, 1]
    for device, result in results.items():
        ax2.plot(result['test_accuracies'], label=f'{device.upper()} 测试准确率', 
                color=colors.get(device, 'gray'), linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('测试准确率对比', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练时间对比（突出GPU优势）
    ax3 = axes[1, 0]
    devices = list(results.keys())
    times = [results[dev]['time'] for dev in devices]
    
    # 使用不同颜色突出GPU优势
    bar_colors = ['red' if dev == 'cpu' else 'green' for dev in devices]
    bars = ax3.bar(devices, times, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # 添加数值标签
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('设备')
    ax3.set_ylabel('训练时间 (秒)')
    ax3.set_title('训练时间对比', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 加速比和准确率对比
    ax4 = axes[1, 1]
    if len(results) > 1:
        cpu_time = results['cpu']['time']
        gpu_time = results['cuda']['time']
        speedup = cpu_time / gpu_time
        
        # 创建双轴图
        ax4_twin = ax4.twinx()
        
        # 准确率柱状图
        accuracies = [results[dev]['final_accuracy'] for dev in devices]
        bars1 = ax4.bar([x-0.2 for x in range(len(devices))], accuracies, 
                       width=0.4, label='最终准确率 (%)', color='lightblue', alpha=0.7)
        
        # 加速比柱状图（只在GPU上显示）
        if 'cuda' in results:
            bars2 = ax4.bar([x+0.2 for x in range(len(devices))], [0, speedup], 
                           width=0.4, label='GPU加速比', color='orange', alpha=0.7)
        
        ax4.set_xlabel('设备')
        ax4.set_ylabel('准确率 (%)', color='blue')
        ax4_twin.set_ylabel('加速比', color='orange')
        ax4.set_title(f'性能对比 (GPU加速比: {speedup:.1f}x)', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(devices)))
        ax4.set_xticklabels(devices)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细统计信息
    print("\n" + "="*60)
    print("详细性能统计")
    print("="*60)
    for device, result in results.items():
        print(f"{device.upper()} 设备:")
        print(f"  训练时间: {result['time']:.2f} 秒")
        print(f"  最终准确率: {result['final_accuracy']:.2f}%")
        print(f"  模型参数: {result['model_params']:,}")
    
    if len(results) > 1:
        cpu_time = results['cpu']['time']
        gpu_time = results['cuda']['time']
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU性能提升:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  时间节省: {cpu_time - gpu_time:.2f} 秒")
        print(f"  效率提升: {(speedup-1)*100:.1f}%")

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