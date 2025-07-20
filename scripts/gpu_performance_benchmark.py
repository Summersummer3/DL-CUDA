#!/usr/bin/env python3
"""
GPU性能基准测试脚本
测试GPU在各种计算任务上的性能增益
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

def print_system_info():
    """打印系统信息"""
    print("=== 系统信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            device_capability = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {device_name}")
            print(f"  显存: {device_memory:.1f} GB")
            print(f"  计算能力: {device_capability[0]}.{device_capability[1]}")
    print()

def benchmark_matrix_operations():
    """矩阵运算性能测试"""
    print("=== 矩阵运算性能测试 ===")
    
    sizes = [100, 500, 1000, 2000, 4000, 6000, 8000]
    results = {'size': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
    
    for size in sizes:
        print(f"测试矩阵大小: {size}x{size}")
        
        # 创建测试矩阵
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # CPU测试
        start_time = time.time()
        for _ in range(5):  # 多次运行取平均
            c_cpu = torch.mm(a, b)
        cpu_time = (time.time() - start_time) / 5
        
        # GPU测试
        if torch.cuda.is_available():
            a_gpu = a.cuda()
            b_gpu = b.cuda()
            
            # 预热
            torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(5):
                c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 5
            
            # 避免除零错误
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: {speedup:.2f}x")
            else:
                speedup = float('inf')
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: 无限大 (GPU时间过短)")
            
            results['size'].append(size)
            results['cpu_time'].append(cpu_time)
            results['gpu_time'].append(gpu_time)
            results['speedup'].append(speedup)
        else:
            print(f"  CPU时间: {cpu_time:.4f}秒")
            print("  GPU不可用")
        
        print()
    
    return results

def benchmark_convolution_operations():
    """卷积运算性能测试"""
    print("=== 卷积运算性能测试 ===")
    
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    image_size = 224
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    
    results = {'batch_size': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
    
    # 创建卷积层
    conv_cpu = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
    if torch.cuda.is_available():
        conv_gpu = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1).cuda()
        # 复制权重
        conv_gpu.load_state_dict(conv_cpu.state_dict())
    
    for batch_size in batch_sizes:
        print(f"测试批次大小: {batch_size}")
        
        # 创建输入数据
        input_cpu = torch.randn(batch_size, in_channels, image_size, image_size)
        
        # CPU测试
        start_time = time.time()
        for _ in range(10):
            output_cpu = conv_cpu(input_cpu)
        cpu_time = (time.time() - start_time) / 10
        
        # GPU测试
        if torch.cuda.is_available():
            input_gpu = input_cpu.cuda()
            
            # 预热
            conv_gpu(input_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(10):
                output_gpu = conv_gpu(input_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 10
            
            # 避免除零错误
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: {speedup:.2f}x")
            else:
                speedup = float('inf')
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: 无限大 (GPU时间过短)")
            
            results['batch_size'].append(batch_size)
            results['cpu_time'].append(cpu_time)
            results['gpu_time'].append(gpu_time)
            results['speedup'].append(speedup)
        else:
            print(f"  CPU时间: {cpu_time:.4f}秒")
            print("  GPU不可用")
        
        print()
    
    return results

def benchmark_neural_network_training():
    """神经网络训练性能测试"""
    print("=== 神经网络训练性能测试 ===")
    
    # 创建大型神经网络
    class LargeNN(torch.nn.Module):
        def __init__(self, input_size=1000, hidden_size=2000, output_size=100):
            super(LargeNN, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden_size // 2, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    batch_sizes = [32, 64, 128, 256, 512]
    results = {'batch_size': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
    
    for batch_size in batch_sizes:
        print(f"测试批次大小: {batch_size}")
        
        # 创建模型和数据
        model_cpu = LargeNN()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.001)
        
        x_cpu = torch.randn(batch_size, 1000)
        y_cpu = torch.randint(0, 100, (batch_size,))
        
        # CPU训练
        start_time = time.time()
        for epoch in range(5):
            optimizer_cpu.zero_grad()
            output = model_cpu(x_cpu)
            loss = criterion(output, y_cpu)
            loss.backward()
            optimizer_cpu.step()
        cpu_time = time.time() - start_time
        
        # GPU训练
        if torch.cuda.is_available():
            model_gpu = LargeNN().cuda()
            model_gpu.load_state_dict(model_cpu.state_dict())
            optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.001)
            
            x_gpu = x_cpu.cuda()
            y_gpu = y_cpu.cuda()
            
            # 预热
            optimizer_gpu.zero_grad()
            output = model_gpu(x_gpu)
            loss = criterion(output, y_gpu)
            loss.backward()
            optimizer_gpu.step()
            torch.cuda.synchronize()
            
            start_time = time.time()
            for epoch in range(5):
                optimizer_gpu.zero_grad()
                output = model_gpu(x_gpu)
                loss = criterion(output, y_gpu)
                loss.backward()
                optimizer_gpu.step()
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # 避免除零错误
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: {speedup:.2f}x")
            else:
                speedup = float('inf')
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: 无限大 (GPU时间过短)")
            
            results['batch_size'].append(batch_size)
            results['cpu_time'].append(cpu_time)
            results['gpu_time'].append(gpu_time)
            results['speedup'].append(speedup)
        else:
            print(f"  CPU时间: {cpu_time:.4f}秒")
            print("  GPU不可用")
        
        print()
    
    return results

def benchmark_element_wise_operations():
    """元素级运算性能测试"""
    print("=== 元素级运算性能测试 ===")
    
    sizes = [1000000, 5000000, 10000000, 50000000, 100000000]
    results = {'size': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
    
    for size in sizes:
        print(f"测试向量大小: {size:,}")
        
        # 创建测试数据
        a = torch.randn(size)
        b = torch.randn(size)
        c = torch.randn(size)
        
        # CPU测试 - 复杂元素级运算
        start_time = time.time()
        for _ in range(10):
            result_cpu = torch.sin(a) * torch.cos(b) + torch.exp(c) * torch.log(torch.abs(a) + 1)
        cpu_time = (time.time() - start_time) / 10
        
        # GPU测试
        if torch.cuda.is_available():
            a_gpu = a.cuda()
            b_gpu = b.cuda()
            c_gpu = c.cuda()
            
            # 预热
            torch.sin(a_gpu) * torch.cos(b_gpu) + torch.exp(c_gpu) * torch.log(torch.abs(a_gpu) + 1)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(10):
                result_gpu = torch.sin(a_gpu) * torch.cos(b_gpu) + torch.exp(c_gpu) * torch.log(torch.abs(a_gpu) + 1)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 10
            
            # 避免除零错误
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: {speedup:.2f}x")
            else:
                speedup = float('inf')
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: 无限大 (GPU时间过短)")
            
            results['size'].append(size)
            results['cpu_time'].append(cpu_time)
            results['gpu_time'].append(gpu_time)
            results['speedup'].append(speedup)
        else:
            print(f"  CPU时间: {cpu_time:.4f}秒")
            print("  GPU不可用")
        
        print()
    
    return results

def benchmark_sorting_operations():
    """排序运算性能测试"""
    print("=== 排序运算性能测试 ===")
    
    sizes = [100000, 500000, 1000000, 5000000, 10000000]
    results = {'size': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
    
    for size in sizes:
        print(f"测试数组大小: {size:,}")
        
        # 创建测试数据
        data = torch.randn(size)
        
        # CPU排序
        start_time = time.time()
        for _ in range(5):
            sorted_cpu, _ = torch.sort(data)
        cpu_time = (time.time() - start_time) / 5
        
        # GPU排序
        if torch.cuda.is_available():
            data_gpu = data.cuda()
            
            # 预热
            torch.sort(data_gpu)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(5):
                sorted_gpu, _ = torch.sort(data_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 5
            
            # 避免除零错误
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: {speedup:.2f}x")
            else:
                speedup = float('inf')
                print(f"  CPU时间: {cpu_time:.4f}秒")
                print(f"  GPU时间: {gpu_time:.4f}秒")
                print(f"  加速比: 无限大 (GPU时间过短)")
            
            results['size'].append(size)
            results['cpu_time'].append(cpu_time)
            results['gpu_time'].append(gpu_time)
            results['speedup'].append(speedup)
        else:
            print(f"  CPU时间: {cpu_time:.4f}秒")
            print("  GPU不可用")
        
        print()
    
    return results

def benchmark_memory_bandwidth():
    """内存带宽测试"""
    print("=== 内存带宽测试 ===")
    
    if not torch.cuda.is_available():
        print("GPU不可用，跳过内存带宽测试")
        return None
    
    sizes = [1000000, 5000000, 10000000, 50000000, 100000000]
    results = {'size': [], 'cpu_bandwidth': [], 'gpu_bandwidth': []}
    
    for size in sizes:
        print(f"测试数据大小: {size:,} 元素")
        
        # 创建大数组
        data_cpu = torch.randn(size)
        data_gpu = data_cpu.cuda()
        
        # CPU内存复制测试
        start_time = time.time()
        for _ in range(100):
            copy_cpu = data_cpu.clone()
        cpu_time = (time.time() - start_time) / 100
        
        # GPU内存复制测试
        start_time = time.time()
        for _ in range(100):
            copy_gpu = data_gpu.clone()
        torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / 100
        
        # 计算带宽 (GB/s)
        data_size_gb = size * 4 / (1024**3)  # 4 bytes per float32
        
        cpu_bandwidth = data_size_gb / cpu_time if cpu_time > 0 else 0
        gpu_bandwidth = data_size_gb / gpu_time if gpu_time > 0 else 0
        
        print(f"  CPU带宽: {cpu_bandwidth:.2f} GB/s")
        print(f"  GPU带宽: {gpu_bandwidth:.2f} GB/s")
        if cpu_bandwidth > 0 and gpu_bandwidth > 0:
            print(f"  GPU/CPU带宽比: {gpu_bandwidth/cpu_bandwidth:.2f}x")
        else:
            print(f"  GPU/CPU带宽比: 无法计算")
        
        results['size'].append(size)
        results['cpu_bandwidth'].append(cpu_bandwidth)
        results['gpu_bandwidth'].append(gpu_bandwidth)
        
        print()
    
    return results

def plot_results(matrix_results, conv_results, nn_results, element_results, sort_results, bandwidth_results):
    """绘制结果图表"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 处理无限大值，用于绘图
    def filter_infinite_speedups(speedups):
        return [s if s != float('inf') else 1000 for s in speedups]  # 将无限大替换为1000
    
    # 矩阵运算结果
    if matrix_results['size']:
        ax = axes[0, 0]
        ax.plot(matrix_results['size'], matrix_results['cpu_time'], 'b-o', label='CPU', linewidth=2, markersize=6)
        ax.plot(matrix_results['size'], matrix_results['gpu_time'], 'r-o', label='GPU', linewidth=2, markersize=6)
        ax.set_xlabel('矩阵大小')
        ax.set_ylabel('时间 (秒)')
        ax.set_title('矩阵乘法性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 卷积运算结果
    if conv_results['batch_size']:
        ax = axes[0, 1]
        ax.plot(conv_results['batch_size'], conv_results['cpu_time'], 'b-o', label='CPU', linewidth=2, markersize=6)
        ax.plot(conv_results['batch_size'], conv_results['gpu_time'], 'r-o', label='GPU', linewidth=2, markersize=6)
        ax.set_xlabel('批次大小')
        ax.set_ylabel('时间 (秒)')
        ax.set_title('卷积运算性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 神经网络训练结果
    if nn_results['batch_size']:
        ax = axes[0, 2]
        ax.plot(nn_results['batch_size'], nn_results['cpu_time'], 'b-o', label='CPU', linewidth=2, markersize=6)
        ax.plot(nn_results['batch_size'], nn_results['gpu_time'], 'r-o', label='GPU', linewidth=2, markersize=6)
        ax.set_xlabel('批次大小')
        ax.set_ylabel('时间 (秒)')
        ax.set_title('神经网络训练性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 元素级运算结果
    if element_results['size']:
        ax = axes[1, 0]
        ax.plot(element_results['size'], element_results['cpu_time'], 'b-o', label='CPU', linewidth=2, markersize=6)
        ax.plot(element_results['size'], element_results['gpu_time'], 'r-o', label='GPU', linewidth=2, markersize=6)
        ax.set_xlabel('向量大小')
        ax.set_ylabel('时间 (秒)')
        ax.set_title('元素级运算性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 排序运算结果
    if sort_results['size']:
        ax = axes[1, 1]
        ax.plot(sort_results['size'], sort_results['cpu_time'], 'b-o', label='CPU', linewidth=2, markersize=6)
        ax.plot(sort_results['size'], sort_results['gpu_time'], 'r-o', label='GPU', linewidth=2, markersize=6)
        ax.set_xlabel('数组大小')
        ax.set_ylabel('时间 (秒)')
        ax.set_title('排序运算性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # 内存带宽结果
    if bandwidth_results and bandwidth_results['size']:
        ax = axes[1, 2]
        ax.plot(bandwidth_results['size'], bandwidth_results['cpu_bandwidth'], 'b-o', label='CPU', linewidth=2, markersize=6)
        ax.plot(bandwidth_results['size'], bandwidth_results['gpu_bandwidth'], 'r-o', label='GPU', linewidth=2, markersize=6)
        ax.set_xlabel('数据大小')
        ax.set_ylabel('带宽 (GB/s)')
        ax.set_title('内存带宽性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('gpu_performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(matrix_results, conv_results, nn_results, element_results, sort_results, bandwidth_results):
    """打印性能总结"""
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)
    
    if matrix_results['speedup']:
        max_speedup = max(matrix_results['speedup'])
        print(f"矩阵乘法最大加速比: {max_speedup:.2f}x")
    
    if conv_results['speedup']:
        max_speedup = max(conv_results['speedup'])
        print(f"卷积运算最大加速比: {max_speedup:.2f}x")
    
    if nn_results['speedup']:
        max_speedup = max(nn_results['speedup'])
        print(f"神经网络训练最大加速比: {max_speedup:.2f}x")
    
    if element_results['speedup']:
        max_speedup = max(element_results['speedup'])
        print(f"元素级运算最大加速比: {max_speedup:.2f}x")
    
    if sort_results['speedup']:
        max_speedup = max(sort_results['speedup'])
        print(f"排序运算最大加速比: {max_speedup:.2f}x")
    
    if bandwidth_results and bandwidth_results['gpu_bandwidth']:
        max_gpu_bandwidth = max(bandwidth_results['gpu_bandwidth'])
        max_cpu_bandwidth = max(bandwidth_results['cpu_bandwidth'])
        print(f"GPU最大内存带宽: {max_gpu_bandwidth:.2f} GB/s")
        print(f"CPU最大内存带宽: {max_cpu_bandwidth:.2f} GB/s")
        if max_cpu_bandwidth > 0:
            print(f"GPU/CPU带宽比: {max_gpu_bandwidth/max_cpu_bandwidth:.2f}x")
        else:
            print(f"GPU/CPU带宽比: 无限大")

def main():
    """主函数"""
    print("GPU性能基准测试")
    print("="*60)
    
    # 打印系统信息
    print_system_info()
    
    # 运行各种性能测试
    print("开始性能测试...\n")
    
    matrix_results = benchmark_matrix_operations()
    conv_results = benchmark_convolution_operations()
    nn_results = benchmark_neural_network_training()
    element_results = benchmark_element_wise_operations()
    sort_results = benchmark_sorting_operations()
    bandwidth_results = benchmark_memory_bandwidth()
    
    # 绘制结果
    print("生成性能图表...")
    plot_results(matrix_results, conv_results, nn_results, element_results, sort_results, bandwidth_results)
    
    # 打印总结
    print_summary(matrix_results, conv_results, nn_results, element_results, sort_results, bandwidth_results)
    
    print("\n" + "="*60)
    print("性能测试完成！")
    print("图表已保存为: gpu_performance_benchmark.png")

if __name__ == "__main__":
    main() 