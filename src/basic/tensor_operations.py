#!/usr/bin/env python3
"""
基础PyTorch张量操作示例
演示GPU加速的张量运算
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

def print_device_info():
    """打印设备信息"""
    print("=== 设备信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print()

def basic_tensor_operations():
    """基础张量操作"""
    print("=== 基础张量操作 ===")
    
    # 创建张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建随机张量
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    print(f"张量形状: {x.shape}")
    print(f"张量数据类型: {x.dtype}")
    print(f"张量设备: {x.device}")
    
    # 基础运算
    print("\n基础运算:")
    print(f"x + y 形状: {(x + y).shape}")
    print(f"x * y 形状: {(x * y).shape}")
    print(f"x^2 形状: {(x ** 2).shape}")
    
    # 统计操作
    print(f"\n统计操作:")
    print(f"x 均值: {x.mean():.4f}")
    print(f"x 标准差: {x.std():.4f}")
    print(f"x 最大值: {x.max():.4f}")
    print(f"x 最小值: {x.min():.4f}")
    print()

def matrix_operations_benchmark():
    """矩阵运算性能基准测试"""
    print("=== 矩阵运算性能基准测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sizes = [100, 500, 1000, 2000, 3000]
    
    gpu_times = []
    cpu_times = []
    
    for size in sizes:
        print(f"测试矩阵大小: {size}x{size}")
        
        # 创建测试矩阵
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # GPU测试
        if torch.cuda.is_available():
            a_gpu = a.to(device)
            b_gpu = b.to(device)
            
            # 预热
            torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            # 计时
            start_time = time.time()
            for _ in range(10):
                c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 10
            gpu_times.append(gpu_time)
            
            print(f"  GPU时间: {gpu_time:.4f} 秒")
        
        # CPU测试
        start_time = time.time()
        for _ in range(10):
            c_cpu = torch.mm(a, b)
        cpu_time = (time.time() - start_time) / 10
        cpu_times.append(cpu_time)
        
        print(f"  CPU时间: {cpu_time:.4f} 秒")
        
        if torch.cuda.is_available():
            speedup = cpu_time / gpu_time
            print(f"  加速比: {speedup:.2f}x")
        print()
    
    # 绘制性能对比图
    if torch.cuda.is_available():
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(sizes, cpu_times, 'b-o', label='CPU')
        plt.plot(sizes, gpu_times, 'r-o', label='GPU')
        plt.xlabel('矩阵大小')
        plt.ylabel('时间 (秒)')
        plt.title('矩阵乘法性能对比')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        speedups = [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]
        plt.plot(sizes, speedups, 'g-o')
        plt.xlabel('矩阵大小')
        plt.ylabel('加速比')
        plt.title('GPU加速比')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('matrix_operations_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()

def tensor_reshaping_operations():
    """张量重塑操作"""
    print("=== 张量重塑操作 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建原始张量
    x = torch.randn(2, 3, 4, 5, device=device)
    print(f"原始张量形状: {x.shape}")
    
    # 重塑操作
    print("\n重塑操作:")
    print(f"view(2, 60): {x.view(2, 60).shape}")
    print(f"reshape(2, 60): {x.reshape(2, 60).shape}")
    print(f"flatten(): {x.flatten().shape}")
    print(f"squeeze(): {x.squeeze().shape}")
    print(f"unsqueeze(0): {x.unsqueeze(0).shape}")
    
    # 转置操作
    print(f"\n转置操作:")
    print(f"transpose(0, 1): {x.transpose(0, 1).shape}")
    print(f"permute(3, 2, 1, 0): {x.permute(3, 2, 1, 0).shape}")
    print()

def memory_operations():
    """内存操作"""
    print("=== 内存操作 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存操作测试")
        return
    
    device = torch.device('cuda')
    
    # 显示初始内存状态
    print("初始内存状态:")
    print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # 分配大张量
    print("\n分配大张量...")
    large_tensor = torch.randn(5000, 5000, device=device)
    print(f"张量形状: {large_tensor.shape}")
    print(f"张量大小: {large_tensor.numel() * 4 / 1024**3:.2f} GB")
    
    print("分配后内存状态:")
    print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # 释放张量
    print("\n释放张量...")
    del large_tensor
    torch.cuda.empty_cache()
    
    print("释放后内存状态:")
    print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"已缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print()

def advanced_operations():
    """高级张量操作"""
    print("=== 高级张量操作 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    
    # 广播操作
    print("广播操作:")
    a = torch.randn(100, 1, device=device)
    b = torch.randn(1, 100, device=device)
    c = a + b
    print(f"广播加法形状: {c.shape}")
    
    # 索引和切片
    print("\n索引和切片:")
    print(f"x[0, :5]: {x[0, :5]}")
    print(f"x[:5, :5] 形状: {x[:5, :5].shape}")
    
    # 条件操作
    print("\n条件操作:")
    mask = x > 0
    print(f"正数元素数量: {mask.sum()}")
    print(f"正数元素: {x[mask][:5]}")
    
    # 聚合操作
    print("\n聚合操作:")
    print(f"按行求和: {x.sum(dim=1)[:5]}")
    print(f"按列求最大值: {x.max(dim=0)[0][:5]}")
    print(f"全局最大值: {x.max()}")
    print()

def main():
    """主函数"""
    print("PyTorch 基础张量操作示例")
    print("=" * 50)
    
    # 检查设备
    print_device_info()
    
    # 运行各种操作
    basic_tensor_operations()
    tensor_reshaping_operations()
    memory_operations()
    advanced_operations()
    
    # 性能基准测试
    matrix_operations_benchmark()
    
    print("=" * 50)
    print("示例程序完成！")

if __name__ == "__main__":
    main() 