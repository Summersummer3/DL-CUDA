#!/usr/bin/env python3
"""
PyTorch CUDA 快速开始脚本
快速验证CUDA环境和运行基础示例
"""

import torch
import time
import sys

def check_environment():
    """检查环境"""
    print("=== 环境检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {device_name} ({device_memory:.1f} GB)")
    else:
        print("⚠️  CUDA不可用，将使用CPU运行")
    
    print()

def quick_tensor_test():
    """快速张量测试"""
    print("=== 快速张量测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试张量
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # 测试矩阵乘法
    start_time = time.time()
    z = torch.mm(x, y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"GPU矩阵乘法时间: {gpu_time:.4f} 秒")
    print(f"结果张量形状: {z.shape}")
    print()

def quick_nn_test():
    """快速神经网络测试"""
    print("=== 快速神经网络测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单神经网络
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    ).to(device)
    
    # 创建测试数据
    x = torch.randn(32, 1000, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 训练一个批次
    start_time = time.time()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.time() - start_time
    
    print(f"神经网络训练时间: {train_time:.4f} 秒")
    print(f"损失值: {loss.item():.4f}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print()

def memory_test():
    """内存测试"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存测试")
        return
    
    print("=== GPU内存测试 ===")
    
    device = torch.device('cuda')
    
    # 显示内存信息
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    cached_memory = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"GPU总显存: {total_memory:.1f} GB")
    print(f"已分配显存: {allocated_memory:.2f} GB")
    print(f"缓存显存: {cached_memory:.2f} GB")
    
    # 测试大张量分配
    try:
        large_tensor = torch.randn(4000, 4000, device=device)
        print(f"成功分配大张量: {large_tensor.shape}")
        print(f"张量大小: {large_tensor.numel() * 4 / 1024**3:.2f} GB")
        
        # 清理内存
        del large_tensor
        torch.cuda.empty_cache()
        print("✓ 内存测试通过")
        
    except RuntimeError as e:
        print(f"✗ 大张量分配失败: {e}")
    
    print()

def run_examples():
    """运行示例程序"""
    print("=== 运行示例程序 ===")
    
    try:
        print("1. 运行基础张量操作示例...")
        import subprocess
        result = subprocess.run([sys.executable, "src/basic/tensor_operations.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ 基础张量操作示例运行成功")
        else:
            print("✗ 基础张量操作示例运行失败")
            print(result.stderr)
    except Exception as e:
        print(f"✗ 运行示例时出错: {e}")
    
    print()

def main():
    """主函数"""
    print("PyTorch CUDA 快速开始")
    print("=" * 50)
    
    # 环境检查
    check_environment()
    
    # 快速测试
    quick_tensor_test()
    quick_nn_test()
    memory_test()
    
    # 询问是否运行完整示例
    print("=" * 50)
    response = input("是否运行完整的示例程序？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是']:
        run_examples()
    
    print("=" * 50)
    print("快速开始完成！")
    print("\n下一步:")
    print("1. 运行 'python scripts/verify_cuda.py' 进行完整验证")
    print("2. 运行 'python src/basic/tensor_operations.py' 查看详细示例")
    print("3. 运行 'python src/basic/neural_network.py' 训练神经网络")
    print("4. 运行 'jupyter notebook notebooks/' 启动Jupyter笔记本")

if __name__ == "__main__":
    main() 