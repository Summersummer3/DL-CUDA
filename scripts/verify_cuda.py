#!/usr/bin/env python3
"""
CUDA和PyTorch安装验证脚本
"""

import sys
import torch
import numpy as np
import time

def check_python_version():
    """检查Python版本"""
    print("=== Python版本检查 ===")
    print(f"Python版本: {sys.version}")
    if sys.version_info >= (3, 8):
        print("✓ Python版本符合要求 (>= 3.8)")
    else:
        print("✗ Python版本过低，需要3.8或更高版本")
        return False
    print()
    return True

def check_pytorch_installation():
    """检查PyTorch安装"""
    print("=== PyTorch安装检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch构建信息: {torch.version.cuda}")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print("✓ PyTorch CUDA版本安装成功")
    else:
        print("✗ PyTorch CUDA版本未安装或CUDA不可用")
        return False
    print()
    return True

def check_gpu_devices():
    """检查GPU设备"""
    print("=== GPU设备检查 ===")
    
    if not torch.cuda.is_available():
        print("✗ 没有可用的CUDA设备")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"GPU设备数量: {device_count}")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"GPU {i}: {device_name}")
        print(f"  计算能力: {device_capability[0]}.{device_capability[1]}")
        print(f"  显存: {device_memory:.1f} GB")
        
        # 检查当前设备
        if i == torch.cuda.current_device():
            print(f"  ✓ 当前设备")
    
    print()
    return True

def test_basic_operations():
    """测试基础张量操作"""
    print("=== 基础张量操作测试 ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA不可用，跳过GPU测试")
        return False
    
    try:
        # 创建测试张量
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # 测试矩阵乘法
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"GPU矩阵乘法时间: {gpu_time:.4f} 秒")
        
        # CPU版本对比
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"CPU矩阵乘法时间: {cpu_time:.4f} 秒")
        print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        # 验证结果
        z_cpu_check = z.cpu()
        if torch.allclose(z_cpu, z_cpu_check, atol=1e-5):
            print("✓ GPU和CPU计算结果一致")
        else:
            print("✗ GPU和CPU计算结果不一致")
            return False
        
        print("✓ 基础张量操作测试通过")
        print()
        return True
        
    except Exception as e:
        print(f"✗ 基础操作测试失败: {e}")
        return False

def test_memory_operations():
    """测试内存操作"""
    print("=== GPU内存操作测试 ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA不可用，跳过内存测试")
        return False
    
    try:
        device = torch.device('cuda:0')
        
        # 获取GPU内存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"GPU总显存: {total_memory:.1f} GB")
        print(f"已分配显存: {allocated_memory:.1f} GB")
        print(f"缓存显存: {cached_memory:.1f} GB")
        
        # 测试大张量分配
        try:
            # 尝试分配一个较大的张量
            large_tensor = torch.randn(5000, 5000, device=device)
            print(f"成功分配大张量: {large_tensor.shape}")
            
            # 清理内存
            del large_tensor
            torch.cuda.empty_cache()
            
            print("✓ GPU内存操作测试通过")
            print()
            return True
            
        except RuntimeError as e:
            print(f"✗ 大张量分配失败: {e}")
            return False
            
    except Exception as e:
        print(f"✗ 内存操作测试失败: {e}")
        return False

def test_neural_network():
    """测试神经网络操作"""
    print("=== 神经网络测试 ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA不可用，跳过神经网络测试")
        return False
    
    try:
        device = torch.device('cuda:0')
        
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
        
        # 前向传播
        start_time = time.time()
        outputs = model(x)
        loss = criterion(outputs, y)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time
        
        # 反向传播
        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time
        
        print(f"前向传播时间: {forward_time:.4f} 秒")
        print(f"反向传播时间: {backward_time:.4f} 秒")
        print(f"损失值: {loss.item():.4f}")
        
        print("✓ 神经网络测试通过")
        print()
        return True
        
    except Exception as e:
        print(f"✗ 神经网络测试失败: {e}")
        return False

def main():
    """主函数"""
    print("PyTorch CUDA 安装验证")
    print("=" * 50)
    
    tests = [
        check_python_version,
        check_pytorch_installation,
        check_gpu_devices,
        test_basic_operations,
        test_memory_operations,
        test_neural_network
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！PyTorch CUDA环境配置成功！")
        return 0
    else:
        print("❌ 部分测试失败，请检查环境配置")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 