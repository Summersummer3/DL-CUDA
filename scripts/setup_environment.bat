@echo off
echo ========================================
echo PyTorch CUDA 环境设置脚本
echo ========================================
echo.

echo 检查Python安装...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python未安装，请先安装Python 3.8+
    echo 推荐下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python已安装
python --version
echo.

echo 检查pip...
pip --version
echo.

echo 创建虚拟环境...
python -m venv pytorch_cuda_env
if %errorlevel% neq 0 (
    echo 创建虚拟环境失败
    pause
    exit /b 1
)

echo 激活虚拟环境...
call pytorch_cuda_env\Scripts\activate.bat
echo.

echo 升级pip...
python -m pip install --upgrade pip
echo.

echo 安装PyTorch (CUDA版本)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo PyTorch安装失败
    pause
    exit /b 1
)

echo.
echo 安装其他依赖...
pip install numpy matplotlib seaborn pandas scikit-learn tqdm jupyter notebook ipywidgets pillow opencv-python tensorboard wandb accelerate transformers datasets
echo.

echo 验证安装...
python scripts\verify_cuda.py
echo.

echo ========================================
echo 环境设置完成！
echo ========================================
echo.
echo 使用方法:
echo 1. 激活环境: pytorch_cuda_env\Scripts\activate.bat
echo 2. 运行示例: python src\basic\tensor_operations.py
echo 3. 启动Jupyter: jupyter notebook notebooks\
echo.
pause 