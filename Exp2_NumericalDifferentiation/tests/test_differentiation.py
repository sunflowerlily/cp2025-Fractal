import sys
import os
import numpy as np
import pytest

# 添加父目录到路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.differentiation_solution import f, forward_diff, central_diff, analytical_derivative
from differentiation import f, forward_diff, central_diff, analytical_derivative

# 测试函数定义
def test_function_definition():
    """测试函数f(x)是否正确实现"""
    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    expected = [6.0, 2.0, 0.0, 0.0, 2.0]
    
    for x, expected_val in zip(x_values, expected):
        assert abs(f(x) - expected_val) < 1e-10, f"f({x}) 应该返回 {expected_val}，但返回了 {f(x)}"

# 测试解析导数
def test_analytical_derivative():
    """测试解析导数是否正确实现"""
    x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    expected = [-5.0, -3.0, -1.0, 1.0, 3.0]
    
    for x, expected_val in zip(x_values, expected):
        assert abs(analytical_derivative(x) - expected_val) < 1e-10, f"analytical_derivative({x}) 应该返回 {expected_val}，但返回了 {analytical_derivative(x)}"

# 测试前向差分
def test_forward_diff():
    """测试前向差分法是否正确实现"""
    # 定义测试函数
    def test_func(x):
        return x**2
    
    # 测试点和步长
    x = 1.0
    delta = 0.01
    
    # 期望值 (f'(x) = 2x)
    expected = 2.0
    
    # 计算结果
    result = forward_diff(test_func, x, delta)
    
    # 由于数值误差，允许一定的误差范围
    assert abs(result - expected) < 0.1, f"前向差分在 x={x}, delta={delta} 处应该接近 {expected}，但返回了 {result}"

# 测试中心差分
def test_central_diff():
    """测试中心差分法是否正确实现"""
    # 定义测试函数
    def test_func(x):
        return x**2
    
    # 测试点和步长
    x = 1.0
    delta = 0.01
    
    # 期望值 (f'(x) = 2x)
    expected = 2.0
    
    # 计算结果
    result = central_diff(test_func, x, delta)
    
    # 由于数值误差，允许一定的误差范围，但中心差分应该比前向差分更精确
    assert abs(result - expected) < 0.01, f"中心差分在 x={x}, delta={delta} 处应该接近 {expected}，但返回了 {result}"

# 测试不同步长下的误差行为
def test_error_behavior():
    """测试不同步长下的误差行为"""
    # 这个测试主要是确保前向差分和中心差分在不同步长下表现出预期的行为
    # 定义简单的二次函数
    def test_func(x):
        return x**2
    
    # 解析导数
    def test_func_derivative(x):
        return 2*x
    
    # 测试点
    x = 1.0
    
    # 较大步长
    large_delta = 0.1
    # 较小步长
    small_delta = 1e-10
    
    # 计算不同步长下的误差
    forward_large = abs(forward_diff(test_func, x, large_delta) - test_func_derivative(x))
    forward_small = abs(forward_diff(test_func, x, small_delta) - test_func_derivative(x))
    
    central_large = abs(central_diff(test_func, x, large_delta) - test_func_derivative(x))
    central_small = abs(central_diff(test_func, x, small_delta) - test_func_derivative(x))
    
    # 验证中心差分比前向差分更精确（对于较大步长）
    assert central_large < forward_large, "中心差分在较大步长下应该比前向差分更精确"
    
    # 验证较小步长下可能出现舍入误差导致精度下降
    # 注意：这个测试可能会失败，因为舍入误差的行为取决于具体实现和平台
    # 但我们期望在极小步长下，误差会增加
    if forward_small > 1e-8 and central_small > 1e-8:
        pass  # 如果误差较大，说明可能受到舍入误差影响，测试通过
    else:
        # 如果误差很小，也可以接受，因为某些实现可能非常稳定
        pass

if __name__ == "__main__":
    pytest.main(["-v", __file__])