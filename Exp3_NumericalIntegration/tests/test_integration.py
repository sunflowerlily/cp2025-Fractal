import sys
import os
import numpy as np
import pytest

# 添加父目录到路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integration import f, rectangle_method, trapezoid_method
#from solution.integration_solution import f, rectangle_method, trapezoid_method
def test_function_definition():
    """测试函数f(x)是否正确实现"""
    x_values = [0.0, 0.5, 1.0]
    expected = [1.0, 0.8660254037844386, 0.0]
    
    for x, expected_val in zip(x_values, expected):
        assert abs(f(x) - expected_val) < 1e-10, f"f({x}) 应该返回 {expected_val}，但返回了 {f(x)}"

def test_rectangle_method_constant():
    """测试矩形法在常函数上的表现"""
    def const_func(x):
        return 2.0
    
    result = rectangle_method(const_func, 0, 1, 10)
    expected = 2.0
    
    assert abs(result - expected) < 1e-10, "矩形法在常函数上计算错误"

def test_rectangle_method_linear():
    """测试矩形法在线性函数上的表现"""
    def linear_func(x):
        return x
    
    result = rectangle_method(linear_func, 0, 1, 100)
    expected = 0.5
    
    assert abs(result - expected) < 1e-2, "矩形法在线性函数上计算错误"

def test_trapezoid_method_constant():
    """测试梯形法在常函数上的表现"""
    def const_func(x):
        return 2.0
    
    result = trapezoid_method(const_func, 0, 1, 10)
    expected = 2.0
    
    assert abs(result - expected) < 1e-10, "梯形法在常函数上计算错误"

def test_trapezoid_method_linear():
    """测试梯形法在线性函数上的表现"""
    def linear_func(x):
        return x
    
    result = trapezoid_method(linear_func, 0, 1, 100)
    expected = 0.5
    
    assert abs(result - expected) < 1e-10, "梯形法在线性函数上计算错误"

def test_convergence():
    """测试两种方法的收敛性"""
    # 测试函数：x^2
    def test_func(x):
        return x**2
    
    # 精确值
    exact = 1/3
    
    # 不同的N值
    N_values = [10, 100]
    
    # 验证误差随N增大而减小
    rect_errors = []
    trap_errors = []
    
    for N in N_values:
        rect_result = rectangle_method(test_func, 0, 1, N)
        trap_result = trapezoid_method(test_func, 0, 1, N)
        
        rect_errors.append(abs(rect_result - exact))
        trap_errors.append(abs(trap_result - exact))
    
    # 验证误差在N增大时确实减小
    assert rect_errors[1] < rect_errors[0], "矩形法误差没有随N增大而减小"
    assert trap_errors[1] < trap_errors[0], "梯形法误差没有随N增大而减小"
    
    # 验证梯形法比矩形法更精确
    assert trap_errors[0] < rect_errors[0], "梯形法精度没有比矩形法高"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
