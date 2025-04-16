import sys
import os
import numpy as np
import pytest

# 添加父目录到路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bessel_recursion import bessel_up, bessel_down
#from solution.bessel_recursion_solution import bessel_up, bessel_down

from scipy.special import spherical_jn

def test_bessel_up_basic():
    """测试bessel_up基本功能"""
    x = 1.0
    lmax = 5
    result = bessel_up(x, lmax)
    
    # 检查返回值类型和长度
    assert isinstance(result, np.ndarray), "返回值类型应为numpy数组"
    assert len(result) == lmax + 1, "返回数组长度应为lmax + 1"
    
    # 检查j_0(x)和j_1(x)的计算是否正确
    assert abs(result[0] - np.sin(x)/x) < 1e-8, "j_0(x)计算错误"
    assert abs(result[1] - (np.sin(x)/x**2 - np.cos(x)/x)) < 1e-8, "j_1(x)计算错误"

def test_bessel_down_basic():
    """测试bessel_down基本功能"""
    x = 1.0
    lmax = 5
    result = bessel_down(x, lmax)
    
    # 检查返回值类型和长度
    assert isinstance(result, np.ndarray), "返回值类型应为numpy数组"
    assert len(result) == lmax + 1, "返回数组长度应为lmax + 1"
    
    # 检查归一化是否正确
    assert abs(result[0] - np.sin(x)/x) < 1e-8, "归一化后j_0(x)应等于解析值"

def test_bessel_special_cases():
    """测试特殊情况"""
    # 测试x=0的情况
    lmax = 3
    with pytest.raises(ZeroDivisionError):
        bessel_up(0, lmax)
    with pytest.raises(ZeroDivisionError):
        bessel_down(0, lmax)
    
    # 测试小x值
    x = 0.1  # 使用较大的x值
    result_up = bessel_up(x, lmax)
    result_down = bessel_down(x, lmax)
    scipy_result = spherical_jn(np.arange(lmax + 1), x)
    
    # 放宽精度要求
    assert np.allclose(result_up[:2], scipy_result[:2], rtol=1e-5, atol=1e-6), "小x值时低阶结果应准确"
    assert np.allclose(result_down[:2], scipy_result[:2], rtol=1e-5, atol=1e-6), "小x值时低阶结果应准确"

def test_bessel_accuracy():
    """测试计算精度"""
    x = 1.0
    lmax = 5
    l = np.arange(lmax + 1)
    
    result_up = bessel_up(x, lmax)
    result_down = bessel_down(x, lmax)
    scipy_result = spherical_jn(l, x)
    
    # 检查与scipy结果的一致性
    assert np.allclose(result_up, scipy_result, rtol=1e-8, atol=1e-10), "向上递推结果应与scipy结果一致"
    assert np.allclose(result_down, scipy_result, rtol=1e-8, atol=1e-10), "向下递推结果应与scipy结果一致"

def test_bessel_stability():
    """测试数值稳定性"""
    x = 1.0
    lmax = 15  # 选择一个较大的lmax以观察稳定性
    l = np.arange(lmax + 1)
    
    result_up = bessel_up(x, lmax)
    result_down = bessel_down(x, lmax)
    scipy_result = spherical_jn(l, x)
    
    # 计算相对误差
    err_up = np.abs((result_up - scipy_result) / scipy_result)
    err_down = np.abs((result_down - scipy_result) / scipy_result)
    
    # 检查l > x区域的稳定性
    high_l = l > x
    assert np.mean(err_down[high_l]) < np.mean(err_up[high_l]), "向下递推在l > x时应更稳定"

if __name__ == "__main__":
    pytest.main(["-v", __file__])