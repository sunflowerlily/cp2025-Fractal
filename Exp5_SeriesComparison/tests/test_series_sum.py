import sys
import os
import numpy as np
import pytest

# 添加父目录到路径，以便导入学生代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from solution.series_sum_solution import sum_S1, sum_S2, sum_S3
from series_sum import sum_S1, sum_S2, sum_S3
def test_sum_S1_basic():
    """测试sum_S1基本功能"""
    assert abs(sum_S1(1) - 0.166666666666667) < 1e-10, "N=1时计算错误"
    assert abs(sum_S1(2) - 0.216666666666667) < 1e-10, "N=2时计算错误"
    assert abs(sum_S1(4) - 0.254365079365079) < 1e-10, "N=4时计算错误"

def test_sum_S2_basic():
    """测试sum_S2基本功能"""
    assert abs(sum_S2(1) - 0.166666666666667) < 1e-10, "N=1时计算错误"
    assert abs(sum_S2(2) - 0.216666666666667) < 1e-10, "N=2时计算错误"
    assert abs(sum_S2(4) - 0.254365079365079) < 1e-10, "N=4时计算错误"

def test_sum_S3_basic():
    """测试sum_S3基本功能"""
    assert abs(sum_S3(1) - 0.166666666666667) < 1e-10, "N=1时计算错误"
    assert abs(sum_S3(2) - 0.216666666666667) < 1e-10, "N=2时计算错误"
    assert abs(sum_S3(4) - 0.254365079365079) < 1e-10, "N=4时计算错误"

def test_consistency_small_N():
    """测试小N值时三种方法结果一致"""
    N = 10
    assert abs(sum_S1(N) - sum_S3(N)) < 1e-10, "S1和S3在N=10时结果应该几乎相同"
    assert abs(sum_S2(N) - sum_S3(N)) < 1e-10, "S2和S3在N=10时结果应该几乎相同"

def test_monotonicity():
    """测试和的单调性"""
    N_values = [10, 100]
    sums_1 = [sum_S1(N) for N in N_values]
    sums_2 = [sum_S2(N) for N in N_values]
    sums_3 = [sum_S3(N) for N in N_values]
    
    # 验证和随N增大而增大
    assert sums_1[1] > sums_1[0], "S1结果应随N增大而增大"
    assert sums_2[1] > sums_2[0], "S2结果应随N增大而增大"
    assert sums_3[1] > sums_3[0], "S3结果应随N增大而增大"

def test_relative_difference():
    """测试大N值时的相对差异"""
    N = 1000
    s1 = sum_S1(N)
    s2 = sum_S2(N)
    s3 = sum_S3(N)
    
    # 验证S2的误差应该大于S1的误差
    err1 = abs((s1 - s3) / s3)
    err2 = abs((s2 - s3) / s3)
    assert err2 > err1, "S2的误差应该大于S1的误差"

if __name__ == "__main__":
    pytest.main(["-v", __file__])