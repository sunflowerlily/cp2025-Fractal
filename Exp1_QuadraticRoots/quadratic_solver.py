import numpy as np

def standard_formula(a, b, c):
    """使用标准公式求解二次方程 ax^2 + bx + c = 0
    
    参数:
        a (float): 二次项系数
        b (float): 一次项系数
        c (float): 常数项
    
    返回:
        tuple: 方程的两个根 (x1, x2) 或 None(无实根)
    """
    # 学生在此处实现代码
    pass

def alternative_formula(a, b, c):
    """使用替代公式求解二次方程 ax^2 + bx + c = 0
    该方法通过将标准公式的分子和分母都乘以 -b∓√(b^2-4ac) 得到
    
    参数:
        a (float): 二次项系数
        b (float): 一次项系数
        c (float): 常数项
    
    返回:
        tuple: 方程的两个根 (x1, x2) 或 None(无实根)
    """
    # 学生在此处实现代码
    pass

def stable_formula(a, b, c):
    """稳定的二次方程求根程序，能够处理各种特殊情况和数值稳定性问题
    
    参数:
        a (float): 二次项系数
        b (float): 一次项系数
        c (float): 常数项
    
    返回:
        tuple: 方程的两个根 (x1, x2) 或 None(无实根)
    """
    # 学生在此处实现代码
    pass

def main():
    test_cases = [
        (1, 2, 1),             # 简单情况
        (1, 1e5, 1),           # b远大于a和c
        (0.001, 1000, 0.001),  # 原测试用例
    ]
    
    for a, b, c in test_cases:
        print("\n" + "="*50)
        print("测试方程：{}x^2 + {}x + {} = 0".format(a, b, c))
        
        # 使用标准公式
        roots1 = standard_formula(a, b, c)
        print("\n方法1（标准公式）的结果：")
        if roots1:
            print("x1 = {:.15f}, x2 = {:.15f}".format(roots1[0], roots1[1]))
        else:
            print("无实根")
        
        # 使用替代公式
        roots2 = alternative_formula(a, b, c)
        print("\n方法2（替代公式）的结果：")
        if roots2:
            print("x1 = {:.15f}, x2 = {:.15f}".format(roots2[0], roots2[1]))
        else:
            print("无实根")
        
        # 使用稳定的求根程序
        roots3 = stable_formula(a, b, c)
        print("\n方法3（稳定求根程序）的结果：")
        if roots3:
            print("x1 = {:.15f}, x2 = {:.15f}".format(roots3[0], roots3[1]))
        else:
            print("无实根")

if __name__ == "__main__":
    main()
