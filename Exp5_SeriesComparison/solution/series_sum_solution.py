import numpy as np
import matplotlib.pyplot as plt

def sum_S1(N):
    """计算第一种形式的级数和：交错级数
    S_N^(1) = sum_{n=1}^{2N} (-1)^n * n/(n+1)
    """
    result = 0.0
    for n in range(1, 2*N + 1):
        result += (-1)**n * n / (n + 1)
    return result

def sum_S2(N):
    """计算第二种形式的级数和：两项求和相减
    S_N^(2) = -sum_{n=1}^N (2n-1)/(2n) + sum_{n=1}^N (2n)/(2n+1)
    """
    sum1 = sum2 = 0.0
    for n in range(1, N + 1):
        sum1 += (2*n - 1) / (2*n)
        sum2 += (2*n) / (2*n + 1)
    return -sum1 + sum2

def sum_S3(N):
    """计算第三种形式的级数和：直接求和
    S_N^(3) = sum_{n=1}^N 1/(2n(2n+1))
    """
    result = 0.0
    for n in range(1, N + 1):
        result += 1.0 / (2*n * (2*n + 1))
    return result

def calculate_relative_errors(N_values):
    """计算相对误差"""
    err1 = []
    err2 = []
    
    for N in N_values:
        s1 = sum_S1(N)
        s2 = sum_S2(N)
        s3 = sum_S3(N)
        
        err1.append(abs((s1 - s3) / s3))
        err2.append(abs((s2 - s3) / s3))
    
    return err1, err2

def plot_errors(N_values, err1, err2):
    """绘制误差分析图"""
    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, err1, 'o-', label='S1 Error', alpha=0.7)
    plt.loglog(N_values, err2, 's--', label='S2 Error', alpha=0.7)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('N')
    plt.ylabel('Relative Error')
    plt.title('Relative Errors vs N')
    plt.legend()
    
    plt.savefig('series_sum_errors.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_results():
    """打印典型N值的计算结果"""
    N_values = [10, 100, 1000, 10000]
    
    print("\n计算结果:")
    print("N\tS1\t\tS2\t\tS3\t\tErr1\t\tErr2")
    print("-" * 80)
    
    for N in N_values:
        s1 = sum_S1(N)
        s2 = sum_S2(N)
        s3 = sum_S3(N)
        err1 = abs((s1 - s3) / s3)
        err2 = abs((s2 - s3) / s3)
        print(f"{N}\t{s1:.8f}\t{s2:.8f}\t{s3:.8f}\t{err1:.2e}\t{err2:.2e}")

def main():
    """主函数"""
    # 生成N值序列
    N_values = np.logspace(0, 4, 50, dtype=int)
    
    # 计算误差
    err1, err2 = calculate_relative_errors(N_values)
    
    # 打印结果
    print_results()
    
    # 绘制误差图
    plot_errors(N_values, err1, err2)

if __name__ == "__main__":
    main()