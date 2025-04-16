# 计算物理实验 - 第八周：深入探究数值计算中的误差与精度

## 简介

欢迎参与本周的计算物理实验！本周我们将深入探讨数值计算的核心挑战之一：**误差与精度**。理论上的完美数学公式在计算机的有限精度世界中执行时，会不可避免地引入各种误差，如**截断误差**（由算法近似引起）和**舍入误差**（由有限的数字表示引起）。这些误差的累积和放大可能导致计算结果严重偏离真实值，甚至产生完全错误的结论——这就是误差的“危害”。本周的系列实验旨在通过具体案例，让大家亲身体验这些现象，理解其背后的原理，并学习评估和控制误差的方法。

## 学习目标

完成本周实验后，你应该能够：

1.  识别和区分截断误差与舍入误差的主要来源。
2.  理解并演示灾难性抵消 (Catastrophic Cancellation) 如何导致精度急剧下降。
3.  分析数值微分中步长选择对总误差的影响，找到最佳平衡点。
4.  比较不同数值积分方法的收敛速度和精度。
5.  认识到简单求和顺序的改变如何影响舍入误差的累积。
6.  理解数学上等价的表达式在数值计算中可能具有截然不同的稳定性和精度。
7.  体验递推关系中的数值不稳定性，并了解稳定计算的策略。

## 实验内容

本周包含以下六个实验，每个实验都聚焦于误差与精度的一个特定方面：

1.  **实验一：二次方程求根的稳定性 (Exp1_QuadraticRoots)**
    *   **核心问题:** 灾难性抵消。
    *   **任务:** 实现并比较标准求根公式与数值稳定公式在特定情况下的精度差异，特别是在单精度和双精度下。

2.  **实验二：数值微分的误差权衡 (Exp2_NumericalDifferentiation)**
    *   **核心问题:** 截断误差 vs. 舍入误差。
    *   **任务:** 使用有限差分法计算导数，并通过绘制误差-步长关系图（log-log），观察并解释误差随步长变化的 V 形曲线。

3.  **实验三：数值积分的收敛性 (Exp3_NumericalIntegration)**
    *   **核心问题:** 截断误差与算法阶数。
    *   **任务:** 实现梯形法则和辛普森法则，计算已知定积分，绘制误差-步长关系图（log-log），验证不同方法的收敛阶数。

4.  **实验四：调和级数求和顺序 (Exp4_HarmonicSum)**
    *   **核心问题:** 舍入误差累积。
    *   **任务:** 计算 $\sum_{n=1}^N \frac{1}{n}$ 和 $\sum_{n=N}^1 \frac{1}{n}$，比较两者在大 N 和不同精度下的结果差异，解释原因。

5.  **实验五：不同形式级数的比较 (Exp5_SeriesComparison)**
    *   **核心问题:** 数学等价与数值稳定性。
    *   **任务:** 计算三个数学上相关的级数 $S_N^{(1)}, S_N^{(2)}, S_N^{(3)}$，比较它们的数值结果和稳定性，理解表达式形式对计算精度的影响。

6.  **实验六：贝塞尔函数递推的不稳定性 (Exp6_BesselRecursion)**
    *   **核心问题:** 递推算法的数值稳定性。
    *   **任务:** 实现球贝塞尔函数的向上和向下递推关系，观察向上递推在特定条件下的发散现象，并验证向下递推的稳定性。

## 操作指南

## 操作指南

1.  **接受作业:** 通过 GitHub Classroom 提供的链接接受本次作业，这将为你创建一个私有的代码仓库。
2.  **克隆仓库:** 将你的作业仓库克隆到本地计算机。
    ```bash
    git clone <your-repository-url>
    cd ComputationalPhysics_Errors_Assignment
    ```
3.  **环境设置:** 确保你安装了所需的 Python 库。建议使用虚拟环境。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    pip install -r requirements.txt
    ```
4.  **完成实验:**
    *   仔细阅读每个实验目录下的 `项目说明.md` 文件，理解实验要求。
    *   在对应的 `.py` 文件（学生代码模板）中填充代码。模板中通常会标出 `### YOUR CODE HERE ###` 或类似标记。
    *   运行你的代码，生成结果（可能需要绘图或打印数值）。
    *   (可选) 在本地运行测试代码，检查你的实现是否基本正确：
        ```bash
        pip install pytest
        pytest Exp1_CatastrophicCancellation/tests/
        pytest Exp2_NumericalDifferentiation/tests/
        pytest Exp3_ODE_Stability/tests/
        ```
5.  **撰写实验报告:** 根据每个实验 `项目说明.md` 中的要求，撰写实验报告。报告可以是一个 Markdown 文件 (`实验报告.md`) 或 PDF 文件，放在仓库的根目录下或每个实验目录下。报告应包含：实验目的、方法简述、代码关键部分（如果需要）、结果（表格、图像）、误差分析、讨论和结论。
6.  **提交作业:**
    *   将你修改过的代码文件 (`.py`) 和实验报告文件添加到 Git暂存区：
        ```bash
        git add Exp*/ *.py
        git add REPORT.md  # 或者其他报告文件
        ```
    *   提交更改：
        ```bash
        git commit -m "完成实验内容"
        ```
    *   推送到 GitHub：
        ```bash
        git push origin main
        ```
    *   确保在截止日期前完成推送。GitHub Classroom 会自动记录你最后一次推送的版本。

## 评分

*   **自动评分 (Autograding):** GitHub Classroom 会自动运行每个实验 `tests/` 目录下的测试代码。这些测试主要检查你的函数是否返回了预期范围内的数值结果。自动评分结果会直接显示在 GitHub Classroom 界面上。
*   **手动评分:** 教师/助教将审阅你的代码逻辑、代码风格、实验报告的完整性、分析的深度和结论的合理性。最终成绩将结合自动评分和手动评分。

## 截止日期

**[在此处填写具体的截止日期和时间]**

## 资源

*   课堂笔记/讲义中关于数值误差、数值微分、ODE求解的部分。
*   NumPy 官方文档: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
*   Matplotlib 官方文档: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
*   浮点数运算参考: What Every Computer Scientist Should Know About Floating-Point Arithmetic ([https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html))

祝你实验顺利，深入理解数值计算的奥秘！

## 目录结构
```
cp2025-error-precision/
├── README.md                 # 项目总说明
├── requirements.txt          # Python依赖包
│
├── Exp1_QuadraticRoots/      # 实验一：二次方程求根
│   ├── 实验说明.md           # 实验说明文档
│   ├── 实验报告.md           # 实验报告文档（需完成）
│   ├── quadratic_solver.py   # 学生代码模板（需实现）
│   ├── solution/             # 参考答案
│   │   └── quadratic_solver.py
│   └── tests/                # 单元测试
│       └── test_quadratic.py
│
├── Exp2_NumericalDifferentiation/  # 实验二：数值微分
│   ├── 实验说明.md
│   ├── 实验报告.md           # （需完成）
│   ├── differentiation.py    # （需实现）
│   ├── solution/
│   │   └── differentiation_solution.py
│   └── tests/
│       └── test_differentiation.py
│
├── Exp3_NumericalIntegration/     # 实验三：数值积分
│   ├── 实验说明.md
│   ├── 实验报告.md           # （需完成）
│   ├── integration.py         # （需实现）
│   ├── solution/
│   │   └── integration_solution.py
│   └── tests/
│       └── test_integration.py
│
├── Exp4_HarmonicSum/             # 实验四：调和级数
│   ├── 实验说明.md
│   ├── 实验报告.md           # （需完成）
│   ├── harmonic_sum.py       # （需实现）
│   ├── solution/
│   │   └── harmonic_sum_solution.py
│   └── tests/
│       └── test_harmonic_sum.py
│
├── Exp5_SeriesComparison/        # 实验五：级数比较
│   ├── 实验说明.md
│   ├── 实验报告.md           # （需完成）
│   ├── series_sum.py         # （需实现）
│   ├── solution/
│   │   └── series_sum_solution.py
│   └── tests/
│       └── test_series_sum.py
│
└── Exp6_BesselRecursion/         # 实验六：贝塞尔函数
├── 实验说明.md
├── 实验报告.md           # （需完成）
├── bessel_recursion.py   # （需实现）
├── solution/
│   └── bessel_recursion_solution.py
└── tests/
    └── test_bessel_recursion.py
```