# 实验项目 1：探索混沌边界：曼德博集合 (Mandelbrot Set) 与朱利亚集合 (Julia Sets)

## 项目说明

**分形概念：** 曼德博集合、朱利亚集合、复数迭代、逃逸时间算法。

**物理连接：** 非线性动力系统、混沌理论的边界、相空间结构、稳定性分析。曼德博集合可以看作是二次映射 $f_c(z) = z^2 + c$ 的参数空间中，使得原点轨道保持有界的复数 $c$ 的集合。

**计算任务：**
1. 实现逃逸时间算法来生成曼德博集合图像。对复平面上的每个点 $c$，迭代 $z_{n+1} = z_n^2 + c$ (从 $z_0 = 0$ 开始)，判断迭代序列 $|z_n|$ 是否保持有界（例如，是否一直小于 2）。
2. 根据迭代次数（逃逸时间）对点进行着色，可视化曼德博集合。
3. 选择几个曼德博集合内部或边界上的点 $c$，生成对应的朱利亚集合（此时 $c$ 固定，迭代 $z_{n+1} = z_n^2 + c$ 从不同的初始 $z_0$ 开始）。
4. 实现图像缩放功能，探索曼德博集合边界的无限复杂细节。

**分析与可视化：** 生成高质量的曼德博集合和朱利亚集合图像。观察曼德博集合的自相似结构（不同尺度下的微型曼德博集合）。比较不同 $c$ 值对应的朱利亚集合的连通性（当 $c$ 在曼德博集合内部时，朱利亚集合是连通的；外部则为康托尘埃）。

**学习目标：** 理解复数迭代和动力系统稳定性，掌握逃逸时间算法，练习二维数据处理和可视化，体会简单规则生成复杂结构的魅力。

**可能扩展：** 探索其他迭代函数（如 $z^3+c$）生成的类似分形；研究曼德博集合不同“芽”和“触角”的结构；优化计算速度（如检测周期性）。