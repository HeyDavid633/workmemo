> 细化方向中的基础知识补充，此部分内容较多、系统、参考集中；故单开Note以记录
>
> 本Note几乎完全Reference 李沐 《Dive into DL》（动手学深度学习）D2L

+ 根目录Github：https://github.com/d2l-ai/d2l-en
  + UCB Course：Intro to DL 
  + 中文版（主要阅读）： https://zh-v2.d2l.ai/chapter_preface/index.html
  + 英文版（对比参考）：https://d2l.ai/chapter_introduction/index.html
    + 英文版涵盖更全，在后面有延伸章节
  + B站课程 抽重要的以思考：[李沐 -【动手学深度学习v2】](https://www.bilibili.com/video/BV1oX4y137bC/?spm_id_from=333.999.0.0&vd_source=04730b3493e6e060da6046ae9c007fd0)

<img src="https://s3.bmp.ovh/imgs/2023/12/05/b1c08acfe52cf407.png" style="zoom:50%;" />

[toc]

# 1. 基本概念

> 建立基本概念，一些常见的、高频的 关键词

+ 典型的训练过程：①设计模型 --- ②获取新数据 --- ③更新模型 --- ④检查模型效果（否则到②）

  + 通过用数据集来确定程序行为 --- *数据编程*（programming with data）

+ 关键组件

  + 数据 data   用来学习
    + example 会有一组 特征features；根据特征进行预测 --- 得到特殊的属性label
    + 每个样本的**特征类别数**都相同时，其**特征向量**是固定 长度--- 数据的**维数**（dimensionality）
  + 模型 model  如何转换数据：这里面其实有“算法”的概念
  + 目标函数objective function   量化模型效果
    + 也称为 损失、代价函数loss/cost function --- 常用“平方差 squared error”
  + 优化算法Algorithm    调整模型参数以优化目标函数的
    + 搜索最佳的参数组合，最小化损失函数，基本方法*梯度下降*（gradient descent）

+ 监督学习supervised learning **（feature - lable pair）**

  + **例子：**
    + 预测目标：心脏病发作 / 没有发作（生成模型，将任何的输入映射到标签 --- **预测**）
    + 观测的特征：心率、舒张压和收缩压
  + <img src="https://s3.bmp.ovh/imgs/2023/12/03/ab0a3515a306c82a.png" style="zoom:40%;" />
  + 问题分类：
    + 任何有关“有多少”的问题很可能就是**回归**问题
    + 是集中的哪一个的问题：**分类**
    + 不互相排斥的类别问题：多标签分类 --- **标记** 问题
    + 推荐系统 / 序列学习

+ 无监督学习 unsupervised learning

+ 强化学习 reinforcement learning：强调如何基于**环境**而行动



# 2. 预备Preliminaries

> 主要关于使用python来进行一系列数据操作

1. 数据操作：n维数组 -- 张量**tensor**，同numpy中的ndarray；数据切片操作 - 同python
2. 数据预处理Preprocessing：**pandas**，读取数据集、处理缺失值-用插值法插入均值、转换为tensor格式
3. 线性代数Linear Algebra：
   + scalar标量（常量）、variable变量
   + vector 向量：其中的标量值为元素element、分量component
   + Matrix矩阵：向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶
   + Tensor张量：*任意数量轴的张量*
     + 图像以$n$维数组形式出现， 其中3个轴对应于高度、宽度，以及一个*通道*（**channel**）轴
     + 对于图像而言，channel轴所表达的其实就是 红、绿、蓝 其中的一个
     + **Reduction 降维**，求元素和 $\sum_{i=0}^d{x_i}$ ,对应还有 非降维求和Non-reduction sum
   + **范数** Norms
     + 向量的*范数*是表示一个向量有多大。 这里考虑的*大小*（size）不涉及维度，而是分量的大小
     + $L_2$范数中常常省略下标2，即$\|\mathbf{x}\|$等同$\|\mathbf{x}\|_2$ = 向量元素平方和的平方根$\|\mathbf{x}\|_{2}=\sqrt{\sum_{i=1}^{n} x_{i}^{2}}$
     + $L_1$范数，向量元素的绝对值之和$\|\mathbf{x}\|_{1}=\sum_{i=1}^{n}\left|x_{i}\right| $
     + *Frobenius范数* ，**矩阵元素**平方和的平方根；可以理解为矩阵形向量的$L_2$范数
4. 微积分 Calculus
   +  ①*积分*（integral calculus）--- 逼近法 ②*微分*（differential calculus）
   + 导数和微分Derivatives and Differentiation
   + 偏导数 Partial Derivatives
   + 梯度 Gradients：函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量$\nabla_{\mathbf{x}} f(\mathbf{x})=\left[\frac{\partial f(\mathbf{x})}{\partial x_{1}}, \frac{\partial f(\mathbf{x})}{\partial x_{2}}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_{n}}\right]$
5. 自动求导 automatic differentiation
   + 自动微分使系统能够随后**反向传播梯度**
   + *反向传播*（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数



# 3. 线性神经网络

> Linear Neural Networks for Regression
>
> 如果初步上手，实际上我理想中的就是看这一章节，包含了“想象中的”全流程：
>
> 定义简单的神经网络架构、数据处理、指定损失函数和如何训练模型

## 举例 

> 根据房屋的面积（平方英尺）和房龄（年）来估算房屋价格（美元）

+ ***特征*（feature）**：自变量（面积和房龄）
+ **标签、目标（target）**： 房屋价格
+ 初步建模结果：$\text { price }=w_{\text {area }} \cdot \text { area }+w_{\text {age }} \cdot \text { age }+b$
+ 其中 度量模型质量的方式 --- **损失函数** $L(\mathbf{w}, b)=\frac{1}{n} \sum_{i=1}^{n} l^{(i)}(\mathbf{w}, b)=\frac{1}{n} \sum_{i=1}^{n} \frac{1}{2}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b-y^{(i)}\right)^{2}$
+ 备注：线性回归的结可以用一个简单的公式表达：解析解 analytical solution

## 线性回归 - SGD

**随机梯度下降** Stochastic Gradient Descent

+ 计算损失函数（数据集中所有样本的损失均值） 关于模型参数的**梯度** $\nabla_{\mathbf{x}}f(\mathbf{x})$
+ **小批量**随机梯度下降：在每次需要计算更新的时候随机抽取一小批样本（从而不需要在每次更新参数前遍历整个数据集）
  + 两个需要预先设定的量 **超参数**hyperparameter ①小批量中的样本数 **batch size**   ②学习率 $\eta$  **learning rate**
  + 所谓调参实际上就是 **选择超参数**的过程 --- 依据训练迭代结果来调整
  + 小批量 的抽取：所以在读取数据集的时候需要一定程度的打乱 ----- 连续地获得不同的小批量，直至遍历完整个数据集

+ 一个直接的例子：定义1️⃣基本模型（线性回归：wx+b）2️⃣损失函数 3️⃣优化算法（SGD）

## Softmax - 分类问题

希望解决“哪一个”的问题，得到的值应该是“概率”

+ 线性层的输出，不能直接作为预测值：①没有限制输出的值（概率）总和为1 ②这里的值可以是负数
+ Softmax的含义：将未规范化的预测变换为非负数并且总和为1
  + $\hat{y}_{j}=\frac{\exp \left(o_{j}\right)}{\sum_{k} \exp \left(o_{k}\right)} $      （显然这**不是**线性函数）
  + 分母的部分：规范化常数（*配分函数*）
  + 对每一项求幂保证非负，**分母**是所有求幂后的 
  + softmax不是线性函数，但因为输出由输入特征的仿射变换决定，所以softmax回归还是线性模型
+ cross-entropy loss 交叉熵 ：衡量两个概率分布之间差异的度量



# 4. 多层感知机 MLP

Multilayer Perceptrons：为了克服**线性**模型的限制，加入一个或多个（带了激活函数的）**隐藏层**

使用**隐藏层**和**激活函数**来得到**非线性模型**

## 关键概念

+ **前向传播** forward propagation：按顺序（输入层 ---> 输出层）计算和存储神经网络中**每层的结果**
+ **反向传播** backward propagation：计算神经网络参数**梯度**的方法
  + 二者关联：沿着依赖的方向**遍历**计算图并计算其路径上的所有变量，把这些变量用于反向传播，计算顺序与前向的相反
  + 初始化模型参数后， **交替使用**前向传播和反向传播，利用反向传播给出的梯度来更新模型参数
  + 反向传播重复利用前向传播中存储的中间值，所以训练比起预测更吃内存（显存）
    + **注意**：中间值的大小与网络层的数量和批量（**batch size**）大致成**正比**，正比这点非常关键；也是爆显存的原因
+ **梯度爆炸**gradient exploding：参数更新过大，破坏了模型的稳定收敛
+ **梯度消失**gradient vanishing：参数更新过小，每次更新的幅度太小

## 激活函数$\sigma$

+ 主要主要是为了 防止模型层数的**塌陷**
  + 中间的层 必须有激活函数，否则层数就 -1了
  + 最后的输出层就没有 因为是末尾了，不需要防止他塌陷

+ 隐藏层的输出应用，克服线性的限制 $\begin{aligned}
  \mathbf{H} & =\sigma\left(\mathbf{X} \mathbf{W}^{(1)}+\mathbf{b}^{(1)}\right) \\
  \mathbf{O} & =\mathbf{H} \mathbf{W}^{(2)}+\mathbf{b}^{(2)} .
  \end{aligned}$   所以大多激活函数都是非线性的
+ 不仅是 按行操作的（row-wise）更是逐元素操作（element-wise）
+ 通过计算加权和并加上偏置来确定神经元是否应该被激活
  + ReLU（Rectified linear unit，*ReLU*）去除了所有负值   $\mathbf{ReLU}(x) = \mathbf{max}(0,x)$
    + 减缓了梯度消失问题，求导表现特别好
  + Sigmod，挤压函数squashing function，
    + 将输入变换为区间(0, 1)上的输出 $sigmod(x) = \frac{1}{1+\mathbf{exp}(-x)}$
    + 常作为二元分类问题的概率，是平滑的、可微的阈值单元近似
  + tanh 双曲正切
    + 函数形状上类似sigmod，但关于0 **中心对称**：将函数挤压到（-1，1）之间

## 模型选择

+ **过拟合**（overfitting） ---- 对抗过拟合的技术：**正则化**（regularization）- 权重衰减 / 启发式的技术  / 暂退法dropout
+ 两个误差 辨别 --- 衡量过拟合与欠拟合
  + 训练误差（training error）模型在**训练集**上得到误差
  + 泛化误差（generalization error）模型应用在 从原始样本抽取的**无限多**的样本上的误差期望
+ 当我们在说 **模型选择** 时：
  + 比较的模型本质上完全不同：决策树 / 线性模型
  + 不同超参数设置下的同一类模型：决定隐藏层数量、激活函数
  + 评定方式上：
    + 划定专门的 验证集（validation set） --- 训练/验证 9/1开、8/2开
    + K折交叉验证：（在训练数据实在稀缺的情况下）原始数据分成K个子集， 然后执行K次模型训练和验证，每次在K−1个子集上进行训练。最后对这K次的结果取平均值

## 对抗过拟合

+ 权重衰减（weight decay）$L_{2}$ **正则化**：在训练集的损失函数中加入惩罚项 $L(\mathbf{w},b) + \frac{\lambda}{2}\|\mathbf{w}\|^2$
+ 暂退法**dropout**：
  + 在前向传播过程中，计算每一内部层的同时注入噪声，面上看是在训练过程中**丢弃（drop out）**一些神经元
  + 实现：在整个训练过程的每一次迭代中，在计算下一层之前将 当前层中的一些节点**置零**（丢弃的效果）

### 另：

+ **参数初始化**：好的参数初始化可以解决以上问题：过拟合、梯度爆炸/消失 --- *是一整个研究领域*
  + 默认初始化：正态分布
  + 新的初始化方法：Xavier初始化、包括专门用于参数绑定（共享）、超分辨率、序列模型和其他情况的**启发式**算法。

+ 分布偏移：训练集和测试集并不来自同一个**分布**（feature --- label的关系不一样）



# 5. 深度学习网络

+ 块block ：比层layers大，但是比整个模型小的单位；在编程上使用class（类）来表示





# 6. 注意力机制anttention
