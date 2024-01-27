>  paper阅读笔记

***【SC’ 07】Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore Platforms***

+ 在几个多核CPU平台上对比实现了SpMV
  + TLB的策略上 选择
  + 循环上做了优化，自动调整框架，软件预取
  + 矩阵非0元结构 有影响

+ 说明：
  + 是科学计算常用的，高频率 瓶颈
  + SpMV对于多核处理器的峰值性能使用率低
  + 稀疏kernel有更高的 指令和存储开销 -- 无论哪种内存方式下

----------



***【SC09】Implementing Sparse Matrix-Vector Multiplication on Troughoutput-Oriented Processors***

+ 在GPU上部署SpMV  
  + 充分利用大带宽、细化到每个线程对于行列的处理
  + 提出了新格式：HYB = COO+ELL  ,这个格式效果最好

-------------------



***【FPGA14】a-scalable-sparse-matrix-vector-multiplication-kernel-for-energy-efficient-sparse-blas-on-fpgas***

+ FPGA实现了CSC格式稀疏 的SpMV
  + 把不规律的随机内存转换为了规则的串行内存

---



~~***【PARCO‘ 01】Towards a fast parallel sparse symmetric matrix-vector multiplication***~~

+ 对称矩阵 稀疏矩阵-向量乘
  + 软件层面作 流水线
  + 矩阵重排序---利用带宽
  + 寄存器 分块
  + 消息传递实现的并行
+ 时间太早，所用的是多CPU，有了GPU以后这个效果不用参考了

----------



***大数据驱动的稀疏矩阵测试问题研究*** --- 慧敏 开题报告

**1 稀疏矩阵 数据集**

不同计算，所用物理意义不同 则稀疏矩阵不同，依据矩阵特征取到的有效集合 **SuiteSparse**【TOMS‘ 11】 ---2789个

**2 矩阵特征**

1. 行、列数 ，非0元数量
2. 非零元分布：非零元密度，平均每行（列）非零元的数量，第i行（列）非零元数量，非零元的最大最小值、标准差

5个不同架构GPU测试性能，矩阵特征---性能图



----



***【FPGA' 22】High-Performance Sparse Linear Algebra on HBM-Equipped FPGAs Using HLS: A Case Study on SpMV***

在支持HBM的FPGA上的有效SpMV，



+ 用到了SpMV、SpMM、SpMSpV算子的 应用：线性系统求解器、图处理、压缩transformer的推理、PageRank

+ FPGA的优势：功耗、定制内存层次结构和计算引擎以细粒度并行
+ Xilinx vitis上已有的稀疏库VSL ，已有SpMV。作为对比项存在
  + 利用组合仲裁器解决bank冲突，它有2个问题，限制了片内buffer的使用
    1. PE和共享bank数量在高频是限制为4个
    2. VSL需要添加来自不同PE的结果，**而不是级联**，遵循部分缓冲区方法处理浮点



+ 选择kernel SpMV的原因：

  + 2种主要的访存模式：（1）稀疏矩阵的流式访问：无数据重用，需要高片外内存带宽（2）稠密向量 随机访问：数据重用，要求有效使用片内缓冲区
  + 和其他稀疏算子一样，==不规则的计算模式**？？？**==
  + 方便扩展SpMV到SpMM、SpMSpV

  

+ CPU与GPU基线，选择MKL与CuSparse为准 --- 都是库而存在

  + FPGA基线Vitis VSL，但==无法 运行大矩阵==？？？
  + FPGA部署平台Xilinx Alveo U280: 16 HBM通道 + 2 DDR 通道 = 总共提供 268 GB/s 带宽
  + CPU平台 Intel Xeon Gold 6242, 384 GB DDR4内存，282 GB/s带宽
  + GPU平台Nvidia 1080Ti，11 GB GDDR5X， 484GB/s带宽



+ **评价指标**
  + 吞吐量：每秒多少Giga个操作（GOPS），乘法与加法都被认为是 2个ops
  + 带宽效率：**单位带宽的吞吐量**--- MOPS/GBPS
  + 能效：**单位功率的吞吐量** --- GOPS/W

----



***【FPGA‘ 21】ThunderGP: HLS-based Graph Processing Framework on FPGAs***

Chen X, Tan H, Chen Y, et al. ThunderGP: HLS-based graph processing framework on fpgas[C]//The 2021 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays. 2021: 69-80.

+ 基于FPGA的开源HLS图处理框架ThunderGP，即一个可以用HLS实现的图处理平台，运用OpenCL。
  + 给到的是API和工作流

+ 作为图处理框架，能快速建模，对于以各种图为数据集的应用(SpMV)上运行

说法：

+ FPGA具有细粒度并行性、能效和可重构性，适宜于处理当前旺盛的这些需求下所广泛应用的 稀疏算子

--------------



***【TPDS‘ 19】HitGraph: High-throughput Graph Processing Framework on FPGA***

搞出出来了一种 基于边缘中心范式加速图形处理的FPGA框架HitGraph，并且设计了HLS工作流



GPU相对于FPGA 处理图计算问题 时的一些缺点：

+ 内存访问粒度太大，以至于浪费了外部内存带宽（存取一整个cacheline的数据，但只想要其中的一部分）
+ 图算法的时间和空间数据局部性差，片上内存使用率不足
+ 原子操作的开销大：内存锁以防止竞争

换句话说，FPGA的优势：

+ FPGA主要还是能效比高，可以提供计算密集型任务的定制加速
+ 可控的片内存储器资源和密集可编程原件，以克服上述GPU存在的问题



------

***A Survey of Coarse-Grained Reconfigurable Architecture and Design: Taxonomy, Challenges, and Applications***

> 粗粒度可重构体系结构与设计综述：分类、挑战和应用   
>
> 清华大学 刘雷波



**注**：此论文还有一中文的版本：魏少军、刘雷波等《中国科学：信息科学》-可重构计算

此中文版本能较好的初步理解



+ 概览：CGRA的主要优点：接近于ASIC的能效和性能 ，有制造后的类似软件的可编程性

#### 1. CGRA定义

+ 特定于领域的灵活性，在特定功能和固定功能之间有一定的灵活性
+ ==空间与时间计算相结合：spatial and Temporal，==
  + 空间：利用并行资源和数据传输通道计算，FPGA，ASIC。提高区域效率
  + 时间：利用时间复用资源，GPP。避免了深度流水和集中通信的开销
+ 配置 或 数据流驱动的执行：通过互连利用高效的显式数据流执行，完全放弃了控制流执行 --- 遵循生产者消费者  数据依赖关系； 执行风格上进一步支持 显式的数据通信
+ **缺点**：可编程性：高级语言的自动编译很难；适应性，整合成本高；涉及复杂控制流性能就严重下降

#### 2. CGRA分类法 

依据3个抽象：编程模型对底层系统的抽象（**线程**）、计算模型对计算语义的抽象、执行模型对微体系结构的抽象（指令的触发配置等），**在4个维度上进行分类**

+ 编程模型：命令编程模型和命令语言（C\C++）、并行编程模型（可以表达并行性 CUDA）、透明编程---动态编译、依赖硬件

+ 计算模型，（基于配置的分类）：（**注 所有CGRA都是MIMD的**）SCSD、SCMD、MCMD

+ 执行模型，（**配置的调度**-从内存中取出配置并映射到硬件；配置的执行，单个配置内执行操作）的分类：静态调度顺序执行SSE、SSD、DDD、DSD

+ Micro-arch模型 ，过于琐碎

  ![classifcation CGRA](C:\Users\dWX1205369\Pictures\CGRA_related\classifcation CGRA.PNG)

  

#### 3.CGRA应用状态

+ 因为牺牲了 细粒度的灵活性和互连灵活性，所以用于特定领域的加速器DSA。区别于FPGA平台，其用于原型开发和演示平台。
+ 给得到的计算资源多，控制流少，所以用于计算密集、数据移动频繁的操作：安全、信号和图像处理、深度学习

####  CGRA的挑战

1. 编程 结构：平衡编程效率与性能
   + 硬件角度：体系结构上的复杂（要调度二维硬件资源阵列、任务分区）
   + 软件角度：模型需要暴露出更多的细节以使得这些资源可以被探索，需要充分暴露但又隔离硬件细节，成功例子CPCPU --- **CUDA**
   + 高级语言 --- 编译器优化的结果 与 手动优化的结果还是有很大差距，又意味着要暴露更多细节
2. 计算模型：计算但有限的并行度
   + 整体：希望包含多个层次的并行（指令TLP、数据DLP、任务TLP、内存MLP、推测并行），但支持更多层次就要导致面积和功率的开销
   + CGRA是**支持时间计算的空间体系结构**，CGRA可以在每个周期重新配置PE和片上网络，从而导致时空映射。难以实施推测并行：开销大、错误代价高 、buffer影响。
3. 虚拟化
   + CGRA体系结构的设计的多样与差异，不可能提取同一的硬件抽象
   + 主要依赖静态编译、难以动态调度执行顺序和同步通信已经解决的配置
4. 内存效率
   + 现有的方案：缓存、暂存 专为GPP开发，没有针对不同访问模式优化
     1. 作为空间计算结构：导致分散和冗余的内存地址
     2. 没有针对应用优化内存访问模式，无法再运行或编译时 探索内存并行性MLP
     3. CGRA适合PIM的集成，但PE的逻辑更复杂，工艺上集成困难

![CGRA challenge and trends](C:\Users\dWX1205369\Pictures\CGRA_related\CGRA challenge and trends.PNG)

#### 4.先进的CGRA与趋势

1. 趋势1：编程驱动的架构设计
   + **对应用程序进行分析，识别热点区域，用该区域引导分区。 模拟工具和编译输出 评估性能---迭代，以更适宜于目标应用程序** ，面向应用程序的设计 追求极致的（在该领域上的）性能、忽略了模型效率和编程难度
   + 在设计CGRA的时候同时考虑 编程模型
2. 趋势2：多级的并行计算 --- 计算模型层面
   + CGRA ILP主要在在于面积和功率的开销
   + DLP：指SIMD计算，CGRA提供API来整合内存访问并避免冗余地址生成、**为数据重用提供高效的内存**，可以更高效率的实现DLP
   + TLP：集成多个集中式的CGRA从而形成一个更大的CGRA，实现粗粒度TLP。数据流实现细粒度TLP
   + 主要的实现的可能性 - **推测并行**：预测不正确代价太大，任务级推测TLS、块级预测
3. 趋势3：虚拟化  --- 集成到异构计算平台和操作系统
   + 从FPGA的虚拟化中提取灵感 ：抽象和标准化
     + 方向：overlay覆盖---抽象FPGA的低级细节、提供 虚拟硬件进程、标准化
   + 前瞻方向：标准化的API、CGRA的叠加、虚拟硬件进程（在CGRA上更有优势）
   + CGRA的执行：从CPU卸载下来、专做 计算密集型和数据密集型任务
   + CGRA资源调度：基于的配置级别有梯度（PE阵列、PE线路、PE）
     + 目前CGRA虚拟化技术主要针对静态编译，而针对操作系统和动态资源管理的CGRA虚拟化即将到来

4. 趋势4：高效的内存系统
   + 压缩技术：用压缩算法离线压缩发送到CGRA上的二进制文件
   + 集成电路工艺 扩大主存带宽：3D堆叠的DRAM--HBM
   + 消除（片外M和处理器间的）传输
     + 放更多内存在最为缓存集成到片上（嵌入式DRAM）--- 但密度较低
     + 计算组件 集成到内存：PIM；c/d都是PIM的方式，b则更近似于近存计算
   + 碎片化和冗余内存访问与地址生成的问题
     + 设计矢量化或流式并行内存接口
     + 动态CGRA框架、编译时调整内存的访问模式
   + 总体而言这些内存方面的改进方向:
     + 高效的内存接口提高分布式内存访问的规则性和并行性
     + 可编程内存管理单元提高内存接口的灵活性
     + 3D堆叠芯片技术提高CGRA和内存之间总线的带宽和延迟

![PIM](C:\Users\dWX1205369\Pictures\CGRA_related\PIM.PNG)

#### CGRA在体系结构和应用的未来和发展

+ 挑战 --- 趋势，取出重要的部分有前景的部分来：（1）声明性的编程语言（2）推测执行技术（3）虚拟化、统一的接口与协议（4）存算的融合（5）可编程的内存管理（6）数据流运行时的优化













+ 需参考论文：



---

***CGRA Survey***

<img src="C:\Users\dWX1205369\Pictures\CGRA_related\classic CGRA.PNG" alt="classic CGRA" style="zoom:80%;" />



典型的处理结构，包含一个2D处理元素阵列，能够执行基本的算术、逻辑、以及使用功能单元（FU）和小寄存器文件（RF）作为临时数据存储的字级存储器操作

CGRA交换矩阵可以在空间（受处理元件数量限制）和时间（受芯片上可存储的配置数量限制）

+ 与FPGA相比：（相比于其 **位级** 的）粒度更粗，则性能更高且功耗更低，配置的时间更短

+ 有限长度的配置序列 来执行循环，于是适合用来加速循环。
  + **编译器**利用CGRA的时空配置来加速应用程序执行。
  + 编译器负责从循环内核中提取尽可能多的**并行性**（**受数据依赖约束**）



1. 编译器把高级语言 处理为 数据流图DFG
2. DFG**映射**到CGRA上 （映射时有很多考量：质量/编译时间）

给一个例子：GEMM的数据流图，然后进行映射（简单映射的例子）

	<img src="C:\Users\dWX1205369\Pictures\CGRA_related\GEMM DFG.PNG" alt="GEMM DFG" style="zoom:67%;" /><img src="C:\Users\dWX1205369\AppData\Roaming\Typora\typora-user-images\image-20221108160235544.png" alt="image-20221108160235544" style="zoom: 50%;" />

+ 简单的分类：基本CGRA如最开始的图、异构CGRA（PE不尽相同）、空间CGRA、片上网络

+ 与CPU的耦合：有松耦合 有紧耦合。主机处理器负责运行非环路代码，配置CGRA，并启动从主存储器到CGRA本地存储器的DMA数据传输

#### **CGRA的编译**（映射）：

给定应用程序和CGRA体系结构的循环，编译：将循环**映射**到CGRA上（即，生成固定周期数的配置），以最大限度地提高吞吐量。

+ ##### 映射过程：放置 + 路由（placement and touting）

  + 放置决定哪个PE将执行每个操作

  + 路由确保数据可以及时路由到依赖操作

    <img src="C:\Users\dWX1205369\AppData\Roaming\Typora\typora-user-images\image-20221108162828017.png" alt="image-20221108162828017" style="zoom:80%;" />

    一个更加实际的例子。这里还是**抽象**映射，真正的映射是要包括每个PE上的详细路由配置

    + 上图中，映射有3个部分：（1）序幕prologue（2）稳态kernel（3）结语epilogue。1/3只执行一次，2稳态kernel被不断重复（迭代中的操作）
    + 其中的“II = 2”表示 kernel的调度长度 即 初始化间隔 Initial Interval (II)，即 迭代启动之间的周期数
    + 给定了DFG和CGRA以后，映射器mapper首先计算最小初始间隔MII  --- 通过遍历DFG

+ ##### 具体CGRA 的映射方法

  1. 启发式方法heuristics

     模拟退火、以边缘为中心的模调度、规划地点和路线(SPR)、List调度、进化算法、机器学习

  2. 数学优化 mathematical optimization, 

     整数线性规划、布尔可满足性（SAT）解法器

  3. 图论启发技术graph theory inspired techniques

     具体不表，主要利用图论的已有概念：子图同态、图表同态、图次要、兼容性图、图形

+ **其他编译相关问题**

  1. **数据访问**：CGRA本地内存库（memory bank）具有非统一的内存访问体系结构，其中只有PE的子集可以访问具有有限数量的读/写端口的内存库。所以内存限制使得实际设置中整体性能下降，编译器应该充分考虑这一点
     + 内存感知编译
     + 内存地址生成
  2. **嵌套循环的映射**
     + 基于多面体、循环扁平化、收缩映射
     + 有限配置内存下的嵌套循环映射
  3. 应用级映射
     + CPU和CGRA中间的分区
     + 同步数据流（SDF）
  4. 使用控制流处理环路
  5. 可扩展的 CGRA映射 --- 更多的节点、更大规模的阵列

-----------



***【IEEE Access‘ 20】A Survey on Coarse-Grained Reconfigurable Architectures From a Performance Perspective***

本文与我所做内容更加遥远，CGRA的发展脉络，当前比较先进的CGRA结构，以及CGRA器件作为一个整体拿出来与GPU、ASIC、FPGA相比，分析这**类**CGRA器件的性能，没有说要去做什么样的一件事。

对于CGRA参数指标的分析方法可参考：阵列size、制造工艺（? nm）、频率、功耗、数据宽度、面积、峰值性能、可达的峰值性能（%）



可重构的体系结构：FPGA、CGRA 被用来处理HPC问题

前者配置时间慢、编译时间长、（相对而言）时钟频率低

#### CGRA结构

当前比较先进3种CGRA体系结构

<img src="C:\Users\dWX1205369\Pictures\CGRA_related\nowdays CGRA.PNG" alt="nowdays CGRA"  />

+ ADRES，较为主流，这方面为基础的后续工作多。第一排RC扩展了编排执行的VLIW处理器的后端管道
+ TRIPS，紧耦合PE，直接进行相邻通信 降低延迟
+ Plasticine，大型的CGRA结构，通过专门的模式地址生成器来并行



CGRA主要指标的 整体进步趋势（图）

-----



***【BOOK】Blocks, Towards Energy-efficient, Coarse-grained Reconfigurable Architectures***

说的很细，页数225。对于他们的Blocks的论文展开叙述

具体结构甚多细节，待有需要时再看

----



换一个视角来看CGRA --- https://zhuanlan.zhihu.com/p/556608481

#### CGRA结构

（以 ADRES和MorphoSys 为例）

从逻辑连线的角度来看

关键概念：当只有第1行的IS、shared RF和Mem的时候，结构上就和VLIW处理器一致

此处的IS（Issue slot）其他材料也称 ALU、**FU**(Function unit)、**PE**(process element)、**RC**(reconfigurable cell)反正就是 进行数据运算的单元

**1.紧耦合**

<img src="C:\Users\dWX1205369\Pictures\CGRA_related\tight coupling CGRA.PNG" alt="tight coupling CGRA" style="zoom:67%;" />

所谓的紧耦合tight coupling就是在于 处理器和可重构整列 融为一体，如上图（ADRES）

在这里指出的 松与紧 耦合 都是指**CGRA与CPU**之间的耦合程度，无论松紧都不能脱离CPU

**2.松耦合**

松，即松在处理器部分和可编程阵列的分离



#### CGRA内部的连线方式

取舍：IS模块要和其他IS连接的越多 传输数据越灵活，但功耗和面积也更大

更加具体的在IS内部的结构（也可以设计）：涉及到数字电路，多路输出选择器、ALU、寄存器组等，不再关注进去

<img src="C:\Users\dWX1205369\Pictures\CGRA_related\CGRA connect line.PNG" alt="CGRA connect line" style="zoom:67%;" />





#### CGRA适合计算密集型应用的原因

将循环部分代码映射(mapping)到硬件上时，可以考虑从空间和时间维度上进行映射

两种映射方式都能够达到执行循环的目的，但a中计算单元的利用率是25%，而b中的利用率是100%

当ab同时映射时，即从空间和时间构成的3维 视角来进行的映射，可以在一个循环内执行不同任务

<img src="C:\Users\dWX1205369\Pictures\CGRA_related\CGRA mapping.PNG" alt="CGRA mapping" style="zoom:80%;" />

-----------



***【IPDPS' 06】Design flow for Optimizing Performance in Processor Systems with on-chip Coarse-Grain Reconfigurable Logic***

关于设计流程 映射流程，可能是我最需要参考的一篇！

#### 设计流程

1. 输入的源代码处 检测kernel的分析程序
2. 创建 中间表示 IR(Intermediate representation)
3. CGRA架构的映射算法   （**强调的部分 对性能影响大**）
4. 编译到微处理器上



#### 执行模型

Optimization优化的示例：

死代码消除， 公共子表达式消除、常量广播、循环转换

之后结合CGRA阵列的结构应用优化：此处更多是循环的展开和循环的标准化

![CGRA design flow model](C:\Users\dWX1205369\Pictures\CGRA_related\CGRA design flow model.PNG)

判断是不是一个kernel从而卸载到CGRA上：

人为设置阈值Threshold，通过分析的到的循环展开在整个程序中的指令数占比（如取10%）来确定



#### 抽象整体结构

处理器和CGRA执行互斥，程序call CGRA时在CGRA上加载了正确的配置，执行kernel；此时处理器进入空闲以降低功耗。 CGRA执行完以后，通知处理器并写回执行剩余程序所需的数据，之后CGRA自己保持空闲

所以**在这个模型**中：**CGRA和处理器不是并行的**

![CGRA implement arch](C:\Users\dWX1205369\Pictures\CGRA_related\CGRA implement arch.PNG)



-----



***粗粒度可重构处理器的结构研究与设计***  硕士毕业论文-上海交通大学 2012



评估结构的性能 --- 建模，三种建模的层次：

1. 事务级建模（Transaction Level Model，TLM）分为各个模块，周期级精确建模，速度快。SystemC、System Verilog
2. 功能级建模（Function Level Model，FLM）开发时间短，**测试是基于软件层面的**，不包含任何周期、时间信息，只验证功能正确与否。
3. 寄存器传输级建模（Register Transfer Level Model，RTL）**开发周期长，**硬件层面建模，工具Modelsim、VCS，语言VHDL、Verilog HDL，周期级精确建模。   可以通过EDA工具进行综合、布局布线。得到硬件实现面积、系统工作频率、关键路径 信息

软件仿真---运行算法实例，作为硬件结果的比对

算法**映射** 硬件仿真---得到仿真结果

----



【中国科学：计算科学】可重构计算: 软件可定义的计算引擎 --- 魏少军, 李兆石, 朱建峰, 刘雷波

**CGRA适合处理的工作：**（运算的特点）

1. 空域上可重构计算依靠处理单元间的  互联 显式实现依赖关系.

2. 时域上可重构计算功能单次配置, 多次执行

   1. 对于需要**重复**执行的功能, 配置信息中可以指定数据通路多次执行该功能  --- **循环**。 

      而对于其他冯诺依曼模型下的机器指令加载和执行耦合： 即便编译器发现一段指令需要重复执行, 它也无法在程序中省掉这段指令在第1次执行后的取指和译码等步骤.

   2. 对于连续变化的功能, 配置信息可以利用配置管理器使数据通路快速连续重构以满足功能要求

      即去处理那些串行的但是功能不一样的指令

*作为疑惑，待补充：关于数据读取形式、循环模式呢*



----

稠密BLAS调研：

优化手段 向量化、数据预取、编译优化、数据重排



----

***【HPCA' 20】SpArch: Efficient Architecture for Sparse Matrix Multiplication***



1. 设计一个合并机制：以管道化处理 部分矩阵的 乘合并阶段，产生后在片上合并了
2. （第一个输入矩阵A）为减少DRAM的访问：提出一种压缩矩阵表示-将部分矩阵的数量减少三个数量级
3. 为增加可扩展性：开发哈夫曼树调度器可处理更大矩阵，同时进一步减少DRAM访问
4. （输入矩阵B）新的表示需读取更多矩阵：使用行预取器和 （近乎最佳的）缓冲区替换策略

以上4点中，第一点更多是预处理，作为一种代价。后面三点提供了加速和减少DRAM访问

整个下来感觉他是大力出奇迹，哪怕付出较大的合并代价也要做压缩，后期再通过专门设计存储结构和专用调度来 补回来加速效果



实现手段：(0)C++构建了循环精确模拟器：实现的性能评估  (1)在verilog实现了阵列合并  (2)在台积电40nm库下来使用 Synops设计编译器 合成了 阵列合并（3）使用xsim RTL模拟器

预取器：（1）提前获取乘法器的数据 --- 隐藏DRAM延迟 ：多个数据提取器实现（2）获取到的数据存在buffer以便复用，片内buffer实现

问题：为什么要做这样所有的非零元之间两两比较，还多出来了一行与一列，含义是怎么样的

-----

***【HPCA‘ 21】SPAGHETTI: Streaming Accelerators for Highly Sparse GEMM on FPGAs***

给出了一张重要的表格，内积外积分别适合的加速 手段

![Spgemm acc](C:\Users\dWX1205369\Pictures\Sparse Kernel\Spgemm acc.PNG)



更密集的矩阵：内积（为不匹配的非零操作数引入了冗余输入获取），但内积与稀疏模式关系不大

高度稀疏的矩阵（<1%）：外积 （大量的部分积矩阵而导致输出局部性较差）适合高度稀疏的模式。外积 高输入数据重用的原因：不需要对冗余的非零元进行输入，适合高度稀疏

内积需要索引匹配，所以内积不好并行和重用这个输入数据，但是外积可以消除掉，所以可以防止内存停顿和最大限度的提高输入重用性。

对于合并阶段，外积因为需要合并大量的 部分积矩阵 来创建最终矩阵，所以破坏了并行性

![inner outer product](C:\Users\dWX1205369\Pictures\Sparse Kernel\inner outer product.PNG)





主要优势：

1. 流的方式静态调度输入：最大限度的提高了DRAM利用率（更少的访问）
2. 并行、带宽使用 ---（外积的特性）流水线处理乘法和合并阶段，并行化的合并，最大限度的利用DRAM带宽
3. 器件通用性好 --- 适用于各种FPGA器件，使用虚拟化的手段生成的加速器，但还是带宽限制

性能上在对角矩阵上性能不佳，非零元分布不规则的矩阵加速更高，但都比当时最先进的SpArch、OutSpace有所提升。对于平台的比较全面，而且代码是公开了的，可以适应于不同配置的FPGA设备、本质是一个硬件生成器和可部署原型（可以在亚马逊云上虚拟实现）

另外就是他这个工作完全是针对于SpArch来做的，没有这个工作可以说他就失去了核心改进点。



问题：模式感知的平铺 和 循环平铺

前者怎么就最大化了B的重用，可以知道前者是考虑了NNZ的负载均衡

----

***【MICRO' 21】SparseAdapt: Runtime Control for Sparse Linear Algebra on a Reconfigurable Accelerator***

串联的论文：CGRA结构上-Transmuter，运行时调度方面Oracle，Ideal Greed，对于内积外积描述的论文OuterSpace



1. 识别在稀疏线性代数运行时 硬件可重构设备 改变状态的时机。这个时机来自于（1）显性状态：代码变化（2）隐性状态：稀疏模式变化
2. 开发了一个框架：SparseAdapt，来适应于状态改变 并进行底层硬件可重构阵列的重构。这个框架集成在了Transmuter
3. 配套做了一套预测模型：包括了决策树、启发式的代价评估策略，使得硬件可重构阵列细粒度得调整。这个模型使用了机器学习训练出来的



+ ABSTRACT:
  + Dynamic adaptation calls for a high-accuracy, low overhead runtime mechanism for adaptation at a fine granularity ---- CGRAs(coarse-grained reconfigurable architectures)
  + SparseAdapt: a lightweight machine learning-based adaptive framework, based on another work called "Transmuter"
  + Evaluate on SpGEMM & SpMSpV get up to 2.9× in terms of energy-efficiency than other runtime reconfiguration
+ 1  INTRODUCTION
  + Sparse linear algebra operations: memory-bounded and thus bottlenecked on data movement rather than compute
  + Recent work: to balance **energy-efficiency & flexibility** then comes to CGRA
    + Real-world sparse datasets are seldom uniform : need pre-processing
    + Compile-time optimizations fail if the dataset evolves over time
  + SparseAdapt: an extension to the runtime of a CGRA Transmuter (their another work), work at two mode :
    + Best Energy-Efficient mode: GFLOPS/W
    + Best Power-Performance mode GFLOPS^3/W
  + Main contribution：
    + **Identifies** hardware reconfiguration opportunities associated with phase transitions during execution for sparse algebra routines
    + **Framework SparseAdapt** ：adapt to phase changes and reconfigure hardware configuration
    + **Predictive model** for hardware reconfiguration at fine granularities
    + **Demonstrates improvements** in energy-efficiency and performance 
+ 2  MOTIVATION AND RELATEDWORK
  + 2.1  Existence of Implicit and Explicit Phases
    + Example SpGEMM : opportunity for dynamic reconfiguration
      + explict phase change： multiply → merge
      + implicit phases : from computation of dense columns with corresponding dense rows
  + 2.2 RelatedWork
    + 2.2.1 Adaptation for Traditional Hardware: on CPU(with MLP model) and no vary implicit phases
    + 2.2.2 Adaptation for CGRA: dynamic scheduling and dataflow techniques but lack dynamic hardware reconfiguration on ex & implicit phase
    + 2.2.3 Comparison with ProfileAdapt: can not adapt implicat phase
+ 3 HARDWARE DESIGN
  + 3.1 Architectural Background
    + a brief overview of Transmuter：co-work with Host CPU, add feedback ,assume they share physical memory
  + 3.2 Configuration Parameters --- Table1
    + 3.2.1 Dynamic Voltage-Frequency Scaling (DVFS).
    + 3.2.2 Cache Capacity: balance perforamnce and power
    + 3.2.3 Sharing Mode: XCU incorporate  L1/L2 cache for shared or privated
    + 3.2.4 On-Chip Memory Type: cache and SPM(scratchpad memory)
    + 3.2.5 Prefetcher Aggressiveness.
  + 3.3 Performance Counters
    + both spatially (across all replicated hardware blocks) and temporally (normalized to the elapsed cycle count of the epoch) by the runtime
  + 3.4 Cost of Telemetry and Reconfiguration: 3 kinds of Grained 
+ 4 PREDICTIVE MODEL
  + by virtue of eliminating the back-andforth switch to the profiling configuration
  + 4.1 Model Construction
    1. Random Sampling
    2. Neighbor Evaluation
    3. Dimension Sweep
  + 4.2 Dataset Construction and Training
  + 4.3 Choice of Predictive Model 
    + decision trees(with pruning)
  + 4.4 Reconfiguration Cost-Aware Prediction
+ 5 EXPERIMENTAL SETUP
  + 5.1 Data Collection and Model Training
    + train a decision tree classifier for each of our configuration parameters
  + 5.2 System Modeling
    + A power estimator is constructed using a combination of RTL synthesis reports for crossbars
    + Reconfiguration Cost: 
      + super fine grained: 100 cycle
      + fine-grained: 100–961k cycles(L1 to L2), 100-122k cycles(L2 to main Mem)
  + 5.3 Comparison Points
    + non-reconfigurin: Baseline、Best Avg、Max Cfg、Ideal Static.
    + dynamically reconfigures： Ideal Greedy、Oracle
  + 5.4 Choice of Dataset and Parameters
+ 6 EVALUATION
  + 6.1 Comparison with Standard Configurations
  + 6.2 Comparison against Ideal and Oracle
  + 6.3 Analysis of Model and Features
  + 6.4 Comparison with ProfileAdapt Scheme
  + 6.5 Effect of Parameter Sweeps
+ 7 DISCUSSION
+ 8 CONCLUSION
+ A ARTIFACT APPENDIX  ---  to install docker and get the exp data localy

---

***【MICRO‘ 21】Distilling Bit-level Sparsity Parallelism for General Purpose Deep Learning Acceleration***

不关注个体权重的内部稀疏并行性。利用一系列权重表现的并行性加速DNN

（1）与位并⾏加速器相⽐，⽤于计算乘积的实际位不是 原始权重⽽是 **交错权重**？？

（2）串⾏化过程不限于 ⼀个权重，⽽是扩展到⼀系列交错的权重 以及每个独立的bit

问题：对于图4 中描述的“比特交织”，他所对其二进制点的方式就是找到最大的吗，这里的数值都比较好表达，那不好表达的数怎么办？这样移动的精度损失？



----

***【ISCA‘ 22】DIMMining: Pruning-Efficient and Parallel Graph Mining on Near-Memory-Computing***

问题：Fig7中的内存地址大小为什么是30bits，不是取到32bits这种数值？没有找到原因，同时这样设置是不是意味着他的可获取的地址大小是定死了的

---

***【MICRO‘ 21】Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture***

混合动态系数模式和可重构架构来加速稀疏自注意力模型

1. 软件部分：稀疏模式、高性能和负载均衡
2. 硬件：设计了可重构的脉动阵列

主要贡献：

1. 软硬件协同的设计框架：Sanger，通过可重构架构利用注意力动态系数性

2. 灵活的细粒度结构化修建技术，用低位计算 来预测稀疏注意力编码--保证负载均衡

3. 一种数据流和可重构架构，统一SDDMM与SPMM消除了了稀疏解码开销和内存传输开销

   



+ ABSTRACT

  + attention-based models work well， But need Sparse-attention
  + main include SDDMM、SpMM
  + this paper Sanger : software&Hardware co-design framework

+ 1 INTRODUCTION

  + brief Intro about the processing of Transformer. Q\K\V\S
  + 2 kind categories
    + static sparsity( limited computation saving)
    + dynamic sparsity(workload imbalance and poor data locality)
  + MAIN contribution
    + Sanger, a hardware and software co-design framework 
    +  a dynamic and fine-grained structured pruning technique with high flexibility and sparsity
    + a score-stationary dataflow and a reconfigurable architecture

+ 2 BACKGROUND AND MOTIVATION

  + 2.1 Primer on Attention
    + query (Q), key (K), value (V) with op as softmax and GELU
    + sparse score matrix is the output of Q ×K(SDDMM) and the input of S ×V(SpMM)
  + 2.2 Existing Sparse Attention Designs
    + See as in table1 & Table 4 divided into co-design & Software only
    + Software-hardware co-design:A3, SpAtten, FTRANS, Sanger
    + Software only:  so on, main coarse-grained structured and unstructured 

+ 3 SANGER OVERVIEW

  + Main about the Fig2 consists of (1)an attention pruning algorithm (2) a reconfigurable attention accelerator
  + software level: quantized to predict, encoding scheme to pack and split the attention mask
  +  hardware level: unified score-stationary dataflow that supports both SDDMM and SpMM operations on one systolic array

+ 4 SOFTWARE PRUNING FOR SPARSE ATTENTION

  + challenges:
    + dynamic sparsity cannot be pre-determined before inference
    + the softmax operation induces implicit sparsit
    + dynamic sparsity : highly unstructured patterns
  + 4.1 Predicting Attention Matrix
    + calculate the quantized prediction of the attention matrix S
    + simulate quantization during training 
    +  approximate gradients of the non-differentiable operator Q 
  + 4.2  Generating Attention Mask
    +  generate a binary attention mask M according to the sparsity pattern it exhibit
    +  threshold T : to trade-off between sparsity and accuracy
  + 4.3  Packing and Splitting Attention Mask
    + partitioning: into sub-matrices along the colum
    + packing: skipping the sub-row
    + splittin: overfull sub-row into multiple row

+ 5 HARDWARE DATAFLOW

  + 5.1 Dense Score Stationary Dataflow
    + 1 Q × K to get S matrix
    + 2 conducts an exponential
    + 3 S × V,reused systolic array 
    + 4 normalization in softmax function
  + 5.2 Sparse Score Stationary Dataflow
    + sparse scores in the same row are packed and then sent to one row of PEs
    + save area: unify the computation of SDDMM and SpMMs

+ 6 ARCHITECTURE OPTIMIZATION

  + (1 )enable the sparsity pattern with uncertain nonzero distribution. (2) SDDMM and SpMM operations differ in the data transfer and accumulation scheme of partial sums
  + 6.1 Reconfigurable Systolic Array Design
    +  decouple the data registers of them from the PE
    + registers are connected one by one
    + each PE is connected to all the registers in one row via a multiplexer controller

  + 6.2 Implementation Details
    + merge the results from the split query vector
    + Modules are fully-pipelined 
    + uses a look-up table 

+ 7 EXPERIMENTS

  + 7.1 Experimental Settings
    + Benchmarks: BRET \ GPT-2 \ GLUE
    + Software pruning implementation : NVida-BERT
    + Hardware implementation: Chisel  UMC 55nm
    + Platforms for comparison
  + 7.2 Pruning Results
  + 7.3 Comparison with CPUs and GPUs
  + 7.4 Comparison with Other Accelerators
  + 7.5 Pattern Visualization and Impact of Packing
  + 7.6 Impact of Binary Thresholding

+ 8 RELATED WORK

  + Sparsity in attention mechanism.
  + Deep learning sparse accelerators.

+ 9 CONCLUSION

+ A ARTIFACT APPENDIX --- privode whole code

---

***【HPCA‘ 22】Griffin: Rethinking Sparse Optimization for Deep Learning Architectures***

1. 稀疏架构的设计空间探索，封装了以前的⼯作并为每⼀类⽹络识别更有效的设计
2. 重⽤双稀疏架构中的逻辑以创建混合架构（Griffin）的技术 --- 稀疏架构模型
3. 根据以前的最佳作品评估，至少8%的性能优势

问题：III中 负载均衡的实现：文中说Fig1简单的排列就足够了 是什么意思呢？  他对于这个洗牌改组还进行了一些限制（4个连续的元素）是为什么？



---

***【ISCA‘ 20】A Multi-Neural Network Acceleration Architecture***

核心思路：通过匹配来⾃不同 ⽹络的计算和内存密集型任务并并⾏执⾏ ---- 充分利⽤加速器的计算资源和内存带宽,尽早驱逐⼤量分配 来最⼩化其⽚上内存容量需求

问题：Fig12中负载均衡的例子：MB中的块调度到CB中，他们的这个块长度是怎么判断的呢？并没有读出来他们的规律

----

***【HPCA‘ 22】CANDLES: Channel-Aware Novel Dataflow-Microarchitecture Co-Design for Low Energy Sparse Neural Network Acceleration***

低功耗稀疏神经网络加速的 通道感知先进数据流微架构协同设计

主要贡献：

1. 采⽤适宜于外积的压缩和适宜于内积的数据流与简单的cross bar开关实现**⾼效的内部连接**，同时绕过辅助索引匹配逻辑
2. 提出了 **累加缓冲区的 2 级组织**，在L1 中包含⼀组低功耗的寄存器，在L2中包含⼀个 6 KB 多组累加器缓冲区
3. 引⼊了**平铺像素优先 (TP) 压缩策略**，以促进部分和更新中的⾼时间局部性，从⽽提⾼ L1 命中率
4. 对跨 PE 的**不同⼯作分区**进⾏了实验，并确定了⽆需离线预处理即可实现⾼⽔平负载平衡的常规分区

综合以上，确定最符合新微体系结构和数据流的容量/重⽤需求 的⽹络和缓冲区层次结构。通过模拟⼀组不同的基于图像的 DNN 的执⾏来评估架构。能效比是最先的架构的5.6x，吞吐量保持在峰值的86-99% 



+ 1 INTRODUCTION
  + promising opportunities to improve the energy efficiency of these accelerators
    +  high level of sparsity exhibited by weights and activations
    +  maximizes data reuse and minimizes data movement --- loop
  + Pixel-first architectures
    + Outer-Product 
    + high activation/kernel reuse and simple indexing schemes
    + need for large accumulator buffers and routing logic
  + Channel-first architectures 
    + Inner-product
    + IF-condition makes Channel-first index generation/matching logic more complex
    + not suffer from architecturally wasted computations

+ 2 BACKGROUND
  + Pixel-First Architectures---Outer-Product 
    + SCNN:64 PEs
      + Each PE has a 4×4 grid of multiplier units
      + PE: Highly non-Load-Banlance
    + STICKER：
      + saves significant storage area 
      + Remain conflict for accumulator buffer resources
  +  Channel-First Architectures---Inner-product
    + SparTen:
      + outperforms SCNN by roughly 4× by better load balancing
    + SNAP：
      + 4 cores， 7×3 PE array per core,each PE has 3 MAC units
      + 2-level partial sum reduction (PE- and Core-level) 

+ 3 CANDLES
  + A.Motivation
    + 3  challenges:(1)Efficient PSUM aggregation (2) Simple indexing logic (3) Load balancing
    + Improve Temporal Locality: 
      + retain the Pixel-first compression strategy
      + dataflow is modified to  similar to Channel-first architectures, 
  + B. High-Level Overview
    + Pixel-first compression and Channel-first dataflow
    + a two-level accumulator buffer captures the reuse of partial sums
    + a memory partitioning scheme
    + a compression algorithm ensures high locality、
    + Key Part：a PSUM filter
  + C. Pixel-first Compression and Channel-first Dataflow
    + a new set of activations and weights are fetched from their buffers every cycle
    + Cacheability：overlap between the partial sums
  + D. The PSUM Filter
    + revisiting the partial sums for the same output neurons in consecutive cycle
    + Fact：A tagged cache
  + E. Tiled Pixel-first Compression
    + distribution of zeros is non-uniform---- miss
    +  tile size can affect in intra-PE underutilization
  + F. Load Balancing across PEs
    + CANDLES allocates the same number of non-zero activations and N × N partition of weights 
    + Partition Design Space: each PE receives 
      + a small share of channels and kernels
      + a large share of each input feature map channel. 
  + G. Microarchitecture Design Choices
    + Weight and Activation Buffers
    + Accumulator Buffer
    + Central Buffer
    + Simpler Crossbar
    + Activation Metadata:
    + Kernel Metadata
    + Wasted Computations

+ 4 METHODOLOGY
  + Verilog, implemented them using industry-standard synthesis
  + place-and-route tools in a 65 nm CMOS process
  + Simulator modeling：SCNN、STICKER
  + Benchmarks：ResNet-50、VGG16

+ 5 RESULTS
  + A. Energy
    + Importance of Microarchitecture-Dataflow Codesign
    + CANDLES Energy Analysis
  + B. Performance
  + C. PSUM-Filter Sensitivity Analysis
    + Replacement Policy
    + PSUM Filter Size
    + Tile size for TP-Compression
    + Space and Complexity of Loop Tiling
  + D. Broader Context Discussion
    + Dense & Quantized Dense Accelerators

+ 6  RELATED WORK
  + A. Similarities with the Baselines
  + B. Other Related Work

+ 7 CONCLUSIONS

  ----

  ***【ISCA‘ 20】Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads***

+ The TSP demonstrates a novel **hardware-software approach** performance on machine-learning workloads within a desired power, based on

  + machine learning workloads exhibit abundant data parallelism, which can be readily mapped to tensors in hardware

  + a simple and deterministic processor with producer-consumer stream programming model enables precise reasoning and control of hardware components, achieving good performance and power efficiency

    

+ 1 INTRODUCTION

  + scalability, performance, and usability challenges\ some project \ application
  + A. Functional slicing --- **Fig1**
    + traditonal one:  heterogeneous units , globally homogeneous
    + TSP:  local functional homogeneity , chip-wide (global) heterogeneity

  + B. Parallel lanes and streams --**Fig2,** 2D
    + 2D：X： dataflow ，Y：Instruction
    + element int stream：1-byte （with Data alignment）
    + producer-consumer model in TSP --- **Fig3**
      + provide a **programming abstraction**

  + C. Paper organization --- Contribution
    + we introduce a tile microarchitecture
    + implementation of the TSP in 14nm ASIC with ISA
    + performance results on ResNet50 
    + discussion of architecture tradeoffs 


+ 2 ARCHITECTURE OVERVIEW --- **Fig5**

  + Explain the photo of TSP
    + 320-lane programming abstraction
      + 16:minVL; 320:maxVL
    + 144 independent instruction queues (ICUs)
    + 64 logical streams
    + 220 MiBytes of globally shared SRAM
  + Explain the funtion element in FIg5
    + e instruction control unit (ICU) : IFetch
    + vector execution module (VXM): VXM-- 4*4
    + matrix execution module (MXM) : 4 independent 2D MACC
    + switch execution module (SXM) : to communicate
    + memory module (MEM) and Chip-to-chip (C2C) modules 
  + **Table1** show the ISA on above hardware-part

  + A. Parallel streams programming model --- **Fig4**
    + producer-consumer model --- each functional slice
  + B. Memory model
    + 2.5 Mibyte per-slice capacity
    + caculate the bandwidth(Instruction and operator)
      + stream registers : 20 TiB/s
      + SRAM bandwidth: 55 TiB/s
  + C. Staggered instruction execution --- **Fig6**
    + Both data and instruction move --- staggered
  + D. Error handling and reliability
    + use  error correcting code (ECC) :SECDED
  + E. Chaining functional slices
    + slice choose the direction of its result stream
  + F. Scalable vectors
    +  minVL 16 Byte to maxVL  320 Byte
    + can powering-down the unused tiles

+ 3 INSTRUCTION SET
  + The TSP programming model **2 critical elements**: 
    + (1) deterministic data paths in hardware
    + (2) exposing temporal information about an instruction’s execution latency through the ISA
  + A. Instruction control unit (ICU)
    + No-op:
    + Synchronization
    + Instruction fetching
  + B. Memory (MEM)
    + Read and write
    + gather or scatter
  + C. Vector (VXM) processor
  + D. Matrix execution module (MXM) --- **Fig7**
  + E. Switch execution module (SXM) --- **Fig8**
+ 4 RESNET50
  + A. Explicitly managing memory
  + B. Resource bottlenecks
  + C. Optimizations
  + D. Quantization
  + E. Model Accuracy
  + F. Deterministic performance
+ 5 DISCUSSION (result)
+ 6 RELATED WORK
+ 7 CONCLUSION



----

【HPCA‘ 20】Parallel Time Batching：Systolic-Array Acceleration of Sparse Spiking Neural Computation

+ 作者信息 ：UCSB，MICRO‘21没中所以改投的HPCA’22
+ Abstract: 
  + SNNs是什么，对比喻ANNs的主要优势
  + 本工作的亮点：把时间窗口打包，更改了时间窗口的处理粒度
  + 最终效果：248x与47x的EDP加速
+ 1 Introduction
  + 对比于ANNs，SNN的不同：更丰富的时间空间信息，简单介绍SNN
  + SNN目前所遇到的问题---DNN不存在的
    + 增加的时间维度：难以管理计算 、 数据移动
    + 因为尖峰的模型：在时间和空间上的稀疏性
  + 目前最流行的两个工作：TrueNorth、Loihi主要问题
    + 多核之间缺少并行 单个核串行
    + 假定所有权重都在片上，没有合理的数据重用（数据流）
    + SpinaFlow，精度差，缺少时间维度的并行
  + 本工作解决的问题：开发了一套脉动阵列模型
    + 把多个点的神经突触打包到单个时间点上
    + StSAP（PTB）组合不重叠的尖峰输入
  + 主要贡献：
    + 并行时间批PTB：以时间窗口TW来处理信息，提高了延迟性能和能耗比
    + 打包时空不重叠尖峰活动StASAP，进一步改进延迟性能
    + 基于脉动阵列的加速器结构
+ 2 Background
  + SNNs的特点
    + 最大的特点：时间数据表达和处理
    + 可以压缩的幅度更大
  + SNN基础信息
    + 对于单个尖峰时间点的操作：3步
      1. 集成尖峰输入
      2. 基于尖峰输入和之前时间点的 更新
      3. 有条件的生成下一个 尖峰输入
  + S-CNNs的基础信息
    + 基本操作上同SNN但有第4步骤：移动到下一个时间点，并且反馈
    + 主要参考图片Fig2与表I
  + 脉动阵列
    + 全局同步管理PE，加速器结构简单
    + 在复杂性、分布带宽、局部性、计算密度上 都有优势，适合做加速器
+ 3 Challenges of SNN accelerators
  + 复杂的时间和空间影响、数据移动  --- 提高PE利用率
  + SNNs 时空稀疏性
    + 空间：同一个时间点上，不是所有神经元都会活动
    + 时间：不同的神经元在相同的时间段，活动次数不同
    + 高度稀疏：百万分之一；结构化稀疏
  + 现存的SNN加速器 --- 主要说问题
    + 时序处理SNN加速器
    + 其余现存的SNN加速器
    + 本工作是第一个 并行加速的，同时利用了时空稀疏性
+ 4 Proposed Architecture
  + 整体结构介绍：a全局结构、b脉动阵列内的PE结构
  + 时间批次TB于与TB-tag：TB是被映射上PE的基本单位
  + PTB一次性处理多个TB
    + 映射 输入输出---Fig6 ，已经突出的神经元
      + 对于给定TW内的神经元 综合突触信息
      + 更新信息、有条件的尖峰输出
    + 减少能耗：
      + 减少了对不同权重的交替访问
      + 支持PE内和PE间的数据重用
    + PE利用率：
      + PTB的打包方式，TB打包了预突触尖峰，隐藏了尖峰的缺席
  + StSAP
    + 本质上是一种压缩手段，打包了尖峰活动
    + 打包策略上：贪婪
    + 利用率：因为打包了，所以PE的利用率也提高了
+ 5 Evaluation Methodology
  + 设备设置：16 * 8 PE的脉动阵列，存储层次：表4
  + 评估性能的模型
  + benchmark：DVS、AlexNet数据集
+ 6 Results
+ Acknowledgements & Disclaimer
  + 本项目由美国政府的机构赞助，出现了比较少见的免责声明



---

【ASPLOS ‘23】HuffDuff: Stealing Pruned DNNs from Sparse Accelerators

使⽤ DRAM 访问量通道从移动级稀疏加速器窃取 修剪后的 DNN 模型架构

HuffDuff，一种具有两种新技术的攻击方案，利用（i）CONV层中存在的边界效应（ii）即时激活压缩的定时侧通道

+ 1 Introduction
  + 为什么窃取DNN模型？
    + 了解模型结构可以对其进行后续攻击
    + 开展逆向工程
  + **DNN推理漏洞**：边缘与数据中心
    + DNN部署在边缘侧，攻击者容易获得访问权限
    + 边缘侧 计算能力有限、使用修剪过的DNN模型
    + 数据中心和边缘侧的攻击模型不通用
    + 针对边缘的攻击，可以从物理上攻击
      + 针对DRAM的总线侦听、冷启动攻击、电磁EM
      + 侵入式攻击，解封装、微探测
  + 本文关注的攻击模型
    + 片上SoC + 插槽中的片外DRAM：HMTT探测
    + SoC + DRAM安装在统一PCB上
  + 传统攻击模式的局限性 --- table1
    + 剪枝使得每个张量传输的数据量被压缩
    + 相比于稠密的，稀疏的情况下方案与时间指数增加
  + Key insights：
    + 到卷积(CONV)层表现出边界效应，根据不同的边界确定过滤器维度、步幅因⼦和池化参数
    + 稀疏加速器的后处理单元执⾏动态编码，将密集部分和压缩为稀疏输出特征
  + Contribution
    + 确定可以馈送到层的输⼊模式，以可预测地触发不同的⽚外流量。这使我们能够确定过滤器尺寸
    + 展⽰了如何构建在下游多个层创建此类模式的输⼊，揭⽰我们⽆法 直接提供输⼊的层的⼏何形状
    + 描述了如何共同使⽤多个探针来克服不可观察的边界效应
    + 确定了⼀个压缩时间侧通道，它揭⽰了所有层的部分和⾜迹之间的 ⽐率，这进⼀步揭⽰了它们的通道数。由于边界效应与通道数⽆关，因此 这会填补探测器⽆法识别的缺失组件

+ 2 威胁模型

  + 图1：稀疏DNN加速器 + 不可信的外部存储器
  + 攻击者的目标：
    + 什么是攻击？：对用于推理的DNN模型进行逆向工程，确定网络的架构超参数，包括 (a) DNN 中每⼀层的层⼏何结构（输⼊⼤⼩、输出维度、过滤器维度）(b)层之间的数据流图  (c) 权重每 层的稀疏因子
    + 架构相似的代理提供了更多比随机代理更准确的梯度信息

  + 攻击者的能力：
    + 物理访问设备，且可以在设备通过 HMTT或 其他探针 等 DRAM 跟踪⼯具执⾏时监视信号

  + 工作负载：
    + 以CNN为例，非结构化的剪枝

  + 执行环境：--- 图1
    + Soc + 片外存储器
    + 允许加速器同时支持权重和激活稀疏性
    + 支持在片外内存通信期间压缩权重和激活张量
    + 还假设加速器执⾏分层执⾏：所有数据都会出现在SoC和存储器中

  + 排除的讨论范围：
    + 没有片外存储器的纯SRAM加速器
    + 片上执行多层的加速器 --- 实际上没有加速器这么做

  + 更广泛的应用：
    + 放宽（执行环境中）中的⼀些假设会使问题更容易解决
    + 具有结构化稀疏性的加速 可能受到现有密集执行技术的攻击

+ 3 密集案例：路径和解决方案 

  + 任务制定
    + 攻击，实际上就是确定table2中的数据参数

  + 之前的解决方案：ReverseCNN
    + 通过制定 **⽚外内存流量与层维度相关联的**约束⽅程来找到 超参数
    + insight：层间的RAW依赖性 必须依赖于微架构细节/映射调度选择
    + 利用公式（2-6）递归求解所有层的几何形状 -- table1

+ 4 处理稀疏模型

  + 挑战
    + 使用稀疏加速器，传输到/从⽚外 DRAM 的数据块不再直接对应于相关的张量维度
    + 因为稀疏加速器通过消除零来压缩⽤于评估和传输的张量（从1-3变成了8-10的不等式）

  + 朴素处理稀疏性
    + 层之间的稀疏度水平差异很大，第一层与最后一层不好剪枝、中间稀疏度大
    + 附加了方程式12-14来进一步约束解空间

+ 5 通过主动探究学习

  + 图2，⼀维卷积中的边界效应及其对不同输⼊的结果
    + (a) 和 (b) 之间nnz的差异告 诉我们在过滤器中⼼的左侧**⾄少有**⼀个过滤器元件
    + 并且nnz对于 (b) 和 (c)是相同的 这⼀事实告诉我们**最多有**⼀个
    + 从而确定了过滤器的尺寸
    + 问题（1）攻击者只能直接控制第⼀层 的输⼊查询，不能直接制作任何中间特征图（2）卷积层是仿射的，即它们要么有加性偏差，要么后 ⾯跟着⼀个批量归⼀化层

  + 处理bias和Batch Normalization
    + 加了+2的bias以后，图2的情况发生了变化
    + 如果探针为负（-1）则经过了relu结果又不一样
      + 可以简单地做**多个** 独⽴的随机探针来放⼤观察到边界效应

    + 边界效应下，可观察（不同的nnz 计数）或不可观察（相同的 nnz 计数）

  + 探测下游层
    + 继续使用图2的例子，推广到任意权重值
    + 右上图，第一层-第二层，连续得分析

  + 处理下游层的错误
    + 前面的例子可知：观测nnz，仅仅有部分是可观测的，于是可能存在错误
    + 图5.4 所有nnz都有不同的、清晰的 边界效应。后两种情况下：可能是（部分可观察性）或（⽆可观察性）
    + 通过重复探测（⼀层**失败的概率**）随着独立随机探测的数量 指数下降

+ 6 自动化攻击（完整的攻击方案）

  + 广义的输入形式A（m，n）
  + 符号卷积引擎此 
    + (1)为当前层的每个 ⼏何假设⽣成预期的nnz模式
    + (2) 将第⼀层输⼊馈送到加速器 
    + (3) ⽐较通过窥探获得的输出nnz⽚外内存流量以确定哪个层的几何形状是正确的

  + 探测算法的伪代码
  + 探测攻击的局限性
    + 边界效应与通道数无关，无法确定

+ 7 使用结构化的属性 --- 获取通道数

  + 稠密PSUMs
    + 与输出激活相反 极不可能包含零，因此在累积过程保持密集

  + 定时边通道
    + 图3 GLB 中的密集psum值如何被压缩为稀疏输出特征
    +  ⼀旦缓冲区中有⾜够的数据，压缩的稀疏块将被写回 DRAM。这⼀直持续到处理完所有 密集的PSUMs
    + （a） 描述了编码过程受 GLB 约束、实际编码一般都是此
    + （b） 描述了编码过程受 DRAM 限制，取决于DRAM带宽够不够使

+ 8 评估

  + Eyeriss v2加速器 双侧稀疏加速器
  + 基于TorchAttacks的实现，使⽤ BIM方法生成对抗样本
  + 效果：在 NVIDIA 2080Ti GPU 上，在不到 10 分钟的时间内获得除输出通道 数之外的所有层⼏何信息

+ 9 讨论和未来方向

  + HuffDuff限制
    + 排除了只有SRAM得加速器，eg ShiDianNao
    + 排除了执⾏层融合的加速器、不稀疏的加速器

  + 潜在的防御策略

+ 10 相关工作

+ 11 结论

  

----

【DAC'20】High PE Utilization CNN Accelerator with Channel Fusion Supporting Pattern-Compressed Sparse Neural Networks

**题目**：支持模式压缩的稀疏神经网络的通道融合 高 PE 利用率 CNN 加速器

**关键词**：稀疏 CNN 加速器、剪枝算法、模式压缩、通道融合、空载率降低

**摘要**：最近，基于 CNN 的方法在广泛领域取得了显着进展。引入了网络剪枝算法和硬件加速器来加速 CNN。然而，现有的剪枝算法并未充分研究**模式剪枝**方法，目前稀疏CNN的索引存储方案效率不高。此外，现有加速器的性能受到稀疏网络上空载 PE 的影响。

这项工作提出了一种软件-硬件协同设计来解决这些问题。该软件包括一种基于 ADMM 的方法，它以可接受的精度损失压缩卷积核的模式，以及一种减少索引存储开销的霍夫曼编码方法。硬件是fusionenabled脉动架构，通过支持channel fusion可以降低PE的空载率，提升性能。在 CIFAR-10 上，这项工作实现了 5.63 倍的索引存储减少，不同层之间有 2-7 个模式，top-1 精度损失为 0.87%。与最先进的加速器相比，这项工作以合理的面积和功率开销实现了 1.54x-1.79x 的性能和 25%-34% 的空载率降低。

---

【DAC'20】3D CNN Acceleration on FPGA using Hardware-Aware Pruning

**题目：**使用硬件感知的 剪枝 在 FPGA 上加速 3D CNN

**关键词：**3D CNNs, video analysis, DNN weight pruning, ADMM, FPGA.

**摘要：**最近有许多尝试通过探索 3D CNN 将卷积神经网络 (CNN) 的成功从 2 维 (2D) 图像分类扩展到 3 维 (3D) 视频识别。考虑到移动或物联网 (IoT) 市场的新兴增长，有必要研究 3D CNN 在边缘设备上的部署。以前的工作已经在硬件平台上实现了标准 3D CNN (C3D)，但是，它们没有利用**模型压**缩来加速推理。这项工作提出了一种硬件感知修剪方法，可以完全适应 FPGA 设计的**循环平铺技术**，并应用于称为 R(2+1)D 的新型 3D 网络。利用强大的 ADMM，所提出的修剪方法在 FPGA 上同时实现了高精度和显着的计算加速。分层修剪率高达 10 倍，精度损失可忽略不计，修剪后的模型在 Xilinx ZCU102 FPGA 板上实现，与未修剪的版本相比，修剪后的模型实现了 2.6 倍的加速；与FPGA上最先进的C3D工作相比，有2.3 倍的加速和 2.3 倍的能效提升



+ 关于作者：
  + Massoud Pedram，USC-**EE**，Citation: 34k+
  + Miriam Leeser，Northearn University, GPU/reconfigrable，citation 4k+
  + Xue Lin, Northearn University, ML/HPC system, citation 5k+



+ 1 Intro

  + BG1：R（2+1）D CNN在图像领域成就大，但视频不行 
    + **2D CNN**无法对时间信息和运动模式进行建模
    + 标准 3D CNN **(C3D)**，并且不区分空间和时间信息
    + **R(2+1)D** CNN 准确且参数少：2D空间 1D时间卷积
  + BG2:边缘计算，DNN的模型压缩
    + 权重量化 和 **权重剪枝** --- 问题：精度损失
  + 贡献点：
    + 提出了一种用于 3D DNN 的分块修剪方案，直接匹配 FPGA 设计的循环平铺技术
    + 基于ADMM 的解决方案框架，实现理想的分块修剪方案，精度损失可以忽略不计
    + R(2+1)D 上测试，相比未剪枝2.6x的加速比；比最先进的3D工作 2.3×性能和能效提升

+ 2 RELATED WORK

  + 主要是关于3D CNN的硬件实现（FPGA / ASIC）
  + 本工作R（2+1）D的加速工作
    + R(2+1)D 是 3D CNN 的高级变体，更少的参数 实现高精度
    + 由不同种类的内核组成，分别探索空间和时间信息，

+ 3 Methodology

  + 3.1-3D CNN和提出的权重稀疏模式
    + M×N×Kd×Kr×Kc，分块blocks size Tm×Tn
    + 分块权重剪枝：基本剪枝单元是一个权重块
      + 对于具体的权重块，有稀疏度的要求
  + 3.2-权重剪枝问题 定义
    + 修剪模型权重，使参数满足稀疏性要求，同时保持精度
    + 反向：不能通过SGD、而是居于ADMM的方法
  + 3.3-ADMM 再定义
    + 对于 增广拉格朗日函数数的进一步变化
  + 3.4-ADMM迭代
    + W、Z最小化的步
  + 3.5-Masking和再训练
    + ADMM导致精度的显著下降，此处提高精度
    + 0权重被mask，其他的权重再训练

+ 4 - FPGA IMPLEMENTATIONS

  + 4.1-FPGA实现
    + 带有块使能信号 的剪枝、非剪枝模型
      + 决定是否将相应的**输入特征**和**权重块**加载到片上
      + 有效：保留权重块，否则表示被剪枝了
    + Tn乘法结果的累加 加法树
      + 在缓冲区的相应维度中执行数组分区以 增加带宽
      + 双缓冲技术用于通过重叠数据传输与计算 减少延迟
  + 4.2 设计空间的探索
    + 平铺方法与5个维度对应
    + 资源利用 - 分析
      + 考虑双缓冲，输出、输入和权重缓冲区的内存利用率
    + 性能分析 - 时间
      + 片外/片上Mem之间的数据**传输延迟** + **计算延迟**
      + 双缓冲，数据集加载和卷积计算可以Max
      + 关注一个**平衡的**问题

+ 5-EXPERIMENTAL RESULTS

  + 不同的分块策略 和 网络参数 --- 消融实验

  





