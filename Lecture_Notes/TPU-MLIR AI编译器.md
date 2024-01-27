AI编译器：TPU-MLIR

> 算能科技 周文婧 --- ICT博士
>
> 主要关注：MLIR如何用于深度学习编译器

# 1- TPU\GPU等的介绍

Global / local mem等

# 2 - AI编译器件

+ 编译步骤：

  + 原代码 --- 中间表达（中间多步可以复用） --- 目标器件 语言

+ MLIR - 基于LLVM开发的框架 [mlir.llvm.org](https://mlir.llvm.org/)

  + 为啥要搞成多层：在高级IR与低级IR之间的轴上做取舍
  + 提取代码的意图 --- 适应硬件的高性能
  + 核心：Dialect 
  + <img src = 'https://s3.bmp.ovh/imgs/2023/09/20/c62062e4d24e3616.png' >
  + Attribute - **Operation** 定义的基本单位 - Value(Type)

  

  # 3 - MLIR中的OP

  + 直接写一个 ODS文件，相比于嵌入到C++中减少了大量的代码 --- 应用到2中的具体语法
  + 最后还是会回到，生成的C++文件
  + Pattern rewritting 去做DAG的匹配，实现**转换**
  + ![](https://s3.bmp.ovh/imgs/2023/09/20/d796d4e8f5515840.png)

  + Dialect 转换 - 部分转换、全转换（主要应用于不同dialect之间的转换）
    + 具体还有状态可言，合法、非法、动态（看情况合不合法）

  # 4 - TPU-MLIR

+ 主要参考paper - AirXiv上面（孙老师分享）

+ MLIR开源的编译器 - 只要能转换成ONNX那就可以支持

  <img src="https://s3.bmp.ovh/imgs/2023/09/20/7b6258ddbce5f1ad.png" style="zoom:50%;" />

  <img src="https://s3.bmp.ovh/imgs/2023/09/20/85c943f7e44d867c.png" style="zoom:50%;" />

## 4.2 **算子融合** 

+ 激活函数融合：例如relu+Conv
+ BatchNorm融合
+ 常量折叠：
+ 矩阵乘融合：多个融合为一个（推导）

## 4.3 量化

eg。FP32 - Int8

优势1.MAC的效率提高  2.存储小了 则数据拌匀效率高

## 4.4 编译实力 - 详情查询手册

+ 使用一个 阈值 来判断能不能进行MLIR的变异

+ 

  