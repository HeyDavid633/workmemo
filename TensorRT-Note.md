> 关于TensorRT的了解笔记（急） --- 2023.5.17

**Reference**:

1. [知乎 - TensorRT入门，简单介绍与初步上手](https://zhuanlan.zhihu.com/p/371239130)，有基本的信息，以及相关链接
2. [Github-TensorRT](https://github.com/NVIDIA/TensorRT)：包含了安装版本对齐
3. [Nvidia-TensorRT Guide](https://developer.nvidia.com/tensorrt)：官方的介绍

+ [官方文档 - TensorRT支持的平台 - （限制）](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)



# 1-TensorRT是什么

## 1.1-基本概念

+ 在**NVIDIA**各种**GPU硬件平台**下运行的一个**C++推理框架**
  + 利用Pytorch、TF或者其他框架训练好的模型，可以转化为TensorRT的格式（通过 **ONNX**）
  + 利用TensorRT推理引擎去运行这个模型，从而提升这个模型在英伟达GPU上运行的速度
  + 所支持的Nvidia平台 - （有一定的**算力**限制）

> **加速效果**：①SSD检测模型，加速3倍(Caffe)；②CenterNet检测模型，加速3-5倍(Pytorch)；③LSTM、Transformer(细op)，加速0.5倍-1倍(TensorFlow)；④resnet系列的分类模型，加速3倍左右(Keras)；⑤GAN、分割模型系列比较大的模型，加速7-20倍左右(Pytorch)



## 1.2-用到的手段 

- 算子融合(层与张量融合)：通过融合一些计算op或者去掉一些多余op来减少数据流通次数、显存的使用

- **量化：**量化即INT8量化或者FP16以及TF32等不同于常规FP32精度的使用，这些精度可以显著提升模型执行速度但是会很小程度上降低精度，**小于 1% 精度下降** 

  >  这一点在效果上和 TensorCore上产生了关联 - 这玩意是硬件，而TensorRT算一种编译优化 - 软硬之间；而TensorRT在包含有 TensorCore确实能起到更好的效果

- 内核自动调整：根据不同的显卡构架、SM数量、内核频率等(例如1080TI和2080TI)，自调优

- 动态张量显存：我们都知道，显存的开辟和释放是比较耗时的，通过调整一些策略可以减少模型中这些操作的次数，从而可以减少模型运行的时间

- 多流执行：使用CUDA中的stream技术，最大化实现并行操作

> 以上的情况中，优化策略的代码本身是闭源的



# 2-实验情况

+ 硬件环境配置

  | CPU  | intel i7-11700k   |
  | ---- | ----------------- |
  | 内存 | 16GB              |
  | GPU  | Nvidia 3090(24GB) |

+ 软件环境配置(Anaconda中实际部署)

  | GCC/G++  | 11.3.0                                                       |
  | -------- | ------------------------------------------------------------ |
  | CUDA     | cuda-12.0.1                                                  |
  | CuDNN    | cuDNN-8.8.0                                                  |
  | TensorRT | 8.6.1                                                        |
  | torch    | torch=2.0.0+cu118， torchvision=0.15.1+cu118 ，torchaudio=2.0.1 |

[知乎 - 最容易部署的一个TensorRT实例](https://zhuanlan.zhihu.com/p/395590559)

<img src = 'https://s3.bmp.ovh/imgs/2023/05/21/8a2a013be0ecedef.png' >