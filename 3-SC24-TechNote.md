# 1 - 复现工作

## 1.1 ByteTransformer

+ 基于pytorch的Transformer -  https://github.com/bytedance/ByteTransformer
+ 首先复现，总体过程最为流畅 （git clone  --- run脚本 ---- 得到 .log） 
  + 感官上：**轻量级？**
+ 首先基于docker来配置，pull下来一个pytorch1.13+cu116
  + 没有N驱动、python版本不匹配；进一步配置要换源、现场下载，外面看不到位置
  + 在anaconda中创建环境 ByteTf来运行，安装使用pytorch - 新安装cuda11.6

#### 问题 ：

+ 在ByteTransformer的代码文件中 找不到 cutlass.h --- 手动安装 github：nvidia/cutlass 但不成功；  最后使用 git clone --recuseive-submoudels .....

#### 重新用Docker来复现该工作：

+ 拉取新镜像 `docker pull pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel`
  + 拉取过程中慢 -- 等待～



## 1.2 TH-GNN

+ JIT的模式 复现报错 --- 某些moudle、so文件 没有生成
+ 环境配置：一系列环境在annaconda中已经对齐；但没有build成功
  + **PyTorch** 1.8.0+、**DGL** 0.7.0+、**Ninja** 1.10+、**GPUtil** 1.4+ 
  + 结论：如果是给的某个环境+的话，则直接参考**最低的**那个
+ 能有docker的话则直接 使用docker --- conda欲速则不达，看起来最笨实际上最顺利的方法

#### 问题1 代码运行

+ `FROM nvidia/cuda:11.3.0-devel-ubuntu18.04 as base` 在dockerhub上并没有找到这个镜像，于是

  + 更换版本为`nvidia/cuda:11.3.1-devel-ubuntu18.04`可以正常安装

+ dokcerfile build文件到一半，其中有一个文件 进度到 300M/1.2GB则停止了，尝试多次，进度都没动

  + Docker info查看`Docker Root Dir: /home/david/.local/share/docker` 确定安装的cache；把这个目录下的文件删掉，才能跑完
  + nv这里要清除的Cache并不是 [Dockrt Build Cache](https://blog.csdn.net/catoop/article/details/128002962) ；删除了这些没有使用的镜像，没有作用

+ Dockerfile中的文件

  `COPY install/ubuntu_build.sh /install/ubuntu_build.sh
  RUN bash /install/ubuntu_build.sh`

  + 其中在ubuntu_build中有一项是：wget 清华镜像站上的Anaconda3-2021.05-Linux-x86_64.sh，始终403 forbiden；在我的ubuntu上（估计是因为proxy的原因）无法正常下载，进而无法执行改脚本
  + 解决：单独手动下载该脚本，在Dockerfile中把这个COPY进Docker；之后正常执行

+ ![](https://s3.bmp.ovh/imgs/2023/09/26/a82daabe50f11832.png)

  + 要想在docker中使用GPU，需要先配置 [Nvidia-Container-Runtime](https://blog.csdn.net/qq_35395195/article/details/131431872)
  + Docker images中可以看到这个镜像，但是docker run的时候说没找到


+ 其安装脚本中的对一些包的配置管理不完全、不细致
  + 直接使用脚本conda安装的torch、dgl等，结果是最新版
  + 缺少的包：h5py
  + [dgl](https://www.dgl.ai/pages/start.html)、[torch](https://pytorch.org/get-started/previous-versions/)没有安装与cuda11.3所匹配的版本
+ [ImportError: version `GLIBCXX_3.4.22‘ not found](https://blog.csdn.net/qq_30653631/article/details/107620137)

#### 问题2 未解决问题


+ 内存不够（32GB） --- 官方给到的配置是64GB

  + dmesg查看内核kill原因，发现是oom -- 宿主机上dmesg查看：容器内的PID与dmesg中的PID并不一样
  + 能够打出一部分的数据.csv文件，不能打出图片
+ 关于DGL库的问题 AttributeError: 'RedditDataset' object has no attribute 'num_labels'

  + DGL的版本过高（1.1.2） --- 在0.5.0以后则弃用 --- readme写的0.7.0+
  + 安装更低的版本有其他问题 RuntimeError: Bool type is not supported by dlpack

    + torch与之关系不匹配

#### 运行停车 小结

+ torch、dgl、cuda版本对齐的问题，依据docker和官网所给 的并不能完全成功运行 ；硬件配置：内存不够 64GB；综上，能跑出来2个小规模的数据 .csv，认为任务已经跑通，不再深究完全跑通



本工作有Dockerfile；可以直接用来拉取镜像

+ 创建容器 ```sudo docker run --gpus 0 --name ByteTranformer -itd -v /home/david/ATC24/ByteTransformer:/Byte gnn:v1 /bin/bash```
  + 可以使用宿主的 Nvidia 驱动；
  + 新建的容器 命名为ByteTranformer
  + 把宿主的文件挂载在docker上
  + 容器创建运行 挂在后台跑 ；再用 docker exec -it 【NAME / container ID】进入



## 1.3 FeatGraph - SC20

> 实现前言 - 完全使用Docker来进行以下的所有操作；由于编译的各种操作，已经不是管理pip包那么简单了，种种环境的影响下，Docker就是最优解（1）不影响其他环境 （2）推倒重来的轻量级

Docker创建容器 ，命名为**FeatGraph-SC20**，使用的镜像还是 ubuntu20 + cu116

```dockerfile
docker run --gpus 0 --name FeatGraph-SC20 -itd -v /home/david/ATC24/FeatGraph:/FeatGraph pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel /bin/bash
```

执行语句 进入命令行的操作

```dockerfile
docker exec -it FeatGraph-SC20 /bin/bash
```

#### 问题：

+ DGL的版本问题 ---l s1.1.X 
  + 但是复现的多数环境，都不支持那么新<img src = 'https://s3.bmp.ovh/imgs/2023/10/15/ecdba632d3c46431.png' >
    + 问题1：[TypeError: adj() got an unexpected keyword argument ‘scipy_fmt’ ](https://discuss.dgl.ai/t/typeerror-adj-got-an-unexpected-keyword-argument-scipy-fmt/3755)
    + 问题2：[Issue with AttributeError: ‘DGLGraph’ object has no attribute ‘adjacency_matrix_scipy’. ](https://discuss.dgl.ai/t/issue-with-attributeerror-dglgraph-object-has-no-attribute-adjacency-matrix-scipy/3562)


+ 下载数据集 `download_reddit_dataset.py`强制下载 - reddit（4.3GB）

+ 运行代码`python bench_vanilla_spmm.py --dataset data/reddit_csr_float32.npz --feat-len 64 --target x86`
  + 错误1：[TVMError: Check failed: bf != nullptr: Target llvm is not enabled](https://discuss.tvm.apache.org/t/tvmerror-check-failed-bf-nullptr-target-llvm-is-not-enabled/5561)
    + 错误1- 解决1 ：在cmake时LLVM选项未开启 需要设置为（**LLVM ON**）
    + 伏笔：那么（**CUDA OFF**）是否也可以开 --- 担心CUDA版本的匹配问题

  + 错误1： 展开： 在源码安装TVM前，**首先**还需要 [从源码安装LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm) 然后再（**LLVM ON**）
    + `git clone https://github.com/llvm/llvm-project.git` LLVM git下来非常大 - 大约2GB
    + 源码安装 cmake失败，cmake到错误的位置耗时约30min<img src = 'https://s3.bmp.ovh/imgs/2023/10/16/34eead6e76b52d8f.png' > 不打算debug 
    + 后查看：源码安装完成以后，查看build文件 --- **50GB！**

  + 错误2：依据 源码安装TVM时对[LLVM的介绍的另外两种方式](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github)  （1）“download pre-built version of LLVM from [LLVM Download Page](http://releases.llvm.org/download.html)” 但是*没太看懂应该如何安装*；而且并不清楚应该安装它提供的那些包中的哪一个？--- 查看Doc 退化到了 源码安装 （2）“You can also use [LLVM Nightly Ubuntu Build](https://apt.llvm.org/)” -- 自动化脚本安装

    + 官网的脚本使用 `https://apt.llvm.org/llvm.sh` 其中安装时会向 `/etc/apt/source.list` 中添加了源： 

      ```shell
      deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-17 main
      #deb-src http://apt.llvm.org/bionic/llvm-toolchain-bionic-17 main
      deb https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/bionic/ llvm-toolchain-bionic-17 main
      #deb-src https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/bionic/ llvm-toolchain-bionic-17 main```
      ```

      显然这个源太慢了，28MB的文件下载时间不可接受（>2h） 

    +  想到能否换源清华，答案是可以，用[tuna.moe - LLVM清华源](https://mirrors.tuna.tsinghua.edu.cn/help/llvm-apt/)；

  + 错误3： 清华源没有最新的版本，最新为15，此处采用llvm-12 --- LLVM安装成功 ；但tvm build不成功，报了error，build到90%停止，报错信息如下（问题猜测是 LLVM的版本过高，与TVM不匹配（当前官方最新为17，清华源最新15，刚安装12成功，但build tvm失败）没有进行Debug：

    ```shell
    FeatGraph/tvm/src/target/llvm/codegen_llvm.cc:480:82: error: no matching function for call to 'llvm::ElementCount::ElementCount(int&, bool)'
    ```

    + 通过翻阅 FeatGraph的发布时间 - 其所要求git的TVM 0.7 的版本号发布时间“Oct 3 2020”；回溯查看当时TVM所应该基于的LLVM版本 推测为“Jul 22, 2020”的这版，也就是LLVM 10
    + 此时选择在安装了LLVM 12的Docker中不再进行uninstall操作。【**Docker的优势体现**】重新基于同一个镜像创建了另一个容器安装LLVM 10后，基于LLVM的TVM build 成功 -- 100% ; 结合刚才下载的数据集成功运行 python代码

#### 运行小结：

+ 运行这个关于编译器的工作尤其体现了Docker的优势：（1）完全的隔离 --- 不影响其他的**所有**环境；（2）轻量化 --- 创建配置不费劲、删除不心疼。 痛点解决：（1）GPU --- 驱动可以与宿主共用（2）文件挂载 --- Docker中的操作、下载痕迹能够一直保留

+ FeatGraph ---》 TVM ---》 LLVM ；种种基于，版本都要注意**向下兼容**，所基于的版本需要以源码进行安装

+ 2组CPU代码运行成功，GPU的没有 --- 估计CUDA版本一定会出错 ，要求10及以下，3090硬件不支持

  `TVMError: Check failed: allow_missing: Device API gpu is not enabled.`



## 1.4 Faith - ATC22

安装之前 - github readme：`conda install tvm-cu102 -c ./conda/pkg`感觉情况不妙；30系的显卡不支持10代cuda很可能跑不了，但对其仍抱有一定的期待

Docker创建容器 ，命名为**Faith-ATC22**，使用的镜像为了配合后续的cuda10.2 所以直接拉了一个新的镜像下来：`bleakie/cuda10.2_cudnn8.0_ubuntu16.04:latest` 创建语句:

```dockerfile
docker run --gpus 0 --name Faith-ATC22 -itd -v /home/david/ATC24/Faith:/Faith bleakie/cuda10.2_cudnn8.0_ubuntu16.04:latest /bin/bash
```

docker启动容器语句

```dockerfile
docker exec -it Faith-ATC22 /bin/bash
```

+ 本次常使用到conda，于是首先拿到新的docker：（1）更新apt源ubuntu16，（2）更新pip源（3）安装conda并换源到清华

#### 问题：

+ `conda env create -f environment.yml` 与`conda env create --file conda/build-environment.yaml` 总是有点小错误，类似

  ```shell
  CondaError: Downloaded bytes did not match Content-Length
    url: https://conda.anaconda.org/anaconda/linux-64/mkl-2021.4.0-h06a4308_640.tar.bz2
    target_path: /root/anaconda3/pkgs/mkl-2021.4.0-h06a4308_640.tar.bz2
  ```

  通过重复执行conda env create -f 语句解决

+ readme上确有提醒必须安装 cudnn与cublas：`Remember to add cudnn and cublas into /usr/local/cuda-10.2. For adding cudnn and cublas`，否则在执行`sh conda/build_cuda.sh` 报错

  ```shell
  CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
  Please set them or make sure they are set and tested correctly in the CMake files:
  CUDA_CUBLAS_LIBRARY
      linked by target "tvm" in directory /root/anaconda3/envs/tvm-build/conda-bld/tvm-cu102-package_1697633308207/work
      linked by target "tvm_runtime" in directory /root/anaconda3/envs/tvm-build/conda-bld/tvm-cu102-package_1697633308207/work
  ```

  + 但是cublas并不单独安装，而是在cuda中已安装；对于cuda11以后的版本来说是在`/usr/local/lib64 或者  /usr/local/include下的某个 .so文件`  对于cuda10来说是在`/usr/lib/x86_64-linux-gnu` --- 这里涉及到了cuda版本的变化带来的cublas路径变化。
  + 单独`export CUDA_CUBLAS_LIBRARY= ` 没有解决问题
  + 我是使用docker直接拉下来的含有cuda10.2的镜像；在cuda10.2的安装步骤中，打了补丁[Patch1](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal) ，下载该.run文件并bash运行，问题解决 --- 找到了cublas

+ 使用  `sh conda/build_cuda.sh`  实际上在 用cmake的手段build tvm-cu102；以上报错在cmake开始以前，cmake到99%时，报错

   ```shell
   make[2]: Leaving directory '$SRC_DIR/build'
   [ 99%] Built target tvm_objs
   make[1]: Leaving directory '$SRC_DIR/build'
   make: *** [Makefile:136: all] Error 2
   Traceback (most recent call last):
     File "/root/anaconda3/envs/tvm-build/bin/conda-build", line 11, in <module>
       sys.exit(main())
     ... ...
     File "/root/anaconda3/envs/tvm-build/lib/python3.7/site-packages/conda_build/utils.py", line 382, in _func_defaulting_env_to_os_environ
       raise subprocess.CalledProcessError(proc.returncode, _args)
   subprocess.CalledProcessError: Command '['/bin/bash', '-o', 'errexit', '/root/anaconda3/envs/tvm-build/conda-bld/tvm-cu102-package_1697635987468/work/conda_build.sh']' returned non-zero exit status 2.
   ```

  + 这个build的错误定位不明确 ---- 查阅资料后认为：尽管各个依赖包、库都参照了 `conda enviroment.yaml`文件来进行安装，确保了这些的版本匹配关系，但是对于python、gcc等版本仍然不能保证对齐；最重要的一点：此处的cuda10 不适应于30系显卡，此处的build可能也会因此失败

  

## 运行停车，移植与重开

+ Docker的从头安装：确保有一个纯净的环境，参考[Tuna.moe](https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/) - Debian/Ubuntu/Raspbian 的安装手段，然后进行安装

  + 为了使在 Ubuntu 上配置无需使用 `sudo` 来运行 Docker ，输入 

    ```shell
    sudo usermod -aG docker $USER  # 然后退出终端 重启
    ```

  + 换源为 [中科大、网易源]( https://blog.csdn.net/m0_37282062/article/details/115770314)，其间通过docker info查看是否换源成功

+ 打包docker环境实现快速的移植

  + `docker pull`的速度很多时候不可接受，而且docker中本身已经有一定的配置了 复用更好。
  + 想法1 - 打包**容器**`docker export <容器ID> > container.tar`，传到对应位置后(并命名) `docker import container.tar test/my:v1` 但打包后的容器太大 - Faith-ATC22【32GB】感觉有很多冗余
  + 想法2 实施 - 打包所需**镜像**  `docker save -o image.tar <镜像名称>`，传到对应位置后 `docker load -i image.tar` 该镜像的大小为5GB

+ 移植过程，重新配置

  + 拉取了多个镜像 - 希望更匹配paper中的环境，但有很多**其他问题**
    + [tuna.moe - LLVM清华源](https://mirrors.tuna.tsinghua.edu.cn/help/llvm-apt/)并不支持可以下载到LLVM-10但不支持ubuntu16 -- ubuntu版本不能太低
    + 有的镜像中所包含的 cuda10.2不完整 --- 没有cuda/bin -- 找不到nvcc
    + 每次都是新系统重新配置，系统上的小问题 --- 修起来快但很杂

  + **结论**：不如直接打包 **docker环境**实现快速的移植 ；即**想法1**，最笨但最稳妥


### 在四节点上运行

+ 登陆：`ssh root@10.18.19.43 -p 2005`密码 root

+ 登陆上去自动进入一个（每个人自己的）docker；我现在的需求等于是 ***在docker里再用docker***

+ Node1 的使用

  1. 正常安装了docker --- 可以在容器里安docker

  2. 创建容器并启动，错误： --- GPU相关 仍然参考配置 [Nvidia-Container-Runtime](https://blog.csdn.net/qq_35395195/article/details/131431872) --- 问题解决：docker里面再套docker可以使用GPU；和单层docker操作完全一样

     ```shell
     docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
     ```

+ 4节点运行小结 ---- 总体上感觉不方便
  + 空间太小 ：共用500GB，一个人100GB多点
  + 目前我能用node1/4；没有node2、3不能用，node4做了网络隔离连不上，node1关机了 

### 重开1 - FeatGraph - SC20

+ 启动语句变更`--gpus all`，否则没法输出`nvidia-smi`（既不报错也没输出）

  ```shell
  docker run --gpus all --name FeatGraph-test -itd -v ~/ATC24/FeatGraph:/FeatGraph test/featgraph-sc20:v1.1 /bin/bash
  ```

+ 打包容器过来的镜像，再新建容器进去，容易没有nvcc -V，需要手动export一下

  ```shell
  export LD_LIBRARY_PATH=/usr/local/cuda/lib
  export PATH=$PATH:/usr/local/cuda/bin

### 重开2 - Faith-ATC22

+ 执行`sh conda/build_cuda.sh`仍然有上述问题，报错保持一致；但**报错截图**得不对 --- 可以定位到build cu102-tvm上来

  + 把cuda10.2的分支2安装了即`cuda_10.2.2_linux.run` ；没有解决cmake的问题
  + 注意到需要安装cudnn，补充安装[cudnn8（cuda10.2对应）](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify); 重新使用[.tar的方式安装](https://blog.csdn.net/h3c4lenovo/article/details/119003405)（否则没有 cudnn_version.h --- 在make时观察到fatal error但没中断）
  + 本路径尝试多次，卡点与以上相同，于是放弃

### 重开2.2 - Faith从头尝试

> 本次解决了每一个小错再往下走

1. Ubuntu换源，pip换源
2. cuda10.2分支补充安装，吧cudnn等信息移到 /usr/local/cuda
3. Conda安装并换源 ----- 直到此步15min

+ `conda env create -f environment.yml` 出现报错，有6个包没有安装成功；

  ```shell
  CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://conda.anaconda.org/anaconda/linux-64/libprotobuf-3.11.4-hd408876_0.tar.bz2>
  Elapsed: -
  
  An HTTP error occurred when trying to retrieve this URL.
  HTTP errors are often intermittent, and a simple retry will get you on your way.
  ```

  + 解决思路1: 在本地分别下载好了这几个 `*.tar.bz2` 上传上去；在远端的`enviroment.yml`中删去对应的包 - 先创建环境A，装其他的依赖；然后再在conda环境A中，手动 `conda install *.tar.bz2` 

    + `conda install ./tensorflow-base-1.14.0-gpu_py37h8f37b9b_0.tar.bz2`错误 报了一串类似的

      ```shell
      CondaVerificationError: The package for tensorflow-base located at /root/anaconda3/pkgs/tensorflow-base-1.14.0-gpu_py37h8f37b9b_0
      appears to be corrupted. The path 'lib/python3.7/site-packages/tensorflow-1.14.0.dist-info/entry_points.txt'
      specified in the package manifest cannot be found.
      ```

      解决：在 `~/anaconda3/pkgs/` 路径下删除对应的包，再手动`conda install`就好了

  + 实际解决：在安装的时候`conda env create -f environment.yml` 删去了对应的包，自动下载了需要的东西

    + 自动下载与解压的错误，关于OpenGL的相关驱动  --- 解决 `apt-get install libgl1-mesa-dev`

      ```shell
      ERROR conda.core.link:_execute(698): An error occurred while installing package 'anaconda::pyopengl-3.1.1a1-py37_0'.
      Rolling back transaction: done
      
      LinkError: post-link script failed for package anaconda::pyopengl-3.1.1a1-py37_0
      location of failed script: /root/anaconda3/envs/NNver/bin/.pyopengl-post-link.sh
      ==> script messages <==
      <None>
      ==> script output <==
      stdout: Warning: Missing OpenGL driver, install with yum install mesa-libGL-devel or equivalent
      ```

+ 完全遵照readme执行，解决了每一步中小错误再往下执行，（之前没有关注这些小错，但仍然可以往下走）。之前关于make的卡点问题解决，`sh conda/build_cuda.sh`，即在这里能够build到100% ，但是出现的新的问题仍然不能产生文件tvm-cu102 

  <img src = 'https://s3.bmp.ovh/imgs/2023/11/09/88b57d8613bf876e.png' >

+ 再次运行，命名为Faith2.3，仍然基于这个流程，但在其间事先解决了apt install的问题；最后build成功-100%，与Faith2.2的问题相同；但是出现的新的问题仍然不能产生文件tvm-cu102 ；
  + 通过观察文件输出，认为有可能是问题出在llvm上，没有事先安装；但LLvm的安装脚本并不支持此镜像ubuntu16了
  + Faith2.3 停工，与Faith出的问题一致，作为实验组 ---- 之后通过源码直接安装（也需要LLVM？）试试看

###  重开3 - Faith tvm源码build

> 创建一个 比较基础的个人镜像 ubuntu20 + cuda102
>
> 其余的操作同Faith2.3 把apt等安装在问题发生以前

+ 在ubuntu20 + cuda102 的docker中走全流程conda-build，进入卡点1 -- 99%处（还不如cuda10.2 + ubuntu16）

+ 常出现的小问题解决 `apt-get install --reinstall ca-certificates` 

```shell
Err:6 https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/focal llvm-toolchain-focal-17 Release
  Certificate verification failed: The certificate is NOT trusted. The certificate chain uses expired certificate.  Could not handshake: Error in the certificate verification. [IP: 101.6.15.130 443]
```

#### conda环境中安装tvm、llvm的路径问题

+ 源码安装tvm前，仍然首先进行llvm安装，然后查看安装是否成功、版本`llvm-as --version`
+ 尽管 有输出，但是仍然在后面找不到 `llvm-config`；
+ 通过查询 `which llvm-config`发现，和我正在使用的conda环境有关系 ，llvm-config位置
  + 我的是在`/root/anaconda3/envs/tvm-build/bin/llvm-config` --- **再次理解conda1**
  + 而以前在非conda的安装环境中是在  `/usr/bin/llvm-config`
+ 于是通过在`config.make` 直接**指定路径** `set(USE_LLVM /path/to/your/llvm/bin/llvm-config)`通过源码安装TVM（最新版 0.14）与LLVM-14 成功

#### Conda的python执行的时候找不到torch

+ 在conda环境 tvm-build中，使用`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch` 但在python执行的时候找不到torch
+ 经过which python查询，发现是conda装的包和我当前的python解释器路径不一致造成的
  + which python结果 `/root/anaconda3/bin/python`
  + 但我当前的安装路径的python为 `/root/anaconda3/envs/tvm-build/bin/python3.7`
  + 根本原因：**再次理解conda2**，conda激活了某个环境，python解释器不一定切换进来了
+ 需要[强制切换python解释器](https://blog.csdn.net/qq_43744723/article/details/122090500)，和切换gcc编译器的方法非常类似；切换后成功

#### 在远端的docker中使用Jupyter notebook

+ `python matmul_ansor_benchmark.py`与`python pytorch_benchmark.py --dir model --data yelp`把报错排完以后，没反应，没有输出也没有终止 ----只有函数定义def

<img src = 'https://s3.bmp.ovh/imgs/2023/11/12/9bd4b5f5700148f1.png' >

+ 在docker中[用jupyter](https://blog.csdn.net/fs1341825137/article/details/109683965)来试试看有没有输出 ---  涉及到端口的映射，最快的方式先打包为镜像再启动容器 

  + 容器启动时绑定端口，把docker内的端口映射到host上（同一端口号） 启动`jupyter notebook` --- 再通过 浏览器访问 `10.18.19.43:8080` 

    ```shell
    docker run --gpus all --name Faith3.1 -p 8080:8080 -itd -v ~/ATC24/Faith:/Faith ubuntu20-cu102/faith:v3 /bin/bash
    ```

  + 在重新启动容器时，即 `source ~/.bashrc`时，python又掉了；`which python`发现又切了回去`/root/anaconda3/bin/python`；于是强行通过`ln -s` 把`/root/anaconda3/envs/tvm-build/bin/python3.7`链到python上去

#### cutlass路径找不到

```shell
(base) root@ae106e5d879f:/Faith/artifact# In file included from matmul_verification_artifact1.cu:16:0:
../tvm_kernels/cuda_kernel/mma/mma_kernel.h:1:10: fatal error: cutlass/cutlass.h: No such file or directory
 #include "cutlass/cutlass.h"
```

+ 去[Github/nvidia/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)参考说明安装了cutlass，但并不直接解决以上报错；
  + ubuntu的C语言编译，include的文件在`/usr/include/` 或者 `/usr/local/include`
  + 将cutlass2.6.0(最后一个支持cuda10.2的版本)的源码下载下来后，单独把其中的[include/cutlass/...](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass)文件拿出来，放在 `/usr/local/include`下，解解决以上报错

  

#### Faith过程的结论

+ 感觉这个工作的代码开源就属于比较不怎样的：（1）各种路径不统一（2）readme中关于conda的执行逻辑不清楚（3）版本限制不明（4）里面一些与硬件有关的编译参数写死（5）安装的库不完全(cloudpickle \ XGboost)
+ `Benchmark/*.ipynb`跑通；但是画图的数据没有完全跑出  --- 先研究再说




## 大卡点记录：

Faith: 

1. conda-build不通：先conda安装依赖，再源码安装tvm

2. python benchmark. py运行没反应 ：使用jupyter再尝试

   

## 复现小结  

+ 当创建了一个新的docker

  + --gpu绑定驱动、--name命名、--itd放在后台跑、-p 端口映射、-v文件夹挂载
  + 检查nvidia-smi\nvcc -V是否可用、检查文件挂载是否成功
  + Ubuntu换源，pip换源
+ 欲速则不达，跑到最远的环境直接打包上另一个平台才是最“快速”的手段
+ 对于cmake、make等的错误，通过报错信息不能立即定位到错误根源，应该回溯全程
+ conda的作用不局限于pip包的集中管理，更在于python解释器的快速切换、llvm编译器的隔离



# 2 - 自主框架搭建 - 摸方法

| 文件名         | 实现目标                                                     | 备注                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| demo1          | 初步实现pytorch对cuda的调用：os.system()的形式调用/ load的形式 | 没有传参出来 、手动管理内存                                  |
| demo2          | 模仿MLSys22 实现 简单的cuda版 tensor_add操作                 | 有传参数出入 、**自动**管理内存 --- 第一个理想的单元         |
| demo3          | demo2上强度，实现GEMM                                        | demo2 量变、CUDA专家知识上强度；引入计时、Mem统计**（CUDA知识复习）** |
| microbenchmark | 对Transformer类模型的测评方法了解                            | 使用huggingface的现成模块参数设置 `config.()`- 主要参考李沐  |
| demo4          | demo3改变矩阵规模：非仿真、批量矩阵---bmm                    | 1.09回学校的复健 ；优化计时时间                              |
|                |                                                              |                                                              |
| demo4          | attention forward过程的算子替换 实现                         |                                                              |

## 2.0 问题插曲 

### 2.0.1 VScode代码高亮

```python
The Pylance extension is not installed but the python.language Server value is set to "pylance". Wouldyou like to install the Pylance extension to use Pylance, orrevert back to Jedi?
```

原来是VScode选的主题不对，`command + shift + p`设置主题为`morden dark`解决

### 2.0.2 Git出现问题

处理git的时候`git clone https://github.com/Tencent/TurboTransformers --recursive`报错 

```shell
Cloning into '/home/david/Documents/other_work/TurboTransformers/3rd/Catch2'...
fatal: unable to access 'https://github.com/catchorg/Catch2.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.
fatal: clone of 'https://github.com/catchorg/Catch2.git' into submodule path '/home/david/Documents/other_work/TurboTransformers/3rd/Catch2' failed
Failed to clone '3rd/Catch2'. Retry scheduled
```

参考https://blog.csdn.net/qq_42921511/article/details/120551306；不要用sudo，而是使用

```shell
git config --global --unset http.proxy 
git config --global --unset https.proxy
```

再git即解决问题



## 2.1 CUDA-torch耦合demo

### 2.1.1 Pytorch-Extension基本信息

> 参考PPT12.18

+ 调用方式1：傻逼的调用方式 os.system() ---- 不对
  + 在命令行中打印 字符串“nvcc xxxx.cu - o xxxx”然后再打印字符串运行该程序
+ 调用方式2：https://pytorch.org/cppdocs/installing.html#visual-studio-extension
  + pytorch提供了C++的方式，完全脱离了“pytorch”，只是一个 ***C torch？***

+ 参考 MLSys22的实现方式，包夹结构

  1. `end2end.py` 最外层的脚本 `os.system()` 执行  

  2. `train_gatconv_our.py `  ：GATConv在里面作为一个函数使用

  3. `layer/gatconv_layer.py ` ：GATConv的函数定义

     + from operators.fused_gat import **fused_gat_op**,fused_gat_stash_op,fused_gat_fusescatter

  4. `operaters / fused_gat.py` 该文件 

     + 定义了同名的：⭐️***fused_gat*** = load(name = "**fused_gat**"，....)  --- **编译的过程？**
       + `fused_gat/fused_gat.cpp`
       + `fused_gat/fused_gat.cu`
     + 定义 **fused_gat_op** (....):  ---- 又封装了一层 cpp/cu
       + return了一个 class - FusedGATFunction 
       + class FusedGATFunction中使用了 fused_gat.cpp / fused_gat.cu 中的函数：**fused_gat**.gat_forward() 
         + 此处的`fused_gat.gat_forward() `应该是load进来的那个 同名文件

     

+ 问题1：编译完成（看样子是编译完成了），但不能运行 --- 查询的多数结果说是环境、版本问题；但实际上原因在于没有写.cpp文件

  + ![](https://s3.bmp.ovh/imgs/2023/12/18/a2d9af01eef7eb01.png)
  + 解决：参考github上的issue：https://github.com/pybind/pybind11/issues/2145；需要写一个专门的与 “.cu”**同名**的cpp文件
    + 一个 A.py需要配合cuda使用的话：需要 .cu + .cpp ----  对于.cpp中的写法：声明.cu中的函数



### 插曲1 参考萌哥所实现的 拓展

1. 在huggingface中替换了一个 模块（GEMM）为自己的
2. pytorch官方

   + PyTorch自定义拓展 https://pytorch.org/docs/master/notes/extending.html；例子中扩展了 [`torch.nn`](https://pytorch.org/docs/master/nn.html#module-torch.nn), [`torch.autograd`](https://pytorch.org/docs/master/torch.html#module-torch.autograd), [`torch`](https://pytorch.org/docs/master/torch.html#module-torch)
     + 其中有不可微的部分、依赖于非pytorch的库；更近一步往下C拓展
       +  子类化（自定义的）函数，需要定义forward()、backward() 算子
     + 自定义 tensor-like/封装了tensor的类
     + 对PyTorch API进行重载
   + 【萌哥参考】关于 CUDA扩展的介绍 https://pytorch.org/tutorials/advanced/cpp_extension.html
     + create PyTorch operators defined *out-of-source*
     + `torch.utils.cpp_extension`就是来自于此https://pytorch.org/docs/master/cpp_extension.html
     + 何时需要这么拓展：对性能要求高（call得频繁 单次call成本高）

3. 试跑结果 未成功 --- `python test_matmul.py ` 出现问题 ：动态库编译有问题（未搜到解决方案）

```shell 
ImportError: /home/david/.cache/torch_extensions/py310_cu121/matmul/matmul.so: undefined symbol: _Z33cublas_tensor_core_matmul_cuda_4dN2at6TensorES0_
```



### 2.1.2  Pytorch Extension例子

+ 重要的参考
  + [Pybind-Doc：python如何与C++绑定]( https://pytorch.org/docs/master/cpp_extension.html)
  + [【教程】：pytorch如何拓展 cpp/CUDA](https://pytorch.org/tutorials/advanced/cpp_extension.html)：后面的信息、链接均来自于此教程

实际上支持两种拓展

+ setuptools：AOT - ahead of time（萌哥的代码中所涉及的部分 ）
  + `<torch/extensions.h>`包含了有：

    + `<ATen/ATen.h>`：The **ATen library**,  Tensor计算的APT，包含了数据类型
    + `<pybind11.h>`：[pybind11](https://github.com/pybind/pybind11),  Python 与 C++ 相绑定的工具

  + 在当前目录下保持 

    ```shell
    pytorch/
      lltm-extension/
        lltm.cpp
        setup.py
    ```

    编译语句使用：`python setup.py install --uesr || exit 1 `

  + 对比脚本下

+ [`torch.utils.cpp_extension.load()`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load)：JIT - just in time（【MLSys22】）

  + 构建过程
    1. 创建一个临时目录`/tmp/torch_extensions/lltm`
    2. 将Ninja构建文件发送到临时目录中
    3. 将源文件编译成进 shared library
    4. 将此shared library导入为Python moudle
  + 构建结论上而言：**Python moudle的使用方式和setuptools构建的是完全一样的**
  + 注意，因为用了Ninjia，所以编译是增量的incremental，如果reload extension时没有修改cpp这些，那么过程会很快
  + JIT的形式中，.cpp和.cu文件可以同名，但是AOT的就不行

+ JIT中更近一步 使用手手写CUDA

  1. 先写一个类似 AOT 那种方式时的 cpp文件 ：此中定义了函数 并且具有pybind11的作用
  2. cpp中的函数也起到了声明 .cu中的函数的作用

+ JIT实例 - Tensor_add的操作

  + 问题1【未解决】：关于`demo2_script.py` 在经过多次修改以以后，似乎**无法编译**；重新写了个与之内容完全一样文件（文件名不一样）`demo2_script_bak.py` 重新编译，编译又成功 --- 玄学。    

  + 问题2：**数据类型**：在指定张量的时候就必须确定为float类型；同时在为结果开辟空间的时候也必须确定类型；结果数据类型如果是多个张量的话应该是`std::vector<torch::Tensor>`；只返回一个张量则是`torch::Tensor`

    ```c++
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(tensor1.device());
    auto result = torch::empty({d1, d2}, options);
    ```

  + 问题3：数据没有搬上GPU --- 但这一步在MLSys的代码中并没有体现？

    ```cpp
    //确保在传递给CUDA内核之前，PyTorch张量的数据已经被移到GPU上
        tensor1 = tensor1.cuda();
        tensor2 = tensor2.cuda();
    ```








### 2.1.3 Pytorch Extension上压力

+ Demo3 的例子 - Matrix Mul

  + 使用python的 `time.time()`进行计时 - 基本思路和C保持一致；

  + 对GPU的显存 记录

    ```python
    GPUs = GPUtil.getGPUs()
    GPU_Memory = GPUs[0].memoryUsed   #此处之所以为GPUs 显卡需要是list，哪怕只有一张都应该是GPUs[0]
    ```

> 从这一步开始发现，除了报错需要及时记录解决过程和分析成因外，较为流畅的过程就不做太多记录；代码中也有相应的注释可用



### 2.1.4 torch中的显存记录

使用GPUtil，记录的似乎是 直接获取的显存信息，不同的方法间显存一样 ---- 不对

```python
import GPUtil

GPUs = GPUtil.getGPUs()#检查GPU显存占用
GPU_Memory = GPUs[0].memoryUsed   #此处之所以为GPUs 显卡需要是list，哪怕只有一张都应该是GPUs[0]
..... （program）
print("Golden GPU Memory: %s MB"%GPU_Memory)
```



### 2.1.5 demo再运行出问题

同样的一段程序我现在编译不了，（除非是以前编译生成过了对应的文件，**此次**没有新编译，直接用的.so文件），出现问题如下

```shel
/local/lib/python3.10/site-packages/torch/include/pybind11/cast.h:%2520In%2520function%2520‘typename%2520pybind11::detail::type_caster<typename%2520pybind11::detail::intrinsic_type<T>::type>::cast_op_type<T>%2520pybind11::detail::cast_op(make_caster<T>&)’:
```

**解决**：降低gcc的版本（由于现在安装的是gcc12 --- `CUDA12.1` 安装的时候所要求的升级）

+ 编译这个还是需要降回`gcc11/g++11`，这两者的版本务必要匹配；否则也有问题

  ```shell
  sudo update-alternatives --config gcc
  sudo update-alternatives --config g++
  ```

  

## 2.2 CUDA-torch实现

实现目标：一个标准的Transformer但使用我们的算子初步进行替换

+ 前置知识

  1. 标准的Transformer（Pytorch）关注inference过程 代码在哪里找？长什么样？

     参考D2L、[李沐视频 - 注意力机制](【64 注意力机制【动手学深度学习v2】】 https://www.bilibili.com/video/BV1264y1i7R1/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)

  2. 性能指标的理解 - 计时方式和具体操作是否合理 - MLSys22的方式也能照搬但不太理解

     这一步转入了2-LearnNote



### 2.2.1 替换torch.bmm()

```python
scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d) 
self.attention_weights = masked_softmax(scores, valid_lens)
return torch.bmm(self.dropout(self.attention_weights), values)
```

其中数据的维度为`queries:torch.Size([256, 10, 8]) keys.transpose(1,2):torch.Size([256, 8, 10]) values:torch.Size([256, 10, 8]) `

希望替换为下面，但显然维度对不上

```python
scores = matrix_mul.run_matrix_mul(queries, keys.transpose(1,2))  / math.sqrt(d) 
self.attention_weights = masked_softmax(scores, valid_lens)
return matrix_mul.run_matrix_mul(self.dropout(self.attention_weights), values)
```

+ 对比`torch.bmm` / `torch.dot` / `@`/`torch.matmul` 
  + `torch.bmm` 用于批量矩阵乘法，
    + 适用于批量矩阵乘法（**Batch** Matrix-Matrix Multiplication）。输入张量的形状应该是 `(batch_size, n, m)` 和 `(batch_size, m, p)`，输出形状是 `(batch_size, n, p)`。每个 `n x m` 的矩阵都与对应位置上的 `m x p` 矩阵相乘。
  + `torch.dot` 用于一维张量的点积
  + `@` 运算符和 `torch.matmul` 用于一般的矩阵乘法，
  + 其中 `torch.matmul` 提供更多的灵活性



## 2.3 测试基准FasterTransformer

+ 依照 Nvidia官方的github进行docker安装https://github.com/NVIDIA/FasterTransformer/blob/main/docs/bert_guide.md#requirements

  + 我的设备4080laptop，所以对应的 Compute Capatity sm_89
  + 在本次尝试中 使用docker来作为运行环境，文件路径挂载到宿主上，所以实际使用VScode来操作

+ ```
  #拉一个docker下来，nv官方提供镜像，大小16GB CUDA 11.8  ubuntu版本为22.04
  docker run --gpus all --name FasterTransformer -itd -v /home/david/Documents/other_work/FasterTransformer:/FasterTransformer nvcr.io/nvidia/pytorch:22.09-py3 /bin/bash
  ```

  + 需要在`/FasterTransformer/FasterTransformer/build`目录下，通过 执行

  + ```python
    python ../examples/pytorch/bert/bert_example.py <batch_size> <layer_num> <sequence_length> <head_number> <size_per_head> <--data_type fp32/fp16/bf16> <--int8_mode 0/1/2/3> <--sparse> <--time>
    python ../examples/pytorch/bert/bert_example.py 1 12 32 12 64 --data_type fp16 --time
    ```

整理了一下比较关键的参数

| 参数          | bert_example | bert_trans_test |      |      |
| ------------- | ------------ | --------------- | ---- | ---- |
| batch_size    | 1            | 16. (1)         |      |      |
| layer_num     | 12 🤗         | 12              |      |      |
| seq_len       | 32 --- 64    | 64              |      |      |
| head_number   | 12🤗          | 12              |      |      |
| size_per_head | 64🤗          | 64              |      |      |
| avg_seq_len   |              | 32              |      |      |

备注：

hidden_size = hidden_dim

num_attention_heads=head_num

num_hidden_layers=layer_num



## 2.4 叫停 期望找官方的版本

> 2.3 中的pytorch版本的  FasterTransformer 所实现的对比 感觉摸到了**对比**的头绪
>
> 总的来说找的这些代码对比起来有个问题：**面向对象编程 --- 难以拆（怕计算要素不全）**不是依照算子组织的

### 1 - 坚持找pytorch官方实现

#### （1） torch.nn.Transformer

+ **Doc** Encoder结构官方实现 [Docs >torch.nn >TransformerEncoder算子的介绍]( https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) ; 但还是面向对象编程、有源码

  + 在他的上一层还有[torch.nn.Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)； 

  + 层次关系 torch.nn.modules.transformer < TransformerEncoder < TransformerEncoderLayer

    ```python
    transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    ```

  + 缺点在只是**Doc** 没有专门做对比的感觉，不如找别人的工作直观 --- 没有更具体的例子

    ```python
    CLASS torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)
    ```

+ **Example**  https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html 相对完整点，用了官方的Encoder(上面)、

  > ```python
  > from torch.nn import TransformerEncoder, TransformerEncoderLayer
  > ```

  + 但主要是多卡的并行的例子
  + 更基础的参考⭐️ [使用 Torchtext 与 nn.Transformer实现的transformer_tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html?highlight=transformer%20encoder) 这个例子更偏向于D2l那种，是功能上的实现：（1）对于时间测评不清晰，不是我想要的（2）编程上过于面向功能设计、还原设计结构 -- 语言功能（3）训练+推理 搞到一

#### （2）Pytorch Hub的实现 - Hgf

+ PyTorch的官网 - 资源Resoureces/Models(Meta) --- [Pytorch Hub - 完全交给了Hg](https://pytorch.org/hub/huggingface_pytorch-transformers/)
  + view on Github则会直接跳转 Hugface 的 Github网页（回到了Hgface）
  + FasterTransformer倒是也对比了Hgf，但人家不是paper的工作，而且参数设置没明白

#### （3）⭐️最可观的例子 - Attention 

+ 【有戏】[高性能的Trasnformer实现 with Scaled Dot Product Attention ：SDPA](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)例子实现，主要是关于Attention的，有完整代码

  + 涉及到了 Fused的 QKV部分

  + 目前名为 `torch.nn.functional.scaled_dot_product_attention` 已经被集成进 `torch.nn.MultiheadAttention` /`torch.nn.TransformerEncoderLayer` 即**（1）**

  + 对于现在集成进 F.scaled_dot_product_attention的操作 已经有了优化 （下面三者之一）

    + [FlashAttention](https://arxiv.org/abs/2205.14135)  / [Memory-Efficient Attention](https://github.com/facebookresearch/xformers) / Python 一般实现的Attention

    ```python
    #对于现在集成进 F.scaled_dot_product_attention
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    
    The default implementation runs in 4298.660 microseconds
    
    The math implementation runs in 22879.881 microseconds #无优化的版本 -- math
    The flash attention implementation runs in 4329.078 microseconds #证明现在pytorch用的是flash attention
    The memory efficient implementation runs in 4457.291 microseconds
    ```

  + **收获点 了解做profile的工具，**然后用 `chrome://tracing`打开所保存的 `.json`文件

    ```python
    from torch.profiler import profile, record_function, ProfilerActivity
    ```

#### （4）Better Transformer

+ **Example** [FAST Transformer Inference with better Trasnformer](https://pytorch.org/tutorials/beginner/bettertransformer_tutorial.html#fast-transformer-inference-with-better-transformer)；

  + 能用这种方式加速的前提是：实现上都基于了**（1）**中的那些， 所以**缺点完全继承**了

  + 样例中基于了 `torchtext.models`已经预训练好了的模型，直接import

    ```python
    import torch, torchtext
    from torchtext.models import RobertaClassificationHead
    
    xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
    model = xlmr_large.get_model(head=classifier_head)
    #测评时 --- 有点像 FasterTransformer 所演示的样子：做推理
    with torch.no_grad():
        for i in range(ITERATIONS = 10):
          output = model(model_input)
    ```

+ 特性上：支持了稀疏---Exploiting **sparsity** in NLP inference：来自于输入长短不一样时候的 padding



### 2 - 看别人的对比消除对.so的恐惧

| 工作项目                                                     | 对比的                                                   | 备注                                                         |
| ------------------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ |
| [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) | 📌**对比明确**hgf的例子；自己的两级优化                   | 从这个[例子](https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/bert/bert_example.py)中摸清了 对比**脚本写法、数据准备、代码组织、jit**、热身+100次迭代对比encoder ---✅对比时间 ✅和hg对比 |
| [TurboTransformer](https://github.com/Tencent/TurboTransformers/)PPoPP21 | ❌ 融合可以看，但项目本身不参考                           | 项目较大，可以选择的模型很多；但对比的过程代码不清晰         |
| [ByteTransformer](https://github.com/bytedance/ByteTransformer)IPDPS23 | 代码中**没有进行对比** - 只有自己怎么跑、📌**正确性验证** | 本[例子](https://github.com/bytedance/ByteTransformer/blob/main/unit_test/python_scripts/bert_transformer_test.py)中的代码前向结构 正确性基准测试是**面向过程**编程，**计时明确** ✅对比正确性 |

+ 对于一些模型设置的初始化 还是会参考 hgf🤗 `import transformers`然后用`BertConfig()\BertModel()`来设置参数

+ TurboTransformer对比过程详解

  + [TurboTransformers](https://github.com/Tencent/TurboTransformers/tree/4532fa118c07375b3650f0768b70982c914be4ce)/[benchmark](https://github.com/Tencent/TurboTransformers/tree/4532fa118c07375b3650f0768b70982c914be4ce/benchmark)/下 ： benchmark.py 选择框架torch--- >  torch_benchmark_helper.py 选择模型Bert ---> benchmark_helper.py

    

### 3 - 上代码试试

| 跑通             | 代码项目    | 主要目的                                                     | 备注                                                         |
| ---------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 后来发现**错误** | benchmk1.py | 摸清**对比时间**的方法、和**别人对比**的方法 --- 主要对比了🤗；把FasterTransformer的对比部分抽取出来，取hgf的基准例子<br />benchmk1.py似乎有错误，计算不正确；batch_size只能为1 | 可以选用jit：不开5.78ms / 开3.3ms                            |
| ❌ 想拆开 但放弃  | benchmk2.py | bhmk1.py中的基准 摘取源码，希望组织成 **面向过程** ---- 组织过程怕计算要素不全，导致基准就是错的！ | 有源码 但**面向对象编程**，算子流程不直观                    |
|                  | benchmk3.py | 摸清对比正确性；来自在 ByteTransformer                       | 正确性验证 有一个前向的基准**(结果不正确)**                  |
|                  | benchmk4.py | 修正benchmk1.py的问题  顶替1算出来的东西确保正确<br />确保了**mask、input**是一致的 | torch.jit在这个版本的torch中跑不通，需要到docker中（torch = 1.13才能用） 两种方式生成的**权重不一致** |



# 3 - 底层代码构建

正式开始构建自己的代码

| 代码项目           | 主要目的                                                     | 备注                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| base1.py           | 把ByteTransformer的例子中基准代码抽取出来，作为我的**基准**  | 加上了备注，和FasterTransformer中的流程图对齐。数据的生成还是随机的 `uniform_(-0.4, 0.4)` |
| profile_sparity.py | 探索其中的稀疏性：非零元数量比例、元素分布、小于某一阈值的数量（一直想做，一直不知道如何下手） | 数据生成随机，计算流程和base1中的保持一致 --- 稀疏性的探索结果和姝慧姐的**结论保持一致** -- 发现是重复造了轮子 |
| base2.py           | 用之前的方法把torch.matmul替换成我自己的算子，用之前的cuda文件**Load** | ❌困难：我自己写的run_matmul与之 **维度对齐**                 |
| fuse.py            | 把ByteTransformer中的（.cu / .h）**融合代码**切出来，放到base1中做一定的替换 - **希望基于它** | ❌因为其中是.cu代码，替换进去也需要load工作。没成功原因：①他实现的融合只针对fp16 --- 用了tensor-core；各处类型对齐不敢保证正确性 ②只有定义没有调用关系- CMake为.so的形式 ③融合后的代码，因为有特殊优化zero-pad,很多索引眼花缭乱 |
| my_sdp.py          | 用我的基准和 `F.scaled_dot_product_attentiond`对齐结果，比较正确性 | Pytorch - `F.scaled_dot_product_attentiond`中对于mask / scale值也不一样；超参规模小时，可以对上；规模大了精度差异会扩大 |
| fuse1.py📌          | 因为fuse的失败，自己构建融合的MHA，放到base1中 ，与End Attention之后的 部分对齐计算结果 | 难点：①与base2相同 ②中间涉及维度变换 索引应看清 ③CUDA编程老大难，多多熟悉 -- - 任务划分<br />涉及很多CUDA编程深水区 --- 算子融合尤其看重`shared Mem` |
|                    |                                                              |                                                              |

+ 在fuse1.py 的构建过程中 考虑要**停手**，补充两方面知识
  1. CUDA编程专家知识 -- 时间线长、深度大，但借助一些例子再次上手
  2. softmax的实现：[SoftMax基本机制](https://www.zhihu.com/question/435368791) , 原来的代码究竟在哪个维度上softmax（dim = -1）✅
  
  ```shell
  #编译完成，但出现问题 ---- 同萌哥代码中的一致 -- 貌似是其中的一个函数
  ImportError: /home/david/.cache/torch_extensions/py310_cu121/my_fused_attention/my_fused_attention.so: undefined symbol: _Z18run_batchMatrixMulN2at6TensorES0_
  ```
  
  解决问题：`.cu`与`.cpp`中的函数并不匹配 --- `run_my_fused_attention（）`在`.cu`中有，但`.cpp`没有，所以出问题
  
  

### sqrt未曾想到的问题 

后来发现是自己的数据维度没有对齐

```python
#python代码的程序
scores1 = torch.matmul(A.float(), B.float().transpose(-2, -1)) / (head_size ** .5)
#CUDA 代码的程序 --- 这个sqrt的结果与head_size有关系
score = score / sqrtf(static_cast<float>(head_size));
```



### q/k被 覆盖 - 交换位置了

```shell
q :  tensor([[[[-0.0089,  -0.4835, 0.1279,  0.3058],
          [0.2417,  -0.2100,  0.1727,  0.2396],
          [0.1062,  -0.4100, 0.0001,   0.2044]]]], device='cuda:0')
k :  tensor([[[[ 0.0864, -0.0528,  0.0878,  0.4383],
          [-0.0909, -0.0006, -0.0388,  0.1715],
          [-0.0174, -0.1024,  0.1518,  0.2891]]]], device='cuda:0')
q( 0, 0): -0.008909 * k ( 0, 0):  0.086396
q( 1, 0):  0.086396 * k ( 0, 0):  0.086396
q( 2, 0):  0.624452 * k ( 0, 0):  0.086396
q( 0, 0): -0.008909 * k ( 0, 1):  0.624452
q( 1, 0):  0.086396 * k ( 0, 1):  0.624452
q( 2, 0):  0.624452 * k ( 0, 1):  0.624452
q( 0, 0): -0.008909 * k ( 0, 2):  0.241676
q( 1, 0):  0.086396 * k ( 0, 2):  0.241676
q( 2, 0):  0.624452 * k ( 0, 2):  0.241676
q( 0, 1): -0.483517 * k ( 1, 0): -0.052824
q( 1, 1): -0.052824 * k ( 1, 0): -0.052824
q( 2, 1): -0.265493 * k ( 1, 0): -0.052824
q( 0, 1): -0.483517 * k ( 1, 1): -0.265493
```

显然其中的 `q(1,0)、q(2，0)`被 k的某些元素给覆盖掉了；而`k(0, 2)`又被`q(1,0)`覆盖了



## 3.1 Fuse1 的展开 debug

| 文件名                                     | 目的                                                         | 备注                                                         |
| ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| fuse1.py📌                                  | 因为fuse的失败，自己构建融合的MHA，放到base1中 ，与End Attention之后的 部分对齐计算结果 | 难点：①与base2相同 ②中间涉及维度变换 索引应看清 ③CUDA编程老大难，多多熟悉 -- - 任务划分<br />涉及很多CUDA编程深水区 --- 算子融合尤其看重`shared Mem` |
| fuse1_fix1.py                              | 试跑代码的一般基准 - QKT计算正确                             | 只针对于 head_num、batch_size = 1的情况                      |
| fuse1_fix2.py                              | QK^T_mask_softmax正确                                        | 只针对于 head_num、batch_size = 1的情况 / seq_len< 16；其中必须head_size >= seq_len --- 一般head_size = 64； |
| fuse1_fix3.py -`fused_attention.cu0124`    | 不追求Attention的全过程；保证`QK^T_mask_softmax`，之后的*V过程返回到python中算 | 确保seq_len > 16与 <16✅的情况都能正确<br />把seq的两种大小分开写；<=16的情况写作一个算子，> 16写成两个算子（线程规划变化） |
| fuse1_fix3.py - `fused_attention.cu0125-1` | ①batch_size 可以不等于1，在1 ，8 ，16 都可以成功 ②head_num 可以不等于1， 在测试时一般设置为 12 | ✅充分使用 `q.storage_offset() &  q.stride()`                 |
| 回归fuse1.py - `fused_attention.cu0125- 2` | seq_len 16 -- 1024(总是8的倍数) 都统一为一个kernel --- 任务规划一个thread处理一行(seq_len)，把GEMM2也融合进来 | 任务规划改为了 一个thread处理一行（对比实验 - 效果很差）🤯    |



## 3.2 上syncfree之后的测试

> 无同步：咨询阳哥以后 - 不需要等待多个kernel的启动；更多是一种解释上的东西

此时的结论是 - 在代码的构建阶段，如果方向逐渐清晰了；不需要写那么多东西 - 太花时间了



## 3.3 ncu与nsys 分析kerenl性能

+ `nsys：Nvidia Nsight Systems`粗粒度分析 / `ncu：Nvidia Nsight Compute`细粒度分析 

  + 前者是Sys级的，不仅对GPU，还对CPU/IP以及OS都有分析到 --- 就是数据指标看球不懂～

+ 两者都是对kenrel层面的分析（哪怕你编译成了`.so`），比如我的任务中是`python XXX.py`其中调用的cuda都能看到

  + 因为是直接对GPU做的监测，无论以什么方式运行的程序，只要用到的CUDA kernel被提交给了GPU，那么就都能看到
    + 所以不仅仅是`nvcc sample.cu -o sample.out `生成执行的文件能用命令`ncu sample.out`监测，像我的python脚本也可以用`ncu python sample.py`来监测

+ 使用：Linux监测 和 查看 分离

  + 本地需要在[Nvidia Developer / CUDA compute]( https://developer.nvidia.com/tools-overview)中按照你的本地系统来下载查看软件

  + 通过先在命令行中生成对应的`.ncu-rep`或`.nsys-rep`；然后拉到本地-用查看软件打开

    ```shell
    #nsy的分析方法 - 生成的报告文件在当前路径下，名为 XXX.nsys-rep
    nsys profile python syncfree1.py 8 1 256 12 64
    
    #ncu的分析 - 可以在命令行就查看 也可以保存为报告
    #在命令行中直接查看；方便，但kernel一多输出就很难接受了；不需要对应的查看软件
    ncu -o python syncfree1.py 8 1 256 12 64  
    #生成报告，名为 XXX.ncu-rep
    ncu -o rep1 python syncfree1.py 8 1 256 12 64
    ```


### 分析实例 - 指标的解释

```shell
syncfree_triangle_attention_kernel(const float *, const float *, const float *, float *, int, int, int, int, int, int, int, int)
Begins: 4.11906s
Ends: 4.12081s (+1.749 ms)
grid:  <<<256, 8, 12>>>           #kernel的任务划分
block: <<<32, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 136 bytes   #每个block的占用，我只有34个float 34*4B=136B 
Dynamic Shared Memory: 0 bytes    #我的代码中没有动态的这部分设置---在<<<gridSize, blockSize>>>中没设置
Registers Per Thread: 37
Local Memory Per Thread: 0 bytes
Local Memory Total: 79,822,848 bytes
Shared Memory executed: 32,768 bytes
Shared Memory Bank Size: 4 B     
Theoretical occupancy: 50 %       #SM占用率:一个SM中active warps数/最大可能的active warps数
Launched from thread: 48208
Latency: ←5.236 ms
Correlation ID: 745448
Stream: Default stream 7
```

+ 关于Occupancy占用率的解释[知乎 - GPU基础：Occupancy、wave and tail effect](https://zhuanlan.zhihu.com/p/657005697)

  + **高占用率**不总是代表高性能；没那么高的时候，就是有限的任务，但资源很丰富能干好 
  + 但是（过）**低占用率**总是会干扰隐藏内存延迟的能力，性能会下降
  + 所以存在一个最佳点，超过这个点以后 ，提高占用率不会提高性能

  

  

## 3.4 专注于提性能

> 总体性能目标：把加速比全部提到1以上
>
> 论文中会有主旨的优化方法，但这种方法可能只造成了20%的加速，剩下80%的加速来自于代码实现上的小trick

1. `dim3 blockSizeDim(head_size/2)`，warp内展开

   + 一个warp干2个warp的事减小同步成本; 

   + 一切`syncthreads()`更换为更轻量级的`__syncwarp()` 

   + 使用warp内的同步/广播机制 `shfl_xor_sync/shfl_sync`替代原来的部分`__syncwarp()`同步

     (由老师的无同步代码带来的启示)  主要参考[cuda-c-programming-guide关于cuda-c-programming-guide的介绍](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)

   + 使用指令`__ldg()`没啥提升 - [常量内存与纹理内存、表面内存使用](https://zhuanlan.zhihu.com/p/680075822)

     帕斯卡架构及其更高架构默认使用 `__ldg()` 函数读取全局内存，故不需要显式使用

2. `dim3 blockSizeDim(head_size/2, head_num)`，

   + 仍然保持warp内展开；但同步只同步warp内的线程；

   + 可能提高了SM的利用率

     

3. 提取别的代码中的抽象思路，放进我的代码中；;主要参考代码来源

+ 参考方法：先看看代码组织框架 ---> 找到核心脚本 与 实现Attention的部分 ---> 主看代码，论文为辅 

+ ByteTransformer 

  + 优化点

  + 核心代码

    `/other_work/ByteTransformer/unit_test/python_scripts/bert_transformer_test.py`

    `/other_work/ByteTransformer/bytetransformer/src` 

+ FlashAttention

+ FasterTransformer

+ TurboTransformer



# 4 - 中层代码构建

| 文件名                   | 主要目标                                                     | 备注                                                         |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1️⃣generate_demo_single.py | 【3.01】对于当前0-14的算子可以随机选一段来进行融合 - 放进一个函数里包起来 | ①函数的名字搞得更专业一些 ②`os.system` 执行尝试 ③恰好融合Attention的部分换成我的算子 |
| 2️⃣generate_multiseg.py    | 【3.05】对于当前0-14的算子可以**随机选多段**来进行融合，决策顺序  ①是否融合②是否调度手写算子③融合几段④几段的融合范围 | `python generate_multiseg.py B L S H W mask_id 0/1 0/1 seg_num [start, end]` |
| 3️⃣translate.py            | 【3.10】给定输入value ( < 65535)  翻译为对应的二进制数，然后翻译为对应的 段数n + n段[start, end] | `python translate.py value`                                  |
| 4️⃣generate_genidx.py      | 【3.10】融合2️⃣3️⃣                                               | `python generate_multiseg.py B L S H W mask_id 0/1 start`    |
| 5️⃣traverse_generated.py   | 【3.10】遍历4️⃣中的生成的代码文件； 排错,解决抽取代码的BUG; 寻找搜索空间 | 搜索空间 初步确定$[32768, 49152] = [1000\thinspace0000\thinspace0000\thinspace0000, 1100\thinspace0000\thinspace0000\thinspace0000]$ |
| 6️⃣                        |                                                              |                                                              |
|                          |                                                              |                                                              |



使用ChatGPT实现value编码的过程 --- translate_demo.py
>现在我有16个位置pos，pos0 - pos15，其中的每一个位置可以取值为0或1
>
>按pos0-pos15的顺序将其中的所有值排列起来，会得到一个值value
>
>
>
>我有这样一个翻译的需求，将这个值value翻译为下面的一组数：
>
>[num] [pos0] [segment_num] [start_1] [end_1] ... [start_n] [end_n]
>
>说明：
>
>1. 翻译的要求是：当 pos_i和pos_i+1的值相同时，我们就认为他们是一个segment，当pos_i到pos_i+n的值都相同，那么他们都是一段。
>2. [num]是value从二进制数 ，由pos0-pos15组成的16位字符串翻译而来的一个整数值
>3. [pos0]就是 pos0的取值，即0或1
>4. 以上的[start_n] [end_n]的组数，取决于segment_num的数量，例如segment_num = 1那么[start_n] [end_n]对只有1组即[start_1] [end_1]；segment_num = 3那么[start_n] [end_n]对有3组即[start_1] [end_1] [start_2] [end_2] [start_3] [end_3]
>5. 我会给你一段连续的输入，是pos0-pos15的值，也就是一串长度为16的由01组成的字符串
>
>
>
>举例：
>
>1. 输入是0001101010101010时，翻译结果是6826 0 2 1 2 3 4
>2. 输入是0000010101010101时，翻译结果是1365 0 1 1 4
>3. 输入是0011111010010110时，翻译结果是16022 0 3 2 6 9 10 13 14
>4. 输入是0101010101010101时，翻译结果是21845 0 0 



针对于

遍历一遍，搜索空间：

+ 使用手写算子：[32768, 49152]
+ 不使用手写算子：[16384, 24576] 恰好是上面的一半



# 5 - 上层 - 采样搜索

用XGboost来训练一个分类器，做到我给出一个16位的字符串，输出后面的字符串

每一个XGboost训练的结果都是针对于当前参数下的（batch_size, seq_len , **mask**）

> 现在有这样一个用XGboost训练的需求，我希望这个XgBoost分类器为我训练一个模型，主要功能是得到一个输入16位2进制的字符串，能够给我返回一个预测的值value，并且有一个关于关于这个value值在总体中属于（High Performance）1或者（Low Performance）0的预测。
>
> 我给到你一个数据集，这个数据集的组成是 一个16位的二进制字符串，和一个值value。注意这些数据中value值越小，则是High Performance，越大则属于Low Performance。
> 总共有数百对这样 “字符串 - value对”，这个数据集的名称为"train_data_xgboost.txt"，里面的数据如下形式
>
> 1111100011001100   54.524
> 0100010111011101   55.053
> 0100110000010101   57.406
>
> 所以对于这个xgboost的训练出来的模型，能够做到：我输入一个16位的 二进制字符串，它输出一个预测值value，并告诉我这个值属于High Performance 1 或者 Low Performance 0
>
> 样例输入：0100010111011101
>
> 样例输出：54.524 1

