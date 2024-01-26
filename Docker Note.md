# 0 - 初见Docker

**为了解决什么样的问题？** ---   部署环境复杂，需要一个整体的容器

> 比如有一个web，但是想部署到服务器上：（1）软件一样（2）相同的操作系统发行版 仍然有差异

是什么东西：开源的 **应用容器引擎**，基于GO语言，开销小启动快



三个重要的概念：

+ 镜像**image** ：环境的快照，有所有的依赖和基础环境（ubuntu、Debian） --- 相对静态的概念 
+ 一个镜像包image含多个容器**containers**, 每个都相当于虚拟机相互独立，运行自己程序
+ Docker脚本**Dockerfile**：生成Docker（image）的脚本
  + FROM、WORKDIR 、COPY、 RUN（创建）、 CMD（运行）


> Docker与Kubernetes：两个东西并不是面向同一个层面，不是取代关系
>
> Kubernetes主要是使用集群：负载均衡、故障转移，使得全自动化管理

<img src = 'https://s3.bmp.ovh/imgs/2023/09/21/abec7ffdf8fb8091.png' style="zoom:33%;"  >



# 1 - 安装使用Docker

+ 在hub.docker.com上所搜索得到的 镜像，实际上已经指定了操作系统

  + 网站上有对应的标签，可以选作FROM的后面
  + 创建时，Docker的端口号和本机的端口号不一样，要单独暴露
  + 首次构建比较慢，但是之后就会有“缓存”了

+ 基本语句

  + docker images **罗列镜像** / docker rmi **-f**  删除镜像（强制）/ docker tag （docker命名）

  +  docker run **-d** 运行镜像(分离后，运行在后台不占用命令行)

  + docker ps **罗列容器** 正在运行

  + docker stop+id 暂停运行的**容器**（关机） / docker rm -f id删除

  + docker exec -it 进入某个**容器** 以命令行的形式进行交互 --- exit退出交互 

  + Docker-compose同时启动多个容器，初始时需要使用.yml来编写配置文件

    + Docker-compose up  -d --build创建 / down删除

      