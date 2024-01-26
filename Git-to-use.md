> git神秘又熟悉，每次提起都说不会，那么要在什么时候才能拍着胸脯说我会呢？我想就是今天吧！
>
> 干瘪瘪得去学效果肯定是不好的，套进具体的开发场景，强制自己用它来管理熟能生巧

【**特别好的Ref**】初步上手 建立概念体系⭐️ [B站 - Git工作流和核心原理 | GitHub基本操作 | VS Code里使用Git和关联GitHub](https://www.bilibili.com/video/BV1r3411F7kn/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)

**Reference：** ①[MIT缺失的一课 - Git](https://missing-semester-cn.github.io/2020/version-control/) - 本身有讲座做讲解，逻辑上自底向上，更合适初步接触；其余介绍都是自顶向下 - 出自接触则记不住，其中**资源衍生**也非常多，妙极了！妙在既可以整体看一遍就完事，也可以深入看； ②[B站-清华自动化协会分享Git](https://www.bilibili.com/video/BV1Ab411Q7JW/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243) - 关于手搓网站的系列培训，1h，整体过一遍 ③[Git Pro](https://git-scm.com/book/en/v2) 深入学习，起到字典的作用；④[游戏 - 练习Git相关知识的小网站](https://learngitbranching.js.org/?locale=zh_CN&NODEMO=)可视化展示git的过程。





# 1-一张图的工作原理

<img src = 'https://s3.bmp.ovh/imgs/2023/05/17/9d406ba609998c2d.png' >

1. workspace/unstage工作区，当前存放项目代码的地方
2. Index/Stage暂存区，用于**临时**存放改动，保存**即将提交**的文件列表信息。
3. Repository仓库区，存放数据的位置，这里面有你提交到所有版本的数据。
   + 其中HEAD指向最新放入仓库的版本
4. Remote远程仓库，托管代码的远程服务器

+ Git安装配置  - 查看配置信息 `git config --list`
+ Git的逻辑：**提交是不可改变的**。实际上不存在覆盖，只有新建版本 - 树形管理；
  + 但并不是错误不能修改 ，而是**修改**实际上成了**全新的提交**
  + 所以他是一款**版本控制系统**




> 发现这里记笔记的思路有错误，不应该做成百科全书，要快查的话网上有资源

# 2-本地操作

+ `git` 命令都对应着对提交树的操作，例如增加对象，增加或删除引用
  + s所以进行一切操作的逻辑和底层思想：思考这个操作对（树形的）数据结构会产生什么影响

| 指令                    | 功能                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `git status`            | 查看仓库状态                                                 |
| `git log`               | git commit后查看历史提交记录$^1$                             |
| `git diff （--cached）` | 比较**1-工作区**与**2-暂存区**，（暂存区与上一次的commit）,还可以比较两个分支差异 |

$^1$提交记录的版本号，是个[SH-1的哈希值](https://baike.baidu.com/item/SHA-1/1699692)，呈现为40位十六进制数，实际上输入**前4位**就可以找到东西啦！

## 工作区与暂存区

| 指令                            | 功能                                          |
| ------------------------------- | --------------------------------------------- |
| `git init`                      | 初始化仓库（把这个文件夹变成可以git操作的东西 |
| `git add <filename>`            | 把文件从**1-工作区**添加到**2-暂存区**        |
| `git rm ( --cached) <filename>` | 把文件从1/2删除，（只从暂存区删除）           |
| `git commit -m <message>`       | 提交信息到**3-仓库**（备注）                  |

对于git的版本控制来说，仓库里的才是正式的东西

+ `git commit` 了以后 ，才产生了新的版本
+ `git branch` 更多只是在工作区自行做了不同位置的修改而已



## 版本控制-回退

+ 备注不知道版本号：git log就知道
+ 版本号一长串实际上是个哈希值，<version>只需要给出**前7位**就可以

| 指令                          | 功能                                            |
| ----------------------------- | ----------------------------------------------- |
| `git reset HEAD^`             | 回退到上一个版本，（^表示相对引用，~<num>也是） |
| `git reset --mixed <version>` | 回退，修改内容进**工作区**                      |
| `git reset --soft <version>`  | 回退，修改内容进**暂存区**                      |
| `git reset --hard <version>`  | 彻底回退，修改内容清除                          |

+ 注意相对引用 前面的名字必须是 HEAD或者<branchname>

## 分支管理

实现版本控制的关键

创建分支实际上是对现有分支的**一个拷贝**

| 指令                                  | 功能                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| `git branch <branchname>`             | 创建分支                                                     |
| `git branch -v`                       | 查看所有分支（更改版本）                                     |
| `git branch -d <branchname>`          | 删除分支 ；如果非常得确定要删除 就是`git branch -D`          |
| `git branch -m <old-name> <new-name>` | 分支改名                                                     |
| `git checkout <branchname>`           | 切换分支                                                     |
| `git checkout -b <branchname>`        | （从当前分支）**创建新分支**并**切换**过去                   |
| `git merge <branchname>`              | 分支合并---这个分支是**别的分支**名，合并到当前分支 --- 合并以后产生了一个**新的版本**；如果产生 分支冲突（两个分支对同一内容做了修改） |

> Git 2.23以后 可以使用 `git switch`代替`git checkout`

还有一种合并分支的方法：`git rebase `	<branchname>，使 得到更线性的提交记录；注意这里的branch是将要rebase到的那个分支名



# 3-远程操作(核心)

更直白的说：和github的联动；gitee一般不做考虑

+ 绑定远程仓库 `git remote add/rm/-v`
+ **拉取** 远程仓库 git **clone** / **fetch** / **pull**(fetch + merge)  --- 是与远程仓库通信的方式

  + clone：在别人的基础上开始干（带有.git文件）；自己有仓库就不要clone
  + fetch：拉取但不（和本地的）合并，自己merge就可以合并
    + 因为网上的代码可能有问题，所以不着急合并
    + 从远程仓库**下载**本地仓库中缺失的**提交记录**，但不改本地的文件

+ **推送** 到远程仓库 `git push origin <远程的branch>`

  + 从本地的仓库推送到远端的仓库
  + remote绑定后，本地和远端github互认则可以推上去自己的；别人的不能随便推上去
  + 推到别人的 **PR**：`pull request`向别人申请拉取自己的代码 

  

### 备注

+ Github 上的issue 一般作为一个评论区，也有很多时候是blog等，BBS？



### 提交事项Gitignore

+ 设置文件的忽略规则，减少提交不必要的文件 ---这部分文件就不提交
+ 创建文件 .gitignore 并写入
  + 文件名  -  重名文件就直接指明目录
  + 支持**正则化表达**：...通配符- 反向操作

```shell
touch .gitignore
# 里面输入不需要提交的 文件名
# 通过 vim .gitignore 来写，其中的文件 git status就看不见了
```









