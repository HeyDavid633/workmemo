> 写在前面的话：首先，宏观上了解好的研究是什么样的，都是非常宏观和抽象的东西。其次，对于审稿的方法论与考量的点做罗列，最后提供了 我个人实际上手的**例子**（对比：我的初次审稿与老师的**反馈**）
>
> 尽管我们大概了解review审稿的精髓 是 拒稿，但其中细节还需要细化下来。网络上杂乱无效信息众多，故写下此Doc作备忘。本文档保持更新，每次更新都是通过实践，与老师的多次沟通，在获得了更多的审稿经验后的总结，对整体组织变化不大。
>
> --- Last Edit by 东环路小浩 30/10/2023

**技术问题参考材料：**① [B站 - 李沐：如何判断（你自己的）研究工作的价值【论文精读】](https://www.bilibili.com/video/BV1oL411c7Us/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)     ②[B站 - CVPR的审稿教程（英文）](【如何进行论文审稿？How to write a good review? - CVPR 2020 Tutorial】 https://www.bilibili.com/video/BV1bg411c7R7/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)   ③[Elseviser - How to conduct a review](https://www.elsevier.com/reviewers/how-to-review)     ④[Elseviser - Becoming a peer reviewer](https://researcheracademy.elsevier.com/navigating-peer-review/becoming-peer-reviewer)     ⑤[【重要】ETH - Timothy Roscoe:”Writing reviews for systems conference”](https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf)

**概念问题参考材料：**①[B站 - 上交科研整体入门、鉴别期刊会议/审稿]( https://www.bilibili.com/video/BV1LW4y1q7Tb/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243&t=1041)     ②[B站 - 博士的角度看（独立）审稿](https://www.bilibili.com/video/BV13r4y1v7KR/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)     ③[马里兰大学 - 审稿人应有的心态](https://sites.umiacs.umd.edu/elm/2016/02/01/mistakes-reviewers-make/) 

**注意**：🧐[【反例】知乎 - 撰写审稿意见 + 模版](https://zhuanlan.zhihu.com/p/468445073) 这篇教程的写作风格上对于我们而言过于拖沓 **不可参考**



# 0 - 了解点抽象的

## 0.1 - 如何判断好的研究

方法核心：有**新意**的方法**有效**解决一个**研究**问题--------（同样也是摘要的写法）

一个简单的判断方式  新意度 × 有效性 × 问题大小 = 价值

对于自己而言某一个维度上拉满也可，但是对于一流论文，不能在这个三个维度上有短板

+ 研究问题：比较困难的问题，而不是工程上的问题（内存不够、精度不够）
  + 有档次上的区别：针对于某一问题、还是能够通用于一整个领域
+ 有效：相对于之前的方法能更有效
  + 模型精度提升1个点、10个点
  + 规模做大，成本降低
  + 做安全
+ 新意：对于这个领域内的研究者来说，方法是有新意的，不一定完全新提出
  + 看了就知道后面要干嘛，那就不行
  + 对于这个领域来说，能够打开新世界的大门

## 0.2 - 李沐的审稿流程参考

+ 对于有些领域的论文（如ACL）可以参考他们官方网站上的 论文格式+review引导

+ 关于写review report 
  +  感谢编辑~  读摘要，对于引言、结论都摘出来，然后给出：interseting study(表示全文已阅)
  + 走流程 
    1. 英文语法正确性 
    2. 摘要：全篇水平-闭环故事：先提出该问题---后来结论解决了该问题）、有没有体现**工作量**（费不费钱-成本估计、样本量大不大）、语法问题、关于缩写有无说明，对于引用不能**扎堆**
    3. 正文：
       + 引言：有没有引用开山之作，既往研究的缺陷，明确的写出研究目的假设等（aim to）
       + 找茬？对比样本量小、验证，如果真的特别了解该领域就直接点出来，对于图表的命名在文中是不是有引用到（检查格式“SO2“下标 -- $SO_2$，数字”10000“分割符 --- $10 000$或$10,000$ ）
       + 对于具体内容：每段的前两句话一定是中心句，然后展开。前两句读不懂就有问题。有没有自相矛盾处
       + 结论：有无夸大实验结果，weaken：对于部分描述方法要求他弱化语气



# 1- 宏观批判的视角

+ 审稿的视角：看的时候就带着**批判的思维**来看，而非来“**学习**”他的内容
  + 仍然要用读文章的方法：通过abstract、conclusion、contribution来快速了解大概
  + 看的时候就可以一边写出相应的意见（比如 批注在一边）；而不是在完全读完后，再来找可攻击的点
+ **Major Concerns** ：要针对方法上多做一些评价，更有深度，可以指出一些方法细节问题 ---- 显得仔细读了论文



# 2 - 审稿方法论的总结

主要牢记：reviewer的天职就是挑刺拒掉审的paper，实在是挑不出刺才能accept

## 2.1 review书写格式

1. 给出接收意见 Accept/Minor revision/Major revision/Rejection + 一段简述总体意见 
2. **2.1**大意见**Major Concerns**  ；**2.2** 小意见 **Minor comment**  

指清楚主要问题，措辞上不要太委婉（见样例）

## 2.2 **review内容结构**

> 针对于具体写review的细节则参考[ETH - Timothy Roscoe:”Writing reviews for systems conference”](https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf)
>
> 该文章适用于小白和有审稿经验的人，涵盖较全：对于结构和具体细节语气委婉的表达都有介绍。同时该文作者为计算机科学家，评论模式参考意义较大。

### 2.2.1 书写结构

1. 总结文章： 文章主题。搞了哪方面内容  
   + 提出该文章的贡献：  通常文章是没有“有用”的贡献，贡献本身有缺陷  

​		以上1与2 对于书写结构的总体意见

2. 指出具体意见：（意见的来源类型参考如下）

书写结构对应如2.1与2.2 ——方法上先列出“问题点”，选择以扩充为大意见/小意见

+ 文章的新颖性，够不够前沿且学术。相关工作会不会已经有比他更好的
  + 和cusparse等的工业方法对比了，看有没有和学术的工作进行对比，而且最好是近年的工作

+ 文章的贡献点、isight
  + 就是没有创意，他在相关工作的基础上没有可观的提升

+ 学术阅读性上是否明确，有无拼写错误，或者改进句子使得读起来流畅。
  + 中心句子够不够中心，不能模糊的一整段
+ 有无明显技术上的硬伤，明显知识错误 --- 不容易遇见
+ 对于该论文，会不会引起别人的兴趣。不要仅仅是感觉坚实但无聊的工作
  + 这时可以说写的像技术报告，而非学术论文 --- 表达问题
  + Intro里面的引用是否足够，不能引用太少；或者说他说的太冗余

+ 论文是不是适应于该被投的该期刊领域

### 2.2.3 此时需要思考的方面

+ **通用性 与 专用优化**：工作的科学性是不是足够，够不够前沿，而不是专用的工作
+ **实用性**： 有的工作“大力出奇迹”，优化成功了，但前期有很大处理成本，则不实用
+ **分析是否详尽**： 实验数据解释的insight够不够
+ **工业界 与 学术性**： 在这一块有没有和 最新的、权威的工作比较

### 2.2.3 需要另找材料补充查证

+ 在相关工作：有没有引用开山之作，不能扎堆引用（应均沾）
+ 对于成果介绍有没有夸大，可以要求他弱化语气
+ 结果的对比，样本是否足够，对比有没有意义
+ 论文的递进关系是不是明确，以至于能够形成一个完整的闭环故事

## 2.3 **评论细节**

+ review 语气
  + 表达上过于委婉，详例见“Tone”，即关于不好的部分如何表达：“如果用...的话会使得论文更加strong”
  + **对于我们的review来说，直接讲“你这块 ... 太weak了不足以...”**

+ 不是问题的问题：一般来说一个自然段conlcusion能充分概括就够了；不需要篇幅太长。所以不用指责对方把这块的篇幅写的太短了



# 3 - 格式 对比实例

SC22审稿经历2022.04 --- 这里的启示主要在于review书写格式和书写风格上

本人在初期参考了不合适的教程，在书写格式上能近乎达到正确的格式~~（看起来相当不错）~~，但在写作风格上，就非常拖沓，自以为表达婉转实际上却让人难抓住要点。

<img src = 'https://s3.bmp.ovh/imgs/2023/05/12/31eb5f2aef761625.png' >

1. 红：文章主要内容的概述不要太长，简单截取就可以
2. 蓝：Major Concerns谈主要问题，列出2-3点；提出问题：从各方面的出发
3. 绿：Minor Comment罗列的点数多于Major～，**直接了当**指出问题



# 4 - 内容 评论实例

这里的问题其实就和科技论文写作的技巧相通，这里的问题比较深重，需要不断补充丰富～～

市面上的科技论文写作教材对于计算机领域来说适用性不强，本人就“深受其害”，（暂时还未找到合适的写作教材或全套教程 ），主要问题主要在于：

1. **作者背景差异**。作者背景常是人文社科背景，尽管简介清晰、有理有据的表达核心相通，但表达思路和对数据的依赖程度和计算机等理工科领域大有不同。甚至包括一些“委婉”的表述在本领域也不适用。
2. **教材展开思路**。其中最典型的反例是”*高考💯作文分析 与 指导* “这类书，主要展开思路：①有了一篇文章摆在这里，②告诉你这篇文章写得好 -- 依据是得分高，③ 以“马后炮”的角度 来分析他为什么好。此处问题在于，首先，文章应该是先构思想好再下笔成文，然后修改细节，而这类分析却颠倒了这个过程不符合常人思维；其次是预先地被告知了文章的好或不好，不是依照读者自己的思路来判断的，后续的分析只是进一步加深了这个被预先告知的概念 -- “好”。
3. **中文的依赖性思维**。中文化的表述相对于英文，更抽象化，就平时口语化表达的惯性思维来说，经常出现省略掉主宾语、不注意时态、不明确表达程度副词等情况。这些情况，如果不真正下笔进行科技论文写作再就已有作品来反思的话，在平时是难以感知的。一个典型的例子，*“如果**用**这个方法的话，**感觉**会**相对**更合理一些”，*对于这里面的动词“用、感觉”，均省去了主语，“相对“没有明确表达对比对象，”感觉“在学术论文中是不可接受的词，但在中文语境中，表达了不确定性的同时，还表达了一种委婉和后退的余地。

在这里，首先从一个反例展开对其进行分析，逐步得到改进版本，在改进后总结提炼出一些可参考通用的点。

## 4.1 - 反例🤔逐步改进 

如果想要说什么要的好，我们首先知道什么样的是不好，这样可能学得更快。（哈哈上句话也是一句典型的中文化表达）更进一步讲，这这个例子是我的犯错经验，呈现了一个完整的改进过程，虽然也是事后的复盘，但相对纯“马后炮”，会更符合常人的理解思维。

### v1 - 经不起细看

> 梦开始的地方，乍一看格式还挺整齐自我感觉良好，但细看下来，内容上有很大问题

<img src = 'https://s3.bmp.ovh/imgs/2023/05/24/d63c4d6fb89278fd.png' >

拿到文章，①以同行的视角初步研读（除非经验和相关知识丰富，否则不好直接确定这个paper的好坏）；②对于其中小小同行才懂的内容，通过其Reference和关键词搜索的顶会文章 进行学习（学习别的小小同行的论文为啥能发出来，知道了啥叫“好”，即“好”的标准）；③对比这个工作的价值（通过学习到的“好”的标准判断这个工作）。  在经过以上步骤后，对于这个文章就有一定的**印象**，然后展开开始写review。我们会顺着这个**印象**来对这个工作做评价，在印象的指导下，落实下来到点上，写成了观点就是review。在这里有一个很大误区：**印象**，本质是抽象的，所以它是**结论**而非**依据**。

如上图，v1 的Major concerns，首先我就犯了一个比较基础的错误 ---【错误1】对于idea的评价，当我们在对工作作做评价的时候，首先应该想想它的idea是否有新意，使用了一些新方法、新思路，是一个可以发表的工作所必备的，否则就是一个工程化的项目了。所以不应该因为它用了新方法、新思路就说这个idea是有创意的，这事一个很基本的要求。 

【错误2】“not enough、thin与lack”，这几个关键词是我在经历了上述步骤①②③以后，得到的**印象**，我把他们直接当作结论，先写出来，然后对其进行举例说哪些地方“not enough、thin”，需要补充哪些点，于是就走进了误区，而实际上**印象**应该作为**结论**出现。尽管无论我把它认识为结论或者依据，最后在review里呈现的形式都是“观点 + 理由”，因为这是我们做任何表达的通用逻辑。但是，当我首先把印象作为了**依据**时，一上来的观点输出就是**依据**，一切后续的展开都是围绕着印象的展开，所以“not enough”后面的举例、“lack”后面说他缺少的东西都是信口开河的，显得根本就不具体。所以当别人依照清晰的思维读起来时，把写在前面的观点当作**结论**，后续的举例说服力就非常弱了。

所以整体看下来，这个整体的表达就非常的含糊，我输出了**观点**（结论），但是自己支撑不起来，因为后续的举例（依据）不具体，没有聚焦在文中的某些点上。所以v1.0所发现的问题是致命的：把**结论**当**依据**，然后信口开河再写自己的“依据” 。

### v2 - 经不起推敲

> 这里把v1.0的把**硬伤**解决了，但含糊的问题没有完全解决

<img src = 'https://s3.bmp.ovh/imgs/2023/05/24/6c915f15a2c4ffa0.png' >**问题核心： 含糊  -- 问题没有量化 ，即没有做到以数据服人**。v1 中所出现的问题是最为致命的，其中问题的逻辑相对最难解释的。v2中问题的逻辑则简单了很多，具体来说是经过推敲就能发现，这里有了一定的依据，但观点表述**含糊**的问题还是没有解决。

如上图我的用词“first time”、“published work”就是纯没依据，这里也是我**印象**中的观点，因为在搜索了一大圈以后每没有看到用他这个方法的工作，和参考了其他工作，所以在这里通过**印象**作为依据，自己说服了自己。但是落到纸面上，在别人眼中，没有经过我的上述步骤①②③，更没有对v1进行一轮修改，因此以上说法就无法说服别别人。在此处就应该在其后紧跟着我的数据化的、落到实处的依据 --- 引用某篇paper或者数据，对于“first time”则采用语气的弱化。 “not enough、thin与lack”是之前就存在的问题，尽管我进行了具体举例，在后续引用的工作中有了依据。但是表达观点不够明确：既然说你这样是不行的，那么什么样的才叫行？ --- 给出了明确的方向然后再展开解释这个方向。更改后见V3。

### v3 - 最后的修改

<img src = 'https://s3.bmp.ovh/imgs/2023/05/24/cdf23f3fd4368073.png' >

在以上的v1，v2中，最主要的问题分别是依据不明确、观点表述表述含糊。前者需要从底层逻辑上去区分结论与依据的展开关系，后者更多是一个表述问题。我们所谓的表述含糊，**即问题没有量化，没有做到以数据服人**，但反过来说，不去量化问题，不用数据说明问题，我们会怎么表达呢？ --- **程度副词、形容词**！！！所以破案了

在观点阐述中，为了简单直接的**指出**问题（印象），我们确实不会这样去做量化、落实到数据上，而会用**程度副词、形容词**来概括，但这不是终点，而是起点。后续的持续解释，展示被量化了的问题和参考数据，这才是**指明了问题**，所以在这个思路下，看到v3的改进版，不意味着我们完全摒弃了程度副词、形容词的使用，而是极为谨慎地使用它们，一旦使用，就必须在后面跟上对应的 **量化了的落实到数据上了的**依据。否则观点含糊的感觉、空中楼阁的感觉就随之产生了。

### v1 v3 v4 总体对比

<img src = 'https://s3.bmp.ovh/imgs/2023/05/26/ecbafb3fd97d0604.png' >

v1/ v3左右对比则更加清晰，对于Minor Comment的进一步简化，这个很好理解，操作手段来说就是把有众多词修饰表述的句子给使用“语句”来表达了。主要是对于Major Concerns的逻辑上、观点表述上的优化，需要细看：其中对于程度副词、形容词 辩证地使用，对于依据的展开方式。

v4是老师最终提交给committe的review版本，也就是最后的返回版。其中的主要的两点，描述他 “not enough、thin”的部分被直接使用了，在这个角度来说这两点的描述是成功的。对于第3个大点来说的话，则是需要在充分了解行业和会议特点以后，才能做出的评价，在此处不强求。同时注意标为绿色的部分，是将我之前的表述改为了更加学术化的表述。



# 5  - 学术化表达

对于学术化表达，可能是我们在撰写英文材料时会担忧的，但实际上先写出来从0到1才比较重要，学术化表达是可以一点点改出来的，“积累”的感觉也更强。如果说以上1-4的问题比较致命且逻辑难以理清楚的话，这一块可能恰好调换过来，不是逻辑上的错误而是表达上的一些技巧，所以可以算做技术上的积累了。

以下是IPDPS24 review的例子，完整经历了4次审稿 即 以上4遍的【以上1-4】的流程后，在逻辑上已经不会犯特别离谱的问题了，主要是review的侧重**「方法」**与学术化表达上。

<img src = 'https://s3.bmp.ovh/imgs/2023/10/30/9b4a5f2cd2f79de7.png' >

以上的v1-v3，v1是完全自己写的初版，v2是经过老师提了些意见后的修改版，v3是老师的最终修改版。可以看到，关于Major concerns的60%的问题在v1其实已经能自己cover到了，但是对于**方法**、创新性上的意见还是不够（有些可以拆成多个）Minor comments的变化不大。在v2-v3主要是格式上的变化（没有对错，只有风格），对学术化写作的精细修改（技术性的积累）。

关于v3的学术化修改方法论，直接去看“学术写作”的课程或书会更直观，但需要实例以上就是了。从v2-v3的过程确实能发现，语言简练了40%的情况下，其实表达反而更清晰了，其中Major concerns全部是2-3句话：①问题性质②直接说问题在哪③对上一句问题的细化解释。Minor comments接近短语形式的表达，Major concerns中的②也就是这句话：少用not，多用可以表达性质的名词。
