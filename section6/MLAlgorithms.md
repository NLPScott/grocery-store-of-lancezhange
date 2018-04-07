# 常用算法总览

>One good thing about doing machine learning at present is that people actually use it!



- 逻辑回归
   损失函数为交叉熵
- 最近邻
- 近似最近邻(ANN, approximate nearest neighbour)

    - LSH(locality-sensitive hashing)

        基本思想：将原始数据空间中的两个相邻数据点通过相同的映射或投影变换（projection）后，这两个数据点在新的数据空间中仍然相邻的概率很大，而不相邻的数据点被映射到同一个桶的概率很小  
        参考博文：[局部敏感哈希介绍](http://blog.csdn.net/icvpr/article/details/12342159)
- 奇异值分解

    [The Singular Value Decomposition, Applications and Beyond](http://arxiv.org/pdf/1510.08532.pdf)
- 决策树
- CART

- 随机森林

    [Awesome Random Forest](http://jiwonkim.org/awesome-random-forest/)

- 感知机
- 支持向量机

- 朴素贝叶斯
- 极大后验概率(MAP)
- 贝叶斯估计
     - 和 MAP 的最大区别：MAP 假设参数是固定值，而贝叶斯估计假设未知参数也是随机变量。


- 隐马尔科夫
    模型 model = (初始状态, 转移矩阵, 观测概率)

    - 学习问题：已知观测，估计模型，采用 EM 算法
    - 概率计算：在已知模型下得到特定观测的概率，采用前向/后向算法
    - 解码问题：已知模型和观测序列，求最大可能的状态序列，采用维特比算法



- 概率图模型
- 条件随机场



---

### 集成方法(Ensembling)专题

集成方法现在真是火的不要不要的。
参考　[Kaggle Ensembling Guide](http://mlwave.com/kaggle-ensembling-guide/) 一文

- AdaBoost
- GradientBoost




---

### 变分推断(variational Inference)

- [Variational Inference for Machine Learning](http://shakirm.com/papers/VITutorial.pdf)

    出自 Shakir Mohamed from Google DeepMind.

================== 

变分方法
Variational bounds

---

### Semi-supervised Learning 半监督学习

针对只有极少部分数据有标记而大量数据无标记的情形
参考 [半监督学习](http://www.cnblogs.com/liqizhou/archive/2012/05/11/2496155.html)

常用假设： 聚类假设、流形假设。在聚类假设下，大量未标记示例的作用就是帮助探明示例空间中数据分布的稠密和稀疏区域, 从而指导学习算法对利用有标记示例学习到的决策边界进行调整, 使其尽量通过数据分布的稀疏区域。流形假设是指处于一个很小的局部邻域内的示例具有相似的性质,因此, 其标记也应该相似。这一假设反映了决策函数的局部平滑性。和聚类假设着眼 *整体特性* 不同, 流形假设主要考虑模型的 *局部特性*。在该假设下,大量未标记示例的作用就是让数据空间变得更加稠密, 从而有助于更加准确地刻画局部区域的特性,使得决策函数能够更好地进行数据拟合。


- transductive learning 直推学习

和半监督学习的不同：直推学习将未标记的数据视作测试样本，学习的目的是在这些测试样本上取得最佳泛化能力，也就是说，学习器的目标就是将有标记的样本上获得的知识迁移到无标记的样本（如何做到？）； 而半监督学习并不知道要预测的示例是什么。简单理解为：直推学习是一种‘短视’的学习，而半监督更具长远眼光。




active learning 主动学习

学习器与外部环境有交互，即进行场外援助：对一些样本，学习器向外界求助。




- PU Learning

Learning from Positive and Unlabeled Examples
属于部分监督学习（Partially Supervised Classification）

能拿到的数据只有正样本，和一大堆未标记的样本。
正样本可能来自人工标记等方式，可能量也不会太多，在这种场景下，我们非常希望有一种方式能够学习到我们在标注这一批正样本的时候所暗含在内的‘智慧’。

这就是 PU Learning 的用武之地了: 通过标注过的正样本和大量未标注的样本训练出一个二元分类器。

由于没有负样本，传统的监督学习方式就不能用了，或者说，不能直接用。


PU learning 的思路，其实也不复杂：没有负样本，就先选出来一些作为负样本嘛。有了负样本就能训练模型了。这就是 two-stage-strategy
问题是，如何选取可靠的负样本呢（Reliable Negative Examples，RN）？

理论上已经证明：如果最大化未标注样本集 U 中负样本的个数，同时保证正样本被正确分类，则会得到一个性能不错的分类器。



遵循这个想法，具体的选择 RN 的方式有：
1. 朴素贝叶斯

（1）把 P 中的每个样本标记为类别 1；
（2）把 U 中的每个样本标记为类别-1；
（3）使用 P 和 U 训练得到贝叶斯分类器；
（4）对 U 中的每个样本使用上述分类器进行分类，如果分类结果为-1，则把该样本加入 RN。


间谍法
从正样本中划分出一部分间谍到负样本集合中去。训练完模型之后，通过间谍们的概率值，确定阈值。对 U 中所有小于 该阈值的，归入 RN




更进一步，假如我们认为，分错负样本和正样本的代价是不同的，那么我们就要用到 cost-sensitive-strategy







参考文献
Building Text Classifiers Using Positive and Unlabeled Examples(2003)

Liu, B., Dai, Y., Li, X. L., Lee, W. S., & Philip, Y. (2002). [Partially supervised classification of text documents](https://www.cs.uic.edu/~liub/S-EM/unlabelled.pdf). In ICML 2002, Proceedings of the nineteenth international conference on machine learning. (pp. 387–394).



Elkan, Charles, and Keith Noto. "Learning classifiers from only positive and unlabeled data." Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.

在样本满足一定分布情况下，根据正样本和未标记样本作为负样本训练出来的分类模型，预测出来的结果和该样本属于正样本的概率存在一个固定常数系数



Fusilier, D. H., Montes-y-Gómez, M., Rosso, P., & Cabrera, R. G. (2015). Detecting positive and negative deceptive opinions using PU-learning. Information Processing & Management, 51(4), 433-443

 Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama. "Positive-Unlabeled Learning with Non-Negative Risk Estimator." Advances in neural information processing systems. 2017.

[基于 PU-learning 的分类方法](https://jlunevermore.github.io/2017/04/18/58.PU-Learning/)

[基于PU-Learning的恶意URL检测](https://xz.aliyun.com/t/2190)


---



Regularized Greedy Forest正则化贪心森林(RGF)


---

### 排序学习(Learning to Rank)

[Learning to Rank using Gradient Descent](http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)






---

### 机器学习不是万能的

机器学习并不是万能的。

- [Machine Learning: The High-Interest Credit Card of Technical Debt](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43146.pdf) by D. Sculley, et al.


- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf) by D. Sculley, et al., Google

    机器学习系统中隐藏的技术债。* Not all debt is bad, but all debt needs to be serviced*．　这些技术债包括：复杂的模型可能会悄然改变抽象边界，CACE(change anything changes everything)，数据依赖等


#### 可解释的模型
模型可解释
https://blog.kjamistan.com/towards-interpretable-reliable-models/

从语料中自动学习的语义可能充满了人类的偏见
https://www.princeton.edu/~aylinc/papers/caliskan-islam_semantics.pdf



---

### 资料
[机器学习](http://wenku.baidu.com/course/view/49e8b8f67c1cfad6195fa705?fr=search) by 余凯、张潼

[Ruls of Machine Learning](http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)



