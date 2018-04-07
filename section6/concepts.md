 # 一些概念

#### 经验风险最小化 VS 结构风险最小化

泛化误差上界

#### 生成模型 VS 判别模型


#### 监督学习 VS 无监督学习 VS 半监督学习

#### multi-task learning VS multiclass learning


- VC 维: 精确定义分类器的复杂度
    - 直观定义：单个分类器可最多分类的点个数（不要求对在同一直线上的点分类）
    - N维空间中的线性分类器的VC维为 N+1

#### 偏差和方差(bias VS variance)
均方误差




#### 常用损失函数

- 0-1损失
- 平方损失
- 绝对损失
- 对数损失
- 合页损失

交叉熵

#### 评价函数
- accuracy
- 精度
- 召回
- F值
- AUC
    - AUC 的计算技巧：利用其与 wilcoxon-mann-witney test 的等价性，求出正样本的 score > 负样本的 score 的pair数，但还是复杂。进一步的改进方法：首先对 score 从大到小排序。


#### 距离函数

EMD(Earth Mover’s Distance) 距离
EMD 实际上是线性规划中运输问题的最优解


机器视觉领域的顶会
- CVPR 国际计算机视觉和模式识别会议
- ICCV 国际计算机视觉大会
- ECCV 欧洲计算机视觉大会


#### 
[机器学习的表现力、训练力和泛化力](http://blog.evjang.com/2017/11/exp-train-gen.html)


















