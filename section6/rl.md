# 强化学习


[莫烦的强化学习课程](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)


1. 确定型的 policy
 A = pi(S)
2. 在 S 下取 A 为一概率值




DFP(Direct Future Prediction)
[Direct Future Prediction - Supervised Learning for Reinforcement Learning](https://flyyufelix.github.io/2017/11/17/direct-future-prediction.html)
直接预测未来
强化学习和监督学习的结合。




### 组合在线学习(combinatorial online learning)


[组合在线学习：实时反馈玩转组合优化](https://mp.weixin.qq.com/s?__biz=MzAwMTA3MzM4Nw==&mid=2649441835&idx=1&sn=abf10e00dd2354a0f256620b9e1fcda9&chksm=82c0afafb5b726b9a4cdb4d9112deba1bfe72803b20fd5f10bd7dd00b798214fbce750d4503f#rd)




### 强化学习(reinforcement learning)

又称为增强学习。策略梯度(policy gradient)是强化学习中重要的方法

- [Policy Gradient Methods](http://www.scholarpedia.org/article/Policy_gradient_methods)





### 示教学习（Learning from Demonstration，LfD）
不同于从经验（experice）中学习，而是通过示教者（teacher）给出的示例（example）进行学习。



 IRL(Inverse Reinforcement Learning) 逆向强化学习




### 深度强化学习

2013年，在DeepMind 发表的著名论文 `Playing Atari with Deep Reinforcement Learning`中，他们介绍了一种新算法，深度Q网络（DQN）。文章展示了AI agent如何在没有任何先验信息的情况下通过观察屏幕学习玩游戏。结果令人印象深刻。这篇文章开启了被我们成为“深度强化学习”的新时代。

在Q学习算法中，有一种函数被称为Q函数，它用来估计基于一个状态的回报。同样地，在DQN中，使用一个神经网络估计基于状态的回报函数。


[Deep Reinforcement Learning through policy optimization](http://people.eecs.berkeley.edu/~pabbeel/nips-tutorial-policy-optimization-Schulman-Abbeel.pdf)



[The Nuts and Bolts of Deep RL Reseach](http://rll.berkeley.edu/deeprlcourse/docs/nuts-and-bolts.pdf)
深度强化学习研究的基本要点





Ray RLLib: A Composable and Scalable Reinforcement Learning Library

可组合的强化学习并行训练，而不是将并行逻辑贯穿在整个程序中、内聚在所有模块中，从而获得更好的扩展性、组合性和重用性，并且不损失性能。
Ray RLlib 是 Ray 的一部分。
Ray 是一个基于Python的分布式执行框架，除了 Ray RLlib， 包括一个超参数优化框架 Ray tune. 17年11月底，Ray 发布了 0.3 版本，因此，是一个相对较新的框架。网文中说Ray 有望取代 Spark 。

Ray 是如何异步执行以实现并行的呢？如何用对象列表去表示远程对象













### todo

层次式强化学习（Hierarchical RL），基于模型的强化学习（model-based RL）

























