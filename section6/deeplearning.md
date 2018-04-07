# 深度学习专题


### 顶尖研究机构
- MSR

- Facebook 的人工智能实验室 (FAIR, facebook AI research)

    人员方面，有 Yann LeCun，有VC维和SVM的缔造者Vladimir Vapnik，提出随机梯度下降法的Léon Bottou，做出高性能PHP虚拟机HHVM的Keith Adams, 以及 Rob Fergus, Jason Weston, Marc'Aurelio Ranzato,  Tomas Mikolov, Florent Perronnin, Piotr Dollar, Hervé Jégou, Ronan Collobert, Yaniv Taigman等

- 百度IDL


---

### 深度学习(deep learning)通用理论基础

深度学习的核心在于学习多层次的表示（distributed representations，对应不同程度的抽象），从而实现DIKW金字塔(data -> information -> knowledge ->  wisdom).


- [Representation Learning: A Review and New Perspectives by Yoshua Bengio, et al., 2012](http://arxiv.org/abs/1206.5538)

    表示学习

greedy layerwise pre-training 将深度网络分解成几个浅层网络，每层用无监督方法训练

gradient-based learning

Restricted-Boltzman Machine

Stacks of RBMs forms a deep belief network (DBN)
即，受限玻尔兹曼机堆叠成为深度信念网络
Stacks of slightly modified RBMs can form a deep Boltzmann machine (DBM).

Autoencoders

Enery-based Models  分为有潜变量和无潜变量的情形；极大似然方法学习参数








---

###  CNN(卷积神经网络)专题

CNN 已经有了诸多成熟的架构

- LeNet
- AlexNet
- googleNet

    [Going deeper with convolutions](http://arxiv.org/pdf/1409.4842.pdf)


- VGG-Net

　[very deep convolutional networks for large-scale image recognition](http://arxiv.org/pdf/1409.1556.pdf)


- OverFeat

    [OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](http://arxiv.org/pdf/1312.6229.pdf)

- R-CNN

    [R-CNN: Regions with Convolutional Neural Network Features](https://github.com/rbgirshick/rcnn)


- ShCNN

    [Shepard Convolutional Neural Networks by Jimmy SJ. Ren, et al., NIPS 2015](https://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf)

    SenseTime(商汤科技，一家专注于计算机视觉和深度学习原创技术的中国公司)研究人员出品，[代码在此](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/Shepard_CNN)。可用在超分辨率重建，图像修补等。

- PlaNet

   [PlaNet - Photo Geolocation with Convolutional Neural Networks](http://arxiv.org/abs/1602.05314)

   谷歌的工作，仅用图片的像素来定位图片位置


---

### RNN（recurrent neural networks, 循环神经网络）专题

本文是一些关于 RNN 的东西，更多资料参见[Awesome RNN](http://jiwonkim.org/awesome-rnn/)

RNN 是 Recursive Neural Nwtwork(递归神经网络，有时也被简写为 RNN) 的一种，其核心在于对当前的状态保留`记忆`（以隐变量的方式存在）。

给定输入序列 $$(x_1,x_2, ..., x_T)$$, 一个标准的RNN通过如下迭代方程计算输出序列$$(y_1,y_2, ..., y_T)$$:

$$
h_t = sigma(W^{hx}x-t + W^{hh}h_{t-1})
$$

$$
y_t = W^{yh}h_t
$$

所谓的记忆重现，就体现在了$$h_t$$ 与 $$h_{t-1}$$ 上了。

此外，注意到，RNN 各层之间共享一套参数（由此极大减少了待学习的参数数量），训练用到的是 BPTT(backpropagation through time)。

针对传统 RNN 的各种缺陷，有如下几种改进的 RNN 模型。

1. LSTM(long short term memory networks)

    一般用 BPTT 训练 RNN 模型时会因为梯度坍塌/爆炸而无法捕捉远距离依赖，而 LSTM 专治此不服。[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 是一篇不错的介绍文章，里面对 LSTM 有很生动的介绍。Andrej Karpathy 所写的 [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 更是值得一读。[RNN Tutorials 系列文章](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)也写的很好，建议一读。

    LSTM 目前已在诸多领域获得了成功，俨然已是最流行的 RNN 模型，[谷歌的邮件智能回复系统](http://googleresearch.blogspot.co.uk/2015/11/computer-respond-to-this-email.html)也采用了 LSTM.

2. BirDirectional RNNs

    核心想法：t时刻的输出不仅依赖于之前的序列，可能也要依赖于之后的序列

3. GRU(gated recurrent units)

    包含一个重置门，一个进化门。相比 LSTM 少了一个输出门。

    [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf), by Bengio 等人

- [An Empirical Exploration of Recurrent Network Architectures by Rafal Jozefowicz, et al., Google](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

    探讨 RNN 的内在机理。非常重要的一篇，待认真研读。

- [A Critical Review of Recurrent Neural Networks for Sequence Learning](http://arxiv.org/abs/1506.00019v1)

- [Scheduled Sampling for Sequence Prediction with
Recurrent Neural Networks](http://arxiv.org/pdf/1506.03099v3.pdf) by Sami Bengio, et al., at Google.

    定时采样


[recurrent.js](https://github.com/karpathy/recurrentjs) Karpathy 大神用　javascript 写的RNN库


更多请参考　[Awesome RNN](http://jiwonkim.org/awesome-rnn/)



---

### 神经变分推断(neural variational inference)





---
### 深度隐变量模型(deep latent variabl models)

[Neural Variational Inference for Text Processing, by Yishu Miao, et al., Oxford, ICLR 2016, under review](http://arxiv.org/pdf/1511.06038v1.pdf)


---

### RNTN（Recursive Neural Tensor Networks）递归神经张量网络

---

### Memory Networks

[End-to-End Memory Networks by Sainbayar Sukhbaatar, et al., 2015](http://arxiv.org/pdf/1503.08895v4.pdf)



- [Memory-based Bayesian Reasoning with Deeep Learning](http://blog.shakirm.com/wp-content/uploads/2015/11/CSML_BayesDeep.pdf)

    to read

- [Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](http://arxiv.org/pdf/1506.07285v1.pdf)


###

如何在 sequence-2-sequence 任务中，实现对一些关键信息的‘保留’？
比如我们在对话过程中，会出现“你好啊，我叫小S”，然后另一方要回答，“很高兴认识啊小S。”那么这个小S，也就是说实体信息或者日期信息等等，就是在对话的 input-output 中，被“复制”的。
这个信息是应该“原封不动”地被保留，从 sequence2sequence 的输入端被复制到输出端的。
现有的 end2end sequence2sequence 等模型，即使加了 attention 机制，也很难做到这点

Ptr-Net(pointer network)
关于指针网络，参考 http://fastml.com/introduction-to-pointer-networks/
指针网络是 序列-到-序列 注意力模型的一种变体，是最近seq2seq比较火的一个分支，在基于深度学习的阅读理解，摘要系统中都被广泛应用。


CopyNet
提出了拷贝机制。


### 贝叶斯深度学习 Bayesian Deep Learning


- [On Bayesian Deep Learning and Deep Bayesian Learning](https://pan.baidu.com/s/1bp8mP1H)

贝叶斯深度学习 和 深度贝叶斯学习，from 剑桥 deepmind 团队。






### 深度学习的理论支持

缺乏足够的理论支撑一直是深度学习广为人诟病的一点。

- [Mathmatics of Deep Learning](https://arxiv.org/abs/1712.04741)

评述了近期对深度网络的全局最优、几何稳定性、表示不变性等诸多性质的数学理论依据研究。 2017年 12 月中。






---

### 深度学习的多种应用

深度学习的主要应用我们分专题讲述，以下是一些未分类的其他应用

- [Predicting online user behaviour using deep learning algorithms by Armando Vieira](http://arxiv.org/abs/1511.06247v1)

    预测在线用户行为

- [Using Neural Networks for Click Prediction of Sponsored Search](http://arxiv.org/abs/1412.6601)

　　点击率预测

- [Convolutional Neural Networks for Pedestrian Detection](https://github.com/DenisTome/DeepPed)

    行人检测

- [Neural Turing Machines](http://arxiv.org/abs/1410.5401)

    代码[在此](https://github.com/shawntan/neural-turing-machines)

- [Reinforcement Learning Neural Turing Machine](http://arxiv.org/abs/1505.00521)


- [Neural-Colorizer](https://github.com/YerevaNN/neural-colorizer)

　　利用卷积自编码为灰度图着色

- [用深度卷积神经网络下围棋](http://jmlr.org/proceedings/papers/v37/clark15.pdf)
这背后用到的原理阐述。

- [Neural Programmer-Interpreters by Scott Reed, et al., ICLR 2016, submitted](http://arxiv.org/abs/1511.06279)

    利用深度学习运行计算机程序，这听起来是如此虚幻。


- [opinion mining with deep recurrent nets](http://www.cs.cornell.edu/~oirsoy/drnt.htm)

    观点挖掘





---



----

### 注意机制(attension)专题

本文严重参考了 [Survey on Attension-based Models Applied in NLP, 2015.10.07](http://yanran.li/peppypapers/2015/10/07/survey-attention-model-1.html)和 [Survey on Attention-based Models Applied in NLP](http://yanran.li/peppypapers/2015/10/07/survey-attention-model-1.html)一文。

基于 *Attension* 的模型近年来越来越受重视。Attension 大约在2014年的时候被引入计算机视觉领域，后来逐渐扩张到了NLP.

注意机制的核心问题在于：如何确定在哪里倾注注意力以及如何量化。就好比老师在期末考试复习的时候告诉大家要抓重点，但真正的问题在于重点在哪里。

软对齐模型(soft alignment models) 和硬对齐模型(hard alignment models)

硬注意机制每次只注意一个地方，而软注意机制只在一个地方倾注相对更多的注意而已。因此，前者需要随即抽样，需要的计算量相对少一些，而后者是确定性的，虽然计算量大一点，但可以用后向传播训练并因此容易嵌入其他网络。

图片描述和MT的区别：后者存在一个粒度(granularity)问题。比如，应该说图片中有一只猫，还是说有一只看起来很慵懒的眯着眼睛的大花猫？



和 saliency 的区别和联系: bottom up saliency, top down attention

- [Recurrent Models for Visual Attention](http://arxiv.org/abs/1406.6247)

- [Neural Machine Translation by Jointly Learning to Align and Translate by Dzmitry Bahdanau, et al., ICLR 2015.](http://arxiv.org/abs/1409.0473)

    这篇文章应该是最早将注意机制从视觉引入自然语言处理的，Yoshua Bengio 也是作者之一。我们知道，基于深度学习的机器语言翻译采用的是 *编码-解码* 的方式，通过对解码端引入视觉机制，减轻了编码端的压力。

- [Encoding Source Language with Convolutional Neural Network for Machine Translation by Fandong Meng, et al., 中科院](http://arxiv.org/abs/1503.01838)

    也是机器翻译领域的。

- [Effective Approaches to Attention-based Neural Machine Translation by Minh-Thang Luong, et al., EMNLP 2015, camera-ready version](http://arxiv.org/abs/1508.04025)

    集成了全局和局部两种注意机制后得到的结果更好。

- [A Neural Attention Model for Abstractive Sentence Summarization by Alexander M.Rush, et al., 2015](http://arxiv.org/abs/1509.00685)

    首次将注意机制引入了句子摘要问题。

- [DRAW: A Recurrent Neural Network for Image Generation by Karol Gregor, et al., Google, 2015](http://arxiv.org/abs/1502.04623)

    提出了一种被称作 *DRAW*(deep recurrent attentive writer) 的用以生成图片的网络架构。主要的一个观察：人在作画的时候也不是一笔挥就一蹴而成，而是经过不断的调整、修改细节才最终成型。因此，DRAW 每次会注意到某个部分，将之修改. 问题在于，如何确定这个注意部分呢？
    和传统的注意机制的不同之处：传统的注意机制中，注意是在解码阶段被引入的，并且不会去影响到编码阶段，而 DRAW 

- [Teaching Machines to Read and Comprehend by Karl Moritz Hermann, et al.,Google, NIPS 2015, to appear](http://arxiv.org/abs/1506.03340)

    教会机器阅读理解。构建了一个 *（（文档，提问），答案）* 数据集。两个模型，*attentive reader* 和 *impatient reader*. 后者需要一遍又一遍地重读文档，

- [Learning Wake-Sleep Recurrent Attention Models by Jimmy Ba, 2015](http://arxiv.org/abs/1509.06812)

    Ruslan Salakhutdinov 也是作者之一。采用了硬注意机制(hard attention mechanism)，

- [Not All Contexts Are Created Equal: Better Word Representations with Variable Attention by Wang Lin, et al.](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015-2.pdf)

    不得不说，文章名称非常有吸引力，

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention by Kelvin Xu, et al., 2015](http://arxiv.org/abs/1502.03044)


- [Predicting the Next Location: A Recurrent Model with Spatial and Temporal
Contexts by Qiang LIu, et al., 中科院](http://www.shuwu.name/sw/Predicting%20the%20Next%20Location%20A%20Recurrent%20Model%20with%20Spatial%20and%20Temporal%20Contexts.pdf)

    时空上下文。


- spatial transformer networks 空间变换网络

spatial transformer ： affine transformasion ，projective transformation、 thin plate spline（薄板样条）

https://github.com//WarBean/tps_stn_pytorch
PyTorch implementation of Spatial Transformer Network (STN) with Thin Plate Spline (TPS)





---

### 数据集
[Street View House Numbers(SVHN)](http://ufldl.stanford.edu/housenumbers/)



---

### 应用

- 图像风格转换


- 音乐风格转化
[Neural Style Transfer for Musical Melodies](https://ashispati.github.io//style-transfer/)


### 可视化

- [VisualDL](https://github.com/PaddlePaddle/VisualDL)



---

### 学习资料

1. [video lecture: 深度学习暑期学校](http://videolectures.net/deeplearning2015_bengio_theoretical_motivations/),2015,蒙特利尔

2. [Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision)

3. [A Statistical View of Deep Learning](http://blog.shakirm.com/wp-content/uploads/2015/07/SVDL.pdf)
    用统计学的视角看深度学习，待认真研读。

4. [Deep Learning](http://goodfeli.github.io/dlbook/)

    Ian Goodfellow, Aaron Courville 以及 Yoshua Bengio　三人合著。

5. [Neural Networks and Deep Learning ](http://neuralnetworksanddeeplearning.com/)

    也许是最好的入门教程？




