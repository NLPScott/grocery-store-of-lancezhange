# 语音识别专题

语音识别的一个难点在于语音和文字的对齐。


#### 特征

MFCC 



CTC(Connection Temporal Classification)






- [Speech Recognition with Deep RNNs](http://arxiv.org/pdf/1303.5778.pdf)

- [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567)

    百度的 *Deep Speech*，无需对噪音和混响人工建模，在 Switchboard Hub5'00 基准数据上的错误率为 16%

- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin by Dario Amodei, et al., 2015](http://arxiv.org/abs/1512.02595)

    Deep Speech ２代，百度出品。列出的作者有３４人之多！宣称在几个基准测试上比亚马逊的土耳其工人的准确率还要高。只需很少的修改就能用到其他语种，并且可以部署到生产环境（用到所谓的批量分发(batch dispatch)）。主要的三个方面：更好的模型结构（结合　CTC 损失函数），更好的数据（11940 小时的英语和　9400 小时的汉语），以及更快的计算（放弃使用参数服务器和异步更新，而采用同步的 SGD，利用高性能计算中的优化策略对 GPU 计算做优化）。

- [Towards End-to-end Speech Recognition with RNNs](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)

    ICML 2014 论文，结合了双向 LSTM 和 CTC(connectionist temporal classification)目标函数



https://deepmind.com/blog/wavenet-launches-google-assistant/


---
### 语音唤醒 keyword spotting
评价指标： 准召率，实时性，能耗






---

### 工具

[Essentia](https://github.com/MTG/essentia)


[Gentle](https://lowerquality.com/gentle/) 语音和文字的强制对齐



---

### 参考资料

- [Speech Recognition with Neural Networks](http://andrew.gibiansky.com/blog/machine-learning/speech-recognition-neural-networks/)

- [语音识别技术的过去、现在和未来](file:///Users/zhangxisheng/Downloads/The-Past-Present-and-Future-of-Speech-Recognition-Technology.pdf)

