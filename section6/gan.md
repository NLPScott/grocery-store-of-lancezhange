# GANs


For example, if we wanted to minimize some error for image compression/reconstruction, often what we find is that a naive choice of error metric (e.g. euclidean distance to the ground truth label) results in qualitatively bad results. The design flaw is that we don’t have good perceptual similarity metrics for images that are universally applicable for the space of all images. GANs use a second “adversarial” network learn an optimal implicit distance function (in theory).



- [SN-GAN 谱归一化GAN](https://openreview.net/pdf?id=B1QRgziT-)
  提出新的归一化方法：谱归一化
  虽然GAN十分擅长于生成逼真的图像，但仅仅限于单一类型，比如一种专门生成人脸的GAN，或者一种专门生成建筑物的GAN，要用一个GAN生成ImageNet全部1000种类的图像是不可能的。
  但是，这篇ICLR论文做到了，SN-GAN是第一个用一种GAN就覆盖ImageNet全部1000种类数据的GAN变体



- [Keras-gan](https://github.com/eriklindernoren/Keras-GAN)
keras 实现的一些论文中的 gan 方法，值得学习。



- [对抗球](https://arxiv.org/abs/1801.02774)

对微小扰动的脆弱，是数据流形上高维几何的自然表现。




to read
https://weibo.com/ttarticle/p/show?id=2309404176849692455630 十种主流GAN

