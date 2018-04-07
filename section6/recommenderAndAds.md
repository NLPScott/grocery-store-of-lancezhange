# 推荐系统/计算广告专题

### 推荐系统
- 协同过滤仅需要一个 **user-item** 矩阵（矩阵元素为打分值），并不需要user和item的属性。可以视作矩阵补全问题。

- 基于内容
    - 缺点： 推荐物较为静态
- 基于社交(社会化推荐)
- 基于当前情景


- 冷启动问题
- 推荐差异化、个性化　
- 推荐动态化
所谓动态化是指，所推荐的东西的疆域不能是封闭的，要对用户进行适当引领，摆脱‘过滤气泡’造成的智识孤立


- 上下文多臂老虎机在推荐系统中的应用
参考 [netflix 在为用户推荐个性化的视频封面图片实践](https://www.iyiou.com/p/62735)

为什么需要将展示图片做个性化呢？因为剧集的题目很多时候并不足以给出足够的信息，以吸引用户的观看，而如果图片能够投其所好的话，则可以提高用户感兴趣的概率。有的用户喜欢某个演员，那么在剧集图片里展示该演员的剧照会更有效；有的演员喜欢喜剧，那么通过图片来告诉用户这是一部喜剧，则更有可能吸引用户；此外，不同用户可能有着不同的审美，那么对其展示更符合其审美的图片也会有更好的效果

第一个挑战，在于每个剧集只能展示一张图片，如果用户点击并观看了这部剧集，我们并不能确认是因为图片选得好起了作用，还是用户无论如何都会观看这部剧集。用户没有点击的情况也是类似。所以第一个要解决的问题时如何正确地对结果进行归因，对于确定算法的好坏至关重要。
那自然就会想到说，去切换 session 之间的图片。这样就能比较图片的切换带来的效果。
不过，切换也有挑战：频繁的切换可能给用户带来困惑，对最终的归因也带来偏差。

还有一个挑战在于理解一副封面图和同一页面或同一session中其他封面图和展示元素之间的关系。一张包含主角大幅特写的图片可能会非常吸引人，因为这可以使得该图片脱颖而出。但如果整个页面中都是这样的图片，这个页面作为一个整体就不那么吸引人了。此外，封面图的效果可能和故事梗概和预告片的效果也紧密相关。所以候选图片需要能够涵盖该剧集吸引用户的多个方面。

文中所提到的重播(replay)评估方法，我在上家公司做推荐的时候也想到并去做了。

Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms


##### 参考资料
1. [Amazon.com Recommendations: Item-to-Item Collaborative Filtering](http://www.cin.ufpe.br/~idal/rs/Amazon-Recommendations.pdf)


### 计算广告

广告的计价方式
1. 按照展示计费
CPM（cost per mail/ cost per thousand impressions） 千人成本,这种计量方式比较粗犷

CPTM (cost per targeted thousand impressions) 有效千人成本
排除无效的人群

2.按照点击计费
cpc(cost per click)

模型的评估
lift5

#### DSP

追踪用户行为
受众选择:  low-level model 做初筛，high-level model 做细选


- [Ad Click Prediction: a View from the Trenches](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf) by H. Brendan McMahan, et al., google, 2013

    来自谷歌广告一线战壕的干货。FTRL-Proximal 在线学习算法

[实时竞价方面的研究文章汇总](https://github.com/wnzhang/rtb-papers)










