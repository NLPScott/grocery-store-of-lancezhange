# 可视化专题

是的，我把可视化也放到了这一章（你咬我啊）

数据可视化对增进人们对数据的理解、捕捉数据中潜藏的模式，起着至关重要的作用，正所谓*一图胜千言*。

好的可视化作品，能够让受众发现那些原本非常容易被忽略却极具价值的东西，甚至更进一步，**Good answers lead to more good questions**.

可视化的终极目的是**更好地传递信息**。所以，首先你得有信息，其次才涉及到如何去传递。关于从数据中获取信息，请参考本书[数据分析](../section6/dataAnalysis.md)等专题。这里我们只涉及**如何传递**部分。


### 顶会/研究机构

- [IEE VIS](http://ieeevis.org/)

[The Visual Analytics Science and Technology (VAST) Challenge](http://vacommunity.org/VAST+Challenge+2015)


- [北大可视分析](http://vis.pku.edu.cn/wiki/)

传统的直方图、折线图、饼图不再赘述。

Chord Diagrams



#### 一般知识

- [No more Rainbows](https://agilescientific.com/blog/2017/12/14/no-more-rainbows)
  `The world has known for at least 20 years that the rainbow colourmap is A Bad Thing, perhaps even A Very Bad Thing`
  为什么说彩虹图是个特别不好的东西呢？原因大概有如下几点：
  1. 黄色和青色附近有色条，，引入了并不存在的结构，可能会误导受众
  2. 亮度不是单调的，因此无法转为灰度图
  3. 色盲同志无法很好感知

- [Why Should Engineers and Scientists Be Worried About Color?](http://researchweb.watson.ibm.com/people/l/lloydt/color/color.HTM)
 to read

- 




- [Multivariate Network Exploration and Presentation: From Detail to Overview via Selections and Aggregations](http://www.computer.org/csdl/trans/tg/preprint/06875972.pdf)
该论文挑战了可视化研究领军人物Ben Shneiderman所提出的可视化设计原则（“魔咒”）：**overview first, zoom and filter, then details on demand（总览为先 缩放过滤 按需查看细节）**, 详细介绍参见[这篇博文](http://www.storagelab.org.cn/zhangdi/2015/11/07/dosa/)



#### 高维数据的可视化





#### 信息在社交网络上的流播


##### 可视化的欺骗性

可视化有时候是具有欺骗性的，例如一些有意或者无意的视觉操纵。这种欺骗的高明之处在于，它不是数据造假这种低级的欺骗，而是利用信息隐藏、引入一些不易察觉的偏置来达到视觉操作的目的。

常见的欺骗手法：　改变纵轴的起始点，改变放缩比例，改变线条粗细。

参见文章　[Disinformation Visualization: How to lie with datavis](https://visualisingadvocacy.org/blog/disinformation-visualization-how-lie-datavis?utm_campaign=Data%2BElixir&utm_medium=email&utm_source=Data_Elixir_40)
.　非常值得一读。



---

### 工具
- Processing

    已经出到版本3了（2015.10.23）

    [Python Mode for Processing](http://py.processing.org/)

    关于　processing 的书籍，现在也是非常多了。推荐《代码本色－用编程模拟自然系统》(The Nature of Code - Simulating Natural Systems With Processing), 这本书并不是专门讲　processing 的，但是书中用　processing 来实现诸多动态模拟，还是很有参考价值。

- [chart.js](http://www.chartjs.org/)

    特点： 简单灵活，基于HTML5, 自适应，交互式。

- [P5.js]()

- D3.js

    我好钟意你啊，D3

- ggplot2
    大神 Hadley 鼎鼎大名的R包

- [visNetwork](http://dataknowledge.github.io/visNetwork/)

    R包，交互可视化

- [ECharts](http://echarts.baidu.com/index.html)

    百度开源。有集大成者之气。

- [ProcessOn](https://www.processon.com/diagrams)

    在线制作流程图，非常好用，对于偶尔画一两次流程图的童鞋来说，安装软件啥的太麻烦且不必要，processon 绝对可以满足需求。

- [TimeLineJS](https://github.com/NUKnightLab/TimelineJS)

    时间流

    他们又搞个[TimelineJS3](https://github.com/NUKnightLab/TimelineJS3)出来

- [Plotly.js](https://plot.ly/javascript/open-source-announcement/)

    2015.11 开源。基于 D3.js 和 [stack.gl](http://stack.gl/)，绝对利器。

- [Three.js](​http://mrdoob.github.io/three.js/)

- [paper.js]()

- Gephi


- MathBox
http://acko.net/files/mathbox2/iframe-rtt-history.html

http://acko.net/files/mathbox2/iframe-quat.html

http://acko.net/files/gltalks/pixelfactory/online.html 
真是要酷死了



---

### 作品廊

[Visualing Algorithms](http://bost.ocks.org/mike/algorithms/)

[更多编程算法可视化集锦](http://visualgo.net/)

[分形和数学艺术](https://fractals-and-mathematical-art.zeef.com/alan.richmond)

[概念可视化集锦(Gallery of Concept visualization.md )](http://conceptviz.github.io/#/e30=)

[Information Is Beautiful Awards](http://www.informationisbeautifulawards.com/)

[visualizing Facebook Friends](http://paulbutler.org/archives/visualizing-facebook-friends/)

[How to Visualize New York City Using Taxi Location Data and ggplot2](http://minimaxir.com/2015/11/nyc-ggplot2-howto/)

[A Day in the Life of Americans
](https://flowingdata.com/2015/12/15/a-day-in-the-life-of-americans/)


[画出一个好直方图原来也并不简单](http://tinlizzie.org/histograms/)

[出行数据的可视化例子](https://github.com/tsloan1377/montreal_open_data/blob/master/traffic_heatmap_timeseries.ipynb)
犹如一个星系，喜欢！

[程序包管理可视化](https://github.com/anvaka/pm#your-own-graphs)


---

### 资料

[gitbook: Data Visualization for All](http://www.datavizbook.org/content/)


