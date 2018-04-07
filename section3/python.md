# Python

日志 log 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



自动执行启动脚本
import site
site.getusersitepackages()

eg. /Users/zhangxisheng/.local/lib/python2.7/site-packages

然后建立 sitecustomized.py 脚本，里面的语句将在 python 启动的时候自动执行。

但是，sitecustomized.py 脚本中 import 的包，还是不能找到。

应该可以 用 PYTHONSTARTUP 变量指向，但是确实这样是不好的做法，所以还是放弃自动引入包。



try:
except:







基本数据结构

带有默认值的字典

>>> from collections import defaultdict
>>> d = dict()
>>> d = defaultdict(lambda: -1, d)
>>> d['a']


list comprehension 
t =  [True if (i == 'n' or i == 'v') else False for i in x.split("/")]

flatMap 的一种实现：利用 reudce 函数
reduce(list.__add__, [i.split(",") for i in x])
或者较为复杂的 list comprehension
flattened_list = [y for x in list_of_lists for y in x]





集合中随机选取元素
import random
for i in random.sample(poi_tag, 2):
    print i, poi_tag[i]


#### 魔力方法/属性

 我们知道，python 中，**一切皆对象**，同时，python 也是一门多范式的语言：面向过程/面向对象/函数式和谐共存，这背后的奥秘就是 **magic method**.

 事实上，许多运算符和内置函数都是用魔力方法实现的。例如，*+* 其实是 `__add__()`, `len()` 为 `__len__()`，而任何具有`__call__()` 方法的对象都被当做是函数。此外，`with `**上下文管理器**也是借助`__enter__()` 和 `__exit__()`魔力函数实现的。

 如下列举一些重要的魔力方法和魔力属性

1. `__dict__` : 属性列表，`object.attr` 其实就是 `object.__dict__[attr]`，许多内置类型，如 list, 都没有该属性；类的属性
2. `__class__`, `__bases__`,`__name__`,
2. `__slots__` 对拥有该属性的类的对象，只能对`__slots__`中列出的属性做设置
3. 用以包构建的，如 `__all__`
4. `__setitem__`
5. `__init__()`,`__new__`
6. `__repr__`, `__str__`
7. `__exit__(self, type, value, traceback)`
8. `__iter__`(通常和 `yield` 一起用以定义可迭代类)
9. 



中文字符串截取
x.decode('utf8')[0:n].encode('utf8')



#### 动态类型

对象是存储在内存中的实体，程序中用到的名称不过是对象的引用。**引用和对象分离**
乃是动态类型的核心。引用可以随时指向新的对象，各个引用之间互相独立。

可变数据类型，如列表，可以通过引用改变自身，而不可变元素，如字符串，不能改变引用对象本身，只能改变引用的指向。

函数的参数传递，本质上传递的是引用，因此，如果参数是不可变对象，则对参数的操作不会影响原对象，这类似于C++中的值传递；但如果传递的是可变参数，则有可能改变原对象。

#### lamda 函数

我最早是在 haskell 中见到匿名函数的，后来它被加入到了 python 以及 java 中。python 中定义 lamda 函数很简单：

        func = lambda x,y: x+y

其他函数式编程的经典函数如 map, filter, reduce 等，我最早也是在 haskell 中见到的。大多类似，不再赘述。

#### 迭代器

- 循环对象 例如open() 返回的就是一个循环对象，具有 `next()` 方法，最终举出 `StopIteration` 错误
- 迭代器 和循环对象没有差别装饰器
- 生成器 用以构建用户自定义的循环对象，这要用到神秘的`yield`关键字

#### 装饰器

函数装饰器接受一个可调用对象作为参数，并返回一个新的可调用对象。装饰器也可以带有参数，从而更为灵活。类装饰器同理。

例如，上下文管理也可以用 *contexlib* 模块用装饰器的方式实现。

@statcimethod

@classmethod

@property

再入，结合 __call__, 装饰器能够


#### 闭式

闭式可以减少定义函数时的参数，例如可以利用闭包来定义泛函。

#### 元类(metaclass)

我们说过，python 中一切皆对象，就连类也是对象！metaclass 便是用来创建类对象的类。`__class__` 的 `__class__` 为 `type`, 是 python 中内置的创建类的元类。

自定义元类： `__metaclass__`，可以被赋值为任意可调用对象

#### 描述符

`__get__`, `__set__` , `__delete__` ： 实现这三个就能做描述符

#### 继承
1. super()
2. MRO(method resolution order): 也就是通常说的继承顺序，并非简单的深度或者宽度优先，而是确保所有父类不会出现在子类之前
3. `self`, `cls`


#### 协程
协程，用户级线程，可以`让原来要使用异步+回调方式写的非人类代码,可以用看似同步的方式写出来`。具体地，它可以保留上一次调用的状态，和线程相比，没有了线程切换的开销。

`yield`, `next()`, `.send()`, `.close()`



#### 并行

#### 自省(introspection)

`dir`, `type`, `id`

`inspect` 模块

####  性能优化建议/debug

debug： *pdb*模块

[官方 Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
##### C 接口
1. CDLL() 加载动态库，和 R 中的做法很相似。
2. C API


### 单元测试
[Python 单元测试和　Mock 测试](http://andrewliu.in/2015/12/12/Python%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E5%92%8CMock%E6%B5%8B%E8%AF%95/)

#### 其他
 1. 使用 *LBYL*(look before you leap, 例如前置 if 条件判断) 还是 *EAFP*(easy to ask forgiveness than permission，例如 `try--catch`)? 这里给出了一个[建议](http://stackoverflow.com/questions/5589532/try-catch-or-validation-for-speed/)

 2. 不定长无名参数 `\*args`(元组) 和 不定长有名参数 `\*\*kwargs`（列表）
 3. `python -m SimpleHTTPServer 8088` 文件共享从此 so easy.
 4. `self` 并非关键字，而只是一个约定俗成的变量名



dict 的方法

shop_id = shops.main_poi_id.values
n_pois= shops.n_pois.values
tag_map = dict((name, value) for name, value in zip(shop_id, n_pois))

注意，字典是 dict， 不是 map



assert len(data.poi_id.unique()) == data.shape[0], "门店有重复！"



python3 环境
conda create -n python3 python=3 anaconda
source deactivate/activate python3



jupyter
ipython notebook 即可启动

jupyterlab  搜索启动


# 快速传入文件
python -m SimpleHTTPServer 8000  注意windows上用powershell

# 细节
函数必须用 reutrn 返回返回值，如果想有返回值的话

unique 是 函数， unique() 才得到值

忽视警告
import warnings
warnings.filterwarnings("ignore")

忽略命令行下警告错误的输出
python -W ignore yourscript.py


df.values dataFrame 转为 ndarray

Series.values 返回 ndarray

mask
使用()括起筛选条件，多个筛选条件之间使用逻辑运算符&,|,~与或非进行连接，特别注意，和我们平常使用Python不同，这里用and,or,not是行不通的


data.ix[[231, 236]] 选取行
iloc是将序列当作数组来访问，下标又会从0开始


实验记录
sacred  
print_config
with arg="value"


降维可视化
hypertools
hyp.plot(sample, n_clusters=2, explore=False, group='label')


时间和日期
datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

计算天数差值
d1 = datetime.datetime.strptime('2017-08-03', '%Y-%m-%d')
d2 = datetime.datetime.strptime('2017-08-02', '%Y-%m-%d')
delta = d1 - d2
print delta.days


======================================== dataFrame／pandas
df = pd.DataFrame({'B': [4,5,6,7],'A': ['a','b', 'a', 'b'], 'C':['china', 'china', 'eglish', 'egnlish']})

pd.read_csv(data_location, sep=feat_separator, header=None, na_values="null")
读入数据时指定 na 值


抑制烦人的科学计数法 设定浮点数的格式
pd.set_option('display.float_format', lambda x: '%.2f' % x)



箱线图
data[['label', 'hcg_t']].boxplot(by='label')




Pandas所支持的数据类型: 
1. float 
2. int 
3. bool 
4. datetime64[ns] 
5. datetime64[ns, tz] 
6. timedelta[ns] 
7. category 
8. object 
默认的数据类型是int64,float64.


from pandas.util import testing
df = testing.makeDataFrame()

df = pd.DataFrame()

pd.read_clipboard()


df_ta['delivery_duration'] = df_ta['delivery_duration'].astype['float']

df.sample(5) # 随机抽取5条数据查看
df.iloc[22] #利用iloc命令做行选取

df.loc[22:25] #利用loc命令做行选取
df.loc[[22,33,44]] #利用loc命令做指定行选取

df['open_int'] = np.nan
df['open_int'] = 999
df['test'] = df.company_type == '民营企业'
df.rename(columns={'test':'乱加的一列'}, inplace=True) #更改列名，并固化该操作

df_test.drop([2,5],axis=0) #删除行
df_test.drop(['列1','列2'], axis=1) #删除列
df.date = df.date.map(lambda x: x.strftime('%Y-%m-%d')) #将时间数据转换为字符串

df['fs_roe'].mean() #计算平均数
df['fs_roe'].idxmax() #返回最大值index
df.loc[df['fs_roe'].idxmin()]

df.fs_net_profit.corr(df.volume) #求解相关系数
df.company_type.unique() #查看不重复的数据
df_test.dropna(axis= 1) #删除含有NaN值的列
df_test.volume.fillna(method= 'ffill',limit= 1) #限制向前或者向后填充行数


tmp = tmp.set_index('order_unix_time')  设置索引 


group by 
文档 http://pandas.pydata.org/pandas-docs/stable/groupby.html#group-by-split-apply-combine
grouped = df.groupby('A')
grouped = df.groupby(['A', 'B'])
grouped.get_group('bar')
按照多列分组
df.groupby(['A', 'B']).get_group(('bar', 'one'))

组内排序
My_Frame['sort_id'] = My_Frame['salary'].groupby(My_Frame['dep_id']).rank()
注意排序的时候，如果是按照多列分组，多列的写法和上面的不太一样：

如果写成 .groupby(samll_shopes['a', 'b']) 则会报错： ValueError: Grouper for '<class 'pandas.core.frame.DataFrame'>' not 1-dimensional



滑动
df['A'] = df.rolling(2).mean()   注意，针对的是index，即index每隔两个。由于默认的 index 是从0开始的数字，因此和每隔多少行无异
但是如果 index 是 时间戳这种，就可以按照 '2s' 这种间距
df.rolling(2, min_periods=1).sum()  
df_re.groupby('A').rolling(4).B.mean() 直接 rolling 会无视索引的区别，但是线分组的话，rolling 也会被限制在索引中
x.B.xs('a').xs('china')

累积
df_re.groupby('A').expanding().sum()


df['new_column'] = pd.Series 这种赋值方式似乎并不是我们想象的那样，例如，当我给一个 df 赋予随机值列的时候，可能因为df 的 index 不连续，造成好多 NUll 随机数。



df_re.groupby('group').resample('1D').ffill()
resample 等频抽样，D为天，S为秒，

过滤 sf.groupby(sf).filter(lambda x: x.sum() > 2)

df.groupby('g').boxplot()



二维表
pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True)

排序
data_sorted = data.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)

箱线图
data.boxplot(column="ApplicantIncome",by="Loan_Status")

mask 注意==判断时，注意每列的类型，不要混淆int64 和 string

df.xs(index)  按索引选择
如果列名是中文，则用.号取值会报错，用中括号的方式即可。

mask = (x['中国'] <=2 )


import pandas as pd
train = pd.read_csv('train.csv')
train.shape
train.head()
train.columnA.describe()

describe(include='all') 描述所有列（默认只数值列）


取 dataframe 的一列成为 array
shop_id = shops.main_poi_id.values



选择和变换
mask = (df_tr.hour.values == 11) | (df_tr.hour.values == 17) & (df_tr.day.values == 17)
注意，.values 取数值， .str 取字符串， 
例如 (df.messgae.str.find('model_e') != -1) 此处 find 用于判断字符串查找
但是，但是，对于中文，好像又不能用 str 和 unicode 的 u
例如，tmp = raw_data.loc[raw_data.mt_city_name == "北京"] 可以，但是 tmp = raw_data.loc[raw_data.mt_city_name.str == "北京"] 或者 tmp = raw_data.loc[raw_data.mt_city_name == u"北京"] 都是不对的！！！



对于中文列名，如何处理？
x[x['中国'] <= 2]  这种选择方法当时也是可以的

df['gen'] = df['gen'].mask(df['gen'] + df['cont'] < 0.01)



df.loc[df['First Season'] > 1990, 'First Season'] = 1   
df['First Season'] = (df['First Season'] > 1990).astype(int)

选出数值特征
numerric_features = train.select_dtypes(includes=[np.number])
numerric_features.dtypes

筛选非数字特征
categorical = train.select_dtypes(exclude=[np.number])


merge
candi = pd.merge(data, mt_poi, left_on=['city', 'geohash_kb'], right_on=['city_name', 'geohash_mt'], how='left')


geohash
import pygeohash as pgh
def geo_encode(x, n=4, by=1):
    return str(pgh.encode(x[0]/by, x[1]/by,n)) + "--" + x[2]





变量的协方差矩阵
corr = numerric_features.corr()
协方差最大最小值查看
corr[columnA].sort_values(assending=false)[:5]
coor[columnA].sort_values(assending=false)[-5:]

去除重复值
train.columnA.unique()
drop_duplicates(subset=None, keep='first', inplace=False) 

数据透视表
pivot = train.pivot_table(index = 某个分类变量, values=某个数值变量, aggfunc=np.mean)
pivot = df_tr.pivot_table(index = 'area_id', values='poi_id', aggfunc=lambda x: len(x.dropna().unique()))
print pivot
还可以加上 columns

多列透视，这个厉害了
p = df_tr.groupby('area_id').aggregate({'delivery_duration':np.mean, 'poi_id':lambda x: len(x.dropna().unique())})

data.groupby(['model', 'bu']).size()


一列构造多列
df.textcol.apply(lambda s: pd.Series({'feature1':s+1, 'feature2':s-1}))

时间序列
日期类型转换
loandata['issue_d']=pd.to_datetime(loandata['issue_d'])

对此透视表，画条形图
pviot.plot(kind='bar', color='blue')

条件筛选
train = train[train[columnA] < 100]

空值
nulls = pd.dataFrame(train.isnull().sum().sort_values(assending=false)[:25])
统计每列的 nan 值 或者 缺失值
df.isnull().sum()
缺失值填充
df['time3_category'] = df['time3_category'].fillna(1)
df.fillna(df.mean()['a', 'b'])  这种在数据量很大的时候似乎总是很慢不可用


设置列名称
null.columns = ['name1']
设置行索引名称
null.index.name = 'name2'


分类变量的值和频次
y = train.columnA.value_counts()
len(y[y>1].unique()) 过滤频次


哑变量 ont-hot 编码  注意，训练集和测试集要编码一致
train['a_encode'] = pd.get_dummies(train.columnA, drop_first=true)
test['a_encode'] = pd.get_dummies(train.columnA, drop_first=true)

Apply 方法
train.columnA.apply(func)

插补缺失值
data = train.select_dtypes(include=[np.number]).interpolate().dropna()


去除列
data.drop(['featureA'], axis=1)

创建dataFrame
df = pd.dataFrame()
df['id'] = train.id

dataFrame 转化为csv
df.to_csv('a.csv', index=False)
d.to_csv('/Users/zhangxisheng/Downloads/big_meal_single_poi.csv', index=False, encoding='utf-8')  有时候有编码问题的时候，加上 encoding 能够解决

一个 to_csv 的疑难杂症：
我的 df 只有一列，类型为字符串，当我用 to_csv 存储下来之后，发现有些行被加上了双引号，有些则没有。
最后在轩哥的帮助下定位到原因： 这些加上了双引号的行，都是因为本身带有都好；解决办法就是，指明 sep="\t"，这样就行了。

obj.combine_first(other)
如果 obj 中有为 null 的值，用对应索引的 other 中的数据填充


某一列为 nan 的行
df = df[np.isfinite(df['EPS'])]




索引
agg.index.values

列与索引之间可以相互转化
df.set_index('date', inplace=True)
x['index'] = x.index.get_level_values('C')
df.reset_index(inplace=True, drop=False)  对多重索引，将会把每个索引变为列，新加索引为从0开始的自然数，非常赞

层次化索引
所谓层次化索引，其实也是多重索引，只不过有些相同的被省略了。
data.unstack()  可以打平层次化索引中的部分索引，即变宽表

多重索引
m_idx = pd.MultiIndex.from_tuples(zip(dates, labels), names=['date', 'label'])
data_dict = {'observation1':obs1, 'observation2':obs2}
df = pd.DataFrame(data_dict, index=m_idx)
参见 http://www.jianshu.com/p/3ab1554fe6f3

类型转换
df_ta['delivery_duration'] = df_ta['delivery_duration'].astype['float']

斜度
train.columnA.skew()


标签编码
lbe = LabelEncoder()
lbe.fit(df_tr['area_id'].values.reshape(-1, 1))
df_ta['area_id_le'] = lbe.transform(df_ta['area_id'].values.reshape(-1, 1))

宽表变窄表
df = pd.melt(df, id_vars=["date"], var_name="condition")

cut 切分和分组
bins = [0, 5, 10, 15, 20] 
group_names = ['A', 'B', 'C', 'D'] 
loandata['categories'] = pd.cut(loandata['open_acc'], bins, labels=group_names)

分列
grade_split = pd.DataFrame((x.split('-') for x in loandata.grade),
    index=loandata.index,columns=['grade','sub_grade'])



df.info(memory_usage='deep') 该表的精确内存使用量，行列个数，以及对应的数据类型个数
在底层，pandas会按照 数据类型 将 列 分组形成数据块（blocks）
对于包含数值型数据（比如整型和浮点型）的数据块，pandas会合并这些列，并把它们存储为一个Numpy数组（ndarray）。
Numpy数组是在C数组的基础上创建的，其值在内存中是连续存储的。基于这种存储机制，对其切片的访问是相当快的。


看每种类型的块所占内存
for dtype in ['float', 'int', 'object']:
    selected = df.select_dtypes(include=[dtype])
    mean_useage = selected.memory_usage(deep=true).mean()
    mean_useage = mean_useage/1024**2

pandas中的许多数据类型具有多个子类型，它们可以使用较少的字节去表示不同数据，比如，
float型就有float16、float32和float64这些子类型

参考 用pandas处理大数据——节省90%内存消耗的小贴士
https://mp.weixin.qq.com/s?__biz=MzAxNTc0Mjg0Mg==&mid=2653286198&idx=1&sn=f8f0ea4845586b1f9b645995aa07d8a0&open_source=weibo_search




========================================  画图
画图记住一句话，属性设置的时候，设置给plt

palette = sns.color_palette()
color = palette.pop()

对index 为 datetime 的，且列只有一列的，可以直接画图
plt.plot(x)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.style.use(stype='ggplot')
plt.rcParams['figure.figuresize'] = (10.6)


散点图
plt.scatter(x=, y=)


mu, sigma = 100, 15
x = mu + sigma*np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

图的展现
plt.show()  注意 seaborn 也是需要做这个语句才能呈现图

直方图
plt.hist(dataA, color='blue')

中文标记
import numpy as np
import pandas as pd
  
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set(color_codes=True)



多个分布图
sns.distplot(df_ta_morning.delivery_duration.values, kde_kws={"color": "k", "lw": 3, "label": "KDE"})
sns.distplot(df_ta_afternoon.delivery_duration.values, label=u'afternoon', color='green')

散点／回归图
sns.lmplot('order_id',  # Horizontal axis
           'delivery_duration',  # Vertical axis
           data=p.xs(1002483),  # Data source
           fit_reg=True,  # Don't fix a regression line
           line_kws={'color': 'green'}
           )

多个散点图拼接在一张图
fig, axs = plt.subplots(ncols=3)
sns.regplot(x='value', y='wage', data=df_melt, ax=axs[0])
sns.regplot(x='value', y='wage', data=df_melt, ax=axs[1])
sns.boxplot(x='education',y='wage', data=df_melt, ax=axs[2])


myplot.set_xticklabels(rotation=30)  横坐标标度倾斜


时间序列 time series
y.index = y['day']
x = y['diff']
plt.show()
或者 df.set_index('date')['value'].plot()  这种对日期斜着展现，很好

多条时间线  针对窄表，分组
fig, ax = plt.subplots()
ax.set_color_cycle(['red', 'green', 'blue'])

for key, grp in df2.groupby(['main_type']): 
    grp['ratio'].plot(label = "{}".format(key))
    # plt.plot(grp.index, grp['ratio'], label = "{}".format(key))
plt.legend(loc='best')    
plt.show()

如果是多列，想要画在一张图中，则直接 df.plot() 即可。






===============================  正则匹配统计
import re

pattern = re.compile(r'(.*)_ck(.*)_(.*)')
cks = {}

with open("hyclicktags.csv", "r") as f:
    for line in f:
        tmp = int(line.split(",")[1])
        match = pattern.match(line)
        if match:
            if match.group(2) in cks:
                c = cks.get(match.group(2))
                cks[match.group(2)] = c + tmp
            else:
                cks[match.group(2)] = tmp

count = len(cks)
print count

if count <= 200:
    for i in cks:
        print i, cks[i]



正则替换
例如，替换括号中的内容
import re
str = '多摩君1（英文版）c(ab)(34) '
out = re.sub('（.*?）|\(.*?\)', '', str)
print out




#  numpy

numpy 101 题  https://www.machinelearningplus.com/101-numpy-exercises-python/

去重的时候顺便统计频次
(values,counts) = np.unique(p,return_counts=True)
ind=np.argmax(counts)
print values[ind], counts[ind]




import numpy as np
np.log() 对序列做对数变换
np.exp()


每行的和
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
注意： 对series ， apply 的方法，不能写 axis=1



每列的和
df.loc['Row_sum'] = df.apply(lambda x: x.sum())

替换
arr[arr % 2 == 1] = -1

where 筛选
 x = np.where(arr % 2 == 1, -1, arr)

重复
b = np.repeat(1, 10).reshape(2,-1)

np.r_[np.repeat(a, 3), np.tile(a, 3)]

公共元素
np.intersect1d(a,b)

# From 'a' remove all of 'b'
np.setdiff1d(a,b)

np.where(a == b)

index = np.where((a >= 5) & (a <= 10))
a[index]

标量函数向量化
pair_max = np.vectorize(maxx, otypes=[float])


==================================== scikit-learn

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size = 0.33)

线性模型
from sklearn import linear_model
lr = linear_model.LinearRegression()

训练模型
model = lr.fit(x_train,y_train)

model.score() 返回被模型解释的方差的占比（即R方）

模型预测
predicts = model.predict(x_test)

模型评估/评测
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predicts)


岭回归
ridge = linear_model.Ridge(alpha=0.5)

ridge_model = ridge.fit(x_train, y_train)

one hot 编码
单列 OneHotEncoder(sparse = False).fit_transform( testdata[['age']] )
多列 OneHotEncoder(sparse = False).fit_transform( testdata['age', 'salary'])
OneHotEncoder 不能对字符串型的值做处理

LabelBinarizer().fit_transform(testdata['pet'])


==================== csv 文件处理

import pandas as pd

from datetime import datetime

data_location = "/Users/zhangxisheng/Documents/projects/商家不接待项目/refuse_samples.csv"

df = pd.read_csv(data_location, sep=',', header=0)


def str2date(date_str, formats="%Y-%m-%d %H:%M:%S"):
    return datetime.strptime(date_str, formats).date()

dt.weekday()  返回一周的第几天，注意，周一是0


df['date'] = df['date'].map(lambda x: str2date(x))

print df.shape

print df.head(10)
print df['date'][:3]

print df['date'].value_counts()


聚合
print pd.pivot_table(df, values='fea#1', index=['fea#5'], columns='label', aggfunc=len).reset_index()


多列转一列 由多列合成一列
df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)


列求和
small.sales.sum()

shuffle 混洗
from sklearn.utils import shuffle  
df = shuffle(df)  

划分的时候不要混洗
train_test_split(y, shuffle=False)






### ============= sklern

有时候将 dataFrame 用 df.values  转为 ndarray 进行训练的时候，爆出如下错误
Input contains NaN, infinity or a value too large for dtype('float32')

可以如下检查
        if np.any(np.isnan(train_feat)):
            print "存在含有 null 值 的列"

        if np.all(np.isfinite(mat)):
            print "存在非有限值的列"




单列作为ndarray， 要重塑
df.col.values.reshape(-1,1)



# ======================================================
最地道的python
def ecode(x): return 1 if x == 'a' else 0




### =========== tensorflow

在tensor的某一维度上求值的函数。如：

求最大值
tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)

求平均值
tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)





### 参考资料
- [stackoverflow 上一些 python 问题合集 ](http://pyzh.readthedocs.org/en/latest/python-questions-on-stackoverflow.html)
- [python tips/intermediate python](http://book.pythontips.com/en/latest/index.html)

- [关于 python 的面试题](https://github.com/taizilongxu/interview_python)


- [The Best of the Best Practices(BOBP) GUide for Python](https://gist.github.com/sloria/7001839)
