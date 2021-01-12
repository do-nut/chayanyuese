# 异常检测概述
异常指与大多数样本数据发生偏离的情况，异常的现象往往具有较高的价值，值得我们去关注，对于成因的探讨能够带来对于流程、步骤、方法等完善的思路。因此，学会异常检测、选择好的方法进行异常检测至关重要。
### 异常的类别
| 异常的类别 | 定义 |
|---|---|
| 点异常 | 最直观的异常类别，即少数个体异常，大多数个体与预期相符；如：患者的健康指标|
| 上下文异常| 这种异常是基于某种特定环境下产生的，在时许数据中非常常见；如：非节假日、纪念日下的高额饮食消费|
| 群体异常| 一组样本数据共同导致了异常，但单独看群体中的个体数据可能是正常的 |

在以上三种异常中，点异常和上下文异常比较好理解，点异常就是通俗情况下对于异常的理解，上下文异常则是更进一步贴合实际情况，考虑到数据的属性；而群体异常不太好理解，参考了[Datawhale](https://github.com/datawhalechina/team-learning-data-mining/blob/master/AnomalyDetection/%E4%B8%80%E3%80%81%E6%A6%82%E8%BF%B0.md#11-%E5%BC%82%E5%B8%B8%E7%9A%84%E7%B1%BB%E5%88%AB)和另外一个up主的[解释](https://zhuanlan.zhihu.com/p/93779599?utm_source=wechat_session)，我觉得可以理解为一个个体可能是属于正常范围内的取值，但是这一批样本数据整体呈现出与其他正常数据**不同的规律、属性或者说是特点**，因此产生了异常。

### 异常检测类型
无监督：最常见的一种，训练集无标签，绝大多数异常检测都属于此类情况；
有监督：训练集有标签，在这种情况下异常检测的用途更多是用来增强特征空间；
半监督：在训练集中只有单一类别（正常实例）的实例，没有异常实例参与训练；

### 异常检测的应用场景
* 故障检测（点异常）
* 物联网异常检测
* 欺诈检测
* 工业异常检测
* 时间序列异常检测（上下文异常）
* 视频异常检测
* 日志异常检测
* 医疗日常检测
* 网络入侵检测（群体异常）
# 异常检测常用的模型与算法
### 传统方法
##### 线性模型
PCA：将数据集降维，对全部数据计算特征向量，数据能够最大程度保有原始特征，而异常样本距离特征向量的距离比较远（在降维过程中遗失较多信息），因此这个距离可以被用作异常值的判断；
* 优点：直觉上简单，运算开销适中；
* 缺点：线性模型较为局限，例如考虑不到非线性的问题；如果特征较多也不适合；

##### 基于相似度度量的算法
与聚类的思想一致，也是大部分异常检测算法的核心思想，**大部分异常检测算法都可以被认为是一种估计相似度，无论是通过密度、距离、夹角或是划分超平面**

KNN：计算每个点距离第k个点近邻的距离，距离越大越可能是异常点，一般k设为5-10之间，不易过大或过小；
* 优点：简单易懂容易实现；
* 缺点：要计算距离，运算开销大，不适合高维数据；

LOF：是对KNN的一种改进，计算每个点在局部区域上的密度和其领域点的密度，如果某个点的密度比其领域点的密度低，那么我们偏向于它是异常点；
* 优点：简单易懂容易实现；
* 缺点：要计算距离，运算开销大，不适合高维数据；

HBOS：CMU在读phd赵越所改进的方法，假设每个维度间都是独立的，分别计算一个样本在不同维度上所处的密度空间并叠加结果，用直方图去模拟密度分布
* 优点：单开销小可并行计算适用于大量数据；
* 缺点：无法考虑不同特征间的相互影响的关系（直觉认识上最终的结果是由不同的特征共同作用下导致的），但实际效果还不错；

### 集成算法
集成是提高数据挖掘算法精度的常用方法。集成方法将多个算法或多个基检测器的输出结合起来。其基本思想是一些算法在某些子集上表现很好，一些算法在其他子集上表现很好，然后集成起来使得输出更加鲁棒。集成方法与基于子空间方法有着天然的相似性，子空间与不同的点集相关，而集成方法使用基检测器来探索不同维度的子集，将这些基学习器集合起来。

孤立森林：利用决策树的特点，通过不断对特征空间进行划分，异常点更容易被切分成孤立，非异常点需要分割更多次，从而进行区分；
* 优点：运行好效果好适用于大量数据，容易并行；
* 缺点：需要调的参数比较多，结果有一定随机性；

XGBOD：这是一种**有标签**的算法，如果数据有标签，结合xgboost与无监督的检测方法，把无监督的检测方法用于强化特征空间；
* 优点：有标签效果当然会更好；
* 缺点：训练集有标签有难度，特别是对新的领域；
![if 图片](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=%E5%AD%A4%E7%AB%8B%E6%A3%AE%E6%9E%97&step_word=&hs=0&pn=0&spn=0&di=77550&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=0&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=undefined&cs=2876152971%2C2580579145&os=3079342458%2C1772448760&simid=4110000566%2C585583883&adpicid=0&lpn=0&ln=820&fr=&fmq=1610440726526_R&fm=&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=https%3A%2F%2Fgimg2.baidu.com%2Fimage_search%2Fsrc%3Dhttp%3A%2F%2Fupload-images.jianshu.io%2Fupload_images%2F4517099-01e49039eaa86636.png%26refer%3Dhttp%3A%2F%2Fupload-images.jianshu.io%26app%3D2002%26size%3Df9999%2C10000%26q%3Da80%26n%3D0%26g%3D0n%26fmt%3Djpeg%3Fsec%3D1613032730%26t%3Db8229476b647ce5bdcd41608cc6ed3fa&fromurl=ippr_z2C%24qAzdH3FAzdH3F3twgfi7_z%26e3Bt5AzdH3FrAzdH3Fcwunvmmja98aAzdH3F&gsm=1&rpstart=0&rpnum=0&islist=&querylist=&force=undefined)

### 神经网络（机器学习）算法

自编码器：核心思想跟PCA类似，但扩展到了非线性模型上，通过自编码器先压缩再还原，异常点会有更大的重建误差，因为异常与大部分相差大，压缩和重建的过程中遗失的信息更多，误差就会更大；
* 优点：理解简单，大数据多特征适用；
* 缺点：神经网络具有一定随机性，运算开销比较大；

# 使用异常检测的技巧

* 数据有标签，优先使用监督学习，比如xgboost；
* 方法选择时可以使用MetaOD获得算法推荐，或者**优先选择孤立森林**；
* 手动选择模型的话，首先考虑数据量和数据结构，当数据量比较大（>10万条，>100个特征）**优先选择可扩展性强的**，比如孤立森林，HBOS，COPOD；
* 数据量不大且追求精度比较高，可以尝试随即训练多个模型；
* 数据量大特征多不要用传统方法，上神经网络；

# 常检测常用开源库
Scikit-learn：

Scikit-learn是一个Python语言的开源机器学习库。它具有各种分类，回归和聚类算法。也包含了一些异常检测算法，例如LOF和孤立森林。

官网：<https://scikit-learn.org/stable/>

PyOD：

**Python Outlier Detection（PyOD）**是当下最流行的Python异常检测工具库，其主要亮点包括：

包括近20种常见的异常检测算法，比如经典的LOF/LOCI/ABOD以及最新的深度学习如对抗生成模型（GAN）和集成异常检测（outlier ensemble）
支持不同版本的Python：包括2.7和3.5+；支持多种操作系统：windows，macOS和Linux
简单易用且一致的API，只需要几行代码就可以完成异常检测，方便评估大量算法
使用JIT和并行化（parallelization）进行优化，加速算法运行及扩展性（scalability），可以处理大量数据

# 练习环节
参考资料：

* <https://zhuanlan.zhihu.com/p/58313521（pyod库作者对pyod库的介绍）>
* <https://pyod.readthedocs.io/en/latest/ （Pyod库官网）>

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.font_manager

from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers
X_train, Y_train = generate_data(n_train=200,train_only=True, n_features=2)

outlier_fraction = 0.1

x_outliers, x_inliers = get_outliers_inliers(X_train,Y_train)

n_inliers = len(x_inliers)
n_outliers = len(x_outliers)

F1 = X_train[:,[0]].reshape(-1,1)
F2 = X_train[:,[1]].reshape(-1,1)
 
xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

plt.scatter(F1,F2)
plt.xlabel('F1')
plt.ylabel('F2') 

#训练一个kNN检测器
clf_name = 'kNN'
clf = KNN() # 初始化检测器clf
clf.fit(X_train) # 使用X_train训练检测器clf

#返回训练数据X_train上的异常标签和异常分值
y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)

y = np.concatenate((y_train_pred,y_train_scores))
y

