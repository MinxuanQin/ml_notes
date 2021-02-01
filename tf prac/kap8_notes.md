## 图像多分类问题

### 手写体识别

1. get_data()

从csv文件读取数据。正确形式的一行应该含有28\*28+1=785个数字，其中有“1”是因为标签。

函数应该返回两个array，一个是标签，一个是图片。

第一行通过if-else语句来跳过。先用temp_images,temp_labels**两个列表**来加入数据，最后再进行类型转换。

`np.array_split(iamge_data,28)`将数组恢复为28\*28的图像。

注意手法：`with open(filename) as training_file`

2. 构建网络

这里与之前的方法相似，识别26个字母，使用稀疏的交叉熵作为损失函数，Adam做优化器。

结果显示测试集的表现比训练集好，说明训练样本不足，没有很好的学习到特征，而测试集的数目比较小，准确率更高一些。

### 手势识别：石头 剪刀 布
三种手势，generator的class_mode改为**categorical**，在全连接层那里添加了一个dropout层来减少过拟合/提高泛化能力。

[理解dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)

预处理过程中之所以两个集合都做了规划是为了减少计算量（图片的大小是150\*150\*3，已经很大了），同时考虑到sigmoid这样的函数在饱和区工作的问题。

`history` as result of `model.fit_generator()` saves data of each epoch

`model.save('rps.h5')`将训练好的参数存储起来，方便之后直接使用（如做迁移学习）

下图是训练25次之后的统计示意图（from mooc course）：

![c8p1](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c8p1.jpeg)

分析：总体还算稳定，出现波动的原因主要在于训练样本不足。可以通过多种方法继续改进。
