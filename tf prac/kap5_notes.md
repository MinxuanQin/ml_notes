## 狗猫分类实验

1. 数据来源


```python
!wget --no-check-certificate \
  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
  -O /tmp/cats_and_dogs_filtered.zip
```

用`os.mkdir()`可以创建文件夹。这里有一个细节是狗和猫文件夹的大小相同。此次试验和之前不同的是图片的大小，形状不尽相同，因此必须要进行预处理。

结果如下：

![c5p5](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p5_cats.png)

2. 数据分割

函数split_data的任务在于此。具体思路如下：

首先对SOURCE文件夹中的所有文件检查其是否为空,files同时存放了所有有效文件的名字的list

第二步确定两个文件夹的大小（**注意用int确保结果为整数**）

第三步为用`random.sample(list,len(list))`生成一个乱序list。**这里的第二个参数是files的大小而不是testing_length or training_length!!**


```python
import random
a = [0,67,40,99,9, 80, 97]
b = random.sample(a,4)
c = b[0:4]
print(a)
print(b)
print(c)

#输出表示c的声明那里不包含4
```

    [0, 67, 40, 99, 9, 80, 97]
    [67, 0, 80, 40]
    [67, 0, 80, 40]


3. 构建网络

方式类同第四章，构建了三层神经网络。有趣的地方在于训练步骤的语句：


```python
history = model.fit_generator(train_generator,
                             epochs = 2,
                             verbose = 1,
                             validation_data = validation_generator)
```

网络可视化如下：

![c5p6](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p6.png)
![c5p7](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p7.png)
![c5p8](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p8.png)
![c5p9](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p9.png)
![c5p10](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p10.png)

4. 结果展示

问答式的程序运行结果如下（训练两次，平均一次五分钟左右）：

![c5p1](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p1.png)
![c5p2](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p2.png)
![c5p3](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p3.png)

讲习式的程序运行结果如下（训练五次，采用部分训练，且数据集也是切片，数据更少,因此出现了**overfit**）：
![c5p4](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c5p4.png)
