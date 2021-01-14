```python
import numpy as np
%config IPCompleter.greedy=True
```


```python
#加载fashion mnist数据集
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()
```


```python
print(train_images.shape)
print(test_images.shape)
```

    (60000, 28, 28)
    (10000, 28, 28)



```python
print(train_labels[:5])
#输出前五个元素
```

    [9 0 0 3 0]



```python
import matplotlib.pyplot as plt
plt.imshow(train_images[1])
```




    <matplotlib.image.AxesImage at 0x7fc65395d3d0>




    
![png](output_4_1.png)
    



```python
#构建神经元模型
import tensorflow as tf
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#尝试删去input_shape = (28,28) in Flatten layer
```

三层结构，第一层作为输入数据，第三层表示输出，一共十个类别所以是十个神经元。

第二层的神经元个数128是随机指定的

示意图如下：

![picture1 for nn](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c2p1.jpeg)

![picture2 for nn](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c2p2.jpeg)
圆圈代表神经元，Relu的特点为输入为正时才有输出；softmax特点为将输出压缩在0和1之间。

可以用以下代码实现：

``
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation = tf.nn.relu))
model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
``


```python
#查看模型
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten (Flatten)            multiple                  0         
    _________________________________________________________________
    dense (Dense)                multiple                  401920    
    _________________________________________________________________
    dense_1 (Dense)              multiple                  5130      
    =================================================================
    Total params: 407,050
    Trainable params: 407,050
    Non-trainable params: 0
    _________________________________________________________________


结果的100480 = （28\*28 + 1）\* 128<br>
1290 = (128 + 1) \* 10<br>
1表示bias，这种网络结构叫做全连接的网络结构。


```python
#训练模型，评估效果
#先对数据进行normalization/scaling
train_images_scaled = train_images/255.

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(train_images_scaled,train_labels,epochs=5)

test_images_scaled = test_images/255.
model.evaluate(test_images_scaled,test_labels)
```

    Epoch 1/5
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.4974 - accuracy: 0.8242
    Epoch 2/5
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.3564 - accuracy: 0.8703
    Epoch 3/5
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.3216 - accuracy: 0.8824
    Epoch 4/5
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2976 - accuracy: 0.8898
    Epoch 5/5
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.2830 - accuracy: 0.8955
    313/313 [==============================] - 0s 811us/step - loss: 0.3602 - accuracy: 0.8737





    [0.36019039154052734, 0.8737000226974487]



optimizer可以用`tf.optimizers.Adam`,loss同理,`tf.losses.sparse_categorical_crossentropy`

Adam很常用，loss是做分类时常用的，sparse代表整数，后续因为此处的output为one_hot(只有一项为1)而采用

将数据转为0到1之间可以有效增强准确率


```python
#0 T-Shirt/top, 1 Trouser/pants, 2 Pullover shirt, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneakers, 8 Bag, 9 Ankle boot
classify = model.predict(test_images_scaled)
print(classify[333])
print(np.argmax(classify[333]))
print(test_labels[333])
```

    [4.5049340e-08 2.9801519e-09 4.6833576e-10 1.2401997e-10 3.3784858e-10
     1.0596131e-04 3.2817879e-10 9.9988437e-01 2.9971893e-06 6.6260568e-06]
    7
    7



```python
a = np.arange(20).reshape((4,5))
print(a)
a_new = a.reshape((1,4,5))
print(a_new)
```

    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]]
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]
      [15 16 17 18 19]]]


上述操作尝试对array a进行升维，也可以用`a = a[np.newaxis, :]`来完成。


Softmax的作用是将更大的值传给下一层网络，这样会节省大量的编码！(to save a lot of coding)

test_images形式与training_images相似(in future if you have data that look like the training data, then it can make a prediction for what that data would look like)

实训平台初步训练结果如下：

![c2p3](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c2p3.png)
![c2p4](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c2p4.png)

增加神经元个数一定程度上会延长训练时间，增加准确度，但是很快会遇到收益递减定律(hit the law of diminishing returns)

教程练习中输入层和输出层错误的选择会使程序报错，原因总是列在最后面

在这个简单的例子中增加层数不会显著影响神经网络的性能。

实践发现是否normalize会对结果产生非常严重的影响！


```python
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs = {}):
        if(logs.get('loss')<0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True
            
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_img, training_lab),(test_img,test_lab) = mnist.load_data()
training_img =training_img/255.0
test_img = test_img/255.0

model = tf.keras.Sequential([tf.keras.layers.Flatten(),
                            tf.keras.layers.Dense(512, activation=tf.nn.relu),
                            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = "adam", loss = tf.losses.sparse_categorical_crossentropy, metrics = ['accuracy'])

model.fit(training_img,training_lab, epochs = 5,callbacks = [callbacks])
```

    Epoch 1/5
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.4712 - accuracy: 0.8306
    Epoch 2/5
    1873/1875 [============================>.] - ETA: 0s - loss: 0.3578 - accuracy: 0.8694
    Loss is low so cancelling training!
    1875/1875 [==============================] - 2s 1ms/step - loss: 0.3577 - accuracy: 0.8694





    <tensorflow.python.keras.callbacks.History at 0x7fc5a5b05a30>



回调示例。以准确度为指标对训练次数进行限制见练习。

`tf.keras.callbacks.Callback`是keras提供的一个类，我们在这里定义了一个新方法`on_epoch_end`来对epochs进行限制。
