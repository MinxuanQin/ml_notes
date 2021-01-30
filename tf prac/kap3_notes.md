## CNN 卷积神经网络

识别物体的特征来判断(focus on specific, distinct details)

卷积=和(像素\*过滤器)

过滤器效果：竖直，水平

训练目标：过滤器神经元中的数值

Max Pooling:通过取最大数据来增强图像特征，减少数据（尺寸减半）

### 网络构造

在全连接网络的基础上添加4层。

`Conv2D`中64代表过滤器数目，input为灰度值,1代表一个通道

注意需要调节input的维度,用reshape函数，第一项参数为-1

**来自实训示例：当epochs提升到比较大的数字例如20时，测试结果可能会下降，这是过拟合的缘故。**


```python
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images/255.
test_images = test_images/255.

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (28,28, 1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu"),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model.compile(optimizer = "Adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
model.fit(train_images.reshape(-1,28,28,1),train_labels, epochs = 5)
```

    Epoch 1/5
    1875/1875 [==============================] - 32s 17ms/step - loss: 0.4473 - accuracy: 0.8373
    Epoch 2/5
    1875/1875 [==============================] - 34s 18ms/step - loss: 0.2977 - accuracy: 0.8909
    Epoch 3/5
    1875/1875 [==============================] - 34s 18ms/step - loss: 0.2520 - accuracy: 0.9077
    Epoch 4/5
    1875/1875 [==============================] - 34s 18ms/step - loss: 0.2197 - accuracy: 0.9176
    Epoch 5/5
    1875/1875 [==============================] - 34s 18ms/step - loss: 0.1924 - accuracy: 0.9272





    <tensorflow.python.keras.callbacks.History at 0x7fa05e42ed60>




```python
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_12 (Conv2D)           (None, 26, 26, 64)        640       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 13, 13, 64)        0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 11, 11, 64)        36928     
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 1600)              0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 128)               204928    
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 243,786
    Trainable params: 243,786
    Non-trainable params: 0
    _________________________________________________________________


### 模型分析

模型一共由7层组成，其中Max pooling层和flatten层并没有参数。

经过一次卷积层会生成64张图片（提取了64中特征），每一层的参数数目是3\*3+1(bias)。


```python
model.evaluate(test_images.reshape(-1,28,28,1),test_labels)
```

    313/313 [==============================] - 1s 3ms/step - loss: 0.2424 - accuracy: 0.9128





    [0.24242854118347168, 0.9128000140190125]




```python
import matplotlib.pyplot as plt
#获取各层输出
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
#七层输出在pred中，对于testing集合的第一张图
pred = activation_model.predict(test_images[0].reshape(-1,28,28,1))
plt.imshow(pred[1][0,:,:,6])
```

### 观察各层的outputs

方法：构建模型(`tf.keras.models.Model`)，包括model中的input和output。再通过`model.predict`来获得各层输出图像。

以下是outputs：(注意图片的大小区别)

1. 第一层及第二层特征2的图像

![c3p1_1](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c3p1_1.png)
![c3p2_1](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c3p2_1.png)

2. 特征7的图像

![c3p3_2](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c3p3_2.png)
![c3p4_2](https://github.com/MinxuanQin/pics/blob/master/tensorflow_prac/c3p4_2.png)
