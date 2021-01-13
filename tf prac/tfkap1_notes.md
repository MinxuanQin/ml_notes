## tensorflow实操课堂笔记 第一部分
使用jupyter lab作为实验环境，本节主题：一个神经元的网络

1. 打开代码自动提示 `%config IPCompleter.greedy=True`
2. keras是tensorflow中集成度很高的API,Sequential是其中的常用模块(Sequential layers)
3. sgd是stochastical gradient descent的缩写，常用优化器
4. optimizer: make another guess
5. 实例中实际上用了epochs=500，输出值为18.97，在这里为了节省空间将训练次数缩小到50，效果明显下降。
6. 存在误差的原因：神经网络只能给出最可能的结果（probability,not certainty）
7. 练习题：房价预测。形式极其类似，Trick: Scaling the house price down(150k to 1.5)


```python
import tensorflow as tf
print(tf.__version__)
```

    2.2.1



```python
import numpy as np
%config IPCompleter.greedy=True
```


```python
from tensorflow import keras
#model building
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#data preparing
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

model.fit(xs, ys, epochs = 50)

print(model.predict([10.0]))
```

    Epoch 1/50
    1/1 [==============================] - 0s 996us/step - loss: 60.1692
    Epoch 2/50
    1/1 [==============================] - 0s 1ms/step - loss: 47.7700
    Epoch 3/50
    1/1 [==============================] - 0s 1ms/step - loss: 38.0061
    Epoch 4/50
    1/1 [==============================] - 0s 1ms/step - loss: 30.3156
    Epoch 5/50
    1/1 [==============================] - 0s 2ms/step - loss: 24.2566
    Epoch 6/50
    1/1 [==============================] - 0s 1ms/step - loss: 19.4813
    Epoch 7/50
    1/1 [==============================] - 0s 814us/step - loss: 15.7162
    Epoch 8/50
    1/1 [==============================] - 0s 872us/step - loss: 12.7460
    Epoch 9/50
    1/1 [==============================] - 0s 1ms/step - loss: 10.4013
    Epoch 10/50
    1/1 [==============================] - 0s 1ms/step - loss: 8.5490
    Epoch 11/50
    1/1 [==============================] - 0s 2ms/step - loss: 7.0842
    Epoch 12/50
    1/1 [==============================] - 0s 1ms/step - loss: 5.9243
    Epoch 13/50
    1/1 [==============================] - 0s 1ms/step - loss: 5.0046
    Epoch 14/50
    1/1 [==============================] - 0s 1ms/step - loss: 4.2740
    Epoch 15/50
    1/1 [==============================] - 0s 905us/step - loss: 3.6923
    Epoch 16/50
    1/1 [==============================] - 0s 1ms/step - loss: 3.2278
    Epoch 17/50
    1/1 [==============================] - 0s 971us/step - loss: 2.8558
    Epoch 18/50
    1/1 [==============================] - 0s 748us/step - loss: 2.5566
    Epoch 19/50
    1/1 [==============================] - 0s 1ms/step - loss: 2.3148
    Epoch 20/50
    1/1 [==============================] - 0s 955us/step - loss: 2.1184
    Epoch 21/50
    1/1 [==============================] - 0s 2ms/step - loss: 1.9577
    Epoch 22/50
    1/1 [==============================] - 0s 763us/step - loss: 1.8253
    Epoch 23/50
    1/1 [==============================] - 0s 878us/step - loss: 1.7153
    Epoch 24/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.6231
    Epoch 25/50
    1/1 [==============================] - 0s 843us/step - loss: 1.5448
    Epoch 26/50
    1/1 [==============================] - 0s 964us/step - loss: 1.4778
    Epoch 27/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.4197
    Epoch 28/50
    1/1 [==============================] - 0s 877us/step - loss: 1.3687
    Epoch 29/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.3233
    Epoch 30/50
    1/1 [==============================] - 0s 2ms/step - loss: 1.2826
    Epoch 31/50
    1/1 [==============================] - 0s 2ms/step - loss: 1.2456
    Epoch 32/50
    1/1 [==============================] - 0s 2ms/step - loss: 1.2117
    Epoch 33/50
    1/1 [==============================] - 0s 871us/step - loss: 1.1802
    Epoch 34/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.1508
    Epoch 35/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.1231
    Epoch 36/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.0968
    Epoch 37/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.0717
    Epoch 38/50
    1/1 [==============================] - 0s 1ms/step - loss: 1.0477
    Epoch 39/50
    1/1 [==============================] - 0s 912us/step - loss: 1.0247
    Epoch 40/50
    1/1 [==============================] - 0s 892us/step - loss: 1.0024
    Epoch 41/50
    1/1 [==============================] - 0s 666us/step - loss: 0.9808
    Epoch 42/50
    1/1 [==============================] - 0s 2ms/step - loss: 0.9599
    Epoch 43/50
    1/1 [==============================] - 0s 1ms/step - loss: 0.9396
    Epoch 44/50
    1/1 [==============================] - 0s 1ms/step - loss: 0.9198
    Epoch 45/50
    1/1 [==============================] - 0s 1ms/step - loss: 0.9006
    Epoch 46/50
    1/1 [==============================] - 0s 984us/step - loss: 0.8818
    Epoch 47/50
    1/1 [==============================] - 0s 1ms/step - loss: 0.8634
    Epoch 48/50
    1/1 [==============================] - 0s 932us/step - loss: 0.8455
    Epoch 49/50
    1/1 [==============================] - 0s 1ms/step - loss: 0.8280
    Epoch 50/50
    1/1 [==============================] - 0s 642us/step - loss: 0.8109
    [[16.2952]]

