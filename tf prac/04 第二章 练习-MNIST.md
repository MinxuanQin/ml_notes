## Exercise 2
In the course you learned how to do classification using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:
1. It should succeed in less than 10 epochs, so it is okay to change epochs to 10, but nothing larger
2. When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
3. If you add any additional variables, make sure you use the same names as the ones used in the class

I've started the code for you below -- how would you finish it?

## 练习2
在课程中，已经学习了如何用Fashion MNIST数据集训练分类器。Fashion MNIST是一个包含服装项目的数据集。还有另一个类似的数据集叫MNIST，它包含很多手写数字（0到9）的图片和标签。

编写一个MNIST分类器，让它可以训练到99%以上的准确率，并且在达到这个准确率时，就通过回调函数停止训练。

一些注意事项。
1. 它应该在小于10个epochs的情况下成功，所以把epochs改成10个也可以，但不能大过10个。
2. 当它达到99%以上时，应该打印出 "达到99%准确率，所以取消训练！"的字符串。
3. 如果你添加了任何额外的变量，请确保你使用与类中使用的变量相同的名称。

我已经为你准备了下面的代码--请完成这个程序。


```python
%config IPCompleter.greedy = True
```


```python
# YOUR CODE SHOULD START HERE
# YOUR CODE SHOULD END HERE
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# YOUR CODE SHOULD START HERE
#callback function

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.94):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

# YOUR CODE SHOULD END HERE

model = tf.keras.models.Sequential([
# YOUR CODE SHOULD START HERE
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    
# YOUR CODE SHOULD END HERE
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# YOUR CODE SHOULD START HERE
model.fit(x_train, y_train, epochs = 10, callbacks=[callbacks])
model.evaluate(x_test, y_test)
# YOUR CODE SHOULD END HERE
```

    Epoch 1/10
    1875/1875 [==============================] - 7s 3ms/step - loss: 2.6424 - accuracy: 0.9064
    Epoch 2/10
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.3473 - accuracy: 0.9361
    Epoch 3/10
    1860/1875 [============================>.] - ETA: 0s - loss: 0.2927 - accuracy: 0.9424
    Reached 99% accuracy so cancelling training!
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.2942 - accuracy: 0.9421
    313/313 [==============================] - 1s 3ms/step - loss: 0.3892 - accuracy: 0.9234





    [0.38918423652648926, 0.9233999848365784]



进行十次训练最高能达到95%左右的准确率。

错误点：没有normalize


```python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

    Epoch 1/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2014 - accuracy: 0.9404
    Epoch 2/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0810 - accuracy: 0.9749
    Epoch 3/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0535 - accuracy: 0.9835
    Epoch 4/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0382 - accuracy: 0.9879
    Epoch 5/10
    1864/1875 [============================>.] - ETA: 0s - loss: 0.0273 - accuracy: 0.9911
    Reached 99% accuracy so cancelling training!
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.0273 - accuracy: 0.9911





    <tensorflow.python.keras.callbacks.History at 0x7fec94fc0e48>


