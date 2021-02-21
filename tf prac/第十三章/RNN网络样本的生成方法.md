```python
# 模拟神经网络输入数据的生成


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```


```python
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
```


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
```

    2.2.0



```python
# 生成序列数据
dataset = tf.data.Dataset.range(10)
for val in dataset:
   print(val.numpy())
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9



```python
# 获得窗口数据，窗口大小为5
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()
```

    0 1 2 3 4 
    1 2 3 4 5 
    2 3 4 5 6 
    3 4 5 6 7 
    4 5 6 7 8 
    5 6 7 8 9 
    6 7 8 9 
    7 8 9 
    8 9 
    9 



```python
# 去掉不完整的数据
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()
```

    0 1 2 3 4 
    1 2 3 4 5 
    2 3 4 5 6 
    3 4 5 6 7 
    4 5 6 7 8 
    5 6 7 8 9 


不满足长度的元素是通过补充前面的元素得到的。

例：\[6 7 8 9\] --\> \[5 6 7 8 9\]


```python
# 转为numpy列表
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
  print(window.numpy())

```

    [0 1 2 3 4]
    [1 2 3 4 5]
    [2 3 4 5 6]
    [3 4 5 6 7]
    [4 5 6 7 8]
    [5 6 7 8 9]


lambda arg1,arg2,arg3… :\<表达式\>

arg1/arg2/arg3为函数的参数（**函数输入**），表达式相当于函数体，运算结果是表达式的运算结果。


```python
# 打散数据
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
  print(x.numpy(), y.numpy())
```

    [0 1 2 3] [4]
    [1 2 3 4] [5]
    [2 3 4 5] [6]
    [3 4 5 6] [7]
    [4 5 6 7] [8]
    [5 6 7 8] [9]



```python
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
for x,y in dataset:
  print(x.numpy(), y.numpy())

```

    [1 2 3 4] [5]
    [5 6 7 8] [9]
    [2 3 4 5] [6]
    [0 1 2 3] [4]
    [4 5 6 7] [8]
    [3 4 5 6] [7]


shuffle方法的参数buffer_size决定了有多少元素参与其中。所以该参数一般大于等于数据集的大小。


```python
# 设置数据批量，每两个数据为一批次
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)

for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())

```

    x =  [[5 6 7 8]
     [4 5 6 7]]
    y =  [[9]
     [8]]
    x =  [[1 2 3 4]
     [0 1 2 3]]
    y =  [[5]
     [4]]
    x =  [[2 3 4 5]
     [3 4 5 6]]
    y =  [[6]
     [7]]


batch是决定每一小批大小的，prefetch在这里确定了每一批中预处理的元素数目。
