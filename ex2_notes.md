### numpy.append(arr,values,axis = None)
1. values可以为矩阵
2. 当axis存在值的时候，arr和values的维度必须一致
3. 数维度的方法：看数组最开始有几个`[]`
4. axis = 0表示两个数组上下排列（各占一行）

### numpy.concatenate((a1, a2, ...), axis=0)
1. 所有的array要用圆括号括起来
2. axis默认值是0

### numpy.hstack() / numpy.vstack()
两者类似，前者将数组横向并列，后者纵向

### q1运行感想
证明lstsq方法对input非常敏感；由于使用了plt.show(),窗口重叠在一起，不关闭显示的图像就无法运行其余内容

实例：

```
import matplotlib.pyplot as plt
import numpy as np

axis_x = np.array([-8, -7, -6, -5, -4, -3, -2, -1])
axis_y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
fig1 = plt.figure(1)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(axis_x, axis_y)
plt.draw()
plt.pause(4)# 间隔的秒数： 4s
plt.close(fig1)
fig2 = plt.figure(2)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(axis_y, axis_x)
plt.draw()
plt.pause(6)# 间隔的秒数：6s
plt.close(fig1)
```
来源： <https://blog.csdn.net/qq_36248632/article/details/90321044>

### numpy.reshape((-1,))
当参数为-1时，函数会根据其余维度判断array的形状，此处会将array转换为行向量
```
import numpy as np
import matplotlib as py

a = [[1,2,3],[4,5,6]]
a = np.array(a)
print(a.reshape((-1,3)))   #a = [[1 2 3]
                           #    [4 5 6]]
print(a.reshape((-1,)))    #a = [1 2 3 4 5 6]
```

### numpy.where(condition, x, y)
```
sv = np.where(alpha > 1e-6, True, False)
```

### numpy.any(a, axis)
Test whether any array element along a given axis evaluates to True.

任意元素为true返回true

### numpy中*与dot区别
前者可以做到各项元素相乘，后者按照矩阵乘法法则，需要注意维度必须匹配

### Q2 notes
1. cvxopt.solver中的参数必须是cvxopt.matrix()
2. 准备运算过程中可以用np.double()或np.astype(np.double)来将矩阵元素改变成double类型
3. 将a_n转化为sv时注意double类型与0判断大小时需要一个范围
4. cvxopt.solver返回值的'x'表示解，但是格式仍为cvxopt.matrix，需要先用np.array转化再reshape
5. b,c两问output图片中的颜色是由apply.py来实现的
6. d中对alpha值判定的阈值很重要：采用1e-6(similar to a)的准确率比1e-8低，但阈值过小(1e-9)会导致准确率大幅下降