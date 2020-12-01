## 100 numpy exercises( 1-25 )

1. print configuration


```python
import numpy as np
```


```python
np.show_config()
```

    blas_mkl_info:
      NOT AVAILABLE
    blis_info:
      NOT AVAILABLE
    openblas_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None)]
    blas_opt_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None)]
    lapack_mkl_info:
      NOT AVAILABLE
    openblas_lapack_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None)]
    lapack_opt_info:
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
        language = c
        define_macros = [('HAVE_CBLAS', None)]


2. find memory size of any array
字符串后的‘%’：标记转换说明符的开始


```python
z = np.zeros((10,10))
print("%d bytes" % (z.size * z.itemsize))
```

    800 bytes


3. get information of a function
in command line: `run ``python -c "import numpy; numpy.info(numpy.add)"```


4. reverse a verctor


```python
z = np.arange(1,10)
z = z[::-1]
print(z)
```

    [9 8 7 6 5 4 3 2 1]


5. np.reshape
newshape的一个形状维度可以是-1，在这种情况下根据数组长度和其余维度推断该值


6. np.nonzero
返回一个数组，提供其中所有非零元素的脚标(indice)


7. np.eye
生成单位矩阵


8. np.random.random(shape)/np.random.rand(d0,d1,...)
作用一致。在`[0,1)`的均匀分布中产生随机数


9. np.min(z) vs np.minimum(x1,x2)
前者返回最小值，后者返回两个数组各个元素较小的组成的新数组


10. 2d array,1 on border,0 inside


```python
z = np.ones((10,10))
z[1:-1,1:-1] = 0
print(z)
```

    [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]


11. np.pad(array, pad_width, constant_values, mode="'constant'")
为array添加边界，务必明确给出contant_values和mode


12. inf与NaN


```python
float("inf")   #正无穷
float("-inf")
```

inf * 0 = NaN;<br>
\> inf = NaN.<br>
NaN的exponent为可表达的最大值，significand部分不为0
inf的significand部分为0。

不要在 Python 中试图用 is 和 == 来判断一个对象是否是正负无穷或者 NaN。

13. np.diag(v,k)
v:vector,k:diaginal<br>
`np.diagonal`for extract diagonal


14. np.unravel_index(index,shape)
返回index（平面）在所给shape中的位置


15. np.tile(array,reps)
复制array reps次


16. np.dtype(\[(r,g,b)\])


17. real matrix product : `np.dot`


18. Given a 1D array, negate all elements which are between 3 and 8, in place. 


```python
# Author: Evgeni Burovski

Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)
```

    [ 0  1  2  3 -4 -5 -6 -7  8  9 10]


19. np.sum vs built-in sum


```python
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

    9
    10


第一行的sum是python内置的，第二个参数表示start value<br>
第三行的sum为numpy特有的，第二个参数只表示axis
<br>*negative*:sum from the last to the first axis;*default* = None

20. operator ** vs np.square

For most appliances, both will give you the same results. Generally the standard pythonic a\*a or a\*\*2 is faster than the numpy.square() or numpy.pow(), but the numpy functions are often more flexible and precise. If you do calculations that need to be very accurate, stick to numpy and probably even use other datatypes float96.

For normal usage a\*\*2 will do a good job and way faster job than numpy. 

21. 2 << z >> 2

array也可以做移位运算符的第二位


```python
z = np.arange(4,9)
print(z)
print(2<<z)
```

    [4 5 6 7 8]
    [ 32  64 128 256 512]



```python
print(2<<z>>2)
```

    [  8  16  32  64 128]


22. np.nan性质


```python
print(np.array(0) / np.array(0))   #浮点数除法
print(np.array(0) // np.array(0))  #返回整数部分
print(np.array([np.nan]).astype(int).astype(float))
```

    nan
    0
    [-9.22337204e+18]


    <ipython-input-6-2a10de0ee61b>:1: RuntimeWarning: invalid value encountered in true_divide
      print(np.array(0) / np.array(0))
    <ipython-input-6-2a10de0ee61b>:2: RuntimeWarning: divide by zero encountered in floor_divide
      print(np.array(0) // np.array(0))


23. np.copysign(x1,x2)
将x2的符号复制给x1

24. np.intersect1d(x1,x2)
返回两数组交集

25. np.random.randint(start,end,numbers)
生成随机整数数组
