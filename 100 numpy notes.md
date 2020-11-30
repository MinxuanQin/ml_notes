## 100 numpy exercises

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
