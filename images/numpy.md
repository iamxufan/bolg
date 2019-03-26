---
title: Numpy初级
date: 2019-03-02 15:01:28
tags: python
thumbnail: /images/suolvetu/WechatIMG245.png
toc: true
---
### numpy


```python
import numpy as np

l = [[1,2,3],[2,3,4]]
# 列表转化成矩阵
array = np.array(l)
print(array)
```

    [[1 2 3]
     [2 3 4]]

<!--more-->
### numpy的几种属性


```python
# 维度
print('number of dim:',array.ndim)
# 行数和列数
print('shape :',array.shape)    
# 元素个数
print('size:',array.size)   
```

    number of dim: 2
    shape : (2, 3)
    size: 6


### 创建array


```python
a = np.array([2,3,4],dtype = np.int)
print(a,a.dtype)
```

    [2 3 4] int64



```python
# 创建全0数组
a = np.zeros((3,4),dtype = np.int16)
print(a)
```

    [[0 0 0 0]
     [0 0 0 0]
     [0 0 0 0]]



```python
# 创建全1数组
a = np.ones((3,4),dtype = np.int64)
print(a)
```

    [[1 1 1 1]
     [1 1 1 1]
     [1 1 1 1]]



```python
# 创建全空数组
a = np.empty((3,4),dtype = np.float)
print(a)
```

    [[-1.49166815e-154 -1.49166815e-154  4.27255699e+180  6.12033286e+257]
     [ 3.83819517e+151  9.77368093e+165  1.03927302e-042  5.24049485e+174]
     [ 4.27796595e-033  5.81088333e+294 -1.49166815e-154  8.38743761e-309]]



```python
# 创建连续数组
a = np.arange(1,13).reshape(3,4)
print(a)
```

    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]



```python
# 生成线段
a = np.linspace(1,10,6).reshape(2,3)
print(a)
```

    [[ 1.   2.8  4.6]
     [ 6.4  8.2 10. ]]


## 基础运算

### 基本运算1

#### 数组运算


```python
a = np.array([10,20,30,40])
b = np.arange(4)
```


```python
a-b
```




    array([10, 19, 28, 37])




```python
b**2
```




    array([0, 1, 4, 9])




```python
np.sin(a)
```




    array([-0.54402111,  0.91294525, -0.98803162,  0.74511316])




```python
b<3
```




    array([ True,  True,  True, False])



#### 矩阵运算


```python
a = np.array([[1,2],[3,4]])
b = np.arange(4).reshape(2,2)
print(a)
print(b)
```

    [[1 2]
     [3 4]]
    [[0 1]
     [2 3]]



```python
# 对应元素相乘
print(a*b)
# 矩阵乘法
print(np.dot(a,b))
```

    [[ 0  2]
     [ 6 12]]
    [[ 4  7]
     [ 8 15]]



```python
a = np.random.random((2,4))
print(a)
# axis=1时每行分别计算，axis=0时每列分别计算
print('sum =',np.sum(a,axis=1))
print('min =',np.min(a,axis=0))
print('max =',np.max(a))
```

    [[0.31357817 0.09926399 0.57284534 0.9692283 ]
     [0.86206853 0.94729865 0.80886452 0.01849844]]
    sum = [1.9549158  2.63673014]
    min = [0.31357817 0.09926399 0.57284534 0.01849844]
    max = 0.9692282997410551


### 基本运算2


```python
A = np.arange(2,14).reshape(3,4)
print(A)
```

    [[ 2  3  4  5]
     [ 6  7  8  9]
     [10 11 12 13]]



```python
# 最小（大）值索引
A.argmax()
```




    11




```python
# 平均值
print(A.mean())
# 对列求平均
print(A.mean(axis=0))
```

    7.5
    [6. 7. 8. 9.]



```python
# 中位数
np.median(A)
```




    7.5




```python
# 累加
print(np.cumsum(A))
```

    [ 2  5  9 14 20 27 35 44 54 65 77 90]



```python
# 累差
print(np.diff(A))
```

    [[1 1 1]
     [1 1 1]
     [1 1 1]]



```python
# 排序
print(np.sort(A))
```

    [[ 2  3  4  5]
     [ 6  7  8  9]
     [10 11 12 13]]



```python
# 转置
print(np.transpose(A))
```

    [[ 2  6 10]
     [ 3  7 11]
     [ 4  8 12]
     [ 5  9 13]]



```python
# A*AT
print(np.dot(A,A.T))
```

    [[ 54 110 166]
     [110 230 350]
     [166 350 534]]



```python
# 滤波
print(np.clip(A,5,9))
```

    [[5 5 5 5]
     [6 7 8 9]
     [9 9 9 9]]



```python
# 铺平
print(A.flatten())
```

    [ 2  3  4  5  6  7  8  9 10 11 12 13]


## numpy索引


```python
A = np.arange(3,15).reshape(3,4)
print(A)
```

    [[ 3  4  5  6]
     [ 7  8  9 10]
     [11 12 13 14]]



```python
A[1,1]
#A[1][1]
```




    8



## array合并


```python
A = np.array([1,1,1])
B = np.array([2,2,2])

np.vstack((A,B))
```




    array([[1, 1, 1],
           [2, 2, 2]])




```python
np.hstack((A,B))
```




    array([1, 1, 1, 2, 2, 2])



## array分割


```python
A = np.arange(12).reshape(3,4)
print(A)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]



```python
np.split(A,3)
# 横向分割
# np.vsplit(A,3)
# 纵向分割
# np.hsplit(A,2)
```




    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]




```python
# 不等量分割
np.array_split(A,3,axis=1)
```




    [array([[0, 1],
            [4, 5],
            [8, 9]]), array([[ 2],
            [ 6],
            [10]]), array([[ 3],
            [ 7],
            [11]])]



## copy


```python
a = np.array([1,2])
# b.copy()没有关联性，为浅拷贝
a is b.copy()
```




    False



