# Python 复习课笔记 (Python Review Session)
*CS224N | 斯坦福大学 (Stanford University)*

---

## 目录
1. [为什么选择 Python？](#1-为什么选择-python)
2. [环境搭建](#2-环境搭建)
3. [语言基础](#3-语言基础)
4. [NumPy 简介](#4-numpy-简介)
5. [实用 Python 技巧](#5-实用-python-技巧)
6. [其他优质参考资料](#6-其他-import-参考资料)

---

## 1. 为什么选择 Python？

* Python 是一种被广泛使用的通用编程语言。
* 易于上手，学习曲线平缓。
* 科学计算功能强大，类似于 Matlab 和 Octave。
* 被主流深度学习框架（如 PyTorch 和 TensorFlow）广泛采用。

---

## 2. 环境搭建

### 环境管理

* **面临的问题**
  * 存在不同版本的 Python。
  * 存在无数的 Python 包及其复杂的依赖关系。
  * 不同的项目需要使用不同的包。
  * 更糟糕的是，不同项目可能需要同一包的不同版本！

* **解决方案**
  * 保持多个相互隔离的 Python 环境。
  * 每个环境：
    * 可以使用不同版本的 Python。
    * 拥有自己独立的一套包（可以指定包的具体版本）。
    * 可以被轻松复制和移植。

### Anaconda

* **介绍**
  * Anaconda 是一个非常流行的 Python 环境和包管理器。
  * 可以从 [Anaconda 官方网站](https://www.anaconda.com/download/) 下载。
  * 支持 Windows、Linux 和 macOS 系统。
  * 可以创建并管理不同的隔离环境（可以指定特定的 Python 版本或从环境配置文件创建）。

* **基础工作流**
  * **创建新环境**
    ```bash
    $ conda create -n <environment_name>
    $ conda create -n <environment_name> python=3.7
    $ conda env create -f <environment.yml>
    ```
  * **激活与停用环境**
    ```bash
    $ conda activate <environment_name>
    # ...在此环境中进行操作...
    $ conda deactivate
    ```
  * **导出环境**
    ```bash
    $ conda activate <environment_name>
    $ conda env export > environment.yml
    ```

### IDE 与文本编辑器

* 常见的 Python IDE / 文本编辑器：
  * PyCharm
  * Visual Studio Code (VS Code)
  * Sublime Text
  * Atom
  * Vim (适用于 Linux 终端)
* 在你选择的 IDE 或文本编辑器中编写 Python 程序。
* 在终端中激活相应的 conda 环境，并运行以下命令执行程序：
  ```bash
  $ python <filename.py>
  ```

### Jupyter Notebook / Google Colab

* **Jupyter Notebook**
  * 允许你在网页浏览器中本地编写和执行 Python 代码。
  * 交互性强，支持代码重复执行、保存运行结果，并可交替插入文本、公式和图像。
  * 可以将不同的 conda 环境添加到 Jupyter Notebook 中。
* **Google Colab**
  * 访问地址：[https://colab.research.google.com/](https://colab.research.google.com/)
  * 谷歌托管的 Jupyter Notebook 服务，完全在云端运行，无需任何本地配置。
  * 提供免费的计算资源，包括 GPU。
  * 预装了许多常用的 Python 库。

---

## 3. 语言基础

### 常用操作

```python
x = 10
y = 3

# 声明两个整型变量
# 注释以井号 (#) 开头

# 算术运算：加法
x + y  # >> 13

# 幂运算 (求幂)
x ** y  # >> 1000

# 两个整数相除
x / y  # >> 3 (注意：在 Python 3 中整数相除会得到浮点数 3.333...)

# 类型转换进行浮点数除法
x / float(y)  # >> 3.333...

# 将整数转换为字符串并进行字符串拼接
str(x) + "+" + str(y)  # >> "10+3"
```

### 内置特殊值

```python
# 布尔值
True, False

# 表示空或“无”
None

# 可以将变量赋值为 None
x = None

# 列表可以包含 None
array = [1, 2, None]

# 函数可以返回 None
def func():
    return None
```

### 逻辑运算符

* 逻辑运算符：`and`, `or`, `not`
* Python 中的布尔逻辑运算符是用英文单词书写的，而不是像 C++ 等语言中使用 `&&`, `||`, `!`。

```python
# 比较运算符 == 和 != 用于检查相等或不等性，返回布尔值 (True/False)
if [] != [None]:
    print("Not equal")
```

### 缩进代替大括号

* 在 Python 中，代码块是通过**缩进**来划分的，而不是像 C++ 等语言中使用花括号 `{}`。
* 缩进可以是 2 个空格或 4 个空格，但在整个文件中必须保持一致。
* 如果使用 Vim，请在你的 `.vimrc` 中配置一致的缩进格式。

```python
def sign(num):
    # 缩进层级 1：函数体
    if num == 0:
        # 缩进层级 2：if 语句体
        print("Zero")
    elif num > 0:
        # 缩进层级 2：elif (else if) 语句体
        print("Positive")
    else:
        # 缩进层级 2：else 语句体
        print("Negative")
```

### 类型系统与执行机制

* Python 是一种**强类型 (strongly-typed)** 且 **动态类型 (dynamically-typed)** 的语言。
  * **强类型**：解释器总是尊重每个变量的类型，不会隐式地强制转换类型（例如 `1 + '1'` 会抛出错误 `TypeError`）。
  * **动态类型**：变量仅仅是一个绑定到某个名称的值，变量可以被重新赋值为不同类型的值（例如：先执行 `foo = [1, 2, 3]`，随后可以执行 `foo = 'hello!'`）。
* **执行机制**：Python 首先被解释为字节码 (`.pyc`)，然后由虚拟机 (VM) 实现（最常用的是用 C 语言编写的 CPython）编译成机器指令。
* **对科学计算的影响**：虽然纯 Python 运行速度较慢，但它可以运行高度优化的 C/C++ 底层子程序，这使得科学计算（例如矩阵乘法）运行极其快速（如 `np.dot(x, W) + b` 非常高效）。

### 集合：列表 (List)

列表是可变的数组（类似于 C++ 中的 `std::vector`）。

```python
names = ['Zach', 'Jay'] 

# 通过索引访问元素
names[0] == 'Zach' 

# 在列表末尾添加元素
names.append('Richard') 
print(len(names) == 3)  # >> True
print(names)            # >> ['Zach', 'Jay', 'Richard'] 

# 拼接列表
names += ['Abi', 'Kevin']
print(names)            # >> ['Zach', 'Jay', 'Richard', 'Abi', 'Kevin'] 

# 创建空列表的两种方式
names = [] 
names = list() 

# 列表内可以混合不同的数据类型
stuff = [1, ['hi', 'bye'], -0.12, None]
```

### 列表切片 (List Slicing)

可以非常便捷地访问列表的子集。
基本格式：`some_list[start_index:end_index]` （左闭右开，即包含 `start_index`，但不包含 `end_index`）。

```python
numbers = [0, 1, 2, 3, 4, 5, 6] 

# 从起始索引(含)到结束索引(不含)
numbers[0:3] == numbers[:3] == [0, 1, 2] 

# 从指定索引到列表末尾
numbers[5:] == numbers[5:7] == [5, 6] 

# 选择整个列表
numbers[:] == numbers == [0, 1, 2, 3, 4, 5, 6] 

# 负数索引表示从列表末尾倒序环绕
numbers[-1] == 6 
numbers[-3:] == [4, 5, 6] 

# 混合正负数索引
numbers[3:-2] == [3, 4]
```

### 集合：元组 (Tuple)

元组是不可变的数组。一旦创建，无法修改。

```python
names = ('Zach', 'Jay')  # 注意使用圆括号

# 访问元素和获取长度的语法与列表相同
names[0] == 'Zach' 
print(len(names) == 2)  # >> True
print(names)            # >> ('Zach', 'Jay') 

# 尝试修改元组元素会抛出类型错误
names[0] = 'Richard'    # >> TypeError: 'tuple' object does not support item assignment

# 创建空元组
empty = tuple() 

# 创建单个元素的元组，末尾的逗号是必需的！
single = (10,) 
```

### 集合：字典 (Dictionary)

字典是哈希映射 (Hash Map)。

```python
# 创建空字典的两种方式
phonebook = {} 
phonebook = dict() 

# 创建带有一个初始元素的字典
phonebook = {'Zach': '12-37'} 

# 添加一个新元素
phonebook['Jay'] = '34-23' 

# 检查字典中是否存在某个键
print('Zach' in phonebook)   # >> True 
print('Kevin' in phonebook)  # >> False 

# 根据键获取对应的值
print(phonebook['Jay'])      # >> '34-23'

# 删除键值对
del phonebook['Zach'] 
print(phonebook)             # >> {'Jay' : '34-23'}
```

### 循环 (Loops)

* **使用 `range()` 控制循环次数**
  相比于 C++ 中的 `for (i=0; i<10; i++)`，Python 中常用 `range()` 函数：
  ```python
  for i in range(10):
      print(i)
  # >> 输出 0 到 9 (不包含 10)
  ```

* **遍历列表**
  ```python
  names = ['Zach', 'Jay', 'Richard']
  for name in names: 
      print('Hi ' + name + '!')
  # >> 输出:
  # Hi Zach!
  # Hi Jay!
  # Hi Richard!
  ```

* **遍历索引与值**
  ```python
  # 方法一：利用索引
  for i in range(len(names)):
      print(i, names[i])
      
  # 方法二：利用 enumerate() 函数 (推荐)
  for i, name in enumerate(names):
      print(i, name)
  # >> 输出:
  # 0 Zach
  # 1 Jay
  # 2 Richard
  ```

* **遍历字典**
  ```python
  phonebook = {'Zach': '12-37', 'Jay': '34-23'}

  # 遍历键
  for name in phonebook:
      print(name)

  # 遍历值
  for number in phonebook.values():
      print(number)

  # 同时遍历键和值
  for name, number in phonebook.items():
      print(name, number)
  ```
  *注：字典遍历的顺序是否能得到保证，取决于 Python 的具体版本（从 Python 3.6 起，默认按插入顺序排序）。*

### 类 (Classes)

```python
class Animal(object): 
    # 构造函数，使用 `a = Animal('human', 10)` 调用
    def __init__(self, species, age):
        self.species = species 
        self.age = age 
        
    # 实例方法，使用 `a.is_person()` 调用
    # 使用 `self` 关键字指代实例本身
    def is_person(self):
        return self.species == 'human'
        
    def age_one_year(self):
        self.age += 1 

# 继承 Animal 的属性和方法
class Dog(Animal): 
    # 重写父类方法以计算“狗龄”
    def age_one_year(self): 
        self.age += 7
```

### PyTorch 模型类

在以后的课程作业中，你将看到并编写 PyTorch 中的模型类，它们继承自 `torch.nn.Module`（这是所有神经网络模块的基类）。

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 在此初始化各层
        
    def forward(self, x):
        # 在此定义前向传播计算逻辑
        return x
```

### 安装第三方包

* `pip` 专门用于安装 Python 包，而 `conda` 可以安装任何语言编写的包/软件。
* 将 `pip` 和 `conda` 混合使用时可能会引发依赖冲突。**最佳实践是：优先使用 `conda` 安装尽可能多的包，随后再使用 `pip` 安装剩余的包**。
* **常用安装命令**：
  ```bash
  # 在指定环境下安装包
  conda install -n myenv [package_name][=可选版本号]
  
  # 如果 conda 没有此包，可在激活的环境中用 pip 安装
  conda install -n myenv pip      # 在环境中安装 pip
  conda activate myenv            # 激活环境
  pip install [package_name][==可选版本号]
  
  # 从依赖文件批量安装包
  pip install -r <requirements.txt>
  ```

### 导入模块

```python
# 导入内置模块 'os' 和 'time'
import os, time

# 导入第三方包并设置别名 (alias)
import numpy as np
np.dot(x, y)  # 通过 pkg.fn 访问相应的方法

# 导入特定的子模块或函数
from numpy import linalg as la, dot as matrix_multiply
# 注意：直接导入可能会引发命名空间冲突
```

---

## 4. NumPy 简介

* **简介**
  * NumPy 是专为矩阵和向量运算设计的高性能库。
  * 利用底层 C/C++ 子程序和内存高效的数据结构，使得数值计算可以高度向量化。
  * 核心数据类型为 `np.ndarray`（注意：其构造函数为 `np.array()`）。

### `np.ndarray` 的形状

```python
x = np.array([1, 2, 3])        # 一维向量 (1-D vector)
y = np.array([[3, 4, 5]])      # 行向量 (Row vector)
z = np.array([[6, 7], [8, 9]]) # 二维矩阵 (Matrix)

print(x.shape)  # >> (3,)
print(y.shape)  # >> (1, 3)
print(z.shape)  # >> (2, 2)
```
> [!IMPORTANT]
> 形状 `(N,)` 与 `(1, N)` 以及 `(N, 1)` 在 NumPy 中并不等价，可能会导致不同的计算结果。

### 规约操作 (Reductions)

* 常见的规约操作包括：`np.max`, `np.min`, `np.amax`, `np.sum`, `np.mean` 等。
* 规约操作默认是在**指定的轴 (axis)** 上进行的（如果未指定 axis，则会对整个数组的所有元素进行规约，最终得到一个标量）。
* 你可以把规约理解为沿着某个维度对数据进行“折叠”或“压缩”。

```python
# 形状为 (3, 2) 的矩阵
x = np.array([[1, 2], [3, 4], [5, 6]])

# 沿着 axis=1 (行方向) 求最大值，结果形状为 (3,)
print(np.max(x, axis=1))  # >> [2 4 6]

# 沿着 axis=1 求最大值，并保持原有维度，结果形状为 (3, 1)
print(np.max(x, axis=1, keepdims=True))  # >> [[2] [4] [6]]
```

### 矩阵运算 (Matrix Operations)

* **常见算术运算符**（`+`, `-`, `*`, `**`, `/`）在 NumPy 中都是**按元素 (element-wise)** 进行运算的。
* **按元素乘积 (Hadamard 积)**：
  ```python
  A * B
  ```
* **点积 (Dot product) 与矩阵-向量乘积**：
  ```python
  np.dot(u, v)
  np.dot(x, W)
  ```
* **矩阵乘法**：
  ```python
  np.matmul(A, B)  # 或 A @ B
  ```
  *注：虽然 `np.dot()` 也可以用于矩阵乘积，但对于两个二维数组相乘，更推荐使用 `np.matmul()` 或 `@`。*
* **转置**：
  ```python
  x.T
  ```
* `SciPy` 和 `np.linalg` 提供了更多高级矩阵计算方法。

### 索引与切片 (Indexing & Slicing)

```python
x = np.random.random((3, 4))  # 随机生成一个形状为 (3, 4) 的矩阵

x[:]                   # 选择整个矩阵 x
x[np.array([0, 2]), :] # 选择第 0 行和第 2 行
x[1, 1:3]              # 选择第 1 行的第 1 到第 2 列元素 (降维成一维向量)
x[x > 0.5]             # 布尔索引：选择所有大于 0.5 的元素
x[:, :, np.newaxis]    # 在最后一个维度新增一个轴，形状变为 (3, 4, 1)
```
*注：当使用 ndarray 或范围 (Range) 对数组进行索引时，会保留被选择部分的维度。*

### 广播机制 (Broadcasting)

广播机制描述了 NumPy 在进行算术运算时，如何处理形状不同但兼容的数组。

#### 通用广播规则

在对两个数组进行运算时，NumPy 会从它们的末尾维度（即最右侧的维度）开始，逐个向左比较它们的形状。两个维度在以下情况下是兼容的：
1. **它们相等**
2. **其中一个维度的大小为 1**（此时该维度会沿着该方向重复/拉伸，以匹配另一个数组的形状）

#### 广播可视化示例

```python
x = np.random.random((3, 4))  # 形状为 (3, 4) 的矩阵
y = np.random.random((3, 1))  # 形状为 (3, 1) 的行方向的向量
z = np.random.random((1, 4))  # 形状为 (1, 4) 的列方向的向量

# y 会在第二个维度上进行广播，以匹配 x 的列数
s = x + y

# z 会在第一个维度上进行广播，以匹配 x 的行数
p = x * z
```

* **`x + y` 的广播过程**
  ```text
  [ 1  2  3  4 ]     [ 1 ]            [ 1  1  1  1 ]     [  2  3  4  5 ]
  [ 5  6  7  8 ]  +  [ 2 ]   等价于 +  [ 2  2  2  2 ]  =  [  7  8  9 10 ]
  [ 9 10 11 12 ]     [ 3 ]            [ 3  3  3  3 ]     [ 12 13 14 15 ]
  ```

* **`x * z` 的广播过程**
  ```text
  [ 1  2  3  4 ]                    [ 1  2  3  4 ]     [ 1   4   9  16 ]
  [ 5  6  7  8 ]  *  [ 1  2  3  4 ] 等价于 * [ 1  2  3  4 ]  =  [ 5  12  21  32 ]
  [ 9 10 11 12 ]                    [ 1  2  3  4 ]     [ 9  30  33  48 ]
  ```

#### 广播思考题

假设有以下数组：
```python
a = np.random.random((3, 4)) # (3, 4) 矩阵
b = np.random.random((3, 1)) # (3, 1) 向量
c = np.random.random((3, ))  # (3, ) 一维向量
```
以下运算的运行结果和形状是什么？
* `b + b.T`
* `a + c`
* `b + c`

### 编写高效的 NumPy 代码

> [!WARNING]
> 在处理 NumPy 数组时，**务必极力避免**对索引或轴使用显式的 `for` 循环。
> 显式的 `for` 循环会导致代码运行速度变慢约 10 到 100 倍。

```python
# 慢速写法
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x[i, j] **= 2

# 高效的向量化写法
x **= 2
```

---

## 5. 实用 Python 技巧

### 列表推导式 (List Comprehensions)

* 类似于函数式编程语言中的 `map()`。
* 能够极大地提高代码的可读性，并使代码更加简洁。
* 基本格式：`[func(x) for x in some_list]`

```python
# 传统的循环写法
squares = [] 
for i in range(10): 
    squares.append(i**2)

# 使用列表推导式
squares = [i**2 for i in range(10)]

# 支持带条件的过滤
odds = [i**2 for i in range(10) if i % 2 == 1]
```

### 便利语法

* **多重赋值与迭代器解包**
  ```python
  age, name, pets = 20, 'Joy', ['cat']
  x, y, z = ('Tensorflow', 'PyTorch', 'Chainer')
  ```
* **从函数返回多个值**
  ```python
  def some_func(): 
      return 10, 1 
      
  ten, one = some_func()
  ```
* **使用特定的分隔符拼接字符串列表**
  ```python
  ", ".join(['1', '2', '3'])  # >> '1, 2, 3'
  ```
  *(注：如果列表中是数字，可以配合 map 使用：`", ".join(map(str, [1, 2, 3]))`)*
* **包含不同引号的字符串**
  ```python
  message = 'I like "single" quotes.'
  reply = "I prefer 'double' quotes."
  ```

### 调试技巧

Python 拥有极具交互性的 Shell，你可以直接在其中测试任意代码。
* 非常适合代替图形计算器（不用担心整数溢出！）。
* 可以导入任何模块（包括当前工作目录下的自定义文件）。
* 当你不确定某些语法（如矩阵操作）时，可以在其中编写小型测试用例进行验证。

```bash
$ python
Python 3.9.7 (default, Sep 16 2021, 08:50:36) 
[Clang 10.0.0 ] :: Anaconda, Inc. on darwin
>>> import numpy as np
>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([[3, 3], [3, 3]])
>>> A * B
array([[ 3,  6],
       [ 9, 12]])
>>> np.matmul(A, B)
array([[ 9,  9],
       [21, 21]])
```

### 调试工具与手段

| 工具 / 代码 | 作用描述 |
| --- | --- |
| `array.shape` | 查看 NumPy 数组的各个维度大小 |
| `array.dtype` | 检查数组的数据类型 (如 float32、int64 等，以排查精度异常) |
| `type(stuff)` | 查看变量所属的 Python 类型 |
| `import pdb; pdb.set_trace()` | 在当前位置设置调试断点 |
| `print(f'My name is {name}')` | 使用 f-string 快捷地格式化并打印字符串 |

### 常见错误与排查

* **ValueError**: 通常是由于广播机制或矩阵乘法中**维度不匹配**引起的。
* **排查建议**：当遇到此类报错，最直接的第一步就是打印相关数组的 `.shape`，核对它们的实际维度是否与你的预期相符。

---

## 6. 其他优质参考资料

* Python 3 官方文档: [https://docs.python.org/3/](https://docs.python.org/3/)
* Anaconda 官方用户指南: [https://docs.conda.io/projects/conda/en/latest/user-guide/index.html](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
* NumPy 官方文档: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
* 斯坦福 CS231n 课程 Python/NumPy 教程: [https://cs231n.github.io/python-numpy-tutorial/](https://cs231n.github.io/python-numpy-tutorial/)
* 斯坦福 Python 专项课程 (CS41): [https://stanfordpython.com/#/](https://stanfordpython.com/#/)
