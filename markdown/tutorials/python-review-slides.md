## 页面 1: Python 复习课

Python 复习课 (Python Review Session)
CS224N - 25 年冬季
斯坦福大学 (Stanford University)

## 页面 2: 关于 Python

* 标志是两条缠绕的蛇，基于玛雅人的图形表示。
* 然而，该语言实际上是以蒙提·派森的飞行马戏团 (Monty Python's Flying Circus) 命名的 😅

## 页面 3: 课程路线图

1. 为什么选择 Python？
2. 环境搭建
3. Python 基础
4. 数据结构
5. NumPy
6. 实用技巧

## 页面 4: 课程路线图（续）

*(学习目标顺序与路线展示)*

## 页面 5: 为什么选择 Python？

* 广泛使用的通用语言。
* 易于学习、阅读和编写。
* 具有类似于 Matlab 和 Octave 的科学计算功能。
* 被各大主流深度学习框架（PyTorch, TensorFlow）采用。
* 活跃的开源社区，拥有非常多的第三方库！

## 页面 6: Python 解释器

Python 代码 $\rightarrow$ 解释为字节码 (`.pyc`) $\rightarrow$ 由虚拟机 (VM) 实现（最常用的是以 C 语言实现）编译为机器指令。
* 虽然被称为“较慢”，但可以通过运行高度优化的 C/C++ 子例程来加速操作。
* **交互模式 (Interactive Mode)**（逐行运行）
* **脚本模式 (Script Mode)**（运行 `.py` 文件）

## 页面 7: 语言基础：强类型

**解释器始终遵循每种变量的类型（严格处理变量类型）**
* **强类型 (Strongly Typed)**
* 类型不会像 JavaScript 或 Perl 那样隐式地进行自动类型转换：
  * `1 + '1'` $\rightarrow$ 报错 (TypeError)！
  * 像 `float` 和 `int` 相加是由于有显式的内置实现而允许（非自动类型转换）。
  * `[1, 2] + set([3])` $\rightarrow$ 报错 (TypeError)！

## 页面 8: 语言基础：动态类型

* 变量仅仅是绑定到名称的值或对象的引用。
* 变量的数据类型在运行时决定（极其灵活！）。
* **动态类型 (Dynamically Typed)**
* 变量可以重新分配给不同类型的值：
```python
num = 1      # 此时是 int
num = "One"  # 重新赋值为 str ✅
```

* 动态参数和多种类型示例：
```python
def find(required_element, sequence):
   for index, element in enumerate(sequence):
       if element == required_element:
           return index
   return -1

print(find(2, [1, 2, 3]))  # 输出：1
print(find("c", ("a", "b", "c", "d")))  # 输出：2 ✅
```

## 页面 9: 快速自测 🥳

在 Python 中，以下代码的输出是什么？
```python
x = 5
y = "3"
print(x + y)
```
A. 8
B. "53"
C. TypeError
D. "53.0"

## 页面 10: 快速自测解答 🥳

正确答案为：**C. TypeError**
* 解释：由于 Python 是强类型语言，无法直接将整型 (int) 与字符串型 (str) 相加。

## 页面 11: 课程路线图

*(进入第二部分：环境搭建)*

## 页面 12: 排版说明

本课件中：
* 代码以 Courier New 字体显示。
* 命令行输入以 `$` 前缀表示。
* 输出以 `>>` 前缀表示。

## 页面 13: Python 安装

详情访问官方下载地址：[Python Downloads](https://www.python.org/downloads/) 🥳

## 页面 14: 常用命令

* **打印版本号**：
  ```bash
  $ python --version
  $ python -v
  $ python -vv
  ```
* **打印 Python 执行路径**：
  ```bash
  $ which python  # macOS, Linux
  $ where python  # Windows
  ```
* **查看已安装的库**：
  ```bash
  $ python -m pip list
  ```
* **以不同模式运行**：
  ```bash
  $ python script.py                  # 运行脚本
  $ python -i script.py               # 运行完脚本后保留在交互模式中
  $ python -c "print('hello there!')" # 运行单行代码片段
  ```
  * `pip` 是 Python 的包安装器。
  * `-m` 将一个模块（如 `pip`）作为脚本运行。

## 页面 15: 环境管理

**面临的问题**：
* 不同的 Python 版本。
* 数不清的 Python 包及其复杂的依赖关系。
* 不同的项目需要不同包 $\rightarrow$ 更糟糕的是，可能会需要同一个包的不同版本！

## 页面 16: 环境管理：虚拟环境

**解决方案：虚拟环境 (Virtual Envs)**
* 维护多个相互隔离的 Python 运行环境。
* 每个环境：
  * 可以使用不同的 Python 版本。
  * 拥有自己独立的一套安装包（可指定具体的版本号）。
  * 可以被轻易地复制/重现。

## 页面 17: 解决方案 1：venv

使用官方自带模块 `venv`：
```bash
$ python -m venv /path/to/new/virtual/env
```
* 在现有 Python 安装的基础上创建，通常称为虚拟环境的“基础 (base)” Python。
* 虚拟环境目录包含一个特定的 Python 解释器、库以及支持项目运行所需的二进制文件。
* 与其他虚拟环境中的软件以及操作系统中安装的全局解释器和库相互隔离。
* 创建新目录后，可根据不同操作系统进行激活 (activate)。

## 页面 18: 解决方案 2：Anaconda (或 Miniconda)

非常流行的 Python 环境和包管理器：
* 支持 Windows, Linux, macOS。
* 可创建并管理不同的隔离环境。
* 可选择特定的 Python 版本。
* 方便地从环境配置文件中导出/创建环境！

**基本工作流程**：
* **创建新环境**：
  ```bash
  $ conda create -n <environment_name>
  $ conda create -n <environment_name> python=3.7
  $ conda env create -f <environment.yml>
  ```
* **激活/退出环境**：
  ```bash
  $ conda activate <environment_name>
  $ conda deactivate
  ```
* **导出环境**：
  ```bash
  $ conda activate <environment_name>
  $ conda env export > environment.yml
  ```

## 页面 19: 安装包

* `pip` 仅安装 Python 包，而 `conda` 可以安装可能包含使用任何语言编写的软件的包。
* 🚨 **最佳实践**：建议先使用 `conda` 安装尽可能多的包，随后使用 `pip` 安装剩余的包。

* **使用 conda 安装**：
  ```bash
  $ conda install -n myenv [package_name][=可选版本号]
  ```
* **在 conda 环境中通过 pip 安装（当 conda 库中不可用时）**：
  ```bash
  $ conda install -n myenv pip  # 先在环境里安装 pip
  $ conda activate myenv        # 激活环境
  $ pip install [package_name][==可选版本号]   # 单独安装
  $ pip install -r <requirements.txt>       # 从依赖文件批量安装
  ```

## 页面 20: IDE / 文本编辑器

选择您喜爱的 IDE 或文本编辑器来编写 Python 程序 😁：
* PyCharm
* Visual Studio Code
* Sublime Text
* Atom
* Vim (用于 Linux 或 Mac)

在终端中，激活您的虚拟环境并执行：
```bash
$ python <filename.py>
```
* 提示：IDE 通常有非常实用的扩展插件（例如 VS Code）！

## 页面 21: Python Notebooks

* **Jupyter Notebook**：
  * `.ipynb` 文件 $\rightarrow$ 在本地浏览器中编写和运行 Python 代码。
  * 支持交互式运行、重新执行代码、结果保存，可混合编排文本、公式和图像。
  * 可添加 conda 虚拟环境。
  * 基于读取-求值-打印循环 (REPL)。
* **Google Colab**：
  * 云端托管的 Jupyter 笔记本，无需任何本地配置，提供免费的 GPU 算力。
  * 预装了许多常用的 Python 库。
  * 可与 Git 仓库、Google Drive 以及本地存储整合。

## 页面 22: 连线测验！

将左侧工具与右侧描述进行匹配：
1. `venv`
2. `Anaconda`
3. `Jupyter Ntbk`
4. `pip`

A. 用于安装和管理第三方库的 Python 包管理器。
B. 用于创建隔离的 Python 环境以进行依赖项管理的工具。
C. 简化包和环境管理的发行版，专为数据科学设计。
D. 交互式平台，可编写和运行代码，并伴有可视化和注释。

## 页面 23: 连线测验答案！

* **1 $\rightarrow$ B** (`venv` 对应创建隔离的环境)
* **2 $\rightarrow$ C** (`Anaconda` 对应数据科学环境发行版)
* **3 $\rightarrow$ D** (`Jupyter Ntbk` 对应交互式平台)
* **4 $\rightarrow$ A** (`pip` 对应包管理器)

## 页面 24: 课程路线图

*(进入第三部分：Python 基础)*

## 页面 25: 常见运算符与基本操作

```python
x = 10
y = 3

# 算术运算
x + y        >> 13

# 幂运算 (Exponentiation)
x ** y       >> 1000

# 整数除法 (Python 3 中默认是浮点数除法)
x / y        >> 3.3333333333333335

# 显式转换为 float 进行除法
x / float(y) >> 3.3333333333333335

# 强制转换并进行字符串拼接
str(x) + " + " + str(y) >> "10 + 3"
```

## 页面 26: 内置常量/特殊值

* **布尔值**：`True`, `False`
* **None**：表示缺失或不存在（类似 null/nil）
```python
x = None
array = [1, 2, None]  # 列表可包含 None

def func():
    return None       # 函数可返回 None
```

## 页面 27: 内置运算符

* 布尔运算符在 Python 中使用英文单词，而不是 C++ 中的 `&&`, `||`, `!`：
  * `and`, `or`, `not`
* 比较运算符 `==` 和 `!=` 检查是否相等/不等，返回布尔值：
```python
if [] != [None]:
    print("Not equal")
```

## 页面 28: 缩进与空格

* 在 Python 中，代码块是通过**缩进 (indents)**和换行来创建的，而不是像 C++ 那样使用大括号 `{}`。
* 缩进可以是 2 个或 4 个空格，但在整个文件中必须保持一致。
* 如果使用 Vim，可以在 `.vimrc` 中配置好缩进空格数以保持一致。

```python
def sign(num):
    # 缩进级别 1: 函数体
    if num == 0:
        # 缩进级别 2: if 语句体
        print("Zero")
    elif num > 0:
        # 缩进级别 2: elif 语句体
        print("Positive")
    else:
        # 缩进级别 2: else 语句体
        print("Negative")
```

## 页面 29: 🎯 找错赛场 (Debugging Derby)

找出以下代码中的所有错误：
```python
0length = 10
float width = 5.0
print "Beginning work..."
area = 0length * Width
if area > 20
    print("Area: " + area)
message = "Completed!'
```

## 页面 30: 🎯 找错赛场答案与分析

* 变量名不能以数字开头 (`0length`)。
* Python 中不需要也不允许显式的类型声明 (`float width`)。
* `print` 需要加括号 (`print(...)`)。
* 变量名大小写不一致 (`Width` 与 `width`)。
* `if` 条件后面遗漏了冒号 (`:`)。
* 字符串拼接时需要对数字 `area` 进行强制类型转换 (`str(area)`)。
* 字符串的单双引号不匹配 (`"Completed!'`)。

## 页面 31: 🎯 修复后的正确代码

```python
length = 10
width = 5.0
print("Beginning work...")
area = length * width
if area > 20:
    print("Area: " + str(area))
message = "Completed!"
```
修复完毕，大功告成！🥳

## 页面 32: 课程路线图

*(进入第四部分：数据结构)*

## 页面 33: 列表 (List)

列表是可变数组 (mutable arrays)，类似于 C++ 中的 `std::vector`。

```python
names = ['Zach', 'Jay']
names[0] == 'Zach'
names.append('Richard')
print(len(names) == 3) >> True
print(names) >> ['Zach', 'Jay', 'Richard']

names += ['Abi', 'Kevin']
print(names) >> ['Zach', 'Jay', 'Richard', 'Abi', 'Kevin']

names = []      # 创建空列表
names = list()  # 同样创建空列表

stuff = [1, ['hi', 'bye'], -0.12, None]  # 可以混用不同类型
```

## 页面 34: 列表切片 (List Slicing)

可以非常方便地访问列表元素。
基本格式：`some_list[start_index:end_index]`

```python
numbers = [0, 1, 2, 3, 4, 5, 6]
numbers[0:3] == numbers[:3] == [0, 1, 2]
numbers[5:] == numbers[5:7] == [5, 6]
numbers[:] == numbers == [0, 1, 2, 3, 4, 5, 6]

# 负索引表示从右向左环绕访问
numbers[-1] == 6 
numbers[-3:] == [4, 5, 6]
numbers[3:-2] == [3, 4]  # 可以混用正负索引
```

## 页面 35: 元组 (Tuples)

元组是**不可变数组 (immutable arrays)**。

```python
names = ('Zach', 'Jay')   # 注意使用圆括号
names[0] == 'Zach'
print(len(names) == 2) >> True
print(names) >> ('Zach', 'Jay')

# 元组不可变，因此以下操作会报错
names[0] = 'Richard' >> TypeError: 'tuple' object does not support item assignment

empty = tuple()  # 空元组
single = (10,)   # 单元素元组，末尾的逗号非常关键！

## 页面 36: 字典 (Dictionary)

字典即哈希映射 (hash maps)。

```python
phonebook = {}            # 空字典
phonebook = dict()        # 同样创建空字典
phonebook = {'Zach': '12-37'} # 包含一个键值对的字典

phonebook['Jay'] = '34-23'    # 添加另一个项
print('Zach' in phonebook)    >> True
print('Kevin' in phonebook)   >> False
print(phonebook['Jay'])       >> '34-23'

del phonebook['Zach']         # 删除一个项
print(phonebook)              >> {'Jay': '34-23'}
```

## 页面 37: 循环 (Loops)

* 在 Python 中，代替 C++ 中 `for (i=0; i<10; i++)` 的循环语法，通常使用 `range()` 函数：
```python
for i in range(10):
    print(i)
# 依次输出 0, 1, ..., 9
```

## 页面 38: 循环遍历的各种方式

* **遍历列表中的元素**：
  ```python
  names = ['Zach', 'Jay', 'Richard']
  for name in names:
      print('Hi ' + name + '!')
  ```
* **遍历索引和值**：
  * 方法一：
    ```python
    for i in range(len(names)):
        print(i, names[i])
    ```
  * 方法二（推荐）：
    ```python
    for i, name in enumerate(names):
        print(i, name)
    ```

## 页面 39: 遍历字典

```python
phonebook = {'Zach': '12-37', 'Jay': '34-23'}

# 默认遍历键 (keys)
for name in phonebook:
    print(name)

# 遍历值 (values)
for number in phonebook.values():
    print(number)

# 同时遍历键和值
for name, number in phonebook.items():
    print(name, number)
```
*注：字典的遍历顺序是否得到保证取决于 Python 的具体版本（Python 3.7+ 开始保证插入顺序）。*

## 页面 40: 类 (Classes)

```python
class Animal(object):
    # 构造函数
    def __init__(self, species, age):
        self.species = species
        self.age = age
    
    # 实例方法，使用 self 引用自身
    def is_person(self):
        return self.species == 'human'
    
    def age_one_year(self):
        self.age += 1

# 继承 Animal 类
class Dog(Animal):
    # 重写 (Override) 方法
    def age_one_year(self):
        self.age += 7  # 狗龄换算
```
* 提示：Python 中实例变量默认都是公开的 (public)。

## 页面 41: 模型类 (Model Classes)

在后续的课程作业中，您将看到并编写继承自 `torch.nn.Module` 的 PyTorch 模型类，这是所有神经网络模块的基类。
```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化网络层
        ...
    
    def forward(self, x):
        # 定义前向传播
        ...
```

## 页面 42: 🎯 脑内解释器 (Inner Interpreter)

以下代码的输出是什么？
```python
v1 = ["Eeyore", "Goofy", "Nemo", "Wall-E"]
v2 = {"Eeyore": 12, "Nemo": 2, "Goofy": 42}
m1 = v1[1:-1]
for n in m1:
    print(f"{n} is {v2[n]} years old.")
```

## 页面 43: 🎯 脑内解释器答案

输出结果：
```
Goofy is 42 years old.
Nemo is 2 years old.
```
* 解释：`v1[1:-1]` 获取的切片为 `["Goofy", "Nemo"]`（不包含索引为 -1 的 "Wall-E"）。

## 页面 44: 课程路线图

*(进入第五部分：NumPy)*

## 页面 45: 包和模块导入的常见语法

```python
# 导入内置模块 os 和 time
import os, time

# 导入包并指定别名
import numpy as np
np.dot(x, y) # 通过别名访问方法

# 导入特定的子模块或函数
from numpy import linalg as la, dot as matrix_multiply
```
*注：`from module import *` 的方式不被推荐，可能会引起命名空间冲突。*

## 页面 46: 步入正题：NumPy！

* **NumPy**：专门针对矩阵和向量计算进行高度优化的库。
* 利用了底层 C/C++ 子程序与高内存效率的数据结构。
* 将大量的计算表示为向量化操作。
* **主要数据类型**：`np.ndarray`（构建函数为 `np.array()`）。
* 平均而言，使用 NumPy 处理计算任务比使用标准的 Python 列表要快 **5-100 倍**！

## 页面 47: np.ndarray

```python
x = np.array([1, 2, 3]) 
y = np.array([[3, 4, 5]]) 
z = np.array([[6, 7], [8, 9]]) 

print(x.shape) >> (3,)   # 1-D 向量！
print(y.shape) >> (1, 3) # 行向量！
print(z.shape) >> (2, 2) # 矩阵！
```
* **重要提示**：在 NumPy 中，形状 `(N,)` $\ne$ `(1, N)` $\ne$ `(N, 1)`！

## 页面 48: np.ndarray 规约操作 (Reductions)

规约函数包括：`np.max`, `np.min`, `np.sum`, `np.mean` 等。

```python
# 形状：(3, 2)
x = np.array([[1, 2], [3, 4], [5, 6]]) 

# 沿着 axis 1 压缩，形状：(3,)
print(np.max(x, axis = 1))  >> [2 4 6]

# 保持维度，形状：(3, 1)
print(np.max(x, axis = 1, keepdims = True)) >> [[2] [4] [6]]
```
* 规约操作默认总是沿着某个特定的轴 (axis) 进行“塌缩”规约。如果不指定，则会作用于整个矩阵（塌缩为一个标量）。

## 页面 49: np.ndarray 运算

在 NumPy 中，中缀运算符（如 `+`, `-`, `*`, `**`, `/`）均属于**按元素对应计算 (element-wise)**。

* **点积 (Dot product)**：`np.dot(u, v)`
* **矩阵-向量积**：`np.dot(x, W)`
* **哈达玛积 (Hadamard product，按元素相乘)**：`A * B`
* **矩阵乘法**：`np.matmul(A, B)` 或 `A @ B`
  * *注：如果 A 和 B 均是 2-D 数组，推荐使用更清晰的 `np.matmul()` 或 `@`。*
* **转置 (Transpose)**：`x.T`

## 页面 50: 索引与切片 (Indexing)

```python
x = np.random.random((3, 4)) # 随机 (3, 4) 矩阵

x[:]                   # 选取全部元素
x[np.array([0, 2]), :] # 选取第 0 行和第 2 行
x[1, 1:3]              # 将第 1 行的第 1 到 2 个元素选为 1-D 向量
x[x > 0.5]             # 布尔值条件过滤索引 (Boolean indexing)
x[:, :, np.newaxis]    # 插入新维度，得到 (3, 4, 1) 形状 of 3-D 向量
```
*注：使用 ndarray 或范围进行选择可以保留选择的维度大小。*

## 页面 51: 广播机制 (Broadcasting)

```python
x = np.random.random((3, 4)) # 随机 (3, 4) 矩阵
y = np.random.random((3, 1)) # 随机 (3, 1) 向量
z = np.random.random((1, 4)) # 随机 (1, 4) 向量

x + y # 将 y 按列复制，与 x 的每一列相加
x * z # 将 z 按行复制，与 x 的每一行按元素相乘
```
*注：如果遇到广播报错，请先打印并核对矩阵的形状 (shape)。*

## 页面 52: 广播机制可视化

* 形象地展示了行/列数包含 1 的向量在与矩阵运算时，如何被隐式沿着对应轴复制展开成矩阵的形状，从而能够进行逐元素运算。

## 页面 53: 广播机制的一般规律

当对两个数组进行操作时，NumPy 会从右至左（即从最后一个维度开始向左）比较它们的形状。两个维度是兼容的，如果：
1. 它们完全相等，或者
2. 其中一个维度为 1（在这种情况下，该维度的元素会沿该轴复制展开）。

*思考题：以下运算得出的结果形状是什么？*
```python
a = np.random.random((3, 4))
b = np.random.random((3, 1))
c = np.random.random((3, ))
```
* `b + b.T` $\Rightarrow$ 结果形状？
* `a + c` $\Rightarrow$ 结果形状？
* `b + c` $\Rightarrow$ 结果形状？

## 页面 54: 广播机制的一般规律答案

* `b + b.T` $\rightarrow$ 形状为 `(3, 3)`
* `a + c` $\rightarrow$ **广播报错 (Broadcast Error)**
* `b + c` $\rightarrow$ 形状为 `(3, 3)`
  * 解释：当数组维度不一致时，NumPy 会在低维数组的形状左侧隐式地补 1。因此，1-D 向量 `c` 形状为 `(3,)` 补全为 `(1, 3)`，与 `b` 形状 `(3, 1)` 运算后广播为 `(3, 3)`。

## 页面 55: 广播算法伪代码

```python
p = max(m, n)
if m < p:
    # 在左侧填充 1，使 A 达到 p 维
    A_shape = left_pad(A.shape, 1)
else if n < p:
    # 在左侧填充 1，使 B 达到 p 维
    B_shape = left_pad(B.shape, 1)

result_dims = new_list(length=p)
for i in range(p-1, -1, -1):
    A_dim_i = A.shape[i]
    B_dim_i = B.shape[i]
    if A_dim_i != 1 and B_dim_i != 1 and A_dim_i != B_dim_i:
        raise ValueError("could not broadcast")
    else: 
        # 挑选最大的维度作为结果维度
        result_dims[i] = max(A_dim_i, B_dim_i)
```

## 页面 56: 编写高效的 NumPy 代码

**请不惜一切代价避免显式地使用 Python for 循环来遍历索引/轴**。这会导致大约 10-100 倍的速度降幅。

* **低效代码 (不推荐)**：
  ```python
  for i in range(x.shape[0]):
      for j in range(x.shape[1]): 
          x[i, j] **= 2 
  ```
* **高效代码 (推荐)**：
  ```python
  x **= 2
  x[100:1000, :] += 5
  ```

## 页面 57: NumPy 小测试

1. 如何使用 NumPy 创建一个包含从 1 到 10 的数字数组？
   A. `np.arange(1, 10)`
   B. `np.arange(1, 11)`
   C. `np.array(range(1, 10))`
   D. `np.linspace(1, 10)`

2. `np.random.rand(3, 4)` 会生成什么？
   A. 3x4 的随机整数数组
   B. 3x4 的介于 0 到 1 之间的随机浮点数数组
   C. 3x4 的介于 -1 到 1 之间的随机浮点数数组
   D. 3x4 的单位矩阵

## 页面 58: NumPy 小测试答案

1. **B. `np.arange(1, 11)`**（左闭右开区间）
2. **B. 3x4 的介于 0 到 1 之间的随机浮点数数组**

## 页面 59: 课程路线图

*(进入第六部分：实用技巧)*

## 页面 60: 列表推导式 (List Comprehensions)

* 类似于函数式编程中的 `map()`，极大地提高了代码的可读性与简洁性。
* 基本格式：`[func(x) for x in some_list]`
* 传统写法：
  ```python
  squares = [] 
  for i in range(10): 
      squares.append(i**2)
  ```
* 列表推导式写法：
  ```python
  squares = [i**2 for i in range(10)]
  ```
* 还可以添加条件过滤：
  ```python
  odds = [i**2 for i in range(10) if i%2 == 1]
  ```

## 页面 61: 便捷的 Python 语法

* **多重赋值 / 解包迭代对象**：
  ```python
  age, name, pets = 20, 'Joy', ['cat']
  x, y, z = ('TF', 'PyTorch', 'JAX')
  ```
* **混合使用单双引号的字符串字面量**：
  ```python
  message = 'I like "single" quotes.'
  reply = "I prefer 'double' quotes."
  ```
* **函数返回多个值**：
  ```python
  def some_func(): 
      return 10, 1 
  ten, one = some_func()
  ```
* **使用分隔符拼接字符串列表**：
  ```python
  ", ".join(['1', '2', '3']) == '1, 2, 3'
  ```
* **单行 if-else 表达式**：
  ```python
  result = "even" if number % 2 == 0 else "odd"
  ```

## 页面 62: 调试建议

* Python 拥有交互式终端环境，可在其中运行任意代码。
* 可用来代替 TI-84 计算器（无整数溢出限制！）。
* 可以导入任何模块（甚至是当前目录下的自定义脚本模块）。
* 在您不确定语法时，或者想要测试小型的边界情况（尤其是矩阵乘法运算）时，这极具帮助。

**常用的 IPython/终端快捷键**：
* `Ctrl-d`：退出当前交互式 Session。
* `Ctrl-c`：中断当前运行的命令。
* `Ctrl-l`：清空终端屏幕。

## 页面 63: 调试工具箱

| 代码操作 | 功用说明 |
| :--- | :--- |
| `array.shape` | 获取 NumPy 数组的形状维度 |
| `array.dtype` | 检查数组的数据精度/类型（用于排查由于精度带来的异常行为） |
| `type(stuff)` | 获取变量的真实类型 |
| `import pdb; pdb.set_trace()` | 在当前位置设置一个交互式调试断点 |
| `print(f'My name is {name}')` | 快速、方便地构造格式化打印字符串 |

## 页面 64: 常见错误排查

* **维度不匹配异常 (`ValueError`)**：通常是由于矩阵乘法或广播机制中的维度不匹配导致的。首选排查手段是直接打印并检查涉及的数组的 `shape`。
* **寻求社区协助**：Python 拥有极其活跃的开源社区。在遇到棘手的报错时，多查阅 Ed 讨论区、StackOverflow 以及相关的 GitHub Issues，往往能迅速找到解决方案。

## 页面 65: 推荐参考资源

* 官方 Python 3 文档：[Python 3 Docs](https://docs.python.org/3/)
* 官方 Anaconda 用户指南：[Conda Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
* 官方 NumPy 文档：[NumPy Docs](https://numpy.org/doc/stable/)
* CS231N 课程的 Python/NumPy 教程：[CS231N Tutorial](https://cs231n.github.io/python-numpy-tutorial/)
* 斯坦福 Python 专门课程 (CS41)：[CS41 Python Course](https://stanfordpython.com/#/)

## 页面 66: 课程结束 🥳

感谢您的倾听！祝您学习愉快！
