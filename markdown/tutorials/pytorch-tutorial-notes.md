# CS224N: PyTorch 教程笔记 (2021冬季)

### 作者：Dilara Soylu

在本笔记本中，我们将对 PyTorch 进行基础介绍，并完成一个玩具级（toy）NLP 任务。在准备本笔记本时参考了以下资源：

- Matt Lamm 编写 of CS224N 2020冬季课程的 "Word Window Classification" 教程笔记本
- Soumith Chintala 编写的 PyTorch 官方深度学习教程：[Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- Sharon Zhou 在 Coursera 上提供的 PyTorch 教程笔记本：[Build Basic Generative Adversarial Networks (GANs) | Coursera](https://www.coursera.org/learn/build-basic-generative-adversarial-networks-gans)

非常感谢 Angelica Sun 和 John Hewitt 提供的反馈。

## 介绍

PyTorch 是一个在学术界和工业界都广泛应用于各种场景的机器学习框架。PyTorch 最初是作为 TensorFlow（另一个流行的机器学习框架）的一个更灵活的替代方案而诞生的。在发布之初，PyTorch 以其用户友好的特性吸引了大量用户：与 TensorFlow 早期版本在执行操作前必须先定义静态计算图不同，PyTorch 允许用户动态地定义他们的操作（即即时执行/动态图），这也被 TensorFlow 在随后的版本中所整合。虽然 TensorFlow 在工业界中更受青睐，但 PyTorch 通常是研究人员首选的机器学习框架。如果你想了解更多关于两者之间差异的信息，可以查看[这篇博客](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/)。

现在我们已经对 PyTorch 的背景有了足够的了解，让我们开始将其导入到我们的笔记本中。要安装 PyTorch，你可以按照[这里](https://pytorch.org/get-started/locally/)的说明进行。或者，你可以使用 Google Colab 打开此笔记本，其基础内核中已经预装了 PyTorch。完成安装过程后，运行以下单元格：

```python
import torch
import torch.nn as nn

# 导入 pprint，这是我们用来使打印语句更美观的模块
import pprint
pp = pprint.PrettyPrinter()
```

我们已经为开始教程做好了准备。让我们深入了解吧！

---

## 张量 (Tensors)

张量是 PyTorch 中最基础的构建块。张量类似于矩阵，但它们具有额外的属性，并且可以表示更高的维度。例如，一个两边均为 256 像素的正方形图像可以用一个 $3 \times 256 \times 256$ 的张量表示，其中第一维大小为 3，代表颜色通道（红、绿、蓝）。

### 张量初始化

在 PyTorch 中有几种实例化张量的方法，下面我们将逐一介绍。

#### 从 Python 列表创建

我们可以从一个 Python 列表（可以包含子列表）初始化张量。当我们使用 `torch.tensor()` 时，维度和数据类型将由 PyTorch 自动推断。

```python
# 从 Python 列表初始化张量
data = [
        [0, 1], 
        [2, 3],
        [4, 5]
       ]
x_python = torch.tensor(data)

# 打印张量
x_python
```

```python
tensor([[0, 1],
        [2, 3],
        [4, 5]])
```

我们也可以在调用 `torch.tensor()` 时传入可选参数 `dtype` 来设置数据类型。需要熟悉的一些常用数据类型有：`torch.bool`、`torch.float` 和 `torch.long`。

```python
# 使用 dtype 创建特定类型的张量
x_float = torch.tensor(data, dtype=torch.float)
x_float
```

```python
tensor([[0., 1.],
        [2., 3.],
        [4., 5.]])
```

```python
# 使用 dtype 创建特定类型的张量
x_bool = torch.tensor(data, dtype=torch.bool)
x_bool
```

```python
tensor([[False,  True],
        [ True,  True],
        [ True,  True]])
```

我们还可以使用 `float()`、`long()` 等方法将现有张量转换为指定的数据类型。

```python
x_python.float()
```

```python
tensor([[0., 1.],
        [2., 3.],
        [4., 5.]])
```

我们也可以使用 `torch.FloatTensor`、`torch.LongTensor`、`torch.Tensor` 类来实例化特定类型的张量。`LongTensor` 在 NLP 中尤为重要，因为许多处理索引的方法都需要将索引作为 `LongTensor`（64位整型）传入。

```python
# `torch.Tensor` 默认是 float 类型
# 等同于 torch.FloatTensor(data)
x = torch.Tensor(data) 
x
```

```python
tensor([[0., 1.],
        [2., 3.],
        [4., 5.]])
```

#### 从 NumPy 数组创建

我们也可以从 NumPy 数组初始化张量。

```python
import numpy as np

# 从 NumPy 数组初始化张量
ndarray = np.array(data)
x_numpy = torch.from_numpy(ndarray)

# 打印张量
x_numpy
```

```python
tensor([[0, 1],
        [2, 3],
        [4, 5]])
```

#### 从其他张量创建

我们还可以使用以下方法，根据另一个张量来初始化新张量：

* `torch.ones_like(old_tensor)`：初始化一个全 1 张量。
* `torch.zeros_like(old_tensor)`：初始化一个全 0 张量。
* `torch.rand_like(old_tensor)`：初始化一个张量，其所有元素都采样自 $[0, 1)$ 之间的均匀分布。
* `torch.randn_like(old_tensor)`：初始化一个张量，其所有元素都采样自标准正态分布。

所有这些方法都会保留传入的原始张量的属性，如形状（shape）和设备（device，我们稍后会介绍）。

```python
# 初始化一个基础张量
x = torch.tensor([[1., 2.], [3., 4.]])
x
```

```python
tensor([[1., 2.],
        [3., 4.]])
```

```python
# 初始化一个全 0 张量
x_zeros = torch.zeros_like(x)
x_zeros
```

```python
tensor([[0., 0.],
        [0., 0.]])
```

```python
# 初始化一个全 1 张量
x_ones = torch.ones_like(x)
x_ones
```

```python
tensor([[1., 1.],
        [1., 1.]])
```

```python
# 初始化一个张量，其每个元素都采样自 0 到 1 之间的均匀分布
x_rand = torch.rand_like(x)
x_rand
```

```python
tensor([[0.8979, 0.7173],
        [0.3067, 0.1246]])
```

```python
# 初始化一个张量，其每个元素都采样自正态分布
x_randn = torch.randn_like(x)
x_randn
```

```python
tensor([[-0.6749, -0.8590],
        [ 0.6666,  1.1185]])
```

#### 通过指定形状创建

我们还可以通过指定张量的形状来实例化张量（稍后会详细介绍形状）。可以使用的方法与上一节类似：

* `torch.zeros()`
* `torch.ones()`
* `torch.rand()`
* `torch.randn()`

```python
# 初始化一个形状为 4x2x2 的全 0 张量
shape = (4, 2, 2)
x_zeros = torch.zeros(shape) # 也可以写成 x_zeros = torch.zeros(4, 2, 2)
x_zeros
```

```python
tensor([[[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]],

        [[0., 0.],
         [0., 0.]]])
```

#### 使用 `torch.arange()`

我们还可以使用 `torch.arange(end)` 创建张量，它会返回一个 1D 张量，元素范围为从 $0$ 到 $end-1$。我们可以使用可选的 `start` 和 `step` 参数来创建具有不同范围和步长的张量。

```python
# 创建一个包含值 0-9 的张量
x = torch.arange(10)
x
```

```python
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

---

### 张量属性

张量有几个我们必须要掌握的重要属性，即数据类型（data type）、形状（shape）和设备（device）。

#### 数据类型

`dtype` 属性可以让我们查看张量的数据类型。

```python
# 初始化一个 3x2 的张量（3 行 2 列）
x = torch.ones(3, 2)
x.dtype
```

```python
torch.float32
```

#### 形状

`shape` 属性告诉我们张量的形状。这可以帮助我们识别张量有多少个维度，以及每个维度中有多少个元素。

```python
# 初始化一个 3x2 张量（3 行 2 列）
x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
x
```

```python
tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])
```

```python
# 打印它的形状
# 等同于 x.size()
x.shape 
```

```python
torch.Size([3, 2])
```

```python
# 打印特定维度中的元素个数
# 第 0 维度对应行
x.shape[0] 
```

```3
```

我们也可以使用 `size()` 方法获取特定维度的大小。

```python
# 获取第 0 维度的大小
x.size(0)
```

```python
3
```

我们可以使用 `view()` 方法改变张量的形状。

```python
# view() 的示例用法
# x_view 与 x 共享相同的内存，因此改变其中一个也会改变另一个
x_view = x.view(3, 2)
x_view
```

```python
tensor([[1., 2.],
        [3., 4.],
        [5., 6.]])
```

```python
# 我们可以使用 -1 让 PyTorch 自动推断该维度的尺寸
x_view = x.view(-1, 3)
x_view
```

```python
tensor([[1., 2., 3.],
        [4., 5., 6.]])
```

我们也可以使用 `torch.reshape()` 方法来达到类似的目的。但 `reshape()` 和 `view()` 之间存在细微的区别：`view()` 要求数据在内存中必须是连续存储的（contiguous）。你可以参考这个 [StackOverflow 的回答](https://stackoverflow.com/questions/49643961/how-to-reshape-tensor-in-pytorch-matching-tensor-flow-reshape)了解更多信息。简单来说，连续存储意味着数据在内存中的排列方式与我们读取元素的顺序一致。这是因为某些方法（例如 `transpose()` 和 `view()`）实际上并不会改变数据在内存中的存储方式，它们只是改变了关于张量的元信息（meta information），使得我们使用它时会按照预期的顺序看到元素。

如果数据是连续存储的，`reshape()` 会在内部调用 `view()`；如果不是连续的，它会返回一个副本。这种差异对于基础张量而言不太重要，但如果你执行了使底层存储变得不连续的操作（例如转置），在使用 `view()` 时就会遇到问题。如果你想让张量在内存中的存储方式与其使用方式一致，可以使用 `contiguous()` 方法。

```python
# 将 x 的形状改变为 2x3
# x_reshaped 可能是对 x 的引用或副本
x_reshaped = torch.reshape(x, (2, 3))
x_reshaped
```

```python
tensor([[1., 2., 3.],
        [4., 5., 6.]])
```

我们可以使用 `torch.unsqueeze(x, dim)` 函数在指定的 `dim` 处增加一个大小为 1 的新维度，其中 `x` 是输入张量。我们也可以使用相对应的 `torch.squeeze(x)` 来消除所有大小为 1 的维度。

```python
# 初始化一个 5x2 的张量（5 行 2 列）
x = torch.arange(10).reshape(5, 2)
x
```

```python
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
```

```python
# 在第 1 维度增加一个大小为 1 的新维度
x = x.unsqueeze(1)
x.shape
```

```python
torch.Size([5, 1, 2])
```

```python
# 挤压（Squeeze）x 的维度，去除所有只有 1 个元素的维度
x = x.squeeze()
x.shape
```

```python
torch.Size([5, 2])
```

如果我们想获取张量中元素的总个数，可以使用 `numel()` 方法。

```python
x
```

```python
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
```

```python
# 获取张量中的元素总数
x.numel()
```

```python
10
```

#### 设备 (Device)

`device` 属性告诉 PyTorch 将我们的张量存储在哪里。张量存储的位置决定了进行相关计算时使用哪种设备（GPU 还是 CPU）。我们可以通过 `device` 属性找到张量所在的设备。

```python
# 初始化一个示例张量
x = torch.Tensor([[1, 2], [3, 4]])
x
```

```python
tensor([[1., 2.],
        [3., 4.]])
```

```python
# 获取张量所在的设备
x.device
```

```python
device(type='cpu')
```

我们可以使用 `to(device)` 方法将张量从一个设备移动到另一个设备。

```python
# 检查 GPU 是否可用，如果可用，则将张量移动到 GPU
if torch.cuda.is_available():
  x = x.to('cuda') 
```

---

### 张量索引 (Tensor Indexing)

在 PyTorch 中，我们可以对张量进行索引，这与 NumPy 非常相似。

```python
# 初始化一个示例张量
x = torch.Tensor([
                  [[1, 2], [3, 4]],
                  [[5, 6], [7, 8]], 
                  [[9, 10], [11, 12]] 
                 ])
x
```

```python
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.]]])
```

```python
x.shape
```

```python
torch.Size([3, 2, 2])
```

```python
# 访问第 0 个元素，即第一块（二维矩阵）
x[0] # 等同于 x[0, :]
```

```python
tensor([[1., 2.],
        [3., 4.]])
```

我们也可以使用 `:` 来对多个维度进行切片和索引。

```python
# 获取张量中每个元素矩阵的左上角元素
x[:, 0, 0]
```

```python
tensor([1., 5., 9.])
```

我们还可以访问每个维度中的任意指定元素。

```python
# 再次打印 x 以查看我们的张量
x
```

```python
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.]]])
```

```python
# 让我们访问第 0 个和第 1 个元素，各访问两次
i = torch.tensor([0, 0, 1, 1])
x[i]
```

```python
tensor([[[1., 2.],
         [3., 4.]],

        [[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]],

        [[5., 6.],
         [7., 8.]]])
```

```python
# 让我们访问第 1 个和第 2 个元素的第 0 行元素
i = torch.tensor([1, 2])
j = torch.tensor([0])
x[i, j]
```

```python
tensor([[ 5.,  6.],
        [ 9., 10.]])
```

我们可以使用 `item()` 方法从张量中获取 Python 的标量值（scalar value）。

```python
x[0, 0, 0]
```

```python
tensor(1.)
```

```python
x[0, 0, 0].item()
```

```python
1.0
```

---

### 张量操作 (Operations)

PyTorch 的算术操作与 NumPy 非常相似。我们可以对标量或其他张量进行计算。

```python
# 创建一个示例张量
x = torch.ones((3,2,2))
x
```

```python
tensor([[[1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.]]])
```

```python
# 执行按元素加法
# 减法使用 -
x + 2
```

```python
tensor([[[3., 3.],
         [3., 3.]],

        [[3., 3.],
         [3., 3.]],

        [[3., 3.],
         [3., 3.]]])
```

```python
# 执行按元素乘法
# 除法使用 /
x * 2
```

```python
tensor([[[2., 2.],
         [2., 2.]],

        [[2., 2.],
         [2., 2.]],

        [[2., 2.],
         [2., 2.]]])
```

我们可以在大小兼容的两个不同张量之间应用相同的操作。

```python
# 创建一个 4x3 且值全为 6 的张量
a = torch.ones((4,3)) * 6
a
```

```python
tensor([[6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.]])
```

```python
# 创建一个 1D 且值全为 2 的张量
b = torch.ones(3) * 2
b
```

```python
tensor([2., 2., 2.])
```

```python
# a 除以 b (广播机制)
a / b
```

```python
tensor([[3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.]])
```

我们可以使用 `tensor.matmul(other_tensor)` 进行矩阵乘法，使用 `tensor.T` 进行转置。矩阵乘法也可以使用 `@` 运算符来执行。

```python
# 等同于 a.matmul(b)
# 既然 b 是 1D 张量，a @ b 也可以正常运行并自动推断第二维度
a @ b 
```

```python
tensor([36., 36., 36., 36.])
```

```python
pp.pprint(a.shape)
pp.pprint(a.T.shape)
```

```python
torch.Size([4, 3])
torch.Size([3, 4])
```

我们可以使用 `mean(dim)` 和 `std(dim)` 方法沿特定维度计算均值和标准差。也就是说，如果我们想要得到一个 $4 \times 3 \times 2$ 的张量中关于第 0 维度的平均 $3 \times 2$ 矩阵，我们将 `dim` 设为 0。如果我们调用这些方法时不带任何参数，则会计算整个张量的均值和标准差。要使用 `mean` 和 `std`，我们的张量数据类型必须是浮点型。

```python
# 创建一个示例张量
m = torch.tensor(
    [
     [1., 1.],
     [2., 2.],
     [3., 3.],
     [4., 4.]
    ]
)

pp.pprint("Mean: {}".format(m.mean()))
pp.pprint("Mean in the 0th dimension: {}".format(m.mean(0)))
pp.pprint("Mean in the 1st dimension: {}".format(m.mean(1)))
```

```python
'Mean: 2.5'
'Mean in the 0th dimension: tensor([2.5000, 2.5000])'
'Mean in the 1st dimension: tensor([1., 2., 3., 4.])'
```

我们可以使用 `torch.cat` 来拼接多个张量。

```python
# 在维度 0 和 维度 1 上进行拼接
a_cat0 = torch.cat([a, a, a], dim=0)
a_cat1 = torch.cat([a, a, a], dim=1)

print("Initial shape: {}".format(a.shape))
print("Shape after concatenation in dimension 0: {}".format(a_cat0.shape))
print("Shape after concatenation in dimension 1: {}".format(a_cat1.shape))
```

```python
Initial shape: torch.Size([4, 3])
Shape after concatenation in dimension 0: torch.Size([12, 3])
Shape after concatenation in dimension 1: torch.Size([4, 9])
```

PyTorch 中的大多数操作都是非原地（not in place）进行的。然而，PyTorch 也提供了原地（in place）执行的操作，只需在方法名后面加上下划线（`_`）即可。

```python
# 打印我们的张量
a
```

```python
tensor([[6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.]])
```

```python
# add() 不是原地操作
a.add(a)
a
```

```python
tensor([[6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.],
        [6., 6., 6.]])
```

```python
# add_() 是原地操作
a.add_(a)
a
```

```python
tensor([[12., 12., 12.],
        [12., 12., 12.],
        [12., 12., 12.],
        [12., 12., 12.]])
```

---

## 自动求导 (Autograd)

PyTorch 和其他机器学习库因其自动微分（automatic differentiation）功能而闻名。也就是说，只要我们定义了需要执行的操作步骤，框架本身就能够算出如何计算梯度。我们可以调用 `backward()` 方法来要求 PyTorch 计算梯度，这些梯度会被存储在 `grad` 属性中。

```python
# 创建一个示例张量
# requires_grad 参数告诉 PyTorch 去记录并存储梯度
x = torch.tensor([2.], requires_grad=True)

# 打印尚未计算的梯度
# 目前是 None，因为还未进行反向传播
pp.pprint(x.grad)
```

```python
None
```

```python
# 计算 y 关于 x 的梯度
y = x * x * 3 # 3x^2
y.backward()
pp.pprint(x.grad) # d(y)/d(x) = d(3x^2)/d(x) = 6x = 6 * 2 = 12
```

```python
tensor([12.])
```

让我们从另一个不同的张量运行反向传播，看看会发生什么。

```python
z = x * x * 3 # 3x^2
z.backward()
pp.pprint(x.grad)
```

```python
tensor([24.])
```

我们可以看到，`x.grad` 更新为了迄今为止计算出的梯度之和。当我们在神经网络中运行反向传播时， we sum up all the gradients for a particular neuron before making an update. 这正是这里所发生的事情！这也是为什么我们需要在每一次训练迭代中都运行 `zero_grad()` 的原因（稍后会详细介绍）。否则，我们的梯度就会从一个训练迭代不断累加到下一个，从而导致我们的更新是错误的。

---

## 神经网络模块 (Neural Network Module)

到目前为止，我们已经了解了张量、其属性以及张量的基本操作。如果我们从头开始构建网络层，熟悉这些内容将非常有用。我们将在作业3中用到这些知识，但在此之后，我们将使用 PyTorch 中 `torch.nn` 模块里预定义的构建块。然后，我们将这些构建块组合在一起，创建复杂的网络。让我们开始导入该模块并使用别名，这样我们就不需要在每次使用时都输入 `torch.nn`。

```python
import torch.nn as nn
```

### 线性层 (Linear Layer)

我们可以使用 `nn.Linear(H_in, H_out)` 来创建一个线性层。这会接收一个维度为 $(N, *, H\_in)$ 的矩阵并输出一个维度为 $(N, *, H\_out)$ 的矩阵。这里的 $*$ 表示中间可以有任意数量的维度。线性层执行操作 $Ax+b$，其中 $A$ 和 $b$ 是随机初始化的。如果我们不希望线性层学习偏置参数，可以使用 `bias=False` 来初始化我们的层。

```python
# 创建输入数据
input = torch.ones(2,3,4)

# 制作一个线性层，将维度为 N,*,H_in 的输入转换为 N,*,H_out 的输出
linear = nn.Linear(4, 2)
linear_output = linear(input)
linear_output
```

```python
tensor([[[0.1796, 0.1423],
         [0.1796, 0.1423],
         [0.1796, 0.1423]],

        [[0.1796, 0.1423],
         [0.1796, 0.1423],
         [0.1796, 0.1423]]], grad_fn=<AddBackward0>)
```

### 其他模块层 (Other Module Layers)

在 `nn` 模块中还有其他几个预配置的层。一些常用的例子包括：`nn.Conv2d`、`nn.ConvTranspose2d`、`nn.BatchNorm1d`、`nn.BatchNorm2d`、`nn.Upsample` 以及 `nn.MaxPool2d` 等等。随着课程的进行，我们将学习更多关于这些内容的知识。目前，唯一需要记住的重要事情是，我们可以将这些层中的每一层都视为即插即用的组件：我们只需提供所需的维度，PyTorch 就会处理好它们的设置工作。

### 激活函数层 (Activation Function Layer)

我们还可以使用 `nn` 模块为我们的张量应用激活函数。激活函数用于向网络中引入非线性。激活函数的一些例子有 `nn.ReLU()`、`nn.Sigmoid()` 和 `nn.LeakyReLU()`。激活函数分别对每个元素进行操作，因此我们作为输出得到的张量形状与我们传入的张量形状是相同的。

```python
linear_output
```

```python
tensor([[[0.1796, 0.1423],
         [0.1796, 0.1423],
         [0.1796, 0.1423]],

        [[0.1796, 0.1423],
         [0.1796, 0.1423],
         [0.1796, 0.1423]]], grad_fn=<AddBackward0>)
```

```python
sigmoid = nn.Sigmoid()
output = sigmoid(linear_output)
output
```

```python
tensor([[[0.5448, 0.5355],
         [0.5448, 0.5355],
         [0.5448, 0.5355]],

        [[0.5448, 0.5355],
         [0.5448, 0.5355],
         [0.5448, 0.5355]]], grad_fn=<SigmoidBackward>)
```

### 组合层 (Putting the Layers Together)

到目前为止，我们已经看到我们可以创建不同的网络层，并将一层的输出作为下一层的输入传入。为了避免创建中间张量并把它们传来传去，我们可以使用 `nn.Sequential`，它能够帮我们完成这个过程。

```python
block = nn.Sequential(
    nn.Linear(4, 2),
    nn.Sigmoid()
)

input = torch.ones(2,3,4)
output = block(input)
output
```

```python
tensor([[[0.3630, 0.3951],
         [0.3630, 0.3951],
         [0.3630, 0.3951]],

        [[0.3630, 0.3951],
         [0.3630, 0.3951],
         [0.3630, 0.3951]]], grad_fn=<SigmoidBackward>)
```

### 自定义模块 (Custom Modules)

除了使用预定义的模块外，我们还可以通过继承 `nn.Module` 类来构建我们自己的模块。例如，我们可以利用前面介绍的张量来自己构建出类似 `nn.Linear`（它也是继承自 `nn.Module`）的线性层！我们还可以构建出新的、更复杂的模块，例如自定义神经网络。你将在后面的作业中对此进行练习。

要创建自定义模块，我们首先需要继承 `nn.Module`。然后我们可以在 `__init__` 函数中初始化我们的参数，第一步要调用父类的 `__init__` 函数。我们定义的作为类的成员属性的所有 `nn` 模块对象都会被视为可学习参数，它们在训练过程中会被自动优化。普通的张量不是模型参数，但如果用 `nn.Parameter` 类将它们包裹起来，它们就会转换为模型参数。

所有继承自 `nn.Module` 的类也都需要实现 `forward(x)` 函数，其中 `x` 是一个张量。当我们将参数传递给我们的模块（例如执行 `model(x)`）时，这个 `forward` 函数就会被调用。

```python
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size, hidden_size):
    # 调用父类的 __init__ 函数
    super(MultilayerPerceptron, self).__init__()

    # 簿记：保存初始化参数
    self.input_size = input_size 
    self.hidden_size = hidden_size 

    # 定义我们的模型
    # `self.model` 的命名并没有什么特殊要求，它可以是任意的名字。
    self.model = nn.Sequential(
        nn.Linear(self.input_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.input_size),
        nn.Sigmoid()
    )
    
  def forward(self, x):
    output = self.model(x)
    return output
```

这里是定义相同类的另一种替代方法。你可以看到，我们可以不使用 `nn.Sequential`，而是通过在 `__init__` 方法中定义单个层并在 `forward` 方法中把它们连接起来。

```python
class MultilayerPerceptron(nn.Module):

  def __init__(self, input_size, hidden_size):
    # 调用父类的 __init__ 函数
    super(MultilayerPerceptron, self).__init__()

    # 簿记：保存初始化参数
    self.input_size = input_size 
    self.hidden_size = hidden_size 

    # 定义我们的网络层
    self.linear = nn.Linear(self.input_size, self.hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(self.hidden_size, self.input_size)
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    linear = self.linear(x)
    relu = self.relu(linear)
    linear2 = self.linear2(relu)
    output = self.sigmoid(linear2)
    return output
```

现在我们已经定义了我们的类，我们可以实例化它并看看它能做什么。

```python
# 制作一个示例输入
input = torch.randn(2, 5)

# 创建我们的模型
model = MultilayerPerceptron(5, 3)

# 将我们的输入传递到模型中
model(input)
```

```python
tensor([[0.5887, 0.4685, 0.4995, 0.6062, 0.3894],
        [0.5292, 0.5144, 0.4310, 0.5821, 0.3681]], grad_fn=<SigmoidBackward>)
```

我们可以使用 `named_parameters()` 和 `parameters()` 方法来检查我们模型的参数。

```python
list(model.named_parameters())
```

```python
[('linear.weight', Parameter containing:
  tensor([[ 0.0517,  0.0466, -0.3616, -0.3459, -0.3524],
          [ 0.0680,  0.0945, -0.3261,  0.1835,  0.4344],
          [-0.3985,  0.1973, -0.1373, -0.2314, -0.1868]], requires_grad=True)),
 ('linear.bias', Parameter containing:
  tensor([-0.0924,  0.1563, -0.3934], requires_grad=True)),
 ('linear2.weight', Parameter containing:
  tensor([[-0.1111,  0.4216, -0.0263],
          [-0.5714, -0.3207, -0.5514],
          [ 0.1727,  0.4809, -0.4426],
          [ 0.3308,  0.1744, -0.2814],
          [ 0.2347,  0.1584,  0.0537]], requires_grad=True)),
 ('linear2.bias', Parameter containing:
  tensor([ 0.1169,  0.0575, -0.2778,  0.3314, -0.5406], requires_grad=True))]
```

---

## 优化 (Optimization)

我们已经展示了如何使用 `backward()` 函数来计算梯度。然而仅仅拥有梯度对于我们的模型学习来说是不够的。我们还需要知道如何去更新模型的参数。这正是优化器（optimizers）发挥作用的地方。`torch.optim` 模块包含了几个我们可以使用的优化器。一些流行的例子包括 `optim.SGD` 和 `optim.Adam`。在初始化优化器时，我们需要传入我们的模型参数（这可以通过 `model.parameters()` 来访问），从而告诉优化器它将优化哪些值。优化器还包含一个学习率（learning rate, `lr`）参数，该参数决定了在每一步中要做出多大程度的参数更新。不同的优化器也各自拥有不同的超参数。

```python
import torch.optim as optim
```

一旦我们确定了我们的优化函数，我们就可以定义我们想要优化的损失值（loss）。我们可以自己定义损失，也可以使用 PyTorch 中预定义的损失函数，比如 `nn.BCELoss()`（二元交叉熵损失）。现在让我们把所有东西都整合在一起吧！我们将从创建一些虚拟数据（dummy data）开始。

```python
# 创建标签数据 y
y = torch.ones(10, 5)

# 向我们的目标值 y 中加入一些噪声来生成我们的输入 x
# 我们希望我们的模型能够克服噪声，去预测我们原始的标签数据
x = y + torch.randn_like(y)
x
```

```python
tensor([[ 0.6017, -0.1942,  1.5013,  0.8077,  0.4822],
        [ 2.4198,  1.6348,  1.7984,  2.7030,  1.8514],
        [-0.2632,  1.1230,  0.4369,  1.1091, -0.7259],
        [ 1.5842,  1.6495, -0.6160,  1.4506,  0.5238],
        [-0.4652,  1.1538,  4.0669, -0.4767,  3.2751],
        [ 1.6152,  0.1758,  0.6275,  1.8919,  2.0594],
        [ 2.4223,  0.6465,  1.0125,  0.2578, -0.2029],
        [ 1.6735,  0.9132,  0.3643,  0.9575,  1.7279],
        [ 3.0019,  0.7942,  2.0360,  1.3991,  1.3139],
        [ 0.8813,  0.7213,  1.6067, -0.5509,  1.3748]])
```

现在，我们可以定义我们的模型、优化器和损失函数。

```python
# 实例化模型
model = MultilayerPerceptron(5, 3)

# 定义优化器
adam = optim.Adam(model.parameters(), lr=1e-1)

# 使用预定义的损失函数定义损失
loss_function = nn.BCELoss()

# 计算模型目前的损失表现
y_pred = model(x)
loss_function(y_pred, y).item()
```

```python
0.7392909526824951
```

让我们看看我们是否能让我们的模型获得更小的损失。现在我们已经有了我们需要的一切，接下来可以设置我们的训练循环了。

```python
# 设置 epoch 的数量，这决定了训练迭代的次数
n_epoch = 10 

for epoch in range(n_epoch):
  # 将梯度清零
  adam.zero_grad()

  # 获取模型预测值
  y_pred = model(x)

  # 计算损失
  loss = loss_function(y_pred, y)

  # 打印状态信息
  print(f"Epoch {epoch}: training loss: {loss}")

  # 计算梯度
  loss.backward()

  # 迈出一步更新权重
  adam.step()
```

```python
Epoch 0: training loss: 0.7392909526824951
Epoch 1: training loss: 0.6668772101402283
Epoch 2: training loss: 0.5994036793708801
Epoch 3: training loss: 0.5073628425598145
Epoch 4: training loss: 0.4047197103500366
Epoch 5: training loss: 0.30285021662712097
Epoch 6: training loss: 0.21625970304012299
Epoch 7: training loss: 0.13478505611419678
Epoch 8: training loss: 0.0776328295469284
Epoch 9: training loss: 0.042155299335718155
```

可以看到，我们的损失正在逐渐减小。现在让我们检查一下我们模型的预测结果，看看它们是否接近我们的原始目标 $y$（全为 1）。

```python
# 查看我们的模型在训练数据上的表现
y_pred = model(x)
y_pred
```

```python
tensor([[0.8869, 0.9687, 0.9948, 0.9771, 0.9804],
        [0.9883, 0.9996, 1.0000, 0.9999, 0.9999],
        [0.8704, 0.9590, 0.9919, 0.9672, 0.9716],
        [0.9508, 0.9937, 0.9996, 0.9973, 0.9979],
        [0.9012, 0.9760, 0.9967, 0.9839, 0.9864],
        [0.9526, 0.9941, 0.9997, 0.9975, 0.9981],
        [0.9466, 0.9926, 0.9995, 0.9967, 0.9973],
        [0.9450, 0.9922, 0.9995, 0.9964, 0.9971],
        [0.9812, 0.9989, 1.0000, 0.9997, 0.9998],
        [0.8866, 0.9685, 0.9948, 0.9769, 0.9803]], grad_fn=<SigmoidBackward>)
```

```python
# 创建测试数据并检查我们的模型在其上的表现
x2 = y + torch.randn_like(y)
y_pred = model(x2)
y_pred
```

```python
tensor([[0.9582, 0.9954, 0.9998, 0.9982, 0.9986],
        [0.9217, 0.9847, 0.9984, 0.9912, 0.9927],
        [0.9209, 0.9844, 0.9984, 0.9909, 0.9925],
        [0.9410, 0.9911, 0.9994, 0.9957, 0.9966],
        [0.9694, 0.9974, 0.9999, 0.9992, 0.9994],
        [0.9360, 0.9896, 0.9992, 0.9947, 0.9957],
        [0.9561, 0.9949, 0.9997, 0.9980, 0.9984],
        [0.9290, 0.9874, 0.9988, 0.9931, 0.9944],
        [0.9754, 0.9983, 1.0000, 0.9995, 0.9996],
        [0.8905, 0.9706, 0.9953, 0.9789, 0.9821]], grad_fn=<SigmoidBackward>)
```

太棒了！看来我们的模型几乎完美地学会了从我们传入的 $x$ 中过滤出噪声！

---

## 演示：词窗口分类 (Word Window Classification)

在笔记本的这一部分之前，我们已经学习了 PyTorch 的基础知识，并构建了一个解决玩具任务的基本网络。现在我们将尝试解决一个真实的 NLP 任务示例。以下是我们将要学习的内容：

1. **数据 (Data)**：创建一个包含分批张量（Batched Tensors）的数据集
2. **建模 (Modeling)**
3. **训练 (Training)**
4. **预测 (Prediction)**

在这一部分，我们的目标是训练一个模型，该模型能够在一个句子中找到对应于地点（`LOCATION`）的单词，这些地点总是长度为 1（这意味着像 "San Francisco" 这样由两个单词组成的地点在这里不会被识别为单个 `LOCATION`）。

我们的任务被称为词窗口分类是有原因的。我们不希望我们的模型在每次前向传播中仅看单个单词，而是希望它能够考虑被查询单词的上下文。也就是说，对于每个单词，我们希望模型能够注意到其周围的单词。让我们开始深入了解吧！

### 数据 (Data)

任何机器学习项目的首要任务都是建立我们的训练集。通常，我们会利用一个训练语料库（corpus）。在 NLP 任务中，语料库通常是一个 `.txt` 或 `.csv` 文件，其中每一行对应一个句子或表格数据点。在这个玩具任务中，我们假设我们已经将数据 and the corresponding labels into a `Python` list.

```python
# 我们的原始数据，由句子组成
corpus = [
          "We always come to Paris",
          "The professor is from Australia",
          "I live in Stanford",
          "He comes from Taiwan",
          "The capital of Turkey is Ankara"
         ]
```

#### 预处理 (Preprocessing)

为了让我们的模型更容易学习，我们通常会对数据应用一些预处理步骤。这在处理文本数据时尤为重要。以下是一些文本预处理的例子：

* **分词 (Tokenization)**：将句子拆分为单词。
* **转为小写 (Lowercasing)**：将所有字母改为小写。
* **去除噪声 (Noise removal)**：去除特殊字符（例如标点符号）。
* **去除停用词 (Stop words removal)**：去除常用词（如介词、冠词等）。

需要哪些预处理步骤完全取决于当前的任务。例如，虽然在某些任务中去除特殊字符很有用，但对其他任务来说，它们可能非常重要（比如在处理多语言文本时）。对于我们的任务，我们将把所有单词转换为小写并进行分词。

```python
# 我们将用来生成训练样本的预处理函数
# 这是一个简单的函数，我们将字母转换为小写，然后对单词进行拆分（分词）。
def preprocess_sentence(sentence):
  return sentence.lower().split()

# 创建我们的训练集
train_sentences = [sent.lower().split() for sent in corpus]
train_sentences
```

```python
[['we', 'always', 'come', 'to', 'paris'],
 ['the', 'professor', 'is', 'from', 'australia'],
 ['i', 'live', 'in', 'stanford'],
 ['he', 'comes', 'from', 'taiwan'],
 ['the', 'capital', 'of', 'turkey', 'is', 'ankara']]
```

对于我们拥有的每个训练示例，我们还应该有一个相对应的标签。回想一下，我们模型的目标是确定哪些单词对应于 `LOCATION`（地点）。也就是说，我们希望模型对所有非 `LOCATION` 的单词输出 `0`，而对是 `LOCATION` 的单词输出 `1`。

```python
# 出现在我们语料库中的地点集合
locations = set(["australia", "ankara", "paris", "stanford", "taiwan", "turkey"])

# 我们的训练标签
train_labels = [[1 if word in locations else 0 for word in sent] for sent in train_sentences]
train_labels
```

```python
[[0, 0, 0, 0, 1],
 [0, 0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 1, 0, 1]]
```

#### 将单词转换为嵌入 (Converting Words to Embeddings)

让我们更仔细地审视一下我们的训练数据。我们拥有的每个数据点都是一个单词序列。另一方面，我们知道机器学习模型只能处理向量中的数字。我们要如何把单词变成数字呢？你可能想到了嵌入（embeddings），你是对的！

假设我们有一个嵌入查找表 $E$，其中每一行对应一个词嵌入。也就是说，我们词汇表中的每个单词都会在这个表中对应一个嵌入行 $i$。每当我们想为一个单词找到其对应的嵌入时，我们将遵循以下步骤：

1. 在嵌入表中找到该单词对应的索引 $i$：即 `单词 -> 索引`。
2. 索引到嵌入表并获取对应的嵌入向量：即 `索引 -> 嵌入`。

让我们来看第一步。我们应该将词汇表中的所有单词分配给一个对应的索引。我们可以按如下方式操作：

1. 找出语料库中所有独一无二的单词。
2. 为每个单词分配一个索引。

```python
# 找出我们语料库中所有独一无二的单词
vocabulary = set(w for s in train_sentences for w in s)
vocabulary
```

```python
{'always',
 'ankara',
 'australia',
 'capital',
 'come',
 'comes',
 'from',
 'he',
 'i',
 'in',
 'is',
 'live',
 'of',
 'paris',
 'professor',
 'stanford',
 'taiwan',
 'the',
 'to',
 'turkey',
 'we'}
```

现在，`vocabulary` 包含了我们语料库中的所有单词。另一方面，在测试期间，我们可能会遇到不在词汇表中的新单词。如果我们能想出一种表示未登录词（unknown words）的方法，我们的模型仍然可以推断它们是否是 `LOCATION`，因为我们同样也会关注每个预测词周边的单词。

我们引入了一个特殊的 Token——`<unk>`，来处理词汇表之外的单词。如果我们愿意，也可以为未登录词选择其他的特殊字符串。这里唯一的的要求是我们的 Token 必须是唯一的：我们只应当在遇到未知单词时使用这个 Token。我们也将这个特殊 Token 添加到我们的词汇表中。

```python
# 将代表未知词的 token 添加到词汇表中
vocabulary.add("<unk>")
```

早些时候我们提到过，我们的任务之所以被称为“词窗口分类”，是因为模型在需要做出预测时，除了看当前给定的单词外，还会查看其周围的单词。

例如，以句子 "We always come to Paris" 为例。该句对应的训练标签是 `0, 0, 0, 0, 1`，因为只有最后一个词 Paris 是一个 `LOCATION`。在一次传播（即调用 `forward()`）中，我们的模型将尝试为某一个单词生成正确的标签。假设模型正在尝试为 Paris 生成正确标签 `1`。如果我们仅允许模型看到 Paris 而没有任何其他信息，我们就会漏掉一个重要线索，即 `to` 这个词常常出现在 `LOCATION` 的前面。

词窗口允许模型在做出预测时，考虑每个单词周围的 $+N$ 或 $-N$ 个单词。在前面 Paris 的例子中，如果我们把窗口大小（window size）设为 1，这意味着模型会查看紧接在 Paris 之前和之后的单词，也就是 `to` 和……好吧，后面什么都没有。但这又引出了另一个问题。Paris 处于句子的末尾，因此它后面没有其他单词。请记住，我们在初始化 PyTorch 模型时就定义了它们的输入维度。如果我们把窗口大小设置为 1，这意味着模型在每次前向传播中必须接收 3 个单词。我们不能让模型有时候期望接收 2 个词，有时候又期望接收 3 个词。

解决方案是引入一个特殊的 Token，比如 `<pad>`（填充字符），它会被添加在我们的句子两端，以确保每个单词的周围都有一个有效的词窗口。类似于 `<unk>` Token，我们也可以根据需要选择其他的字符串来作为填充 Token，只要确保它被用于这一唯一的用途即可。

```python
# 将填充 token <pad> 添加到词汇表中
vocabulary.add("<pad>")

# 对给定句子进行词窗口填充的函数
# 我们在这里引入这个函数作为一个示例
# 我们将在本教程的后面部分使用它
def pad_window(sentence, window_size, pad_token="<pad>"):
  window = [pad_token] * window_size
  return window + sentence + window

# 展示填充示例
window_size = 2
pad_window(train_sentences[0], window_size=window_size)
```

```python
['<pad>', '<pad>', 'we', 'always', 'come', 'to', 'paris', '<pad>', '<pad>']
```

Now that our vocabularly is ready, let's assign an index to each of our words.

```python
# 我们只需将词汇表转换为列表形式以对其进行索引
# 排序并非必需，这里进行排序是为了展示一个有序的 word_to_ix 字典。
# 话虽如此，我们会发现将填充 token 的索引设为 0 是很方便的，
# 因为一些 PyTorch 函数（如 nn.utils.rnn.pad_sequence，稍后会介绍）将其用作默认值。
ix_to_word = sorted(list(vocabulary))

# 创建一个字典来查找给定单词的索引
word_to_ix = {word: ind for ind, word in enumerate(ix_to_word)}
word_to_ix
```

太棒了！我们现在已经准备好将训练句子转换为与每个 Token 对应的索引序列了。

```python
# 给定一个 token 组成的句子，返回对应的索引列表
def convert_token_to_indices(sentence, word_to_ix):
  indices = []
  for token in sentence:
    # 检查该 token 是否存在于词汇表中。如果存在，获取其索引。
    # 如果不存在，获取未知词 token <unk> 的索引。
    if token in word_to_ix:
      index = word_to_ix[token]
    else:
      index = word_to_ix["<unk>"]
    indices.append(index)
  return indices

# 上述函数的更紧凑版本
def _convert_token_to_indices(sentence, word_to_ix):
  return [word_to_ix.get(token, word_to_ix["<unk>"]) for token in sentence]

# 展示一个例子
example_sentence = ["we", "always", "come", "to", "kuwait"]
example_indices = convert_token_to_indices(example_sentence, word_to_ix)
restored_example = [ix_to_word[ind] for ind in example_indices]

print(f"Original sentence is: {example_sentence}")
print(f"Going from words to indices: {example_indices}")
print(f"Going from indices to words: {restored_example}")
```

```python
Original sentence is: ['we', 'always', 'come', 'to', 'kuwait']
Going from words to indices: [22, 2, 6, 20, 1]
Going from indices to words: ['we', 'always', 'come', 'to', '<unk>']
```

在上面的例子中，`kuwait` 显示为了 `<unk>`，因为在我们的词汇表中并没有它。让我们将 `train_sentences` 转换为 `example_padded_indices`。

```python
# 将我们的句子转换为索引表示
example_padded_indices = [convert_token_to_indices(s, word_to_ix) for s in train_sentences]
example_padded_indices
```

```python
[[22, 2, 6, 20, 15],
 [19, 16, 12, 8, 4],
 [10, 13, 11, 17],
 [9, 7, 8, 18],
 [19, 5, 14, 21, 12, 3]]
```

既然我们为词汇表中的每个单词都有了一个索引，我们就可以使用 PyTorch 的 `nn.Embedding` 类来创建嵌入表。其调用方式如下：`nn.Embedding(num_words, embedding_dimension)`，其中 `num_words` 是我们词汇表中单词的数量，而 `embedding_dimension` 是我们希望拥有的嵌入维度大小。`nn.Embedding` 并没有什么神奇的：它仅仅是一个对可训练的 $N \times E$ 维张量的包装类，其中 $N$ 是我们词汇表的大小，$E$ 是词嵌入维度。该表在初始时是随机初始化的，但它会在训练过程中改变。当我们训练我们的网络时，梯度会一直反向传播到嵌入层，从而使我们的词嵌入得到更新。我们在模型内部初始化将要用到的嵌入层，不过这里我们先展示一个例子。

```python
# 为我们的单词创建一个嵌入表
embedding_dim = 5
embeds = nn.Embedding(len(vocabulary), embedding_dim)

# 打印我们嵌入表中的参数
list(embeds.parameters())
```

为了获取词汇表中某个词的嵌入表示，我们所要做的就是创建一个查找张量（lookup tensor）。这个查找张量只是一个包含我们要查阅的索引的张量。`nn.Embedding` 类期待传入的索引张量必须是 `LongTensor` 类型，所以我们需要依此创建我们的张量。

```python
# 获取单词 Paris 的词嵌入
index = word_to_ix["paris"]
index_tensor = torch.tensor(index, dtype=torch.long)
paris_embed = embeds(index_tensor)
paris_embed
```

```python
tensor([ 0.6732,  0.4117, -0.5378,  0.6632, -2.7096],
       grad_fn=<EmbeddingBackward>)
```

```python
# 我们也可以一次获取多个词的嵌入
index_paris = word_to_ix["paris"]
index_ankara = word_to_ix["ankara"]
indices = [index_paris, index_ankara]
indices_tensor = torch.tensor(indices, dtype=torch.long)
embeddings = embeds(indices_tensor)
embeddings
```

```python
tensor([[ 0.6732,  0.4117, -0.5378,  0.6632, -2.7096],
        [ 0.8021,  1.5121,  0.8239,  0.9865, -1.3801]],
       grad_fn=<EmbeddingBackward>)
```

通常，我们将嵌入层定义为我们模型的一部分，这在笔记本的后续小节中可以看到。

#### 句子分批 (Batching Sentences)

我们在课上已经学过了批次（batches）。等待整个训练语料库被完全处理后再做权重更新的成本是非常高的。另一方面，在处理完每个训练样本后就更新参数会使得更新之间的损失值非常不稳定。为了解决这些问题，我们在训练完一个批次（batch）的数据后更新我们的参数。这使我们能够获得对全局损失梯度的更好估计。在这一节中，我们将学习如何使用 `torch.utils.data.DataLoader` 类将我们的数据整理成批次。

我们将按如下方式调用 `DataLoader` 类：`DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)`。参数 `batch_size` 决定了每个批次中样本的数量。在每个 epoch 中，我们都将使用 `DataLoader` 迭代所有的批次。默认情况下，批次的顺序是确定性的，但是我们可以通过将 `shuffle` 参数设为 `True` 来让 `DataLoader` 打乱批次顺序。通过这种方式，我们能确保自己不会多次遇到较差的批次组合。

如果提供了参数，`DataLoader` 会将它准备好的批次传递给 `collate_fn`。我们可以编写一个自定义函数来传入 `collate_fn` 参数，以打印关于我们批次的统计数据或执行额外的处理。在我们的例子中，我们将使用 `collate_fn` 来做以下事情：

1. 对我们的训练句子进行词窗口填充（Window padding）。
2. 将训练样本中的单词转换为索引。
3. 对训练样本进行填充，使得同一个批次里的所有句子和标签具有相同的长度。同样，我们也需要填充标签。这会引入一个问题，因为在计算损失时，我们需要知道给定样本中真实的单词数量。我们还将在传递给 `collate_fn` 参数的函数中追踪并记录该实际长度值。

由于我们的 `collate_fn` 函数版本需要访问 `word_to_ix` 字典（这样它才能把单词转换成索引），我们将使用 Python 中的 `partial` 函数，它能方便地把我们给定的参数绑定到被调用的目标函数上。

```python
from torch.utils.data import DataLoader
from functools import partial

def custom_collate_fn(batch, window_size, word_to_ix):
  # 将我们的 batch 拆分为训练样本 (x) 和标签 (y)
  # 我们需要将 x 和 y 转换为张量，因为 nn.utils.rnn.pad_sequence 
  # 方法期望传入张量。这同样有用，因为我们的模型期望输入是张量。
  x, y = zip(*batch)

  # 现在我们需要对训练样本进行窗口填充。我们已经定义了一个处理窗口填充的函数。
  # 我们在这里再次把该函数写一遍，以便于所有内容都在一处。
  def pad_window(sentence, window_size, pad_token="<pad>"):
    window = [pad_token] * window_size
    return window + sentence + window

  # 填充训练样本
  x = [pad_window(s, window_size=window_size) for s in x]

  # 现在我们需要将训练样本中的单词转换为索引。出于和上面相同的原因，
  # 我们在这里复制先前定义的函数。
  def convert_tokens_to_indices(sentence, word_to_ix):
    return [word_to_ix.get(token, word_to_ix["<unk>"]) for token in sentence]

  # 将训练样本转换为索引
  x = [convert_tokens_to_indices(s, word_to_ix) for s in x]

  # 我们现在将对样本进行零填充（padding），以使一个批次中所有样本的长度都相同，
  # 从而让矩阵操作成为可能。
  # 我们将 batch_first 参数设为 True，使返回的矩阵将 batch 维度作为第一维。
  pad_token_ix = word_to_ix["<pad>"]

  # pad_sequence 函数要求输入是一个张量，所以我们把 x 转换为张量
  x = [torch.LongTensor(x_i) for x_i in x]
  x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_ix)

  # 我们也将对标签进行填充。在此之前，我们将记录标签的数量，
  # 从而获知每个样本中原本存在多少个单词。
  lengths = [len(label) for label in y]
  lenghts = torch.LongTensor(lengths)

  y = [torch.LongTensor(y_i) for y_i in y]
  y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

  # 我们现在可以返回我们的变量了。这里我们返回变量的顺序
  # 将与我们在训练循环中读取它们的顺序相一致。
  return x_padded, y_padded, lenghts
```

This function seems long, but it really doesn't have to be. Check out the alternative version below where we remove the extra function declarations and comments.

```python
def _custom_collate_fn(batch, window_size, word_to_ix):
  # Prepare the datapoints
  x, y = zip(*batch)  
  x = [pad_window(s, window_size=window_size) for s in x]
  x = [convert_tokens_to_indices(s, word_to_ix) for s in x]

  # Pad x so that all the examples in the batch have the same size
  pad_token_ix = word_to_ix["<pad>"]
  x = [torch.LongTensor(x_i) for x_i in x]
  x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_ix)

  # Pad y and record the length
  lengths = [len(label) for label in y]
  lenghts = torch.LongTensor(lengths)
  y = [torch.LongTensor(y_i) for y_i in y]
  y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

  return x_padded, y_padded, lenghts  
```

Now, we can see the DataLoader in action.

```python
# Parameters to be passed to the DataLoader
data = list(zip(train_sentences, train_labels))
batch_size = 2
shuffle = True
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=window_size, word_to_ix=word_to_ix)

# Instantiate the DataLoader
loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Go through one loop
counter = 0
for batched_x, batched_y, batched_lengths in loader:
  print(f"Iteration {counter}")
  print("Batched Input:")
  print(batched_x)
  print("Batched Labels:")
  print(batched_y)
  print("Batched Lengths:")
  print(batched_lengths)
  print("")
  counter += 1
```

```python
Iteration 0
Batched Input:
tensor([[ 0,  0, 22,  2,  6, 20, 15,  0,  0],
        [ 0,  0, 19, 16, 12,  8,  4,  0,  0]])
Batched Labels:
tensor([[0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]])
Batched Lengths:
tensor([5, 5])

Iteration 1
Batched Input:
tensor([[ 0,  0, 19,  5, 14, 21, 12,  3,  0,  0],
        [ 0,  0, 10, 13, 11, 17,  0,  0,  0,  0]])
Batched Labels:
tensor([[0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0]])
Batched Lengths:
tensor([6, 4])

Iteration 2
Batched Input:
tensor([[ 0,  0,  9,  7,  8, 18,  0,  0]])
Batched Labels:
tensor([[0, 0, 0, 1]])
Batched Lengths:
tensor([4])
```

The batched input tensors you see above will be passed into our model. On the other hand, we started off saying that our model will be a window classifier. The way our input tensors are currently formatted, we have all the words in a sentence in one datapoint. When we pass this input to our model, it needs to create the windows for each word, make a prediction as to whether the center word is a LOCATION or not for each window, put the predictions together and return.

We could avoid this problem if we formatted our data by breaking it into windows beforehand. In this example, we will instead how our model take care of the formatting.

Given that our window_size is N we want our model to make a prediction on every 2N+1 tokens. That is, if we have an input with 9 tokens, and a window_size of 2, we want our model to return 5 predictions. This makes sense because before we padded it with 2 tokens on each side, our input also had 5 tokens in it!

We can create these windows by using for loops, but there is a faster PyTorch alternative, which is the unfold(dimension, size, step) method. We can create the windows we need using this method as follows:

```python
# Print the original tensor
print(f"Original Tensor: ")
print(batched_x)
print("")

# Create the 2 * 2 + 1 chunks
chunk = batched_x.unfold(1, window_size*2 + 1, 1)
print(f"Windows: ")
print(chunk)
```

```python
Original Tensor: 
tensor([[ 0,  0,  9,  7,  8, 18,  0,  0]])

Windows: 
tensor([[[ 0,  0,  9,  7,  8],
         [ 0,  9,  7,  8, 18],
         [ 9,  7,  8, 18,  0],
         [ 7,  8, 18,  0,  0]]])
```

---

### 模型 (Model)

现在我们已经准备好了数据，可以开始构建我们的模型了。我们已经学习了如何编写自定义的 `nn.Module` 类。我们将在此做同样的事情，并将迄今为止我们学到的所有东西整合在一起。

```python
class WordWindowClassifier(nn.Module):

  def __init__(self, hyperparameters, vocab_size, pad_ix=0):
    super(WordWindowClassifier, self).__init__()
    
    """ 成员变量 """
    self.window_size = hyperparameters["window_size"]
    self.embed_dim = hyperparameters["embed_dim"]
    self.hidden_dim = hyperparameters["hidden_dim"]
    self.freeze_embeddings = hyperparameters["freeze_embeddings"]

    """ 嵌入层
    接收一个包含嵌入索引的张量，并返回对应的嵌入向量。
    输出维度为 (索引数量 * 嵌入维度)。

    如果 freeze_embeddings 为 True，则将嵌入层的参数设置为不可训练。
    如果我们只希望更新嵌入参数以外的模型参数，这非常有用。
    """
    self.embeds = nn.Embedding(vocab_size, self.embed_dim, padding_idx=pad_ix)
    if self.freeze_embeddings:
      self.embeds.weight.requires_grad = False

    """ 隐藏层
    """
    full_window_size = 2 * self.window_size + 1
    self.hidden_layer = nn.Sequential(
      nn.Linear(full_window_size * self.embed_dim, self.hidden_dim), 
      nn.Tanh()
    )

    """ 输出层
    """
    self.output_layer = nn.Linear(self.hidden_dim, 1)

    """ 概率值
    """
    self.probabilities = nn.Sigmoid()

  def forward(self, inputs):
    """
    令 B := batch_size
       L := 经过窗口填充后的句子长度
       D := self.embed_dim
       S := self.window_size
       H := self.hidden_dim
        
    inputs: 一个形状为 (B, L) 的 Token 索引张量
    """
    B, L = inputs.size()

    """
    重塑形状 (Reshaping)。
    输入：形状为 (B, L) 的 LongTensor
    输出：形状为 (B, L~, 2S+1) 的 LongTensor
    """
    # 首先，在输入中为每个单词获取其词窗口。
    token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)
    _, adjusted_length, _ = token_windows.size()

    # 在代码（最起码是注释中）进行内部张量尺寸的安全检查是一个好习惯！
    assert token_windows.size() == (B, adjusted_length, 2 * self.window_size + 1)

    """
    获取嵌入 (Embedding)。
    输入：尺寸为 (B, L~, 2S+1) 的 torch.LongTensor
    输出：尺寸为 (B, L~, 2S+1, D) 的 FloatTensor
    """
    embedded_windows = self.embeds(token_windows)

    """
    重塑形状 (Reshaping)。
    输入：尺寸为 (B, L~, 2S+1, D) 的 FloatTensor
    输出：尺寸为 (B, L~, (2S+1)*D) 的 FloatTensor
    -1 参数表示基于剩余 of axes 自动推断出该维度的具体尺寸。
    """
    embedded_windows = embedded_windows.view(B, adjusted_length, -1)

    """
    第一层 (Layer 1)。
    输入：尺寸为 (B, L~, (2S+1)*D) 的 FloatTensor
    输出：尺寸为 (B, L~, H) 的 FloatTensor
    """
    layer_1 = self.hidden_layer(embedded_windows)

    """
    第二层 (Layer 2)。
    输入：尺寸为 (B, L~, H) 的 FloatTensor
    输出：尺寸为 (B, L~, 1) 的 FloatTensor
    """
    output = self.output_layer(layer_1)

    """
    Sigmoid 转换。
    输入：尺寸为 (B, L~, 1) 的 FloatTensor，包含未归一化的类别分数。
    输出：尺寸为 (B, L~, 1) 的 FloatTensor，包含归一化概率。
    """
    output = self.probabilities(output)
    output = output.view(B, -1)

    return output
```

---

### 训练 (Training)

我们现在准备好把一切拼装在一起了。让我们先准备好我们的数据并初始化模型。然后，我们可以初始化优化器并定义损失函数。这一次，与其像之前那样使用一个预定义的损失函数，我们将定义我们自己的损失函数。

```python
# Prepare the data
data = list(zip(train_sentences, train_labels))
batch_size = 2
shuffle = True
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=window_size, word_to_ix=word_to_ix)

# Instantiate a DataLoader
loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Initialize a model
# It is useful to put all the model hyperparameters in a dictionary
model_hyperparameters = {
    "batch_size": 4,
    "window_size": 2,
    "embed_dim": 25,
    "hidden_dim": 25,
    "freeze_embeddings": False,
}

vocab_size = len(word_to_ix)
model = WordWindowClassifier(model_hyperparameters, vocab_size)

# Define an optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define a loss function, which computes to binary cross entropy loss
def loss_function(batch_outputs, batch_labels, batch_lengths):   
    # Calculate the loss for the whole batch
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())

    # Rescale the loss. Remember that we have used lengths to store the 
    # number of words in each training example
    loss = loss / batch_lengths.sum().float()

    return loss
```

与我们之前的例子不同，这次我们不在每个 epoch 里一次性把所有训练数据传给模型，而是使用批次（batches）。因此，在每个训练 epoch 迭代中，我们还要迭代遍历所有的批次。

```python
# Function that will be called in every epoch
def train_epoch(loss_function, optimizer, model, loader):
  
  # Keep track of the total loss for the batch
  total_loss = 0
  for batch_inputs, batch_labels, batch_lengths in loader:
    # Clear the gradients
    optimizer.zero_grad()
    # Run a forward pass
    outputs = model.forward(batch_inputs)
    # Compute the batch loss
    loss = loss_function(outputs, batch_labels, batch_lengths)
    # Calculate the gradients
    loss.backward()
    # Update the parameteres
    optimizer.step()
    total_loss += loss.item()

  return total_loss


# Function containing our main training loop
def train(loss_function, optimizer, model, loader, num_epochs=10000):

  # Iterate through each epoch and call our train_epoch function
  for epoch in range(num_epochs):
    epoch_loss = train_epoch(loss_function, optimizer, model, loader)
    if epoch % 100 == 0: print(epoch_loss)
```

让我们开始训练吧！

```python
num_epochs = 1000
train(loss_function, optimizer, model, loader, num_epochs=num_epochs)
```

```python
0.3274914249777794
0.24941639229655266
0.1968013420701027
0.1381114460527897
0.11672545038163662
0.09148690290749073
0.07141915801912546
0.05857925023883581
0.04900792893022299
0.04107789508998394
```

---

### 预测 (Prediction)

让我们看看我们的模型做预测的表现有多好。我们先从创建测试数据开始。

```python
# Create test sentences
test_corpus = ["She comes from Paris"]
test_sentences = [s.lower().split() for s in test_corpus]
test_labels = [[0, 0, 0, 1]]

# Create a test loader
test_data = list(zip(test_sentences, test_labels))
batch_size = 1
shuffle = False
window_size = 2
collate_fn = partial(custom_collate_fn, window_size=2, word_to_ix=word_to_ix)
test_loader = torch.utils.data.DataLoader(test_data, 
                                           batch_size=1, 
                                           shuffle=False, 
                                           collate_fn=collate_fn)
```

让我们循环遍历测试用例，看一看模型的表现。

```python
for test_instance, labels, _ in test_loader:
  outputs = model.forward(test_instance)
  print(labels)
  print(outputs)
```

```python
tensor([[0, 0, 0, 1]])
tensor([[0.0339, 0.1031, 0.0500, 0.9770]], grad_fn=<ViewBackward>)
```
