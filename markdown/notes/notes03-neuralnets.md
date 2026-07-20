# CS224n 课程笔记 3：神经网络与反向传播

**课程讲师**：Christopher Manning, Richard Socher  
**作者**：Rohit Mundra, Amani Peddada, Richard Socher, Qiaojing Yan  
**授课学期**：2019 冬季  

---

### 摘要
本篇笔记首先介绍了单层和多层神经网络，以及它们在分类任务中的应用。随后，我们详细讨论了如何使用反向传播（Backpropagation）算法来训练这些网络。反向传播是一种分布式梯度下降技术，利用微积分的链式法则来依次更新模型参数。在深入推导神经网络的数学基础之后，我们将分享在实际训练神经网络时的实用技巧和经验，包括：神经元激活函数、梯度检查、Xavier 参数初始化、学习率衰减策略以及自适应优化器（如 AdaGrad、RMSProp、Adam）等。

---

## 1 神经网络基础 (Neural Networks: Foundations)

在前面的讨论中，我们已经确立了对**非线性分类器**的需求。因为在现实世界中，大多数数据都不是线性可分的，线性分类器在其上的性能非常受限。神经网络作为一类具有非线性决策边界的分类器（如图 1 所示），能够很好地解决此类复杂分类问题。

### 1.1 神经元 (A Neuron)
**神经元（Neuron）**是神经网络的基本构建块。它是一个通用的计算单元，接收 $n$ 个输入并产生单个输出。区分不同神经元行为的关键在于它们的参数（权重）。

最常用的神经元设计之一是**Sigmoid 神经元**（或称为二进制逻辑回归单元）。它接收一个 $n$ 维的输入向量 $x$，并输出一个标量激活值 $a$。该神经元关联着一个 $n$ 维权重向量 $w$ 和一个偏置标量 $b$，其计算公式为：
$$a = \frac{1}{1 + \exp(-(w^\top x + b))} \tag{1}$$

我们也可以将权重和偏置项合并，写成如下等价的向量化形式：
$$a = \frac{1}{1 + \exp\left(- \begin{bmatrix} w^\top & b \end{bmatrix} \begin{bmatrix} x \\ 1 \end{bmatrix} \right)} \tag{2}$$

```
   输入 x_i -------> [ 权重 w_i 缩放 ] ---\
   输入 x_2 -------> [ 权重 w_2 缩放 ] ----+---> [ 求和 + 偏置 b ] ---> [ Sigmoid 激活函数 ] ---> 输出 a
   ...
```

### 1.2 单层神经元 (A Single Layer of Neurons)
我们可以将上述概念扩展到多个神经元。如图 3 所示，同一个输入向量 $x$ 可以被同时喂给多个并行的神经元。

设这些不同神经元的权重为 $\{w^{(1)}, \dots, w^{(m)}\}$，偏置为 $\{b_1, \dots, b_m\}$，对应的激活值为 $\{a_1, \dots, a_m\}$：
$$a_1 = \frac{1}{1 + \exp(w^{(1)\top} x + b_1)}$$
$$\vdots$$
$$a_m = \frac{1}{1 + \exp(w^{(m)\top} x + b_m)}$$

为了让表示法对更复杂的网络有用，我们定义以下矩阵抽象：
$$\sigma(z) = \begin{bmatrix} \frac{1}{1 + \exp(-z_1)} \\ \vdots \\ \frac{1}{1 + \exp(-z_m)} \end{bmatrix}, \quad b = \begin{bmatrix} b_1 \\ \vdots \\ b_m \end{bmatrix} \in \mathbb{R}^m, \quad W = \begin{bmatrix} \text{--- } w^{(1)\top} \text{ ---} \\ \vdots \\ \text{--- } w^{(m)\top} \text{ ---} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

现在我们可以将缩放和偏置相加的中间状态写为：
$$z = Wx + b \tag{3}$$

随后，这层神经元的最终输出激活向量可以表示为：
$$a = \sigma(z) = \sigma(Wx + b) \tag{4}$$

直观上，这些激活值可以被视为指示某些特征组合是否存在。我们可以进一步将这些激活值组合起来，去执行各种分类任务。

### 1.3 前向传播计算 (Feed-forward Computation)
现在，我们来看输入向量 $x \in \mathbb{R}^n$ 是如何通过一层 Sigmoid 单元产生激活值 $a \in \mathbb{R}^m$ 的。我们以自然语言处理中的**命名实体识别（NER）**问题为例：
> *"Museums in Paris are amazing"*（巴黎的博物馆棒极了）

我们希望分类窗口的中心词 *"Paris"* 是否为命名实体。在这样的任务中，我们不仅想捕捉窗口中单词的独立出现，还希望能捕捉词与词之间的某种交互作用。例如，只有当第二个词是 *"in"* 时，第一个词是 *"Museums"* 这一事实才对分类起作用。

这类非线性决策无法直接通过将输入喂给 Softmax 函数来实现，而需要借助于中间隐藏层进行非线性映射。我们可以引入另一组权重矩阵 $U \in \mathbb{R}^{m \times 1}$，根据隐藏层激活值计算出一个未归一化的分类得分 $s$：
$$s = U^\top a = U^\top f(Wx + b) \tag{5}$$
其中 $f$ 是非线性激活函数。

> [!NOTE]
> **维度分析示例**：
> 假设我们用 $4$ 维词向量表示每个单词，并使用长度为 $5$ 的单词窗口作为输入，那么输入向量 $x \in \mathbb{R}^{20}$。
> 若隐藏层包含 $8$ 个 Sigmoid 神经元，并产生 $1$ 个标量输出分数，则：
> - $W \in \mathbb{R}^{8 \times 20}$
> - $b \in \mathbb{R}^8$
> - $U \in \mathbb{R}^{8 \times 1}$
> - $s \in \mathbb{R}$

### 1.4 最大边际目标函数 (Maximum Margin Objective Function)
像大多数机器学习模型一样，神经网络也需要一个优化目标（即损失函数），以此来衡量误差的程度。在这里，我们讨论在 NER 结构中常用的**最大边际目标函数（Max-margin Objective）**。该目标的核心思想是确保正确标注的数据点得分，要比错误标注的数据点得分高出一个安全边际。

在上面的例子中，我们称正确标注的窗口 *"Museums in Paris are amazing"* 的得分得为 $s$；而被破坏（corrupt）的错误标注窗口 *"Not all museums in Paris"* 的得分为 $s_c$（下标 $c$ 表示已被损坏）。

我们自然希望最大化两者的差值 $(s - s_c)$。但为了实现更稳健的分类，我们定义一个安全边际 $\Delta = 1$。只有当正确样本的得分没有比错误样本的得分高出至少 $1$ 时，系统才会计算惩罚。因此，对单个样本的损失定义为：
$$J = \max(1 + s_c - s, 0) \tag{6}$$

在整个训练过程中，我们的优化目标是最小化所有训练窗口上的总损失：
$$\text{minimize } J = \sum \max(1 + s_c - s, 0) \tag{7}$$
其中 $s_c = U^\top f(Wx_c + b)$ 且 $s = U^\top f(Wx + b)$。

---

## 1.5 元素级反向传播推导 (Training with Backpropagation – Elemental)

当损失 $J > 0$ 时，我们需要通过梯度下降（如 SGD）更新模型中的参数 $\theta$：
$$\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_{\theta^{(t)}} J$$

**反向传播（Backpropagation）**是一种利用微积分链式法则来计算神经网络各层参数梯度的技术。为了理解它，我们来看一个 4-2-1 结构的玩具网络：

```
   输入层 (k=1)      隐藏层 (k=2)       输出层 (k=3)
      x_1 ---------\
      x_2 ----------+---> 隐藏元 z_1^(2) ---> a_1^(2) ---\
      x_3 ----------+---> 隐藏元 z_2^(2) ---> a_2^(2) ----+---> 输出 s
      x_4 ---------/
```

- $a_j^{(k)}$ 代表第 $k$ 层的第 $j$ 个神经元的激活值（输出）。特别地，对于输入层（$k=1$），有 $a_j^{(1)} = x_j$。
- $z_j^{(k)}$ 代表第 $k$ 层的第 $j$ 个神经元在被激活函数处理前的加权输入之和。
- $W^{(k)}$ 是将第 $k$ 层的输出映射到第 $k+1$ 层输入的参数矩阵。例如，在前面的例子中，$W^{(1)} = W$，$W^{(2)} = U^\top$。
- 我们将反向传播至 $z_j^{(k)}$ 的误差信号定义为 $\delta_j^{(k)} \equiv \frac{\partial J}{\partial z_j^{(k)}}$。

假设当前损失 $J = 1 + s_c - s > 0$。由公式易知 $\frac{\partial J}{\partial s} = -1$。为简化推导，我们先求解输出得分 $s$ 关于输入层权重 $W_{ij}^{(1)}$ 的偏导数。

由于 $W_{ij}^{(1)}$ 在前向传播中仅对 $z_i^{(2)}$ 有贡献，因此根据链式法则：
$$\frac{\partial s}{\partial W_{ij}^{(1)}} = \frac{\partial s}{\partial z_i^{(2)}} \frac{\partial z_i^{(2)}}{\partial W_{ij}^{(1)}} \tag{8}$$

我们知道 $z_i^{(2)} = \sum_k a_k^{(1)} W_{ik}^{(1)} + b_i^{(1)}$，因此：
$$\frac{\partial z_i^{(2)}}{\partial W_{ij}^{(1)}} = a_j^{(1)} \tag{9}$$

对于第一部分，利用输出层关系 $s = \sum_p a_p^{(2)} W_p^{(2)}$：
$$\frac{\partial s}{\partial z_i^{(2)}} = \frac{\partial s}{\partial a_i^{(2)}} \frac{\partial a_i^{(2)}}{\partial z_i^{(2)}} = W_i^{(2)} f'(z_i^{(2)}) \tag{10}$$

将式 (9) 和式 (10) 代入式 (8)，我们得到：
$$\frac{\partial s}{\partial W_{ij}^{(1)}} = \left( W_i^{(2)} f'(z_i^{(2)}) \right) a_j^{(1)} = \delta_i^{(2)} \cdot a_j^{(1)} \tag{11}$$
其中 $\delta_i^{(2)}$ 代表从第二层第 $i$ 个节点向后传播的误差。

#### 反向传播的“误差分配/分布式流动”直观解释
以更新 $W_{14}^{(1)}$ 为例，误差可以通过流图的形式在网络中向后传递：
1. 输出层激活节点 $a_1^{(3)}$ 传回初始误差信号 $1$。
2. 乘以局部梯度 $f'(z_1^{(3)})$（在此玩具模型中为 $1$），得到 $\delta_1^{(3)} = 1$。
3. 误差信号沿着连接权重 $W_1^{(2)}$ 回流，在隐藏激活层 $a_1^{(2)}$ 处接收到的误差为 $\delta_1^{(3)} \times W_1^{(2)} = W_1^{(2)}$。
4. 将误差穿过激活函数，乘以局部梯度 $f'(z_1^{(2)})$，得到 $z_1^{(2)}$ 处的误差 $\delta_1^{(2)} = f'(z_1^{(2)}) W_1^{(2)}$。
5. 最终，分配给参数 $W_{14}^{(1)}$ 的梯度就是该处误差乘以其负责的前向输入值 $a_4^{(1)}$。即最终梯度为 $a_4^{(1)} f'(z_1^{(2)}) W_1^{(2)}$。

#### 偏置的更新 (Bias Updates)
偏置项 $b_i^{(k)}$ 在数学上相当于前向输入永远为 $1$ 的权重。因此，偏置对应的梯度直接等于流回该神经元节点的误差本身：
$$\frac{\partial J}{\partial b_i^{(k)}} = \delta_i^{(k)} \tag{12}$$

#### 误差递推公式
我们可以将误差从第 $k$ 层向第 $k-1$ 层回传的过程概括为：
$$\delta_j^{(k-1)} = f'(z_j^{(k-1)}) \sum_i \delta_i^{(k)} W_{ij}^{(k-1)} \tag{13}$$

---

## 1.6 向量化反向传播 (Training with Backpropagation – Vectorized)

为了在大规模计算中提高速度，我们需要将上述元素级公式推广到矩阵和向量形式。

对于权重矩阵 $W^{(k)}$，由于每一个元素 $W_{ij}^{(k)}$ 对应的导数为 $\delta_i^{(k+1)} a_j^{(k)}$，我们可以将整张矩阵的梯度写成两个向量的**外积（Outer Product）**：
$$\nabla_{W^{(k)}} J = \delta^{(k+1)} (a^{(k)})^\top \tag{14}$$

误差向量 $\delta^{(k)}$ 的层间递归关系也可以写为向量化形式：
$$\delta^{(k)} = f'(z^{(k)}) \circ \left( (W^{(k)})^\top \delta^{(k+1)} \right) \tag{15}$$
其中，$\circ$ 符号代表向量的**按元素相乘（Hadamard 积 / Element-wise Product）**。

> [!TIP]
> **计算效率提示**：
> - 向量化的实现能够极大利用 Python (NumPy) 等矩阵加速库，运算速度要远远快于使用嵌套循环的元素级实现。
> - 在反向传播的计算过程中，保存并重用 $\delta^{(k+1)}$ 来推导 $\delta^{(k)}$ 是消除冗余计算、使反向传播时间复杂度保持可接受的核心。

---

## 2 神经网络训练实用技巧 (Neural Networks: Tips and Tricks)

### 2.1 梯度检查 (Gradient Check)
虽然反向传播公式（解析导数）运算极快，但在编写求导代码时极易犯错。为了验证解析梯度的正确性，我们可以通过**中心差分公式**来数值近似梯度：
$$f'(\theta_i) \approx \frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2\epsilon} \tag{16}$$
其中 $\epsilon$ 是一个很小的数（通常设为 $10^{-5}$）。$\theta^{(i+)}$ 是将参数向量 $\theta$ 的第 $i$ 个维度微调 $+\epsilon$ 后的参数。

```python
def eval_numerical_gradient(f, x):
    """
    一个简单的数值梯度计算实现
    - f: 损失函数，接收 x 并返回标量损失
    - x: 当前待求导的参数向量 (numpy array)
    """
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        
        x[ix] = old_value + h
        fxh_left = f(x)
        
        x[ix] = old_value - h
        fxh_right = f(x)
        
        x[ix] = old_value  # 恢复原值 (非常重要!)
        
        # 计算偏导数并保存到对应位置
        grad[ix] = (fxh_left - fxh_right) / (2 * h)
        it.iternext()
        
    return grad
```
*代码段 2.1：数值梯度检查的简单实现*

> [!WARNING]
> 数值梯度检查每次都需要对网络进行两次完整的前向传播，其计算开销非常高，因此**它仅用于调试和验证解析梯度求导代码的正确性**，绝对不要直接用于训练优化过程。

### 2.2 正则化 (Regularization)
为了减小过拟合，我们常在原始损失函数中引入 $L_2$ 正则化惩罚（也称权重衰减）：
$$J_R = J + \frac{\lambda}{2} \sum_{i=1}^L \|W^{(i)}\|_F^2 \tag{17}$$
其中，$\|U\|_F$ 代表矩阵 $U$ 的 **Frobenius 范数**，其定义为矩阵所有元素平方和的开方：
$$\|U\|_F = \sqrt{\sum_i \sum_j U_{ij}^2} \tag{18}$$
$\lambda$ 是平衡任务损失与正则化强度的超参数。正则化项倾向于让权重矩阵内的参数值向 $0$ 靠拢，有效限制了模型的有效容量。
注意：**偏置项 $b$ 通常不参与正则化**，因为偏置仅用于控制神经元是否被激活，不涉及输入特征权重的控制，对其加正则化往往会导致欠拟合。

### 2.3 丢弃法 (Dropout)
Dropout 是一种极具威力的正则化技术。在训练期间，我们在每一次前向/反向传播中，以概率 $(1-p)$ 随机将一层中的部分神经元输出设为 $0$。在测试阶段，我们则恢复使用完整的网络进行预测。

这一方法能够工作的一个直观解释是：它相当于在训练过程中隐式训练了指数级个共享权重的子网络，并在测试时通过完整网络对它们的预测取了平均。

> [!IMPORTANT]
> **测试时的缩放问题**：
> 在测试时，由于所有神经元都参与了前向计算，神经元的预期输入会比训练时（有 Dropout 发生）大出 $\frac{1}{p}$ 倍。为了保持期望值一致，我们通常会在**训练阶段**前向传播时直接将未被丢弃的激活值除以 $p$（称为 **Inverted Dropout**），这样测试时就无需进行任何额外缩放。

### 2.4 激活函数单元 (Neuron Units)
为了引入非线性，隐藏层神经元会使用激活函数。以下是几种常用的非线性激活函数及其导数：

#### 1. Sigmoid
$$\sigma(z) = \frac{1}{1 + \exp(-z)} \tag{19}$$
- 输出范围：$(0, 1)$。
- 导数：$\sigma'(z) = \sigma(z)(1 - \sigma(z))$。
- 缺点：容易发生梯度消失（Saturating Gradients），且输出是非零中心（Non-zero Centered）的。

#### 2. Tanh
$$\tanh(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)} = 2\sigma(2z) - 1 \tag{20}$$
- 输出范围：$(-1, 1)$。
- 导数：$\tanh'(z) = 1 - \tanh^2(z)$。
- 优点：由于输出是零中心的，在实践中收敛速度往往快于 Sigmoid。

#### 3. Hard Tanh
$$\operatorname{hardtanh}(z) = \begin{cases} -1 & \text{if } z < -1 \\ z & \text{if } -1 \le z \le 1 \\ 1 & \text{if } z > 1 \end{cases} \tag{21}$$
- 导数：
$$\operatorname{hardtanh}'(z) = \begin{cases} 1 & \text{if } -1 \le z \le 1 \\ 0 & \text{otherwise} \end{cases}$$
- 优点：计算开销极低。

#### 4. Softsign
$$\operatorname{softsign}(z) = \frac{z}{1 + |z|} \tag{22}$$
- 导数：
$$\operatorname{softsign}'(z) = \frac{1}{(1 + |z|)^2}$$
- 优点：比 Hard Tanh 的截断更平滑，且不像 Tanh 那么容易饱和。

#### 5. ReLU (Rectified Linear Unit)
$$\operatorname{rect}(z) = \max(z, 0) \tag{23}$$
- 导数：
$$\operatorname{rect}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$$
- 优点：在正区间不饱和，且计算极其简单，是计算机视觉与深度模型中的常用首选。

#### 6. Leaky ReLU
$$\operatorname{leaky}(z) = \max(z, k \cdot z) \quad (0 < k < 1) \tag{24}$$
- 导数：
$$\operatorname{leaky}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ k & \text{otherwise} \end{cases}$$
- 优点：在负区间仍允许一部分微弱的梯度回传，防止神经元“死掉”。

### 2.5 数据预处理 (Data Preprocessing)
- **中心化（Mean Subtraction）**：将输入矩阵 $X$ 减去其均值。需要注意的是，均值应该**仅在训练集上计算**，然后应用到训练、验证和测试集中。
- **标准化（Normalization）**：将各个特征维度除以其标准差，使其动态范围一致。
- **白化（Whitening）**：通过奇异值分解（SVD）对数据进行去相关，使特征的协方差矩阵为单位矩阵。但由于计算成本大，现已不太常用。

### 2.6 参数初始化 (Parameter Initialization)
将权重矩阵全部初始化为零会引发对称性问题，导致同一层的神经元在前向和反向传播中表现完全一致。通常应将权重初始化为零附近的小随机数。

对于 Sigmoid 和 Tanh 激活函数，为了保持各层激活值和梯度的方差在传递中不衰减，推荐使用 **Xavier 初始化**。对于 $W \in \mathbb{R}^{n^{(l+1)} \times n^{(l)}}$：
$$W \sim U \left[ -\frac{\sqrt{6}}{\sqrt{n^{(l)} + n^{(l+1)}}}, \frac{\sqrt{6}}{\sqrt{n^{(l)} + n^{(l+1)}}} \right] \tag{25}$$
其中 $n^{(l)}$ 是输入的神经元数（fan-in），$n^{(l+1)}$ 是输出的神经元数（fan-out）。此时偏置项初始化为 $0$。

### 2.7 学习率与退火策略 (Learning Strategies & Annealing)
设置过大的学习率 $\alpha$ 容易导致损失函数发散（Divergence），而太小则会导致训练过慢或陷入局部最小值。

常用的学习率**退火（Annealing）**策略有：
- **分阶段衰减**：每隔 $n$ 个 epoch 将学习率乘以一个衰减因子。
- **指数衰减**：$\alpha(t) = \alpha_0 e^{-kt}$。
- **倒数衰减**：
$$\alpha(t) = \alpha_0 \frac{\tau}{\max(t, \tau)} \tag{26}$$
其中 $\alpha_0$ 和 $\tau$ 是可调参数，表示在第 $\tau$ 步之后开始进行学习率衰减。

### 2.8 动量更新 (Momentum Updates)
动量法引入了“速度”的概念，能够帮助参数在梯度方向一致的维度上加速更新，并在波动较大的维度上抑制震荡。
```python
# 动量更新算法
v = mu * v - alpha * grad_x
x += v
```
*代码段 2.2：动量更新的伪代码*

### 2.9 自适应优化方法 (Adaptive Optimization Methods)
- **AdaGrad**：根据参数历史梯度的平方和，为每个参数自适应调整学习率。频繁更新的参数学习率下降快，稀疏更新的参数能保持较大步长：
$$\theta_{t,i} = \theta_{t-1,i} - \frac{\alpha}{\sqrt{\sum_{\tau=1}^t g_{\tau,i}^2 + \epsilon}} g_{t,i} \tag{27}$$
```python
cache += dx ** 2
x += - learning_rate * dx / np.sqrt(cache + 1e-8)
```
*代码段 2.3：AdaGrad 的 Naive 实现*

- **RMSProp**：对 AdaGrad 进行改进，使用指数移动平均来限制累积的平方梯度，避免训练后期学习率单调减小至 0：
```python
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```
*代码段 2.4：RMSProp 核心代码*

- **Adam**：结合了动量法（一阶矩）和 RMSProp（二阶矩）自适应学习率的长处，是目前最常用的主流优化器：
```python
m = beta1 * m + (1 - beta1) * dx
v = beta2 * v + (1 - beta2) * (dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```
*代码段 2.5：Adam 核心代码*
