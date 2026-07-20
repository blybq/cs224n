## 页面 1: 神经网络基础

自然语言处理与深度学习 (Natural Language Processing with Deep Learning)

CS224N/Ling284
Diyi Yang
第 3 讲：神经网络基础 (Lecture 3: Neural Network Foundations)

## 页面 2: 课程大纲

**第 3 讲：神经网络基础**
1. 课程行政教务 (3 分钟) + Word2vec 评估回顾 (7 分钟)
2. 引入神经网络 (10 分钟)
3. 矩阵微积分 (25 分钟)
4. 反向传播算法 (35 分钟)

**核心目标**：掌握神经网络是如何通过反向传播 (backpropagation) 进行训练的数学原理与实际实现。

## 页面 3: 课程教务信息

* **作业 2 (Assignment 2)**：旨在确保您真正理解神经网络背后的数学原理……然后我们再让软件框架来自动计算！它还会教授我们如何进行**依存句法分析 (dependency parsing)**。
* 这对部分同学来说会是比较艰难的一周！$\rightarrow$ 如果需要帮助，请务必寻求协助：参加答疑时间！阅读大纲上的教程材料！
* **PyTorch 教程**：本周五 1:30-2:20pm，NVIDIA Aud.。这是入门主流深度学习库 PyTorch 的绝佳机会！
* **海报展示环节 (Poster Session)**：3 月 16 日 12:15-3:15pm，AOERC。线下注册学生必须到场，其他特殊情况请参阅 Ed 论坛贴（第 3 周前截止申请）。

## 页面 4: 复习：如何评估词向量？

* 与 NLP 的通用评估方式一致：**内部评估 (Intrinsic)** vs. **外部评估 (Extrinsic)**。
* **内部评估**：
  * 在特定的/中间子任务上进行评估。
  * 计算速度快。
  * 有助于理解子系统。
  * 除非确立了与实际终极任务的相关性，否则其有效性不够明确。
* **外部评估**：
  * 在真实的终极任务上进行评估。
  * 评估准确率可能需要很长时间。
  * 若表现不佳，难以断定是该子系统的问题、交互问题还是其他子系统的问题。
  * 如果仅替换其中一个子系统就能提高实际任务准确率 $\Rightarrow$ 成功提升！

## 页面 5: 词向量内部评估：词类比 (Word Analogies)

* **词向量类比**：
  * 通过评估向量相加减后的余弦距离，检验其捕获直观语义和句法类比问题的能力。
  * 例如：$$\text{man} : \text{woman} \Longleftrightarrow \text{king} : \text{?}$$
  * 即寻找满足：$$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}$$ 距离最近的词向量。
  * *在检索搜索中，需要将输入的词汇从搜索结果中排除（！）。*
* **问题**：如果词义信息存在本不是线性的关系怎么办？

## 页面 6: GloVe 词向量空间可视化

* 展示了在训练后的向量空间中，不同词对（如人称代词、动词时态）之间呈现出的平行线性映射结构。

## 页面 7: 词义相似度评估

* **词向量距离与人类主观判断的相关性**。
* **常用基准数据集示例**：WordSim353

| Word 1 | Word 2 | Human (mean score) |
| :--- | :--- | :--- |
| tiger | cat | 7.35 |
| tiger | tiger | 10.00 |
| book | paper | 7.46 |
| computer | internet | 7.58 |
| plane | car | 5.77 |
| professor | doctor | 6.62 |
| stock | phone | 1.62 |
| stock | CD | 1.31 |
| stock | jaguar | 0.92 |

## 页面 8: 相关性评估

* 评估词向量计算出的余弦相似度与人类评分之间的相关系数（如 Spearman 相关系数）。

## 页面 9: 词向量外部评估

* 一个优秀词向量能直接提供帮助的经典示例：**命名实体识别 (Named Entity Recognition, NER)** —— 识别文本中对人名、组织或地理位置的引用：
  * *“Chris Manning lives in Palo Alto.”*

## 页面 10: 命名实体识别 (NER)

* **任务目标**：在文本中寻找并分类名称，通过给词元标注标签。例如：
  * *Last night , **Paris Hilton** wowed in a sequin gown .*
    * 标注：`Paris (PER)` `Hilton (PER)`（人名）
  * *Samuel Quinn was arrested in the **Hilton Hotel** in **Paris** in **April 1989** .*
    * 标注：`Samuel Quinn (PER)`, `Hilton Hotel (ORG/FAC)`, `Paris (LOC)`, `April 1989 (DATE)`
* **潜在用途**：
  * 追踪文档中对特定实体的提及。
  * 用于问答系统（答案通常是命名实体）。
  * 将情感分析与正在讨论的实体关联起来。
  * 通常后续会进行**实体链接/规范化 (Entity Linking/Canonicalization)** 以并入知识库（如 Wikidata）。

## 页面 11: 简单的 NER 方案：基于窗口分类的逻辑回归分类器

* **基本思想**：在中心词的邻近上下文窗口内对每个词进行分类。
* 在手工标注的数据上训练逻辑回归分类器，基于窗口内拼接 (concatenation) 起来的词向量，来对中心词进行二分类（是/否某类实体）。
* *在实际中，我们通常使用多分类 Softmax，但这里我们尝试保持模型简单。*
* **示例**：在窗口长度为 2 的句子上下文中，将 “Paris” 分类为是否是地理位置（LOC）：
  * 文本：*the museums in **Paris** are amazing to see .*
  * 窗口特征向量：$$\mathbf{x}_{\text{window}} = [x_{\text{museums}}, x_{\text{in}}, x_{\text{Paris}}, x_{\text{are}}, x_{\text{amazing}}]^T$$
  * 结果向量：$$\mathbf{x}_{\text{window}} = x \in \mathbb{R}^{5d}$$
* 对句子中的每个词的向量运行分类器，以完成全句单词分类。

## 页面 12: 分类算法复习与符号表示

* **监督学习 (Supervised learning)**：我们有一个包含样本的训练数据集：$$\lbrace x_i, y_i \rbrace_{i=1}^N$$
* $x_i$ 是输入（例如单词索引或向量、句子、文档等），维度为 $d$。
* $y_i$ 是我们要预测的标签（$C$ 个类别之一），例如：
  * 情感类别（正向/负向）
  * 命名实体类别
  * 买入/卖出决策
  * 预测下一个词（语言模型）
  * 序列到序列生成（如机器翻译）

## 页面 13: 神经网络分类

* **典型的机器学习/统计 Softmax 分类器**：
  * 学习到的参数 $\theta$ 仅仅是权重矩阵 $W$ 中的元素，而不包含输入表示 $x$（输入通常是稀疏的符号特征）。
  * 分类器给出**线性决策边界 (linear decision boundary)**，这可能会有限制。
* **神经网络分类器的不同之处**：
  * 我们**同时学习**权重 $W$ 以及词的分布式表征 (distributed representations) $x$。
  * 词向量 $x$ 重新表征了独热向量，将它们移动到中间隐藏层的向量空间中，以便使用（线性）Softmax 分类器进行轻松分类。
  * 从概念上讲，我们有一个嵌入层 (embedding layer)：$x = L \cdot e$。
  * 我们使用**深层网络（多隐藏层）**，这使得我们可以对数据进行多次重新表征和组合，从而提供**非线性分类器**。
  * *但在最终层表征面前，它通常依然是线性的。*

## 页面 14: NER 二分类（判断中心词是否为地理位置）

* 我们进行监督训练，并在其为地理位置时期望获得高分。
  $$J_t(\theta) = \sigma(s) = \frac{1}{1 + e^{-s}}$$
  $$s = u^T a$$
  $$a = f(Wx + b)$$
  $$x = \mathbf{x}_{\text{window}} \in \mathbb{R}^{5d}$$
* 其中：
  * $f$ 是按元素计算的**非线性激活函数 (non-linear activation function)**，例如 Sigmoid (logistic)、$\tanh$ 或 ReLU。
  * $x$ 是独热词表示经过嵌入层拼接而成的特征向量。

## 页面 15: 经典与现代非线性激活函数

* $\tanh$ 只是一个缩放并平移后的 Sigmoid 函数（陡度是其 2 倍，值域为 $[-1, 1]$）：
  $$\tanh(z) = 2 \cdot \text{logistic}(2z) - 1$$
* 虽然 Sigmoid 和 $\tanh$ 仍在使用（例如 Sigmoid 用于计算概率），但在深层网络中，通常首选的尝试是 **ReLU (Rectified Linear Unit)**：它训练迅速，并且由于具有良好的梯度回流特性，表现极其出色。
  $$\text{ReLU}(z) = \max(z, 0)$$
* ReLU 具有负值“死亡区”，最近的许多提案对此进行了改进（如 Leaky ReLU / Parametric ReLU）。

## 页面 16: 现代 Transformer 中的激活函数

* **GELU (Gaussian Error Linear Unit)**：在 Transformer 中被广泛使用。
  $$\text{GELU}(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0,1)$$
  $$\text{GELU}(x) \approx x \cdot \text{logistic}(1.702x)$$
* **SiLU / Swish (Sigmoid Linear Unit)**：
  $$\text{SiLU}(x) = x \cdot \sigma(x) \quad \text{Swish}(x) = x \cdot \sigma(\beta x)$$
* **GLU (Gated Linear Unit，门控线性单元)**：使用门控机制。
  $$\text{GLU}(x) = (xV + v) \otimes \sigma(xW + b)$$
* **SwiGLU (Swish-Gated Linear Unit)**：广泛应用于现代大语言模型（如 Llama 3, Qwen 等）。
  $$\text{SwiGLU}(x) = (xV + c) \otimes \text{Swish}_{\beta}(xW + b)$$

## 页面 17: 为什么神经网络需要非线性激活函数？

* 神经网络进行函数逼近（如回归或分类）。
* **如果没有非线性激活函数，深度神经网络无法执行比单层线性变换更复杂的运算**。
* 无论叠加多少层，都可以编译合并为单一的线性变换：$$W_1 (W_2 x) = (W_1 W_2) x = W x$$
* 但是，一旦引入了非线性激活函数的多隐藏层，神经网络就可以逼近**任何复杂的非线性函数**！

## 页面 18: 使用“交叉熵损失”进行训练

我们在 PyTorch 中常常使用交叉熵损失 (Cross Entropy Loss)！
* 之前我们的目标是最大化正确类别 $y$ 的概率，或者等价于最小化该类别的负对数概率。
* 现在我们可以用信息论中的**交叉熵 (Cross Entropy)** 概念来重新表述它。
* 设真实概率分布为 $p$；我们计算出的模型预测概率分布为 $q$。
* 交叉熵定义为：
  $$H(p, q) = -\sum_{c=1}^{C} p_c \log q_c$$
* 假设真实（或标准、黄金）概率分布在正确类别上为 1，在其他地方均为 0（独热分布）：$p = [0, \dots, 0, 1, 0, \dots, 0]$。此时，求和中唯一剩下的项就是正确类别 $y_i$ 的负对数概率：
  $$-\log q_{y_i} = -\log P(y_i | x_i)$$

## 页面 19: 记住：随机梯度下降 (SGD)

参数更新方程：
$$\theta^{\text{new}} = \theta^{\text{old}} - \alpha \nabla_{\theta} J(\theta)$$
即对于每个参数 $\theta_j$：
$$\theta_j^{\text{new}} = \theta_j^{\text{old}} - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

在深度学习中，$\theta$ 也包括数据的内部表示（例如词向量）！
我们该如何计算 $\nabla_{\theta} J(\theta)$？
1. **手动推导 (By hand)**
2. **算法自动计算**：**反向传播算法 (Backpropagation Algorithm)**

其中 $\alpha$ 是步长或学习率。

## 页面 20: 手动计算梯度

* **矩阵微积分 (Matrix calculus)**：推导完全向量化的梯度。
  * *“如果使用矩阵，多元微积分就和单元微积分一样简单。”*
  * 比非向量化的逐个元素求导要快得多，且在编程实现中极其有用。
* 当然，做一次非向量化的逐元素求导有利于建立直观的物理理解（回顾第二讲的示例）。
* 课程讲义和矩阵微积分指南（Matrix Calculus Notes）中包含这些内容的详细推导。
* 推荐复习 Math 51（其有在线教材：[Math 51 Textbook](http://web.stanford.edu/class/math51/textbook.html)）。

## 页面 21: 梯度 (Gradients)

* 给定一个单输入单输出函数：
  $$f(x) = x^3$$
* 其梯度（斜率）就是它的导数：
  $$\frac{df}{dx} = 3x^2$$
* 意义：“如果我们将输入微调一点点，输出会发生多大变化？”
  * 在 $x = 1$ 时，变化大约是输入的 3 倍：$1.01^3 \approx 1.03$
  * 在 $x = 4$ 时，变化大约是输入的 48 倍：$4.01^3 \approx 64.48$

## 页面 22: 多元梯度

* 给定一个单输出、$n$ 维输入的函数 $f: \mathbb{R}^n \to \mathbb{R}$。
* 其梯度是一个由关于每个输入的偏导数 (partial derivatives) 组成的向量：
  $$\nabla_x f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

## 页面 23: 雅可比矩阵：梯度的推广

* 给定一个具有 $m$ 维输出和 $n$ 维输入的函数 $f: \mathbb{R}^n \to \mathbb{R}^m$。
* 其**雅可比矩阵 (Jacobian Matrix)** 是一个 $m \times n$ 的偏导数矩阵：
  $$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = J = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix}$$
  其中每一行是输出分量关于输入向量的梯度。

## 页面 24: 链式法则 (Chain Rule)

* 对于单变量复合函数的链式法则：**相乘导数**。
* 对于多变量复合函数的链式法则：**相乘雅可比矩阵 (Multiply Jacobians)**：
  若 $\mathbf{z} = g(\mathbf{y})$ 且 $\mathbf{y} = f(\mathbf{x})$，则：
  $$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

## 页面 25: 雅可比矩阵示例：按元素激活函数

* 设激活函数是按元素独立作用的：$\mathbf{h} = f(\mathbf{z})$，即 $h_i = f(z_i)$。
* 因为每个输出分量 $h_i$ 仅仅依赖于对应的输入分量 $z_i$，因此其雅可比矩阵是一个**对角矩阵 (diagonal matrix)**：
  $$J = \text{diag}(f'(z_1), f'(z_2), \dots, f'(z_n))$$

## 页面 26: 激活函数的雅可比矩阵计算（二）

## 页面 27: 激活函数的雅可比矩阵计算（三）

## 页面 28: 激活函数的雅可比矩阵计算（四）

## 页面 29: 激活函数的雅可比矩阵计算（五）

## 页面 30: 其他常用的雅可比矩阵

* 建议在课后自行推导以下常见操作的雅可比矩阵以作练习，并与课程讲义核对答案。

## 页面 31: 其他常用的雅可比矩阵（续）

## 页面 32: 其他常用的雅可比矩阵（续）
* 细微说明：这是纯数学意义上的雅可比矩阵。稍后我们讨论“形状约定 (shape convention)”，根据形状约定，对偏置项和权重的梯度的输出形状会进行转置或调整。

## 页面 33: 其他常用的雅可比矩阵（续）

## 页面 34: 回到我们的神经网络！

回顾网络前向公式：
$$\mathbf{x}_{\text{window}} = [x_{\text{museums}}, x_{\text{in}}, x_{\text{Paris}}, x_{\text{are}}, x_{\text{amazing}}]^T$$
$$z = Wx + b$$
$$h = f(z)$$
$$s = u^T h$$

## 页面 35: 回到我们的神经网络！

* 让我们求关于输入 $x$ 的导数：$$\frac{\partial s}{\partial x}$$
* *在实际中，我们最终关心的是损失 $J$ 的梯度，但为了叙述简便起见，我们在此计算得分 $s$ 的梯度。*

## 页面 36: 第一步：将公式拆解为简单的局部片段

仔细定义您的变量并跟踪它们的维度！
* 输入特征：$x \in \mathbb{R}^{5d}$
* 仿射变换：$z = Wx + b \in \mathbb{R}^{D_h}$（其中 $W \in \mathbb{R}^{D_h \times 5d}$）
* 激活函数：$h = f(z) \in \mathbb{R}^{D_h}$
* 得分输出：$s = u^T h \in \mathbb{R}$（其中 $u \in \mathbb{R}^{D_h}$）

## 页面 37: 第二步：应用链式法则

$$\frac{\partial s}{\partial x} = \frac{\partial s}{\partial h} \frac{\partial h}{\partial z} \frac{\partial z}{\partial x}$$

## 页面 38: 第二步：应用链式法则（细节）

## 页面 39: 第二步：应用链式法则（细节）

## 页面 40: 第二步：应用链式法则（细节）

## 页面 41: 第三步：写出局部的雅可比矩阵

从前面的拆解中，我们可以依次写出各部分的导数形式。

## 页面 42: 写出局部的雅可比矩阵（二）

* 第一项：$$\frac{\partial s}{\partial h} = u^T \in \mathbb{R}^{1 \times D_h}$$

## 页面 43: 写出局部的雅可比矩阵（三）

* 第二项：$$\frac{\partial h}{\partial z} = \text{diag}(f'(z)) \in \mathbb{R}^{D_h \times D_h}$$
* 前两项结合：$$\frac{\partial s}{\partial h} \frac{\partial h}{\partial z} = u^T \text{diag}(f'(z))$$

## 页面 44: 写出局部的雅可比矩阵（四）

* 第三项：$$\frac{\partial z}{\partial x} = W \in \mathbb{R}^{D_h \times 5d}$$

## 页面 45: 写出局部的雅可比矩阵（五）

最终组合起来：
$$\frac{\partial s}{\partial x} = u^T \text{diag}(f'(z)) W$$

利用**哈达玛积 (Hadamard product, $\odot$)** 可以将对角矩阵乘法重写为按元素对应相乘的形式（计算更高效）：
$$\frac{\partial s}{\partial x} = (u \odot f'(z))^T W$$
* 这里的 $\odot$ 表示两个维度相同的向量进行逐元素对应相乘。

## 页面 46: 重用计算结果

* 假设我们现在想要计算关于偏置项 $b$ 的导数：$$\frac{\partial s}{\partial b}$$
* 再次应用链式法则：
  $$\frac{\partial s}{\partial b} = \frac{\partial s}{\partial h} \frac{\partial h}{\partial z} \frac{\partial z}{\partial b}$$

## 页面 47: 重用计算结果（二）

* 注意观察前两项：
  $$\frac{\partial s}{\partial b} = \underbrace{\frac{\partial s}{\partial h} \frac{\partial h}{\partial z}}_{\text{与前面计算完全相同！}} \frac{\partial z}{\partial b}$$
* 它们是完全相同的项！为了避免重复计算，我们应当将其缓存重用。

## 页面 48: 重用计算结果（三）

* 我们引入上游梯度（“误差信号”，error signal）定义为 $\delta$：
  $$\delta = \frac{\partial s}{\partial h} \frac{\partial h}{\partial z} = u^T \text{diag}(f'(z)) = u \odot f'(z)$$
* 此时：
  $$\frac{\partial s}{\partial b} = \delta^T \frac{\partial z}{\partial b}$$
  因为 $\frac{\partial z}{\partial b} = I$（单位矩阵），所以：
  $$\frac{\partial s}{\partial b} = \delta^T$$

## 页面 49: 关于矩阵的导数：输出形状问题

* 关于参数矩阵 $W$ 的导数 $\frac{\partial s}{\partial W}$ 会是什么样子？
* 1 个标量输出，$n \times m$ 个输入参数：这会是一个 $1 \times (n \cdot m)$ 的长雅可比行向量吗？
* 用这种长向量来执行 SGD 梯度更新将会非常不便。

## 页面 50: 关于矩阵的导数与形状约定 (Shape Convention)

* 为了便于编写程序，我们脱离纯数学的展开，采用**形状约定 (Shape Convention)**：**梯度的形状必须与对应参数的形状完全一致**！
* 既然 $W$ 的形状是 $D_h \times 5d$ 的矩阵，那么梯度 $\frac{\partial s}{\partial W}$ 的形状也应该是 $D_h \times 5d$ 维矩阵：
  $$\frac{\partial s}{\partial W} = \delta x^T$$

## 页面 51: 关于矩阵的导数计算

* 最终推导得出：
  $$\frac{\partial s}{\partial W} = \delta x^T$$
* 其中 $\delta$ 是到达隐藏激活层 $z$ 的上游梯度（“误差信号”），$x$ 是输入信号（即局部前向输入）。
* 这正是 $\delta$ 和 $x$ 的**外积 (outer product)**！

## 页面 52: 为什么需要转置？

* **直观的小窍门**：转置是为了让矩阵乘法的维度能够匹配！这是检查手算推导是否出错的极好方式。
* **物理机制说明**：因为每个输入分量都会与每个输出分量发生作用，所以求导后形成了外积。详见课程讲义。

## 页面 53: 反向传播中隐藏层局部梯度的推导

* 偏导数单项贡献分析：对于权重 $W_{ij}$，它仅对 $z_i$ 产生直接贡献（例如 $W_{23}$ 只用于计算 $z_2$ 而不影响 $z_1$）。
* 通过展开标量求导并合并，我们能够证实形状约定下外积的正确性：
  $$\frac{\partial s}{\partial W} = \delta x^T$$

## 页面 54: 导数应该具有什么形状？

* 从雅可比矩阵链式法则的角度看，偏导数可能是一个行向量。
* 但根据形状约定，梯度必须与原参数（如偏置项 $b$）具有相同的列向量形状。
* 这导致了雅可比形式（便于应用链式法则推导）与形状约定（便于编写 SGD 优化器代码）之间的差异。
* **我们要求课程作业中的答案均遵循形状约定！**
* 但雅可比形式是您推导过程中的有力武器。

## 页面 55: 导数形状处理的两种策略

当您推导具体问题时，有两种选择：
1. **先用雅可比矩阵形式计算，最后转置以符合形状约定**：
   * 这正是我们刚才的做法，最后将结果转置为列向量形式。
2. **全程遵循形状约定**：
   * 通过观察维度大小来判定何时进行转置或调整项的顺序。
   * 牢记：到达某一隐藏层的误差信息 $\delta$ 的维度一定与该隐藏层本身的维度一致。

## 页面 56: 4. 反向传播算法 (Backpropagation)

* 我们实际上已经展示了反向传播的全貌。
* 它是对网络计算求偏导数，并系统化地应用（推广的多变量/矩阵）链式法则的过程。
* **关键策略**：我们将高层计算出的梯度缓存重用，以计算低层的梯度，从而将计算复杂度降到最低（避免重复计算）。

## 页面 57: 计算图与反向传播

* 软件框架将我们的神经网络公式表示为一个**计算图 (Computation Graph)**：
  * **源节点 (Source nodes)**：输入数据/参数。
  * **内部节点 (Interior nodes)**：基本操作算子（如加法、乘法等）。

## 页面 58: 计算图与反向传播（二）

* **边 (Edges)** 传递着操作算子的计算结果。

## 页面 59: 计算图与前向传播

* 沿着计算图的方向计算并传递数据的过程即为**前向传播 (Forward Propagation)**。

## 页面 60: 计算图与反向传播

* 然后，沿着图的相反方向（逆着边的方向）返回，传递**梯度信号**的过程即为**反向传播 (Backpropagation)**。

## 页面 61: 反向传播：单个节点

* 每个操作算子节点接收一个“上游梯度 (upstream gradient)”。
* 其任务是计算并传递正确的“下游梯度 (downstream gradient)”。

## 页面 62: 反向传播：单个节点与局部梯度

* 每个节点内部都包含一个**局部梯度 (local gradient)**。
* 局部梯度是指该节点的输出关于其自身输入的导数。

## 页面 63: 反向传播：单个节点与链式法则

* 局部梯度在节点的前向传播过程中即可计算出。

## 页面 64: 反向传播：单个节点基本公式

$$\text{[下游梯度]} = \text{[上游梯度]} \times \text{[局部梯度]}$$

## 页面 65: 反向传播：多输入节点

* 对于具有多个输入分支的算子节点（例如乘法算子）。

## 页面 66: 反向传播：多输入节点与多局部梯度

* 多个输入 $\Rightarrow$ 拥有多个对应的局部梯度：
  * 下游梯度分别等于上游梯度乘以对应的局部梯度。

## 页面 67: 梯度在分叉分支处相加

* 当计算图中某个节点有多条输出边流向后续节点时，在反向传播时，来自这些不同路径的梯度会在分叉汇合处进行**累加 (sum)**。

## 页面 68: 梯度在分叉分支处相加（续）

## 页面 69: 常见算子节点的梯度直观理解

* **加法节点 (+)**：将上游梯度原封不动地**分发 (distribute)** 给每个加项输入。

## 页面 70: 常见算子节点的梯度直观理解（二）

* **最大值节点 (max)**：将上游梯度**路由 (route)** 给前向传播时取得最大值的那个通道，其他通道梯度为 0。

## 页面 71: 常见算子节点的梯度直观理解（三）

* **乘法节点 (*)**：将上游梯度乘以另一个输入分量进行**互换开关 (switch)** 后输出。

## 页面 72: 效率：一次性计算所有梯度

* **不正确的反向传播执行方式**：
  * 独立地、重复地为每个变量从头运行链式法则，导致大量冗余的路径遍历。

## 页面 73: 效率：一次性计算所有梯度（续）

* 这种做法造成了大量重复的求导计算。

## 页面 74: 效率：一次性计算所有梯度（三）

* **正确的执行方式**：
  * 沿着反向路径拓扑排序，一次性计算出所有变量的梯度。
  * 类似于我们在手算梯度时引入 $\delta$ 信号来缓存重用中间导数。

## 页面 75: 通用计算图中的反向传播算法

1. **前向传播 (Fprop)**：按照计算图的拓扑排序顺序遍历节点：
   * 在给定前驱节点输出的情况下，计算当前节点的值，并保存中间激活值。
2. **反向传播 (Bprop)**：
   * 初始化最终输出的梯度 = 1。
   * 按照相反的拓扑顺序遍历节点：
     * 利用后继节点传回的梯度，计算当前节点的下游梯度。

只要实现得当，反向传播的计算复杂度与前向传播是同一数量级的（均为大 O 复杂度）。
神经网络通常具有规整的层级结构，因此我们可以方便地使用矩阵与雅可比矩阵来进行运算。

## 页面 76: 自动微分 (Automatic Differentiation)

* 梯度的计算逻辑可以完全由前向传播的符号表达式自动推导得出。
* 每种节点算子仅需要注册两个逻辑：如何根据输入计算前向输出，以及在给定关于输出的梯度时如何计算关于其输入的局部梯度。
* 现代深度学习框架（Tensorflow, PyTorch 等）为您代劳了反向传播的流式计算，但底层的自定义层算子通常仍需要算子编写者手动计算并提供局部导数。

## 页面 77: 反向传播的实现方式

* 深度学习框架内部通过构建动态或静态计算图来自动调度反向传播的执行。

## 页面 78: 手动梯度检查：数值梯度 (Numeric Gradient)

* 为了检验反向传播的代码实现是否正确，可以通过有限差分计算**数值梯度**（对于微小的 $h \approx 10^{-4}$）：
  $$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$
* **特点**：
  * 极易正确实现。
  * 但只是近似值，且计算极其缓慢（必须为模型的每个参数单独重新运行前向计算）。
  * **主要用作调试工具**，用于检验手写解析梯度（Analytic Gradient）是否正确。

## 页面 79: 课程总结

我们已经掌握了神经网络的核心训练技术！
* **反向传播 (Backpropagation)**：沿着计算图递归地（因而极其高效地）应用链式法则。
  $$\text{[下游梯度]} = \text{[上游梯度]} \times \text{[局部梯度]}$$
* **前向通道**：计算各项算子结果并保存中间值。
* **反向通道**：应用链式法则计算并传递所有梯度。

## 页面 80: 既然深度学习框架能自动求导，为什么我们还要学习梯度的细节？

* 现代深度学习框架确实会自动为您计算梯度！
* *欢迎参加本周五的 PyTorch 介绍会！*
* 但正如为什么哪怕编译器和操作系统都已经如此成熟，我们仍然需要学习编译原理或操作系统一样：
* **理解底层的运行机制是极其有用的！**
* 反向传播并非总是能开箱即用地完美工作：
  * 理解其物理过程对于调试、排查和优化模型至关重要。
  * 参见 Andrej Karpathy 的文章：《是的，你必须理解反向传播》 [Karpathy's Article](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
  * 未来课程示例：梯度爆炸 (exploding gradients) 与梯度消失 (vanishing gradients) 问题。
