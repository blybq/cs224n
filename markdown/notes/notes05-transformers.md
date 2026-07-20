# CS224n 课程笔记 5：自注意力机制与 Transformer

**课程讲师**：Christopher Manning, John Hewitt  
**作者**：John Hewitt (johnhew@cs.stanford.edu)  
**授课学期**：2023 冬季  

---

### 摘要
本篇笔记首先阐述了放弃自然语言处理中经典的循环架构（RNN）的根本动机。接着，系统性地引入了自注意力机制（Self-Attention），并以此构建了一个极简的自注意力神经网络。最后，笔记深入剖析了目前（截至2023年及以后）在 NLP 领域占据绝对统治地位的 **Transformer** 架构的设计细节，包括多头注意力、层归一化、残差连接以及编码器-解码器架构等。

---

## 1 神经网络架构及其局限性 (Neural architectures and their properties)

自然语言处理的每一次飞跃，往往都伴随着通用架构的革新。从隐马尔可夫模型 [Baum and Petrie, 1966]、条件随机场 [Lafferty et al., 2001]，到循环神经网络（RNN）[Rumelhart et al., 1985] 和支持向量机 [Cortes and Vapnik, 1995]，NLP 的建模技术在不断演进。

在本节中，我们将探讨为什么 2017 年之前作为主流的 RNN 会被逐步放弃，以及这种局限性是如何催生出目前风靡全球的自注意力（Self-Attention）与 Transformer 架构的。

### 1.1 符号与基础知识 (Notation and basics)
设 $w_{1:n}$ 为一个单词序列，其中每个 $w_i \in V$（$V$ 是有限词汇表）。我们可以将 $w_{1:n}$ 简化看作一个独热（1-hot）向量组成的矩阵 $w_{1:n} \in \mathbb{R}^{n \times |V|}$。
在概率语言模型中，我们采用自回归方式预测下一个单词：
$$w_t \sim \operatorname{softmax}(f(w_{1:t-1})) \tag{1}$$
这代表在模型下，$w_t$ 的概率分布由其前面的词序列经过映射后的输出决定，其中 $f(w_{1:t-1}) \in \mathbb{R}^{|V|}$。

> [!NOTE]
> **关于张量 Softmax 的定义**：
> 如果输入为一个二维张量 $A \in \mathbb{R}^{\ell \times d}$，其 Softmax 沿最后一维归一化的计算公式为：
> $$\operatorname{softmax}(A)_{i,j} = \frac{\exp(A_{i,j})}{\sum_{j'=1}^d \exp(A_{i,j'})} \tag{2}$$
> 类似地，对于三维张量 $B \in \mathbb{R}^{m \times \ell \times d}$，其最后一维的 Softmax 为：
> $$\operatorname{softmax}(B)_{q,i,j} = \frac{\exp(B_{q,i,j})}{\sum_{j'=1}^d \exp(B_{q,i,j'})} \tag{3}$$

设词嵌入矩阵为 $E \in \mathbb{R}^{d \times |V|}$，非上下文表示词向量为 $x = Ew$。序列的嵌入表示记为 $x_{1:n} = w_{1:n} E^\top \in \mathbb{R}^{n \times d}$。
- **非上下文表示（Non-contextual Representation）**：如 $E_{w_i}$，该词的向量表示完全独立于它所在的具体上下文环境（例如，不管是“苹果手机”还是“吃个苹果”，“苹果”一词的词向量都相同）。
- **上下文表示（Contextual Representation）**：我们希望得到包含上下文环境特征的向量表示 $h_i$，它代表单词 $w_i$，但其值是整个输入序列 $x_{1:n}$（或者自回归下的前缀 $x_{1:i}$）的函数。

### 1.2 2017年前的默认选择：循环神经网络 (RNN)
在 2017 年之前，解决绝大多数 NLP 任务的标配起点是 RNN：
$$h_t = \sigma(W h_{t-1} + U x_t) \tag{4}$$

然而，到了 2017 年，研究界公认了这种依赖时间/序列索引的递推公式存在两个根本性瓶颈：

#### 1. 难以进行并行化加速
现代图形处理器（GPU）和张量处理器（TPU）极其擅长并行的矩阵乘法计算。然而，在公式 (4) 中，隐藏状态的计算存在着严苛的**串行依赖**关系：
$$h_2 = \sigma(W \sigma(W h_0 + U x_1) + U x_2) \tag{6}$$
在得到 $h_1$ 的结果之前，GPU 无法并行开始计算 $h_2$；在得到 $h_2$ 之前，无法计算 $h_3$（如图 1 所示）。随着序列长度 $n$ 的增长，这种时间轴上的串行限制使得 GPU 的并行算力根本无法得到完全释放。

```
   时间步 (t=1): x_1 ---> [ RNN 单元 ] ---> h_1
                                            |
                                            v (必须串行等待)
   时间步 (t=2): x_2 ------------------------> [ RNN 单元 ] ---> h_2
                                                                |
                                                                v
   时间步 (t=3): x_3 --------------------------------------------> [ RNN 单元 ] ---> h_3
```

#### 2. 线性交互距离过长 (Linear Interaction Distance)
当序列中两个相距很远的单词需要发生交互（互相影响其表示）时，在 RNN 中，它们必须通过与其距离呈**线性比例**数量级的中间步骤（矩阵相乘和非线性映射）来进行信息的层层传递。
例如，在句子：
> *“The **chef** who ran out of blackberries and went to the stores **is** ...”*

单词 ***chef***（主语）与 ***is***（谓语动词）之间隔着大量其他词。它们之间的语义交互必须跨越 10 多个时间步的隐状态传递。在此传递过程中，主语的信息极易因梯度消失或梯度爆炸而在中途衰减丢失，使模型极难准确捕获这种远距离的依存关系（如图 2 所示）。

这两个痛点启发了全新的设计思想：**我们能否完全放弃循环结构，直接基于注意力机制来构建一整套神经网络？** 这一方案将完美消除串行依赖，并将任意两个位置单词之间的直接交互距离缩短为常数 $O(1)$。

---

## 2 极简自注意力神经网络架构 (A minimal self-attention architecture)

宽泛地讲，**注意力机制（Attention）**就是在一个键值对（Key-Value）存储器中，使用一个查询值（Query）去软性检索信息的过程。它通过计算 Query 与各 Key 的相似度，来对对应的 Values 进行加权平均。而**自注意力（Self-Attention）**，则是指检索中的 Query、Key 和 Value 都是由相同的输入序列衍生而来的。

### 2.1 键-查询-值自注意力机制 (Key-Query-Value Self-Attention)
当前最流行的是 **K-Q-V 自注意力**。
对于输入序列中的任意词向量 $x_i \in x_{1:n}$，我们通过三个学习矩阵 $Q, K, V \in \mathbb{R}^{d \times d}$，分别将该词映射为三种不同的角色向量：
- **查询向量（Query）**：$q_i = Q x_i$ （用于寻找与当前词相关的其他信息）
- **键向量（Key）**：$k_j = K x_j$ （作为被检索词的索引标签）
- **值向量（Value）**：$v_j = V x_j$ （代表被检索词本身所携带的特征内容）

其生成的上下文向量 $h_i$ 是序列中所有值向量的加权和：
$$h_i = \sum_{j=1}^n \alpha_{ij} v_j \tag{7}$$

其中，注意力权重 $\alpha_{ij}$ 决定了检索强度。它通过计算当前查询与各个键的内积相似度 $q_i^\top k_j$，并在整个输入序列上进行 Softmax 归一化得到：
$$\alpha_{ij} = \frac{\exp(q_i^\top k_j)}{\sum_{j'=1}^n \exp(q_i^\top k_{j'})} \tag{8}$$

```
                          Q_i * K_j (内积)
   查询 q_i --------------> [ 点积相似度 ] ---> [ Softmax 归一化 ] ---> 权重 alpha_ij ---\
   键组 k_1, k_2, ... -----/                                                               \
                                                                                            +===> 上下文表示 h_i
   值组 v_1, v_2, ... -------------------------------------------------> [ 加权求和 ] -----/
```

使用这三个不同的映射矩阵 $Q, K, V$，使得同一个单词在作为“查询方”、“被检索方的标签”和“被检索的内容本身”时，能展现出全然不同的语义视角。

### 2.2 位置表示 (Position representations)
在自注意力公式中，由于我们直接计算了任意单词对之间的相似度并求和，整个操作是**具有置换不变性（Permutation Invariant）**的。也就是说，**自注意力本身完全无法感知单词出现的顺序**。

> **位置缺失示例**：
> 句子一：*The oven cooked the bread.*（烤箱烤了面包。）
> 句子二：*The bread cooked the oven.*（面包烤了烤箱。）
> - 对于自注意力而言，由于它仅仅把单词视为一个集合，那么在没有位置信息的情况下，这两个句子中任何一个词（例如 *cooked*）所算出的自注意力上下文向量 $h$ 是**完全一样**的。然而它们的语义已经天差地别。

为了让自注意力能够感知顺序，我们必须显式引入位置信息。常用方案如下：

#### 1. 相加位置编码 (Position Representation through Additive Embeddings)
最常见的策略是引入一个可学习的位置嵌入矩阵 $P \in \mathbb{R}^{N \times d}$（$N$ 是模型支持的最大序列长度）。在进行自注意力计算之前，直接将位置编码向量与词嵌入向量进行相加：
$$\tilde{x}_i = P_i + x_i \tag{12}$$
这样，自注意力在计算相似度时，便能够区分在位置 $i$ 处的单词和在位置 $j$ 处的同一个单词了。

#### 2. 直接在分数上施加线性位置偏置 (Linear Biases on Attention Scores)
另一种方法是直接修改自注意力的计算规则，加入一个与相对距离成反比的固定惩罚项（例如 ALiBi [Press et al., 2022]）：
$$\alpha_i = \operatorname{softmax}\left( k_{1:n} q_i + \begin{bmatrix} -i & \dots & -1 & 0 & -1 & \dots & -(n-i) \end{bmatrix} \right) \tag{13}$$
这会强迫注意力机制本能地更关注距离较近的单词，而惩罚距离较远的单词。

### 2.3 逐元素非线性 (Elementwise nonlinearity)
如果我们直接堆叠多层自注意力机制，会发生什么？
其实，两层级联的自注意力依然是输入的线性组合：
$$o_i = \sum_{j=1}^n \alpha_{ij} V^{(2)} \left( \sum_{k=1}^n \alpha_{jk} V^{(1)} x_k \right) = \sum_{k=1}^n \alpha^*_{ik} V^* x_k \tag{14-16}$$
也就是说，**多层自注意力连续堆叠，其效果在数学上等价于单层自注意力**，无法像深度神经网络那样拟合复杂的非线性决策边界。

为了打破这种线性叠加，我们必须在每一个自注意力层之后，对每个单词独立且并行地应用一个多层前馈网络（Feed-forward Network, FFN）：
$$h_{\text{FF}} = W_2 \operatorname{ReLU}(W_1 h_{\text{self-attn}} + b_1) + b_2 \tag{17}$$
在实践中，前馈隐藏层的维度通常被设计得相当宽（例如设定 $W_1$ 的输出维度为 $5d$ 或 $4d$）。由于该操作在各个时间步上是完全并行的，因此这是一个能高效利用参数和计算力的设计。

### 2.4 因果掩码 / 未来遮蔽 (Future Masking)
在进行单向语言建模（自回归预测）时，预测当前词 $w_t$ 的模型 $f(w_{1:t-1})$ 绝不能“偷看”未来的单词 $w_{t:n}$。

在自注意力中，任何一个位置都可以直接与未来的任意位置计算点积。为了强加“不能偷看未来”的限制，我们在计算 Softmax 之前，对所有 $j > i$（即未来位置）的分数加上一个巨大的负数常数（通常设为 $-10^5$，以防使用 $-\infty$ 造成浮点数溢出 `NaN`）。这使得模型计算的未来位置的注意力权重在经过 Softmax 归一化后恰好收敛为 $0$：
$$\alpha_{ij, \text{masked}} = \begin{cases} \alpha_{ij} & \text{if } j \le i \\ 0 & \text{otherwise} \end{cases} \tag{21}$$

```
   (行 i=当前词，列 j=上下文词。遮蔽使右上三角全为 -inf / 0)
   "Zuko"   [ Zuko ]  -inf    -inf    -inf
   "made"   [ Zuko    made ]  -inf    -inf
   "his"    [ Zuko    made    his  ]  -inf
   "uncle"  [ Zuko    made    his    uncle ]
```
*图 3：自回归未来掩码示意图。每一行代表当前位置，未来位置在输入 Softmax 前被置为 $-\infty$。*

---

## 3 Transformer 架构细节 (The Transformer)

**Transformer** 是一种由 Vaswani 等人于 2017 年在《Attention Is All You Need》中提出的、完全基于自注意力机制的深度神经网络架构。它由多层 Block 堆叠而成，除了上面提到的核心组件，还引入了以下关键设计：

### 3.1 多头自注意力机制 (Multi-head Self-Attention)
单个自注意力头（Single-head）在提取信息时，往往只能关注到一处语义重心。如果想同时提取多种特征组合，多头自注意力机制（Multi-head Self-Attention）提供了绝佳的设计方案。它通过将输入投影到多个独立的、低维度的特征空间（称为“头”，Heads），并行进行自注意力计算，最后再进行拼接融合。

对于 $k$ 个注意力头，我们独立定义 $k$ 组投影矩阵 $Q^{(\ell)}, K^{(\ell)}, V^{(\ell)} \in \mathbb{R}^{d \times \frac{d}{k}}$（对于 $\ell \in \{1, \dots, k\}$）：
$$h_i^{(\ell)} = \sum_{j=1}^n \alpha_{ij}^{(\ell)} v_j^{(\ell)} \tag{22}$$
$$\alpha_{ij}^{(\ell)} = \frac{\exp\left(q_i^{(\ell)\top} k_j^{(\ell)}\right)}{\sum_{j'=1}^n \exp\left(q_i^{(\ell)\top} k_{j'}^{(\ell)}\right)} \tag{23}$$

注意，每个头的输出向量维度被压缩到了 $\frac{d}{k}$。最终，我们将这 $k$ 个头的输出沿着特征维度拼接起来，并通过一个大输出矩阵 $O \in \mathbb{R}^{d \times d}$ 进行线性融合：
$$h_i = O \begin{bmatrix} h_i^{(1)} ; \dots ; h_i^{(k)} \end{bmatrix} \tag{24}$$

> [!TIP]
> **代码中的高效矩阵计算**：
> - 实际上，多头自注意力并不会显式地使用循环来计算每个头，那样效率太低。
> - 在代码实现中，输入矩阵 $x_{1:n} \in \mathbb{R}^{n \times d}$ 会与大矩阵相乘后被重塑（Reshape）为形如 $(n, k, \frac{d}{k})$ 的三维张量。
> - 随后，将张量转置（Transpose）为 $(k, n, \frac{d}{k})$。这在数学上相当于开辟了一个大小为 $k$ 的“头批次（head batch）”轴，使得所有的头能在 GPU 上以单次并行的批处理矩阵乘法同时完成计算。

### 3.2 层归一化 (Layer Norm)
**层归一化（Layer Normalization）**用于稳定各层激活值的分布。在 Transformer 中，层归一化是针对每个样本的每一个单词，在其特征维度 $d$ 上独立计算均值 $\hat{\mu}_i$ 和标准差 $\hat{\sigma}_i$ 的：
$$\hat{\mu}_i = \frac{1}{d} \sum_{j=1}^d h_{ij}, \quad \hat{\sigma}_i = \sqrt{\frac{1}{d} \sum_{j=1}^d (h_{ij} - \hat{\mu}_i)^2} \tag{27}$$
$$\operatorname{LN}(h_i) = \frac{h_i - \hat{\mu}_i}{\hat{\sigma}_i} \tag{28}$$

通过归一化，排除了由于特征幅度波动带来的干扰。这在反向传播中起到了稳定和加速梯度流动的作用。

### 3.3 残差连接 (Residual Connections)
残差连接直接将层的输入累加到层的输出中：
$$f_{\text{residual}}(h) = f(h) + h \tag{29}$$
- **优点一**：由于恒等映射（Identity function）的局部梯度在反向传播中恒为 $1$，极好地保全了梯度流，使得训练上百层的极深神经网络成为可能。
- **优点二**：在数学上，学习一个层与输入之间的“差值”（残差映射）要比从头学习一个复杂的映射容易得多。

#### Add & Norm (残差加与归一化)
在 Transformer 架构中，残差连接和层归一化总是成对出现。主要有以下两种架构设计方案：

- **Pre-LN（前置归一化）**：在输入层之前进行归一化（如图 4 所示）：
  $$h^{(l+1)} = f(\operatorname{LN}(h^{(l)})) + h^{(l)} \tag{30}$$
- **Post-LN（后置归一化）**：在残差相加之后再进行层归一化（最初的 Transformer 架构）：
  $$h^{(l+1)} = \operatorname{LN}(f(h^{(l)}) + h^{(l)}) \tag{31}$$

> [!IMPORTANT]
> 实验表明，**Pre-LN 架构在模型初始化时的梯度特性要远远好于 Post-LN**。使用 Pre-LN 可以使模型省去复杂的学习率预热（Warmup）阶段，且训练极其稳定、收敛迅速。目前的现代大模型几乎全部采用 Pre-LN 结构。

### 3.4 注意力缩放 (Attention Logit Scaling)
当特征向量维度 $d$ 变得很大时，随机初始化的两向量的点积 $q^\top k$ 其绝对值期望会趋近于 $\sqrt{d}$。巨大的点积值会把 Softmax 函数推入极其陡峭的饱和区，导致反向传播时对应的梯度近乎消失为 0。

为了解决这一问题，我们将点积值除以 $\sqrt{d/k}$（即单个头部的特征维度根号）进行缩放归一化：
$$\alpha = \operatorname{softmax}\left( \frac{x_{1:n} Q K^\top x_{1:n}^\top}{\sqrt{d}} \right) \tag{32}$$

---

## 3.5 Transformer 编码器 (Transformer Encoder)

**Transformer 编码器**用于处理不需要遵循自回归限制（即不需要因果掩码）的文本。每个单词的表示都可以同时利用到整句序列的全局双向上下文。

```
   [ 输入文本 ] ---> [ 词嵌入 + 位置编码 ] ---> [ 编码器 Block 堆叠 ] ---> [ 全局双向上下文表示 ]
```

#### 编码器 Block 的内部组成：
1. **多头自注意力（Multi-Head Attention）**，后接 **Add & Norm**（残差连接与层归一化）。
2. **逐层前馈网络（Feed-Forward Network）**，后接 **Add & Norm**。

**典型应用**：BERT [Devlin et al., 2019]。它非常适合用于文本理解、信息抽取和分类任务。

---

## 3.6 Transformer 解码器 (Transformer Decoder)

为了执行自回归文本生成任务，我们使用 **Transformer 解码器**。
解码器 Block 与编码器的唯一区别在于：**其多头自注意力层必须包含因果未来掩码（Masked Multi-Head Attention）**。这确保了在任何一个时间步，模型只能向历史方向看，不能偷看未来的内容。

**典型应用**：GPT 系列模型（GPT-2, GPT-3）以及 BLOOM 模型。当前业界绝大多数最先进的自然语言生成式大模型都基于纯解码器（Decoder-only）架构。

---

## 3.7 Transformer 编解码器架构 (Transformer Encoder-Decoder)

经典的 Transformer 编解码器架构将编码器和解码器组合在了一起（如图 6 所示）。这适用于输入和输出是不同序列的 Seq2Seq 任务（如机器翻译、文本摘要）。

```
   [ 源序列 x ] ---> [ 编码器 ] ---------------------------\
                                                          v
   [ 目标前缀 y ] --> [ 解码器 (Causal) ] ---> [ 交叉注意力 (Cross-Attention) ] ---> [ 预测下一个词 ]
```

除了自身内部的自注意力之外，解码器内部还引入了**交叉注意力（Cross-Attention）**机制来与编码器沟通：
- **Queries（查询）**：来自解码器前一层的隐藏表示 $h^{(y)}$。
- **Keys（键）和 Values（值）**：来自编码器最终输出的全局上下文表示 $h^{(x)}$。

$$\begin{aligned}
q_i &= Q h_i^{(y)} \quad (i \in \{1, \dots, m\}) \tag{34} \\
k_j &= K h_j^{(x)} \quad (j \in \{1, \dots, n\}) \tag{35} \\
v_j &= V h_j^{(x)} \quad (j \in \{1, \dots, n\}) \tag{36}
\end{aligned}$$

这使得解码器在自回归生成当前单词时，可以软性检索源文本中的所有重要信息（这与经典机器翻译中的注意力机制作用相同）。

虽然在中小规模参数下，编解码器架构（如 T5 [Raffel et al., 2020]）在很多翻译和摘要任务上的效果要优于纯解码器模型；但由于需要将参数分摊在编码器和解码器两套模块中，目前参数量最大的主流大语言模型仍倾向于使用结构更为简单的纯解码器（Decoder-only）架构。

---

## 参考文献 (References)

- [Ba et al., 2016] Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer normalization. *arXiv preprint*.
- [Bahdanau et al., 2014] Bahdanau, D., Cho, K., and Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint*.
- [Baum and Petrie, 1966] Baum, L. E. and Petrie, T. (1966). Statistical inference for probabilistic functions of finite state markov chains. *The Annals of Mathematical Statistics*.
- [Bengio et al., 2003] Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A neural probabilistic language model. *JMLR*.
- [Brown et al., 2020] Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*.
- [Cortes and Vapnik, 1995] Cortes, C. and Vapnik, V. (1995). Support-vector networks. *Machine learning*.
- [Devlin et al., 2019] Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- [Elman, 1990] Elman, J. L. (1990). Finding structure in time. *Cognitive Science*.
- [Lafferty et al., 2001] Lafferty, J. D., McCallum, A., and Pereira, F. C. N. (2001). Conditional random fields: Probabilistic models for segmenting and labeling sequence data. *ICML*.
- [LeCun et al., 1989] LeCun, Y., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*.
- [Press et al., 2022] Press, O., Smith, N., and Lewis, M. (2022). Train short, test long: Attention with linear biases enables input length extrapolation. *ICLR*.
- [Radford et al., 2019] Radford, A., et al. (2019). Language models are unsupervised multitask learners.
- [Raffel et al., 2020] Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *JMLR*.
- [Rumelhart et al., 1985] Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1985). Learning internal representations by error propagation. *Technical report*.
- [Schütze, 1992] Schütze, H. (1992). Dimensions of meaning. *Supercomputing '92*.
- [Vaswani et al., 2017] Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- [Workshop et al., 2022] Workshop, B., et al. (2022). Bloom: A 176b-parameter open-access multilingual language model.
- [Xiong et al., 2020] Xiong, R., et al. (2020). On layer normalization in the transformer architecture. *ICML*.
- [Xu et al., 2019] Xu, J., et al. (2019). Understanding and improving layer normalization. *NeurIPS*.
