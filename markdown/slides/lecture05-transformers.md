# Lecture 5: Attention and Transformers

## 页面 1: 标题
* **课程名称**：基于深度学习的自然语言处理 (Natural Language Processing with Deep Learning)
* **课程编号**：CS224N / Ling284
* **主讲人**：Diyi Yang
* **主题**：第 5 讲：注意力机制和 Transformer (Lecture 5: Attention and Transformers)

---

## 页面 2: 课程计划
1. **梯度消失**：(10分钟)
2. **机器翻译**：(10分钟)
3. **从循环 (RNN) 到基于注意力机制的模型**：(15分钟)
4. **自注意力机制 (Self-Attention)**：(15分钟)
5. **Transformer 模型**：(15分钟)
6. **Transformer 的优异成果、局限性及其变体**：(5分钟)

*课程进展提示：候补名单更新；作业 2 将于 1 月 22 日截止；下周四将进行课程项目讨论会。*

---

## 页面 3: 1. RNN 的问题：梯度消失与梯度爆炸

---

## 页面 4 - 8: 梯度消失的直观理解 (Vanishing Gradient Intuition)
* 根据链式法则，在随时间反向传播时，如果雅可比矩阵的特征值（或权重矩阵的元素值）较小，梯度信号在向早期时间步回传时会呈指数级减小。
* 这会导致**梯度消失问题 (Vanishing Gradient Problem)**。

---

## 页面 9: 为什么梯度消失是个问题？
* 远处的梯度信号由于极其微弱，在回传时会被近处的强梯度信号所**淹没**。
* 导致模型参数更新时仅能捕捉到近距离的上下文关系，而无法学习到**长期依赖关系 (Long-term effects)**。

---

## 页面 10: 梯度消失对 RNN-LM 的负面效果示例
* **例句**：
  > *"When she tried to print her **tickets**, she found that the printer was out of toner. She went to the stationery store to buy more toner. It was very overpriced. After installing the toner into the printer, she finally printed her **________**"* (此处应填 `tickets`)
* **分析**：RNN-LM 需要对跨越数十步的 "tickets" 与最后的空格之间的依赖关系进行建模。但在梯度消失的情况下，模型无法在训练中学习到该依赖，导致测试时预测失败。

---

## 页面 11: 为什么梯度爆炸是个问题？
* 梯度若变得过大，SGD 的更新步长将失控：
  $$\theta \leftarrow \theta - \alpha \nabla_{\theta} J(\theta)$$
* 这会导致参数陷入极差的配置区间，使得损失值飙升（所谓的“Iowa 效应”）。
* 在最坏的情况下，这会导致网络内部数据溢出，产生 `Inf` 或 `NaN`，从而被迫中断训练。

---

## 页面 12: 解决方案：梯度裁剪 (Gradient Clipping)
* 如果梯度的范数 (norm) 超过了给定的阈值，则在应用 SGD 更新前，等比例缩减其模长。
  $$\text{if } \|\mathbf{g}\| > \text{threshold}, \quad \mathbf{g} \leftarrow \text{threshold} \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
* 这能够有效限制更新步长，是一个简单且行之有效的工程技巧。

---

## 页面 13: 如何解决梯度消失问题？
* **主要难点**：在 Vanilla RNN 中，隐状态一直在不断被**重写 (Rewrite)**，导致历史信息无法有效保留。
* **改进策略**：
  1. 使用带独立可加记忆的循环网络 $\rightarrow$ **长短期记忆网络 (LSTM)**。
  2. 在模型中建立直接的线性传递通道 $\rightarrow$ **注意力机制 (Attention)**、**残差连接 (Residual Connections)**。

---

## 页面 14: 2. 机器翻译 (Machine Translation)
* **机器翻译 (MT)**：将源语言句子 $x$ 翻译为目标语言句子 $y$ 的任务。
  * 例如：$x$: `I like deep learning` $\rightarrow$ $y$: `我喜欢深度学习`

---

## 页面 15: 神经机器翻译 (NMT) 的崛起
* 神经机器翻译仅用两年时间（2014-2016），就从学术界边缘的前沿尝试跃升为工业界主流的标准算法。
  * **2014 年**：第一篇 Seq2Seq 论文发表 [Sutskever et al. 2014]。
  * **2016 年**：Google 翻译全面从传统的统计机器翻译 (SMT) 切换为神经机器翻译 (NMT)。
  * **震撼性**：传统 SMT 由数百名工程师历经多年开发维护，而 NMT 仅由少数研究员用数月训练即可在翻译质量上将其超越。

---

## 页面 16: 神经机器翻译 (NMT) 的经典 Seq2Seq 架构
* **编码器 RNN (Encoder RNN)**：对源语言句子进行编码，并将其最终的隐状态作为解码器的初始状态。
* **解码器 RNN (Decoder RNN)**：本质上是一个条件语言模型 (Conditional LM)，在源句子编码向量的条件约束下，逐步解码生成目标词。

---

## 页面 17: 序列到序列 (Seq2Seq) 架构的多功能性
Seq2Seq 实际上是一个通用的**编码器-解码器模型**：
* 编码器处理输入，构造隐层神经表示。
* 解码器基于此表示生成输出。
* **广泛的应用场景**：
  * **文本摘要**：长文章 $\rightarrow$ 短摘要
  * **对话**：上一句 $\rightarrow$ 下一句
  * **句法分析**：输入文本 $\rightarrow$ 序列化的解析树
  * **代码生成**：自然语言 $\rightarrow$ Python 代码

---

## 页面 18: 条件语言模型 (Conditional Language Model)
* Seq2Seq 属于**条件语言模型**：
  * 解码器是“语言模型”，在预测下一个词。
  * 它的预测受到源句子 $x$ 的“条件约束”。
* NMT 计算联合概率：
  $$P(y \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)$$

---

## 页面 19: 训练神经机器翻译系统
* 整个 Seq2Seq 作为一个统一的模型联合优化。
* 反向传播是**端到端 (End-to-End)** 进行的。
* 损失 $J$ 为所有解码时间步交叉熵损失的平均值。

---

## 页面 20: 多层深层编码器-解码器机器翻译网络
* 可以通过堆叠多层循环神经网络来强化表示能力。
* 下一层的隐状态作为上一层的输入。

---

## 页面 21: 瓶颈问题 (Bottleneck Problem)
> [!WARNING]
> 传统的 Seq2Seq 架构在编码源句子时，必须将全部语义压缩进一个固定长度的隐向量中。
> 这种限制成为了长距离依赖传输的**瓶颈 (Bottleneck)**，会导致长句子信息丢失严重。

---

## 页面 22: 3. 注意力机制 (Attention)
* **注意力机制**为瓶颈问题提供了完美的解决方案。
* **核心思想**：在解码器的每一步中，使用与编码器的**直接连接**，从而聚焦于源语言序列的特定部分。
* 下面首先通过图示展示其工作机制，随后给出数学公式。

---

## 页面 23: 引入背景：RNN 的均值池化 (Mean-pooling)
* 最基础的从编码器传递句向量的方法是对所有时间步的隐状态取平均值（均值池化）或最大值（最大池化），但这无法突出不同词的重要性。

---

## 页面 24: 注意力机制是加权平均（允许进行检索！）
* 注意力机制本质上是一种**加权平均 (Weighted Average)**，当权重是可学习的时，它非常强大。
* **类比键值检索**：
  * 在数据库查询中，存在一个“键-值”表 (Keys-Values)。输入查询 (Query)，精确匹配某个 Key，并返回其 Value。
  * 在注意力机制中，Query 与所有的 Key 进行**软匹配**（产生介于 0 到 1 之间的权重）。随后将各个 Key 对应的 Value 乘以权重并求和。

---

## 页面 25 - 31: 结合注意力机制的 Seq2Seq 运行机制图解
1. **计算注意力得分 (Attention Scores)**：解码器当前步的隐状态（作为 Query $\mathbf{s}_t$）与编码器各步的隐状态（作为 Keys $\mathbf{h}_i$）进行点积计算：
   $$e_{t, i} = \mathbf{s}_t^\top \mathbf{h}_i$$
2. **计算注意力分布 (Attention Distribution)**：通过对得分应用 Softmax 函数，将其归一化为概率分布 $\alpha_{t, i}$：
   $$\alpha_{t} = \text{softmax}(\mathbf{e}_t)$$
3. **计算注意力输出 (Attention Output)**：将概率分布作为权重，对编码器隐状态（此时作为 Values $\mathbf{h}_i$）进行加权求和，得到注意力上下文向量 $\mathbf{a}_t$：
   $$\mathbf{a}_t = \sum_{i} \alpha_{t, i} \mathbf{h}_i$$
4. **生成预测**：将注意力输出 $\mathbf{a}_t$ 与解码器当前的隐状态 $\mathbf{s}_t$ 进行拼接 (Concatenate)，用于计算当前步的最终输出概率分布 $\hat{\mathbf{y}}_t$。

---

## 页面 32 - 36: 逐词解码生成过程 (以法译英为例)
* 随着解码的推进，解码器在预测每个新词时，其计算的注意力分布都会软聚焦在源句子对应的词上。
* *注：有时前一步的注意力输出也会作为输入反馈喂给解码器的下一步（与常规的解码器输入一同输入）。*

---

## 页面 37: 注意力机制的优势
* **显著提升 NMT 性能**：让解码器聚焦于最相关的源端区域，符合人类翻译时“边看边译”的直觉。
* **彻底消除瓶颈问题**：解码器可以直接“看”编码器的任意位置，绕过了单一隐状态的瓶颈。
* **缓解梯度消失问题**：提供了跨越遥远时序的捷径 (Shortcut)，梯度可以直接回传至最相关的早期状态。
* **带来一定程度的可解释性 (Interpretability)**：通过观察注意力矩阵（权重分布），我们可以得知解码器在预测当前词时在关注源句子的哪些词，无偿获得了**软对齐 (Soft alignment)** 关系。
* > [!WARNING]
  > **劣势**：注意力的计算复杂度对于序列长度 $n$ 是二次方的（即 $O(n^2)$ 复杂度）。

---

## 页面 38: 注意力机制是一项通用的深度学习技术
* 注意力并不局限于机器翻译和 Seq2Seq，它是一个更抽象的概念：
* **更通用的定义**：给定一组向量 Values 和一个向量 Query，注意力机制是根据 Query 计算 Values 加权和的通用技术。
* 我们常说：**Query 关注 (attends to) 哪些 Values**。

---

## 页面 39: 4. 我们真的需要循环结构 (RNN) 吗？
* **抽象来看**：注意力是一种从序列 $\mathbf{x}$ 向神经网络输入 $\mathbf{h}_t$ 传递信息的方法。
* 这也正是 RNN 存在的核心目的。
* **思考**：既然注意力传递信息的效果更好，那我们能否完全抛弃 RNN？
  * 2014-2017：RNN 是主导。
  * 2017 年之后：Transformer 带来了只用注意力机制的革命。

---

## 页面 40: 引入自注意力机制 (Self-Attention)
* **交叉注意力 (Cross-Attention)**：解码器状态去关注编码器输入以生成当前步。
* **自注意力 (Self-Attention)**：在处理序列时，每个位置的表示需要去关注**同一个序列**中的其他所有位置，以建立全局关联。

---

## 页面 41 - 42: 自注意力机制的数学定义
给定输入词向量序列 $\mathbf{x}_{1:n}$：
1. **生成 Queries, Keys, Values**：
   使用三个不同的投影矩阵 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{d \times d}$，将每个输入的词向量映射为三种表示：
   $$\mathbf{q}_i = \mathbf{Q}\mathbf{x}_i, \quad \mathbf{k}_i = \mathbf{K}\mathbf{x}_i, \quad \mathbf{v}_i = \mathbf{V}\mathbf{x}_i$$
2. **计算注意力权重**：
   通过 Queries 与 Keys 的点积度量相似度，并使用 Softmax 归一化：
   $$e_{ij} = \mathbf{q}_i^\top \mathbf{k}_j, \quad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'=1}^{n} \exp(e_{ij'})}$$
3. **加权求和输出**：
   输出是 Values 的加权和：
   $$\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j$$

---

## 页面 43 - 44: 自注意力作为网络积木的障碍与对策：时序顺序
* **障碍**：自注意力机制是**无序的**。对输入词的物理位置不敏感，洗牌打乱后得到的注意力输出依然相同。
* **对策：引入位置编码 (Position Representations)**：
  * 我们将表示索引位置的向量 $\mathbf{p}_i \in \mathbb{R}^d$ 直接加到输入的词嵌入上：
    $$\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{p}_i$$
  * 这样位置信息在第一层就被融合进了网络中。

---

## 页面 45 - 46: 位置表示的类型
* **正弦位置编码 (Sinusoidal Position Representations)**：
  * 使用不同频率的 sin 和 cos 函数拼接而成。
  * 优点：周期性特点可能使模型对绝对位置不敏感，且理论上有望外推到更长的序列。
  * 缺点：不可学习，且实际外推效果不佳。
* **可学习的绝对位置编码 (Learned Absolute Position Representations)**：
  * 将所有 $\mathbf{p}_i$ 作为参数进行学习。
  * 优点：灵活性强，可以更好地贴合训练数据。
  * 缺点：无法外推到超过训练长度的序列。目前大多数模型（如 GPT、BERT）都使用此种表示。
* **其他变体**：
  * 相对线性位置注意力 (Relative linear position attention) [Shaw et al., 2018]
  * 基于依存句法的非线性位置 [Wang et al., 2019]

---

## 页面 47 - 48: 自注意力的障碍与对策：缺乏非线性
* **障碍**：自注意力中没有逐元素的非线性激活函数，叠加多层自注意力仅相当于对 Value 向量进行反复的加权平均。
* **对策：加入前馈神经网络 (Feed-Forward Networks, FFN)**：
  * 在自注意力输出后，对每个向量独立应用一个前馈网络（也称为 MLP）：
    $$\mathbf{m}_i = \text{FFN}(\mathbf{o}_i) = \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 \mathbf{o}_i + \mathbf{b}_1) + \mathbf{b}_2$$

---

## 页面 49 - 51: 自注意力的障碍与对策：不能“偷看未来”
* **障碍**：在做语言建模或机器翻译解码时，预测当前词不能用到未来的词的信息，但自注意力会默认看到全局。
* **对策：掩码自注意力 (Masked Self-Attention)**：
  * 为了实现高效的并行训练，我们不改变 Query 和 Key 的大小，而是通过将未来位置的注意力得分人为设为 $-\infty$，使其在 Softmax 后权重归零：
    $$e_{ij} = \begin{cases} \mathbf{q}_i^\top \mathbf{k}_j, & j \le i \\ -\infty, & j > i \end{cases}$$

---

## 页面 52: 自注意力网络积木的必需组件总结
* **自注意力**：计算的核心基石。
* **位置编码**：赋予网络感知顺序的能力。
* **非线性层 (MLP/FFN)**：在注意力输出后应用，赋予深度表示学习的能力。
* **掩码 (Masking)**：防止在序列预测时发生未来信息的泄漏。

---

## 页面 53 - 55: 5. Transformer 解码器 (Transformer Decoder)
* Transformer 解码器是构建 GPT 等语言模型的核心架构。
* 它由上述的自注意力积木和一些优化机制构成：包含多头自注意力、残差连接以及层归一化。

---

## 页面 56: 自注意力的矩阵并行计算形式
给定拼接后的输入矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$：
* 映射矩阵：$\mathbf{X}_Q = \mathbf{X}\mathbf{Q}$, $\mathbf{X}_K = \mathbf{X}\mathbf{K}$, $\mathbf{X}_V = \mathbf{X}\mathbf{V}$。
* 并行计算公式：
  $$\text{output} = \text{softmax}\left(\frac{\mathbf{X}_Q \mathbf{X}_K^\top}{\sqrt{d_k}}\right) \mathbf{X}_V$$
* 通过一次大矩阵乘法，我们可以高效并行地算出所有位置对的注意力权重并求和。

---

## 页面 57 - 58: 多头自注意力机制 (Multi-Head Self-Attention)
* **动机**：我们可能希望词 $i$ 同时关注序列中不同位置的多个词。单个注意力头只能把权重聚焦在某一个区域。
* **定义**：通过定义多个头，每个头拥有独立的 $\mathbf{Q}_\ell, \mathbf{K}_\ell, \mathbf{V}_\ell \in \mathbb{R}^{d \times d/h}$ 矩阵。
* 各自独立计算注意力输出：
  $$\text{output}_\ell = \text{softmax}\left(\frac{\mathbf{X}\mathbf{Q}_\ell (\mathbf{X}\mathbf{K}_\ell)^\top}{\sqrt{d/h}}\right) \mathbf{X}\mathbf{V}_\ell$$
* 将所有头的输出进行拼接，并使用线性投影矩阵 $\mathbf{Y}$ 进行融合：
  $$\text{output} = [\text{output}_1, \dots, \text{output}_h] \mathbf{Y}$$
* > [!TIP]
  > **效率说明**：虽然包含多个头，但通过对矩阵进行分块和 Reshape，可以使用并行的 Tensor 操作高效完成，基本不增加额外的计算负担。

---

## 页面 59: 缩放点积 (Scaled Dot Product)
* 为什么在点积后要除以 $\sqrt{d/h}$？
* 当向量维度大时，两向量点积的绝对值会偏大。这会导致输入 Softmax 的数值差很大，使得 Softmax 输出的概率趋向于极端的 one-hot 状态，进而导致**梯度变得极小（梯度消失）**。
* 除以 $\sqrt{d/h}$（通常为每个头的维度）可以控制方差，使训练更为平稳。

---

## 页面 60: Transformer 优化机制
除了自注意力外，Transformer 的成功高度依赖两个优化机制（常合称为 Add & Norm）：
1. **残差连接 (Residual Connections)**
2. **层归一化 (Layer Normalization)**

---

## 页面 61: 残差连接 (Residual Connections)
* 传统的层映射为 $\mathbf{X}^{(i)} = \text{Layer}(\mathbf{X}^{(i-1)})$。
* 残差层映射为：
  $$\mathbf{X}^{(i)} = \mathbf{X}^{(i-1)} + \text{Layer}(\mathbf{X}^{(i-1)})$$
* **优势**：
  * 恒等映射的梯度流是 1，极大地缓解了深层网络训练时的梯度消失问题。
  * 使损失函数的能量图景观 (Loss Landscape) 变得更平滑，极易优化。

---

## 页面 62: 层归一化 (Layer Normalization)
* 层归一化旨在通过将每个位置的特征向量归一化为均值为 0、方差为 1 的分布，以稳定并加速训练。
* 对于向量 $\mathbf{x} \in \mathbb{R}^d$，均值为 $\mu$，标准差为 $\sigma$：
  $$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
  其中 $\gamma$ 和 $\beta$ 是可学习的增益和偏置参数，用于恢复网络的表达能力。

---

## 页面 63: Transformer 解码器结构总结
* Transformer 解码器由多个解码器块 (Block) 堆叠而成，每个 Block 包含：
  1. **Masked 多头自注意力机制** (Masked Multi-Head Self-Attention)
  2. **Add & Norm** (残差连接与层归一化)
  3. **前馈神经网络** (FFN/MLP)
  4. **Add & Norm**

---

## 页面 64: Transformer 编码器 (Transformer Encoder)
* Transformer 解码器限制了信息的流动，仅支持单向上下文。
* 如果需要双向上下文（如 bidirectional RNN），则可以使用 **Transformer 编码器**。
* **唯一区别**：移除了自注意力机制中的遮蔽掩码 (Masking)，每个词可以同时关注左边和右边的所有词。

---

## 页面 65 - 66: Transformer 编码器-解码器架构 (Encoder-Decoder)
* 广泛应用于 Seq2Seq 任务（如机器翻译）：
  * **编码器**（不使用掩码的自注意力）构建源句子的特征表示。
  * **解码器**在每一步不仅做 Masked 自注意力，还做一次**交叉注意力 (Cross-Attention)**：
    * 交叉注意力的 Keys 和 Values 来自**编码器的输出**（相当于外部记忆）。
    * 交叉注意力的 Queries 来自**解码器上一层的输入**。

---

## 页面 67: 实验效果展示 (BLEU 分数)
* 在 WMT 2014 英德、英法翻译任务中，基于 Transformer 的机器翻译系统相比于先前的循环网络不仅取得了更高的 BLEU 分数，而且**训练效率大幅提高**。

---

## 页面 68: 扩展应用：文档生成
* Transformer 被广泛扩展至文本摘要和长文档生成任务中，表现出了极佳的长序建模能力。

---

## 页面 69 - 70: Transformer 的局限与未来挑战
1. **二次复杂度开销 (Quadratic Compute)**：
   * 自注意力计算中所有位置两两交互，导致计算量和内存占用随着序列长度 $n$ 呈**二次方级**增长 ($O(n^2)$)。
   * 如果序列长度达到 50,000 以上（处理超长文档），计算代价将变得不可接受。
2. **位置编码的改进**：
   * 简单的绝对位置编码是最佳选择吗？学者们探索了相对位置编码 (Relative Position Attention) 等方案。

---

## 页面 71: 真的需要消除注意力机制的二次复杂度吗？
* 虽然二次复杂度开销很大，但随着模型变大，自注意力部分的耗时占比实际上在下降，而 MLP 的开销在上升。
* 此外，许多尝试降低复杂度的线性注意力变体，在超大规模参数下效果往往不如标准自注意力。
* 那么，设计更廉价的注意力机制是否还有意义？或者我们能否通过正确的设计彻底解锁超长上下文（如 >100k tokens）模型的潜力？
