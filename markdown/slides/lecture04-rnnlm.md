# Lecture 4: Language Models and Recurrent Neural Networks

## 页面 1: 标题
* **课程名称**：基于深度学习的自然语言处理 (Natural Language Processing with Deep Learning)
* **课程编号**：CS224N / Ling284
* **主讲人**：Diyi Yang
* **主题**：第 4 讲：语言模型和循环神经网络 (Lecture 4: Language Models and Recurrent Neural Networks)

---

## 页面 2: 课程计划
1. **新的 NLP 任务**：语言建模 (Language Modeling) (20分钟)
2. **基于神经网络的语言模型**：循环神经网络 (Recurrent Neural Networks, RNNs) (25分钟)
3. **RNN 的问题**：梯度爆炸与梯度消失 (Exploding and Vanishing Gradients) (20分钟)
4. **机器翻译**：(Machine Translation) (10分钟)

> [!IMPORTANT]
> **提示**：作业 2 (Assignment 2) 将于 1 月 22 日（星期四）截止。
> 本节课内容是本门课程中最重要的概念！它孕育并引领了绝大多数现代 NLP 技术。

---

## 页面 3: 1. 语言建模 (Language Modeling)
* **语言建模**是预测下一个单词的任务：
  * 例如：`the students opened their ______`
* **更正式的定义**：给定一个单词序列 $x^{(1)}, x^{(2)}, \dots, x^{(t)}$，计算下一个单词 $x^{(t+1)}$ 的概率分布：
  $$P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)})$$
  其中 $x^{(t+1)}$ 可以是词表 $V$ 中的任意单词。
* 执行该预测的系统称为**语言模型 (Language Model)**。
* 例如：下一个词可能是 `exams` (0.4), `minds` (0.1), `laptops` (0.2), `books` (0.3) 等。

---

## 页面 4: 语言建模 (Language Modeling)
* 你也可以将语言模型看作是一个**为文本片段分配概率**的系统。
* 例如，若有一段文本 $x^{(1)}, \dots, x^{(T)}$，则根据语言模型，该文本的概率为：
  $$P(x^{(1)}, \dots, x^{(T)}) = \prod_{t=1}^{T} P(x^{(t)} \mid x^{(t-1)}, \dots, x^{(1)})$$
* 这正是我们的语言模型 (LM) 所提供的核心能力。

---

## 页面 5 & 6: 语言模型的日常应用
* 你每天都在使用语言模型！
  * **搜索引擎的联想输入/自动补全** (Search autocomplete)
  * **智能手机的输入法预测** (Predictive typing)

---

## 页面 7: 为什么我们需要关注语言建模？
* **基准任务**：语言建模是一项基准任务，能帮助我们衡量在预测语言使用方面的进展。
* **NLP 任务的基石**：语言建模是许多 NLP 任务的核心子组件，尤其是涉及生成文本或估计文本概率的任务：
  * 预测性输入 (Predictive typing)
  * 语音识别 (Speech recognition)
  * 手写识别 (Handwriting recognition)
  * 拼写/语法纠错 (Spelling/grammar correction)
  * 作者身份识别 (Authorship identification)
  * 机器翻译 (Machine translation)
  * 文本摘要 (Summarization)
  * 对话系统 (Dialogue) 等
* **现代 NLP 的基石**：NLP 的其他所有内容都已经基于语言建模进行了重构——例如，ChatGPT 就是一个语言模型！

---

## 页面 8: 下一个词预测能做些什么？
足够强大的语言模型可以完成极其多样化的任务：
* **常识/问答 (Trivia)**: `Stanford University is located in __________, California.` $\rightarrow$ `Stanford` / `Palo Alto`
* **句法 (Syntax)**: `I put ___ fork down on the table.` $\rightarrow$ `the` / `my`
* **共指消解 (Coreference)**: `The woman walked across the street, checking for traffic over ___ shoulder.` $\rightarrow$ `her`
* **词汇语义/主题 (Lexical semantics/topic)**: `I went to the ocean to see the fish, turtles, seals, and _____.` $\rightarrow$ `sharks` / `whales`
* **情感分析 (Sentiment)**: `Overall, the value I got from the two hours watching it was the sum total of the popcorn and the drink. The movie was ___.` $\rightarrow$ `bad` / `terrible`
* **逻辑推理 (Reasoning - 较难)**: `Iroh went into the kitchen to make some tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the ______.` $\rightarrow$ `kitchen`
* **基础算术 (Basic arithmetic)**: `I was thinking about the sequence that goes 1, 1, 2, 3, 5, 8, 13, 21, ____` $\rightarrow$ `34`

---

## 页面 9: n-gram 语言模型 (n-gram Language Models)
* **问题**：如何学习语言模型？
* **旧方法（深度学习之前）**：学习一个 **n-gram 语言模型**！
* **定义**：n-gram 是由 $n$ 个连续单词组成的文本片段。
  * **unigram (一元)**: `"the"`, `"students"`, `"opened"`, `"their"`
  * **bigram (二元)**: `"the students"`, `"students opened"`, `"opened their"`
  * **trigram (三元)**: `"the students opened"`, `"students opened their"`
  * **four-gram (四元)**: `"the students opened their"`
* **核心想法**：收集关于不同 n-gram 出现频率的统计数据，并利用这些数据来预测下一个单词。

---

## 页面 10: n-gram 语言模型 (续)
* **马尔可夫假设 (Markov assumption)**：假定当前词 $x^{(t+1)}$ 仅取决于前面的 $n-1$ 个单词（统计近似）：
  $$P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)}) \approx P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(t-n+2)})$$
* **根据条件概率定义**，我们可以通过在大型文本语料库中计数来估计概率：
  $$P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(t-n+2)}) = \frac{P(x^{(t+1)}, x^{(t)}, \dots, x^{(t-n+2)})}{P(x^{(t)}, \dots, x^{(t-n+2)})} \approx \frac{\text{Count}(x^{(t-n+2)}, \dots, x^{(t+1)})}{\text{Count}(x^{(t-n+2)}, \dots, x^{(t)})}$$

---

## 页面 11: n-gram 语言模型：示例
假设我们正在学习一个 **4-gram 语言模型**。
* 上下文：`as the proctor started the clock, the students opened their _____`
* 根据马尔可夫假设，舍弃远期上下文，仅对最近的 3 个词进行条件约束：
  * `students opened their` 出现了 1000 次
  * `students opened their books` 出现了 400 次 $\rightarrow P(\text{books} \mid \text{students opened their}) = 0.4$
  * `students opened their exams` 出现了 100 次 $\rightarrow P(\text{exams} \mid \text{students opened their}) = 0.1$
* **思考**：我们直接抛弃 `"proctor"` 等上下文是否合理？

---

## 页面 12: n-gram 语言模型的稀疏性问题 (Sparsity Problems)
> [!WARNING]
> 增加 $n$ 会使稀疏性问题呈指数级恶化。通常我们无法使 $n$ 大于 5。

* **稀疏性问题 1**：如果 `"students opened their w"` 在数据中从未出现过怎么办？
  * 导致分子为 0，使得单词 $w$ 的预测概率为 0！
  * **(部分) 解决方案**：为词表中每个单词 $w \in V$ 的计数都加上一个微小的 $\delta$。这被称为**平滑 (Smoothing)**。
* **稀疏性问题 2**：如果 `"students opened their"` 在训练数据中从未出现过怎么办？
  * 导致分母为 0，我们无法计算任何单词的条件概率！
  * **(部分) 解决方案**：退而仅以 `"opened their"` 作为条件。这被称为**回退 (Backoff)**。

---

## 页面 13: n-gram 语言模型的存储问题 (Storage Problems)
* **存储瓶颈**：模型需要存储在语料库中观察到的所有 n-gram 的计数。
* 增加 $n$ 或增大训练语料库的大小都会导致模型体积急剧增大！

---

## 页面 14: 实践中的 n-gram 语言模型
* 你可以在几秒钟内在笔记本电脑上基于 170 万词的语料库 (Reuters 新闻数据) 构建一个简单的 trigram 语言模型。
* 给定上下文：`today the _______`，得到的概率分布如下（新闻风格）：
  * `company` (0.153)
  * `bank` (0.153)
  * `price` (0.077)
  * `italian` (0.039)
  * `emirate` (0.039)
  * ...
* **稀疏性问题**显而易见：概率分布的粒度非常粗糙，许多合理的词由于未被观测而分不到合理的概率。

---

## 页面 15 - 17: 使用 n-gram 语言模型生成文本
* 可以通过对估计得到的概率分布进行**重复采样**来生成文本：
  1. 给定 `today the`，计算分布，并从中采样出 `price`。
  2. 将新词拼接，以 `today the price` 作为新的上下文，计算分布，并从中采样出 `of`。
  3. 拼接得到 `today the price of`，采样出 `gold`。

---

## 页面 18: 文本生成效果示例
* **生成的文本样例**：
  > *"today the price of gold per ton , while production of shoe lasts and shoe industry , the bank intervened just after it considered and rejected an imf demand to rebuild depleted european stocks , sept 30 end primary 76 cts a share ."*
* **特点**：
  * 局部句法看起来惊人地符合语法规范！
  * 但整体**完全不连贯** (incoherent)。
* **结论**：如果想很好地建模语言，我们需要同时考虑比 3 个词长得多的上下文。然而，增加 $n$ 会导致稀疏性急剧增加且模型体积爆炸。

---

## 页面 19: 如何构建神经网络语言模型？
* **任务回顾**：
  * **输入**：词序列 $x^{(1)}, x^{(2)}, \dots, x^{(t)}$
  * **输出**：下一个词的概率分布 $P(x^{(t+1)} \mid x^{(t)}, \dots, x^{(1)})$
* 考虑使用一个**基于固定窗口的神经网络模型 (Window-based Neural Model)**？（类似于在命名实体识别 NER 中使用的窗口模型）。

---

## 页面 20 & 21: 固定窗口神经网络语言模型 (A Fixed-Window Neural LM)
* **结构图解**：
  1. **输入**：窗口内的前几个词（例如 4 个词：`the`, `students`, `opened`, `their`）。
  2. **向量化**：转换为 one-hot 向量。
  3. **词嵌入**：查表得到对应的词向量并进行**拼接 (Concatenate)**，得到向量 $\mathbf{e}$。
  4. **隐藏层**：输入到线性层加激活函数 $\mathbf{h} = f(\mathbf{W}\mathbf{e} + \mathbf{b}_1)$。
  5. **输出层**：$\hat{\mathbf{y}} = \text{softmax}(\mathbf{U}\mathbf{h} + \mathbf{b}_2)$，在整个词表 $V$ 上输出概率分布。

---

## 页面 22: 固定窗口神经网络语言模型：优缺点
* **相较于 n-gram LM 的改进**：
  * **无稀疏性问题**：词向量的相似性能够共享泛化信息。
  * **省空间**：无需存储所有观测到的 n-gram 的组合计数。
* **仍未解决的问题**：
  * **窗口固定且太小**。
  * **无法处理任意长度的输入**：一旦增大窗口，权重矩阵 $\mathbf{W}$ 就会线性增长。
  * 输入中每个位置 of 词对应不同的权重列，**没有参数共享与对称性 (Symmetry)**。
* **我们需要一种可以处理任意长度输入的全新神经网络架构！**
* *注：此经典模型由 Yoshua Bengio 等人于 2000/2003 年提出 (A Neural Probabilistic Language Model)。*

---

## 页面 23: 2. 循环神经网络 (Recurrent Neural Networks, RNN)
* **核心思想**：在时间步上**重复应用相同的权重矩阵**。
* **架构特点**：
  * 可以处理任何长度的输入序列。
  * 隐状态（Hidden States）会将信息向后传递。
  * 共享参数，利用时序上的循环机制对序列进行建模。

---

## 页面 24: 简单的 RNN 语言模型 (Vanilla RNN LM)
* **计算机制**：
  * 在时刻 $t$，词序列中的词 $x^{(t)}$（以词向量形式）与前一步的隐状态 $h^{(t-1)}$ 结合：
    $$\mathbf{h}^{(t)} = \text{sigmoid} (\mathbf{W}_{hh} \mathbf{h}^{(t-1)} + \mathbf{W}_{xh} \mathbf{x}^{(t)} + \mathbf{b}_1)$$
  * 计算输出的概率分布（在词表 $V$ 上）：
    $$\hat{\mathbf{y}}^{(t)} = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}^{(t)} + \mathbf{b}_2)$$
  * 其中 $\mathbf{h}^{(0)}$ 是一个初始隐状态向量（可以是全零或可学习参数）。

---

## 页面 25: RNN 语言模型：优缺点分析
* **RNN 的优势 (Advantages)**:
  1. 可以处理**任意长度**的输入。
  2. 理论上，第 $t$ 步的计算可以**利用很多步之前的信息**。
  3. 模型大小不会因为输入上下文变长而增加（权重矩阵大小固定）。
  4. 每一步均应用相同的权重 $\mathbf{W}_{hh}, \mathbf{W}_{xh}$，体现了处理的时序对称性。
* **RNN 的劣势 (Disadvantages)**:
  1. **串行计算速度慢**：无法并行化，因为必须先计算出 $\mathbf{h}^{(t-1)}$ 才能计算 $\mathbf{h}^{(t)}$。
  2. **长期依赖丢失**：在实际训练中，很难访问很多步之前的信息（梯度消失/爆炸）。

---

## 页面 26: 训练 RNN 语言模型
* **数据准备**：获取一个由词序列组成的大型文本语料库。
* **前向传播**：将词序列输入 RNN-LM，在每一个时间步 $t$ 计算输出分布 $\hat{\mathbf{y}}^{(t)}$。
* **损失函数**：第 $t$ 步的损失是预测分布 $\hat{\mathbf{y}}^{(t)}$ 与真实下一个词 $\mathbf{y}^{(t)}$（即 $x^{(t+1)}$ 的 one-hot 向量）之间的**交叉熵损失 (Cross-entropy Loss)**：
  $$J^{(t)}(\theta) = -\sum_{w \in V} y^{(t)}_w \log \hat{y}^{(t)}_w = -\log \hat{y}^{(t)}_{x_{t+1}}$$
* **总损失**：对训练集中所有时间步的损失取平均：
  $$J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta)$$

---

## 页面 27 - 31: 训练过程与 Teacher Forcing
* **Teacher Forcing (教师强制/教师迫导)**：
  * 在训练阶段，无论模型在第 $t$ 步的预测结果是否正确，我们在第 $t+1$ 步都将**真实单词**作为输入喂给模型。
  * 每一步的损失会累加起来得到整体的语料库损失 $J(\theta)$。

---

## 页面 32: 训练中的工程实践
* **挑战**：一次性在整个语料库上计算损失 and 梯度的显存/内存开销过大。
* **实践做法**：将文本拆分为句子（或固定长度的段落），利用**随机梯度下降 (SGD)** 按批次 (Batch) 进行更新：
  * 输入一个 Batch 的句子，执行前向传播，计算损失，通过反向传播计算梯度并更新权重，然后重复此过程。

---

## 页面 33: RNN 中的反向传播 (Backpropagation)
* **核心问题**：由于相同的权重矩阵 $\mathbf{W}_{hh}$（简记为 $\mathbf{W}$）被重复使用，如何求损失 $J^{(t)}$ 关于 $\mathbf{W}$ 的偏导数？
* **结论**：关于重复权重的梯度是**该权重在每一次出现时所产生的梯度的累加和**。
  $$\frac{\partial J^{(t)}}{\partial \mathbf{W}} = \sum_{i=1}^{t} \left. \frac{\partial J^{(t)}}{\partial \mathbf{W}} \right|_{\text{第 } i \text{ 步}}$$

> **[学习注释：整体梯度计算拼图澄清]**
> * **这仅是计算的一部分**：求单步损失 $J^{(t)}$ 对重用权重 $\mathbf{W}_{hh}$ 的偏导数只是整个网络梯度计算的一部分。我们还需要计算对输入权重 $\mathbf{W}_{xh}$、输出权重 $\mathbf{W}_{hy}$ 及其相应偏置项的偏导。
> * **从单步损失到总损失**：在训练中，我们实际上要最小化整个序列的总损失 $J(\theta) = \frac{1}{T} \sum_{t=1}^{T} J^{(t)}(\theta)$。因此，在求出每个单步损失 $J^{(t)}$ 产生的梯度后，需遍历 $t$ 从 $1$ 到 $T$ 对其进行累加，最后取平均，才是用于参数更新的完整梯度。

---

## 页面 34: 多元微积分链式法则
* 这是由多元微积分的链式法则决定的：如果一个变量通过多条路径影响最终结果，则总导数是所有路径导数的和。

> **[学习注释：链式法则中“加法”与“乘法”融合的核心洞见]**
> * **困惑澄清**：为什么共享参数的求导会产生“加法累加”？直觉上隐状态是通过“乘法链式法则”相连的。
> * **核心洞见推导**：以一个 3 步标量 RNN ($h_3=f(w, h_2), h_2=f(w, h_1), h_1=f(w, h_0)$) 为例。由于每一时间步的隐状态都显式且直接依赖于 $w$，根据多元微积分全导数公式，对隐状态 $h_3$ 递归代入求导：
>   $$\frac{d h_3}{d w} = \frac{\partial h_3}{\partial w} + \frac{\partial h_3}{\partial h_2} \frac{d h_2}{d w}$$
>   $$\frac{d h_2}{d w} = \frac{\partial h_2}{\partial w} + \frac{\partial h_2}{\partial h_1} \frac{d h_1}{d w}$$
>   因为 $h_0$ 与 $w$ 无关，所以 $\frac{d h_1}{d w} = \frac{\partial h_1}{\partial w}$。代入展开可得：
>   $$\frac{d h_3}{d w} = \underbrace{\frac{\partial h_3}{\partial w}}_{\text{第3步直接梯度}} + \underbrace{\frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial w}}_{\text{第2步经 } h_3 \text{ 传导的梯度}} + \underbrace{\frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial h_1} \frac{\partial h_1}{\partial w}}_{\text{第1步经 } h_3, h_2 \text{ 传导的梯度}}$$
>   这三项在数学上是**加法累加关系**。它们每一项的内部则是通过**乘法链式法则**逐层向前传播的。这就是为什么随时间反向传播（BPTT）可以写成求和形式的本质原因。

---

## 页面 35: 随时间反向传播 (Backpropagation Through Time, BPTT)
* 在计算梯度时，我们需要从当前步 $t$ 反向传播到先前的各个时间步 $i = t, \dots, 0$，并将梯度累加。
* 该算法被称为**随时间反向传播 (Backpropagation Through Time, BPTT)** [Werbos, 1988]。
* 链式法则展开：
  $$\frac{\partial J^{(t)}}{\partial \mathbf{W}} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial \mathbf{h}^{(t)}} \frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{h}^{(i)}} \frac{\partial \mathbf{h}^{(i)}}{\partial \mathbf{W}}$$
* > [!TIP]
  > **截断 BPTT (Truncated BPTT)**：在实际应用中，为了节省算力和显存，我们通常会在反向传播大约 20 步后进行截断。

---

## 页面 36: 使用 RNN 语言模型生成文本
* **自动生成 (Roll-outs)**：
  * 通过采样方式进行文本生成：
    1. 给定起始符 `<s>`，输入模型得到分布，采样得到 `my`。
    2. 将 `my` 输入模型，采样得到 `favorite`。
    3. 重复此过程，直到采样到结束符 `</s>`。

---

## 页面 37 & 38: 趣味文本生成实例
* **多样化训练**：可以在任何风格的文本上训练 RNN-LM。
  * 例如：哈利波特小说风格生成、烹饪菜谱文本生成。虽然逻辑有时很荒诞，但句式和用词风格模仿得很逼真。

---

## 页面 39: 评估语言模型：困惑度 (Perplexity)
* 语言模型的标准评估指标是**困惑度 (Perplexity, PPL)**。
* 它是交叉熵损失的指数形式：
  $$\text{Perplexity} = \exp(J(\theta)) = \exp \left( -\frac{1}{T} \sum_{t=1}^{T} \log \hat{y}^{(t)}_{x_{t+1}} \right) = \prod_{t=1}^{T} \left( \frac{1}{\hat{y}^{(t)}_{x_{t+1}}} \right)^{1/T}$$
* **物理意义**：它是词表大小的加权有效分支数。**困惑度越低，代表模型预测得越准，模型性能越好！**

---

## 页面 40: 3. RNN 的问题：梯度消失与梯度爆炸

---

## 页面 41 - 45: 梯度消失直观理解 (Vanishing Gradient Intuition)
根据随时间反向传播的链式法则，计算梯度时包含这一项：
$$\frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{h}^{(i)}} = \prod_{j=i+1}^{t} \frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}}$$
如果雅可比矩阵 $\frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}}$ 中的元素很小（或者权重矩阵 $\mathbf{W}_{hh}$ 的最大特征值小于 1），在连续连乘多步之后，梯度信号在反向传播的过程中会呈指数级衰减。
这被称为**梯度消失问题 (Vanishing Gradient Problem)**。

---

## 页面 46: 为什么梯度消失是个严重问题？
* 远处传入的梯度信号太弱，会被近处传入的强梯度信号所**淹没**。
* **参数更新失衡**：模型参数更新时，只会根据近距离的上下文进行调整，无法学到文本的长期依赖关系 (Long-term effects)。

---

## 页面 47: 梯度消失对 RNN-LM 的负面效果示例
* **例句**：
  > *"When she tried to print her **tickets**, she found that the printer was out of toner. She went to the stationery store to buy more toner. It was very overpriced. After installing the toner into the printer, she finally printed her **________**"* (此处应填 `tickets`)
* **挑战**：为了学到这个依赖，RNN-LM 的梯度需要跨越几十个单词的时间步。
* **结果**：如果梯度消失，模型无法在训练中建立两处 `tickets` 之间的关联，最终在测试阶段将无法预测类似的远距离相关词。

---

## 页面 48: 为什么梯度爆炸是个问题？
* **梯度爆炸 (Exploding Gradients)**：若权重矩阵的特征值大于 1，梯度在连乘时会呈指数级增长。
* **后果**：
  * 在 SGD 更新中，步长过大：$\theta \leftarrow \theta - \alpha \nabla_{\theta} J(\theta)$。
  * 会导致参数被更新到非常糟糕的状态，损失函数值急剧增加（所谓的“ Iowa 效应”）。
  * 严重时，会导致网络计算出现 `Inf` 或 `NaN`，从而被迫中断训练。

---

## 页面 49: 解决方案：梯度裁剪 (Gradient Clipping)
* **梯度裁剪**：如果梯度的范数 (norm) 超过某个设定的阈值，则先将其等比例缩减，然后再用于 SGD 更新。
  $$\text{if } \|\mathbf{g}\| > \text{threshold}, \quad \mathbf{g} \leftarrow \text{threshold} \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|}$$
* **直观理解**：沿着梯度方向迈出，但强行控制每一步的步长。
* 在实践中，梯度裁剪非常有效，使得梯度爆炸成为了一个相对容易解决的工程问题。

---

## 页面 50: 如何从架构上根本解决梯度消失问题？
* **问题本质**：Vanilla RNN 每次计算隐状态都在不停地进行**重写 (Rewrite)**，导致历史信息难以保留。
* **改进思路**：
  1. 引入能够通过“相加”而不是“相乘”来保留信息的独立记忆机制 $\rightarrow$ **长短期记忆网络 (LSTM)**。
  2. 在模型中设计更直接、更线性的直连通道 (Shortcut/linear pass-through) $\rightarrow$ **注意力机制 (Attention)**、**残差连接 (Residual Connections)**。

---

## 页面 51: 5. 机器翻译 (Machine Translation)
* **机器翻译 (MT)**：将源语言（source language）的句子 $x$ 翻译成目标语言（target language）的句子 $y$。
  * 例如：$x$: `I like deep learning` $\rightarrow$ $y$: `我喜欢深度学习`

---

## 页面 52: 神经机器翻译 (NMT) 的崛起
* **重大飞跃**：神经机器翻译从 2014 年的一项前沿尝试，到 2016 年仅用两年时间便成为了行业的主流标准。
  * **2014 年**：第一篇 Seq2Seq 论文发表 [Sutskever et al. 2014]。
  * **2016 年**：Google 翻译系统从传统的统计机器翻译 (SMT) 切换至神经机器翻译 (NMT)。
  * **震撼之处**：基于数百名工程师多年心血构建的 SMT 系统，在短短几个月内就被少数工程师训练的 NMT 系统在性能上全面超越。

---

## 页面 53: 神经机器翻译 (NMT)：架构
* **Sequence-to-Sequence (Seq2Seq) 架构**：
  * **编码器 RNN (Encoder RNN)**：读取源句子，生成源句子的编码向量，作为解码器 RNN 的初始隐状态。
  * **解码器 RNN (Decoder RNN)**：一个条件语言模型，接收编码器的表示，在每一步逐步输出目标语言的单词。
  * *注：测试时，前一步解码器采样的预测词会作为当前步的输入提供给解码器。*

---

## 页面 54: Seq2Seq 的广泛应用
Seq2Seq 模型本质上是通用的**编码器-解码器 (Encoder-Decoder) 架构**：
* 编码器将输入映射为隐层神经表示。
* 解码器基于此神经表示生成目标序列。
* **适用任务**：
  * **文本摘要**：长文本 $\rightarrow$ 短文本
  * **对话系统**：历史对话 $\rightarrow$ 下一回复
  * **句法分析**：输入文本 $\rightarrow$ 解析树序列
  * **代码生成**：自然语言说明 $\rightarrow$ 代码序列

---

## 页面 55: 条件语言模型 (Conditional Language Model)
* Seq2Seq 是**条件语言模型**的典型代表：
  * 解码器是预测目标词的“语言模型”。
  * 它的预测以源句子 $x$ 为“条件”。
* NMT 直接计算如下联合概率：
  $$P(y \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x)$$
* **训练方式**：使用双语平行语料库 (Parallel Corpus) 进行联合优化。

---

## 页面 56: 训练 NMT 系统
* NMT 作为**单个统一系统 (Single System)** 进行联合优化，反向传播是**端到端 (End-to-End)** 的。
* 损失函数 $J$ 是每个时间步负对数概率的平均值：
  $$J = \frac{1}{T} \sum_{t=1}^{T} J_t$$

---

## 页面 57: 多层深层编码器-解码器网络
* 编码器与解码器可以通过堆叠多层来增强表达能力。
* 隐状态在层与层之间向上流动：层 $i$ 的隐状态输出作为层 $i+1$ 的输入。

---

## 页面 58: 瓶颈问题 (Bottleneck Problem)
> [!WARNING]
> **RNN 架构的致命弱点**：编码器必须将源句子的所有信息压缩进一个固定长度的隐状态向量中。
> 这种强制压缩成为了信息的**瓶颈 (Bottleneck)**，随着源句子长度的增加，信息丢失会非常严重。

---

## 页面 59: 课程计划回顾与总结
* 我们本节课学习了：
  1. 语言建模 (Language Modeling) 任务。
  2. 用于语言建模的循环神经网络 (RNN)。
  3. RNN 存在的梯度消失与梯度爆炸问题。
  4. 机器翻译 (MT) 任务及基于 Seq2Seq 架构的神经机器翻译 (NMT) 解决方案。
  5. 明确了 Seq2Seq 面临的编码器信息瓶颈问题。
