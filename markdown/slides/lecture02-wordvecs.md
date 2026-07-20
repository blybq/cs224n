## 页面 1: 深度学习自然语言处理

深度学习自然语言处理 (Natural Language Processing with Deep Learning)

CS224N/Ling284
Diyi Yang
第 2 讲：词向量 (Lecture 2: Word Vectors)

## 页面 2: 课程大纲

**第 2 讲：词向量**
1. 课程组织结构 (3 分钟)
2. Word2vec 介绍 (15 分钟)
3. Word2vec 目标函数梯度计算 (25 分钟)
4. 优化算法基础 (5 分钟)
5. 我们能否通过计数更有效地捕获词义的本质？ (10 分钟)
6. 评估词向量 (10 分钟)

**核心目标**：理解词义可以由实数组成的高维向量来表示，并在本节课结束时能够阅读词嵌入 (word embeddings) 相关的学术论文。

## 页面 3: 课程组织结构

* **旁听/候补名单**
* 如有其他疑问，请发送邮件至 cs224n-win2526-staff@lists.stanford.edu
* **欢迎参加答疑时间 (Office Hours)/辅导课 (Help Sessions)！**
  * 均于今日开始。
  * 欢迎前来讨论期末项目构想以及作业问题。
  * 尽量早来、多来，避开高峰期！
  * **助教答疑时间**：周一至周六，每段 3 小时，多位助教同时值班。
  * 直接过来即可！我们友善的课程工作人员将随时为您提供协助！
  * 详情访问：[Office Hours Page](https://web.stanford.edu/class/cs224n/office_hours.html)
* **授课教师答疑时间（默认线下进行）**：
  * Diyi: 每周二下午 3:30-4:30
  * Yejin: 每周五下午 4:30-5:30

## 页面 4: 如何表示单词的意思？

**定义：意思 (meaning) —— 《韦氏词典》**
* 单词、短语等所代表的思想/概念。
* 一个人想通过使用文字、符号等来表达的思想。
* 在写作、艺术作品等中表达的思想。

**语言学上最普遍的词义思考方式**：
能指 (signifier/符号) $\Longleftrightarrow$ 所指 (signified/概念或事物)
即**指称语义 (denotational semantics)**
“树 (tree)” $\Longleftrightarrow$ $\lbrace$各种树的图示或概念$\rbrace$

## 页面 5: 计算机如何获取可用的词义？

**以前最常用的 NLP 解决方案**：使用例如 WordNet 这样的词典，其中包含同义词集 (synonym sets/synsets) 和上位词 (hypernyms) 列表（“是一个” [is-a] 关系）。

*示例：包含 “good” 的同义词集，以及 “panda” 的上位词结构（使用 NLTK 库代码演示）：*

```python
from nltk.corpus import wordnet as wn

# 输出 "good" 的同义词集
poses = {'n':'noun', 'v':'verb', 's':'adj (s)', 'a':'adj', 'r':'adv'}
for synset in wn.synsets("good"):
    print("{}: {}".format(poses[synset.pos()], 
            ", ".join([l.name() for l in synset.lemmas()])))

# 输出 "panda" 的上位词链条
panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
```

## 页面 6: 类似 WordNet 的资源存在的问题

* **虽有用但缺乏细微差别**：
  * 例如：“proficient”（熟练的）被列为 “good”（好的）的同义词，但这仅在某些语境下正确。
* WordNet 在某些同义词集中列出了具有冒犯性的同义词，且完全没有涵盖单词的内涵意义 (connotations) 或适用性。
* **缺失单词的新含义**：
  * 例如：*wicked, badass, nifty, wizard, genius, ninja, bombest*
* **无法保持实时更新！**
* 具有主观性。
* 需要大量人工来创建和维护。
* **无法用于精确计算单词相似度**（见接下来的幻灯片）。

## 页面 7: 将单词表示为离散符号

在传统的 NLP 中，我们将单词视为离散符号：
*hotel, conference, motel* —— 这是一种**局部表征 (localist representation)**。

这些单词符号可以通过**独热向量 (one-hot vectors)** 来表示：
$$\text{motel} = [0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 1 \ 0 \ 0 \ 0 \ 0]$$
$$\text{hotel} = [0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 1 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0]$$

* 向量维度 = 词表大小 (vocabulary size)（例如，500,000+）
* 独热向量意味着只有一个元素为 1，其余全为 0。

## 页面 8: 离散符号表示单词存在的问题

**示例**：在网页搜索中，如果用户搜索“Seattle motel”（西雅图汽车旅馆），我们希望能够匹配包含“Seattle hotel”（西雅图酒店）的文档。
$$\text{motel} = [0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 1 \ 0 \ 0 \ 0 \ 0]$$
$$\text{hotel} = [0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 1 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0 \ 0]$$

但是，这两个向量是**正交的 (orthogonal)**。
对于独热向量，没有天然的相似度概念！

**解决方案**：
* 尝试依赖 WordNet 的同义词列表来获取相似度？但众所周知其效果极差（存在不完整性等）。
* **更好做法**：学习直接在向量内部对相似度进行编码。

## 页面 9: 通过上下文表示单词

* **分布语义学 (Distributional semantics)**：单词的含义由经常出现在其附近的词汇决定。
  * *“你应该通过一个词所处的环境（上下文）来认识它”* —— (J. R. Firth 1957: 11)
  * 这是现代统计 NLP 最成功的思想之一！
* 当单词 $w$ 出现在文本中时，其**上下文 (context)** 是出现在其附近（在固定大小的窗口内）的单词集合。
* 我们使用 $w$ 的众多上下文来构建 $w$ 的表征。

例如：
* "...government debt problems turning into **banking** crises as happened in 2009..."
* "...saying that Europe needs unified **banking** regulation to replace the hodgepodge..."
* "...India has just given its **banking** system a shot in the arm..."
* *（周围加粗的这些上下文词汇将用来表示 banking 的语义）*

## 页面 10: 词向量 (Word Vectors)

我们将为每个单词构建一个**稠密向量 (dense vector)**，使其与在相似上下文中出现的词向量相似。我们使用向量的点积（标量积）来衡量相似度。

*注：词向量也被称为**词嵌入 (word embeddings)** 或**（神经）词表征 (neural word representations)**。它们是一种**分布式表征 (distributed representation)**。*

$$\text{banking} = \begin{bmatrix} 0.286 \\ 0.792 \\ -0.177 \\ -0.107 \\ 0.109 \\ -0.542 \\ 0.349 \\ 0.271 \end{bmatrix} \quad \text{monetary} = \begin{bmatrix} 0.413 \\ 0.582 \\ -0.007 \\ 0.247 \\ 0.216 \\ -0.718 \\ 0.147 \\ 0.051 \end{bmatrix}$$

## 页面 11: Word2vec 概述

**Word2vec** 是学习词向量的框架 *(Mikolov et al. 2013)*。

**基本思想**：
* 准备一个大型语料库（“文本主体”）：一个长单词列表。
* 固定词表中的每个单词都用一个向量表示。
* 遍历文本中的每个位置 $t$，该位置包含一个**中心词 $c$ (center word)** 和**上下文（“外部”）词 $o$ (outside words)**。
* 利用 $c$ 和 $o$ 的词向量相似度来计算在给定 $c$ 时出现 $o$ 的概率（反之亦然）。
* 不断调整词向量以**最大化该概率**。

*（本页重点介绍 Skip-gram 模型）*

## 页面 12: Word2Vec 概述：示例窗口与计算过程

遍历过程及计算 $P(w_{t+j} | w_t)$ 的过程：
对于文本片段：*... crises banking into turning problems as ...*
* 中心词（位于位置 $t$）：*into*
* 窗口大小为 2 的外部上下文词：
  * 左侧：*crises*, *banking*
  * 右侧：*turning*, *problems*
* 分别计算条件概率：
  * $P(w_{t-2} | w_t) = P(\text{crises} | \text{into})$
  * $P(w_{t-1} | w_t) = P(\text{banking} | \text{into})$
  * $P(w_{t+1} | w_t) = P(\text{turning} | \text{into})$
  * $P(w_{t+2} | w_t) = P(\text{problems} | \text{into})$

## 页面 13: Word2Vec 概述：窗口移动示意

*(随着位置 $t$ 的移动，计算不同窗口下的条件概率，继续进行词向量调整)*

## 页面 14: Word2Vec：目标函数

对于每个位置 $t = 1, \dots, T$，在给定中心词 $w_t$ 的情况下，预测固定窗口大小 $m$ 内的上下文词。**数据似然度 (Data Likelihood)**：
$$L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \le j \le m \\ j \ne 0}} P(w_{t+j} | w_t; \theta)$$

**目标函数 $J(\theta)$** 是平均负对数似然度 (average negative log likelihood)：
$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} | w_t; \theta)$$

* **最小化目标函数 $\Longleftrightarrow$ 最大化预测准确率**
* $\theta$ 是所有待优化的模型参数。
* 目标函数有时也称为代价函数 (cost function) 或损失函数 (loss function)。

## 页面 15: Word2Vec：概率的公式表示

* 我们想要最小化目标函数：
  $$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} | w_t; \theta)$$
* **问题**：如何计算 $P(w_{t+j} | w_t; \theta)$？
* **答案**：我们为每个单词 $w$ 使用两个向量：
  * 当 $w$ 是**中心词**时，使用 $v_w$
  * 当 $w$ 是**上下文词**时，使用 $u_w$
* 此时，对于中心词 $c$ 和上下文词 $o$：
  $$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

## 页面 16: 用向量表示 Word2Vec

计算 $P(w_{t+j}|w_t)$ 过程中的向量运算形式：
* 简称：$P(u_{\text{problems}} | v_{\text{into}})$，即给定中心词向量 $v_{\text{into}}$ 时，输出上下文词向量 $u_{\text{problems}}$ 的概率。
* 分别计算各个上下文词的概率：
  * $P(u_{\text{banking}} | v_{\text{into}})$
  * $P(u_{\text{crises}} | v_{\text{into}})$
  * $P(u_{\text{turning}} | v_{\text{into}})$
  * $P(u_{\text{problems}} | v_{\text{into}})$

## 页面 17: Word2Vec：预测函数与 Softmax

$$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

这是 **Softmax 函数** $\mathbb{R}^n \to (0,1)^n$ 的一个应用：
$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)} = p_i$$

* Softmax 函数将任意实数值 $x_i$ 映射到概率分布 $p_i$。
* **“max”**：因为它会放大最大值 $x_i$ 的概率。
* **“soft”**：因为它依然会给较小值分配一些概率。
* 广泛应用于深度学习中。

**公式拆解**：
1. **点积 (Dot product)** $u^T v = u \cdot v = \sum_{i=1}^n u_i v_i$ 比较了 $o$ 和 $c$ 的相似度。点积越大 $\Rightarrow$ 概率越大。
2. **指数化 (Exponentiation)** 保证了概率值均为正数。
3. **归一化 (Normalize)** 在整个词表上进行求和归一，以形成合法的概率分布。

## 页面 18: 训练模型：优化参数以最小化损失

* 在训练模型时，我们逐渐调整参数以最小化损失。
* 参数 $\theta$ 代表模型中的所有参数，被拼接成一个极长的向量。
* 在我们的案例中，包含 $d$ 维词向量与大小为 $V$ 的词表，每个单词拥有中心词和上下文词两套向量，所以优化参数总量很大。
* 我们通过沿着梯度的相反方向（反向）移动来优化这些参数（即梯度下降）。
* 我们计算所有向量的梯度！

## 页面 19: 互动环节！

回顾核心计算公式：
* 目标函数（平均负对数似然度）：
  $$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} | w_t; \theta)$$
* 给定中心词 $c$ 和上下文词 $o$ 下的概率：
  $$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$

## 页面 20: 优化算法：梯度下降 (Gradient Descent)

* 我们拥有想要最小化的代价函数 $J(\theta)$。
* **梯度下降**是一种用于最小化 $J(\theta)$ 的迭代算法。
* **基本思想**：对于当前的 $\theta$ 值，计算 $J(\theta)$ 的梯度，然后在负梯度的方向上迈出微小的一步。不断重复此过程。
* *注：我们的优化目标函数可能不是凸的 (non-convex)，但实际应用中效果依然很好。*

## 页面 21: 梯度下降更新公式

* **矩阵形式的更新方程**：
  $$\theta^{\text{new}} = \theta^{\text{old}} - \alpha \nabla_{\theta} J(\theta)$$
* **单个参数的更新方程**：
  $$\theta_j^{\text{new}} = \theta_j^{\text{old}} - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$
* 其中 $\alpha$ 表示步长 (step size) 或**学习率 (learning rate)**。

## 页面 22: 随机梯度下降 (Stochastic Gradient Descent, SGD)

* **问题**：$J(\theta)$ 是整个语料库中所有窗口的函数（可能有数以十亿计！），因此计算完整梯度会非常昂贵。在进行单次更新之前，需要等待极长的时间。对于几乎所有的神经网络来说，这都是一个非常糟糕的主意。
* **解决方案：随机梯度下降 (SGD)**：
  * 重复对窗口进行采样，并在每个窗口后进行参数更新。
* **小批量梯度下降 (Mini-batch Gradient Descent)**：
  * 对一小批窗口（batch）进行采样并计算梯度，介于全量梯度下降和纯 SGD 之间。

## 页面 23: Word2vec 参数与计算过程

* 外部词矩阵 $U$ 和中心词矩阵 $V$ 的乘积计算，得出点积，并经过 Softmax 得到预测概率。
* 模型在每个位置都进行相同的预测。
* 我们希望模型对所有（经常）在上下文中出现的词给予合理较高的概率估计。
* 这是一种**“词袋”模型 (Bag of words model)**，因为不考虑上下文词在窗口内的相对顺序。

## 页面 24: Word2vec 空间分布特点

* Word2vec 通过在向量空间中将相似的单词放置在彼此临近的位置，来最大化目标函数。

## 页面 25: Word2vec 算法家族详解

**为什么设计中心词和上下文词两个向量？** $\Rightarrow$ 使得优化更容易。在训练结束时将两者取平均值。
* 当然，也可以只使用一个词向量来实现算法（这会有细微的性能提升）。

**两种模型变体**：
1. **Skip-gram (SG)**：在给定中心词的情况下，预测上下文（“外部”）单词（与位置无关）。
2. **连续词袋模型 (Continuous Bag of Words, CBOW)**：根据上下文词的（词袋）表征来预测中心词。
*本课程主要介绍的是 **Skip-gram** 模型。*

**训练损失函数**：
1. **原始 Softmax (Naïve softmax)**（简单但当输出类别过多时计算极其昂贵）。
2. 更优化的变体，如**层次 Softmax (hierarchical softmax)**。
3. **负采样 (Negative sampling)**。
*(到目前为止，我们解释的是原始 Softmax)*

## 页面 26: 带负采样的 Skip-gram 模型 (SGNS)

* 归一化项（Softmax 中的分母）的计算开销极大（因为需要对整个词表 $V$ 求和）：
  $$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}$$
* 因此，标准的 Word2vec 实现了**带负采样的 Skip-gram 模型**。
* **主要思想**：训练二分类逻辑回归 (binary logistic regressions) 来区分真实对（中心词及其上下文窗口内的词）与若干“噪声”对（中心词与随机挑选的非上下文词）。

> **[学习注释：关于“二分类逻辑回归”的理解澄清]**
> * **非额外模型**：此处的“训练二分类逻辑回归”并非额外训练一个分类模型，词向量本身就是逻辑回归的参数。它只是将原本在整个词表 $|V|$ 上的多分类（Softmax）简化为针对正样本（真实对，标签为 1）和 $k$ 个负样本（噪声对，标签为 0）的二分类逻辑回归损失（Sigmoid + 交叉熵）进行优化。

## 页面 27: 带负采样的 Skip-gram 模型公式

* 我们获取 $k$ 个负样本（使用词频分布概率）。
* **最大化**真实上下文词出现的概率；**最小化**随机单词出现在中心词周围的概率。
* 使用本课程一致的符号表示，我们最小化以下损失：
  $$J_{\text{neg-sample}}(u_o, v_c, U) = -\log \sigma(u_o^T v_c) - \sum_{k \in K_{\text{sampled}}} \log \sigma(-u_k^T v_c)$$
* 这里使用 **Sigmoid 函数**，而不是 Softmax：
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
* 采样概率：$P(w) = \frac{U(w)^{3/4}}{Z}$，即将一元语法分布 (unigram distribution) $U(w)$ 提升至 $3/4$ 次幂值后进行归一化。
  * $3/4$ 次幂的作用是提升低频词被采样的概率。

## 页面 28: 负采样下的随机梯度［补充说明］

* 我们在 SGD 的每个窗口中迭代地获取梯度。
* 在每个窗口中，由于采用了负采样，我们至多只有 $2m+1$ 个词加上 $2km$ 个负采样词，因此梯度 $\nabla_{\theta}J_t(\theta)$ 是**非常稀疏的 (very sparse)**！

## 页面 29: 负采样下的随机梯度［补充说明续］

* 我们可能只需要更新实际出现的词向量！
* **解决方案**：要么需要**稀疏矩阵更新操作 (sparse matrix update)** 以仅更新完整嵌入矩阵 $U$ 和 $V$ 的某些行，要么需要为词向量维护一个哈希表。
* 如果有数百万个词向量并进行分布式计算，避免发送巨大的完整矩阵更新是至关重要的！
* *注：在实际的深度学习包中，嵌入矩阵通常按行（Rows）而非列存储。*

## 页面 30: 为什么不直接捕获共现计数？

为什么我们要遍历整个语料库（可能很多遍）来逐步调整？为什么我们不直接累积所有关于“哪些单词出现在彼此临近位置”的共现统计数据呢？

**构建共现矩阵 (Co-occurrence Matrix) $X$**：
* **两个选择**：基于窗口 (Windows) 与基于全篇文档 (Full Document)。
* **基于窗口**：类似于 Word2vec，使用每个词周围的窗口 $\Rightarrow$ 捕获了一些句法和语义信息（构建“词空间”）。
* **词-文档共现矩阵**：将给出宏观主题（所有体育词汇都会有相似的项），进而实现**潜在语义分析 (Latent Semantic Analysis, LSA)**（构建“文档空间”）。

## 页面 31: 示例：基于窗口的共现矩阵

* 窗口长度为 1（通常更常用的是 5-10）。
* 对称矩阵（不区分左侧或右侧的上下文）。
* **示例语料库**：
  * *I like deep learning.*
  * *I like NLP.*
  * *I enjoy flying.*

**共现计数矩阵**：

| | I | like | enjoy | deep | learning | NLP | flying | . |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| **like** | 2 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| **enjoy** | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **deep** | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| **learning**| 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| **NLP** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| **flying** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| **.** | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |

## 页面 32: 共现向量与降维

* **简单计数共现向量**：
  * 向量的维度随着词表大小的增长而增长。
  * **维度极高**：需要大量的存储空间（即使其非常稀疏）。
  * 随后的分类模型会遇到稀疏性问题 $\Rightarrow$ 导致模型不够健壮。
* **低维向量 (Low-dimensional vectors)**：
  * **基本思想**：在一个固定的、较小的维度中存储“绝大部分”重要信息，形成一个**稠密向量**。
  * 通常是 25–1000 维，类似于 Word2vec。
  * 如何降低维度？

## 页面 33: 经典方法：对矩阵 $X$ 进行降维 (见 HW1)

对共现矩阵 $X$ 进行**奇异值分解 (Singular Value Decomposition, SVD)**：
* 将 $X$ 分解为 $U\Sigma V^T$，其中 $U$ 和 $V$ 是正交矩阵（Orthonormal matrices）。
* 为了进行泛化，仅保留前 $k$ 个最大的奇异值。
* $\hat{X}$ 是在最小二乘法意义下对 $X$ 的最佳秩 $k$ 近似（Rank $k$ approximation）。
* 这是经典的线性代数结果。但是，对于大型矩阵，计算 SVD 的开销非常昂贵。

> **[学习注释：关于 SVD 的掌握程度]**
> * **对后续理解无负面影响**：SVD 作为传统计数型词向量降维的核心方法，在现代 NLP（如 Word2vec, BERT, GPT 等）中已被基于梯度优化的方法取代。即使不知道其严格证明也不影响后续深度学习课程的理解。
> * **几何/物理直观**：它是一种代数工具，能够将高维稀疏的共现计数矩阵 $X$ 压缩为一个低维稠密的实数矩阵 $U$。其行向量即为降维后的词向量，在最小二乘意义下保留了原空间中的主要信息方差。

## 页面 34: 对矩阵 $X$ 的启发式处理 (Hacks)

* 直接在原始计数矩阵上运行 SVD 效果并不理想！
* 对单元格中的计数进行缩放（Scaling）可以提供很大帮助。
* **问题**：虚词/功能词（如 *the, he, has*）出现频率过高 $\Rightarrow$ 导致句法结构占主导。一些修补方法：
  * 对频数取对数（Log the frequencies）。
  * 设定阈值上限限制：$\min(X, t)$，其中 $t \approx 100$。
  * 直接忽略虚词。
  * 使用渐进衰减窗口（越近的词比越远的词计数权重更大）。
  * 使用相关系数（Correlations）代替纯计数，然后将负值设为 0。

> **[学习注释：为什么要对计数矩阵进行缩放（Scaling）？]**
> * **原因**：语料中存在大量如 *the, is* 等无语义的功能词/虚词，其共现频率极高。如果不做缩放，它们的高频共现会主导矩阵的方差，导致 SVD 学到的向量主要反映与这些虚词的共现频次，而不是真正的语义关联。
> * **缩放常用方法**：对数缩放（$\log(X_{ij}+1)$）以压缩高低频的量级差距；阈值上限截断（$\min(X_{ij}, t)$）以削弱超高频词影响；使用 PPMI（正点互信息）代替绝对计数，从而凸显强特异性的词语共现。

## 页面 35: 缩放向量中呈现的有趣语义模式

* **COALS 模型**（来自 Rohde et al. 2005，《一种基于词汇共现的改进语义相似度模型》）。在经过适当缩放后的向量空间中，可以自然浮现出有趣的词义语义模式。

## 页面 36: GloVe 模型

**GloVe 模型** *(Pennington, Socher, and Manning, EMNLP 2014)*：
* **核心思想**：在向量差异中编码语义要素。
* **关键问题**：我们如何才能将**共现概率的比率**捕获为词向量空间中的**线性语义要素**？

## 页面 37: GloVe：对数双线性模型

* **解答：对数双线性模型 (Log-bilinear model)**
* 使用向量差进行计算，其损失函数形式为：
  $$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$
* **优势**：
  * 训练速度极快。
  * 可扩展至超大型语料库。

> **[学习注释：GloVe 的设计理念与损失函数变量拆解]**
> * **解决的问题**：传统 LSA（基于全局共现矩阵，训练快，但线性类比建模差）与 Word2vec（基于局部窗口，类比效果好，但重复扫描全局效率低）的结合，旨在利用全局统计的同时保留线性类比语义。
> * **损失函数变量含义**：
>   * $w_i, \tilde{w}_j$：中心词 $i$ 与上下文词 $j$ 的 $d$ 维向量。
>   * $b_i, \tilde{b}_j$：词 $i$ 与 $j$ 的偏置项，用于吸收各词自身频次带来的影响。
>   * $X_{ij}$：全局语料中 $i$ 与 $j$ 的共现计数；$\log X_{ij}$ 为其对数。
>   * $f(X_{ij})$：权重函数。若 $X_{ij}=0$，则 $f(0)=0$，避免了 $\log 0$ 无意义；随共现频次递增，并在高频时达到饱和上限（如 $x_{\max}=100$ 权重为 1），防止高频功能词主导损失函数。
> * **推导直观逻辑**：GloVe 出发点是利用向量差 $(w_i-w_j)$ 和探针词 $\tilde{w}_k$ 的点积来拟合三词共现比率 $P_{ik}/P_{jk}$。借由指数与对数的同态转换性质（$\exp(A-B)=\exp(A)/\exp(B)$），将三词比率关系拆解并化简为两两词对之间的对数共现值逼近关系：$w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log X_{ij}$。

## 页面 38: 6. 如何评估词向量？

评估方法与 NLP 通用评估手段一致：**内部评估 (Intrinsic)** vs. **外部评估 (Extrinsic)**。

* **内部评估 (Intrinsic Evaluation)**：
  * 在一个特定的/中间子任务上进行评估。
  * 计算速度快。
  * 有助于理解该特定系统。
  * 除非能证明与实际终极任务存在正相关，否则其有效性可能不够明确。
* **外部评估 (Extrinsic Evaluation)**：
  * 在真实的终极任务上进行评估。
  * 计算准确率可能需要花费很长时间。
  * 如果表现不好，不清楚到底是该子系统的问题，还是它与其他子系统的交互问题，或者是其他子系统的问题。
  * 如果仅替换其中一个子系统就能提高真实任务的准确率 $\Rightarrow$ 那说明这是一次成功的提升！

## 页面 39: 词向量内部评估：词类比 (Word Analogies)

* **词向量类比任务**：通过评估向量加减后的余弦距离，检验其捕获直观语义和句法类比问题的能力。
  * 例如：$$\text{man} : \text{woman} \Longleftrightarrow \text{king} : \text{?}$$
  * 即寻找满足：$$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}}$$ 距离最近的向量。
  * *在检索搜索中，需要将输入的词汇从搜索结果中排除（！）。*
* **问题**：如果词义信息存在但并不是线性的关系怎么办？

## 页面 40: GloVe 词向量空间可视化

* 展示了在训练后的向量空间中，不同词对（如国家-首都、人称代词）之间呈现出的平行线性映射结构。

## 页面 41: 词义相似度评估

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

## 页面 42: 相关性评估

* 评估词向量计算出的余弦相似度与人类评分之间的相关系数（如 Spearman 相关系数）。

## 页面 43: 词向量外部评估

* 一个优秀词向量能直接提供帮助的经典示例：**命名实体识别 (Named Entity Recognition, NER)** —— 识别文本中对人名、组织或地理位置的引用：
  * *“Chris Manning lives in Palo Alto.”*

## 页面 44: 课程总结

**第 2 讲：词向量**
1. 课程组织结构
2. Word2vec 介绍
3. Word2vec 目标函数梯度计算
4. 优化算法基础
5. 我们能否通过计数更有效地捕获词义的本质？
6. 评估词向量

**核心目标达成**：理解词义可以由实数组成的高维稠密向量来表示。
