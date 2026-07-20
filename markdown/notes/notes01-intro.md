# CS224n 课程笔记 1：导论与 Word2Vec

**课程讲师**：Christopher Manning, John Hewitt  
**作者**：John Hewitt (johnhew@cs.stanford.edu)  
**往届笔记贡献者**：Francois Chaubard, Michael Fang, Guillaume Genthial, Rohit Mundra, Richard Socher  
**授课学期**：2023 冬季  

---

### 摘要
本篇笔记首先简要介绍了自然语言处理（NLP）领域，随后详细讨论了 Word2Vec，以及通过从分布信号（distributional signal）中进行学习，将单词表示为低维实值向量这一基础且美妙的核心思想。

---

## 1 自然语言处理简介 (Introduction to Natural Language Processing)

**自然语言处理（NLP）**是一门专注于开发和研究能够自动理解与生成自然语言（即人类语言）的系统的科学与工程学科。

### 1.1 人类与语言 (Humans and language)
人类语言是一种极其高效的沟通工具，能够让人们便捷地分享和存储复杂的想法、事实和意图。正如 [Manning, 2022] 所指出的，语言所实现的极高沟通复杂度是人类在所有物种中独有的智能体现。对于致力于创建和研究智能系统的科学家与工程师而言，人类语言不仅是一个迷人的研究对象（毕竟，语言在演化过程中变得既易于学习又非常实用），同时也是在各种人机交互场景（即使在同时存在视觉等其他模态的场景中）中与人类进行交互的绝佳媒介。

### 1.2 语言与机器 (Language and machines)
人类儿童在与丰富的多模态世界互动以及接收各种反馈的过程中，以极高的数据效率（无需观察海量的语言数据）和极高的计算效率（人类大脑是极其高效的计算器官！）学会了语言。尽管在过去的几十年中，NLP 领域取得了令人瞩目的进展，但我们目前仍然无法开发出在语言习得能力上能达到儿童一小部分水平的学习机器。

在构建语言学习机器的过程中，一个根本（且目前仍未完全解决）的问题是**表示（Representation）**问题：我们应该如何在计算机中表示语言，以便计算机能够稳健地处理 and/or 生成语言？这正是本课程的焦点所在。我们将深入探讨**深度学习（Deep Learning）**提供的工具，这是一套对于表示自然语言的千变万化及其所遵循的规则与结构非常有效的工具包。本课程的大部分内容都将围绕“表示”这一问题展开，而本篇笔记的其余部分则将探讨一个更基础的子问题：**我们如何表示单词（Words）？** 

在此之前，让我们先简要讨论一些在学习了现代 NLP 技术后有望构建的实际应用。

### 1.3 NLP 的一些应用 (A few uses of NLP)
自然语言处理算法的实用性日益增加，并且得到了广泛的应用，但它们的失败和局限性在很大程度上仍然不够透明，有时也难以检测。以下是几个主要的 NLP 应用领域（此列表仅为激发兴趣，并非详尽无遗）：

- **机器翻译 (Machine Translation, MT)**：这可能是自然语言处理最早、最成功且最具推动性的应用之一。机器翻译系统学习在不同语言之间进行翻译，在当今的数字世界中无处不在。然而，系统在世界上大约 7000 种语言中的大多数上面仍然面临失败，且在翻译长文本以及确保译文的上下文准确性方面也存在困难，因此这仍然是一个硕果累累的研究领域。
- **问答与信息检索 (Question Answering and Information Retrieval)**：“问答”这个概念可能看起来过于宽泛——难道我们不能把任何问题都表示为问答吗？但在 NLP 中，问答通常指与寻找信息相关的疑问（例如，“阿布扎比的埃米尔是谁？”、“我如何获得英国的实习生签证？”）。不断拓宽可回答问题的范围、为答案提供出处（provenance）、在交互式对话中回答问题，这些是目前演进最快的研究方向。
- **文本摘要与分析 (Summarization and Analysis of Text)**：了解（1）人们在谈论什么，以及（2）他们对这些事物的看法是非常有价值的。企业希望进行市场调研，政治家希望了解公众舆论，个人则希望将复杂的议题浓缩为易于理解的摘要。对于公众而言，NLP 工具可以成为提高信息获取能力的强大武器；但同时，它们也可能被用于企业或政府的监控。随着学习的深入以及你决定自己要构建什么样的系统时，请时刻牢记这种“双重用途（dual use）”的特性。
- **语音（或手语）转文本 (Speech (or Sign)-to-Text)**：将口头或手势语言（音频或视频）自动转录为文本表示是一项规模庞大且非常实用的应用。不过在本课程中，我们将基本不涉及这部分内容。这在一定程度上是历史和方法论原因：原始信号处理的方法 and 专业知识通常在其他课程（例如 224s）及其他研究社区中讲授，尽管近年来两者的技术出现了一些融合。

在 NLP 的所有维度中，大多数现有的工具仅适用于世界上大约 7000 种语言中极少数的几种（通常是一种，最多到 100 种），并且在面对使用人口较少的语言和/或边缘化的方言、口音等方面时，其错误率会不成比例地增高。此外，近年来构建更好系统的成功速度已经远远超出了我们对其进行特征刻画和审计的能力。从种族、性别到宗教等文本中编码的偏见，往往会在 NLP 系统中得到反映甚至放大。

带着这些挑战与思考，同时怀揣着做优秀的科学研究以及构建能够改善人们生活的、值得信赖的系统的渴望，让我们一起来探究 NLP 中第一个迷人的问题。

---

## 2 词表示 (Representing Words)

### 2.1 能指与所指 (Signifier and Signified)
考虑以下句子：
> *Zuko makes the tea for his uncle.*（祖寇给他的叔叔沏了茶。）

这里的单词 **Zuko** 是一个记号（sign），一个在某些（现实或想象的）世界中代表实体“祖寇”的符号。单词 **tea** 同样也是一个符号，指向它所代表的事物（所指，signified）——这里或许指某杯具体的茶。

如果我们换个说法：
> *Zuko likes to make tea for his uncle.*（祖寇喜欢给他的叔叔沏茶。）

注意，此时符号 **Zuko** 仍然指向祖寇，但 **tea** 指代的则是一个更广泛的类别——泛指茶，而不是某杯具体的热茶。

现在考虑以下两个句子：
1. *Zuko makes the coffee for his uncle.*（祖寇给他的叔叔冲了咖啡。）
2. *Zuko makes the drink for his uncle.*（祖寇给他的叔叔倒了饮料。）

哪一个句子与关于“茶（tea）”的句子“更相似”？“饮料（drink）”可能是茶（但也可能非常不同），而“咖啡（coffee）”绝对不是茶，但它和茶却非常相似，不是吗？另外，**Zuko** 和 **uncle** 是否相似，因为它们都描述了人？**the** 和 **his** 是否相似，因为它们都指向了一个类别的具体实例？

词义（Word meaning）是无穷无尽且极其复杂的，它源于人类互相交流和在世界上实现目标的意图。人类使用连续的媒介（如语音、手势）来进行表达，但却在一个离散的、符号化的结构（即语言）中产生记号，以表达复杂的含义。在保留语言所要达到的强信息传递效果的同时，表达和处理语言中所有的细微差别与千变万化，使得词表示成为一个极其迷人的问题。下面我们来看看具体的方法。

### 2.2 独立单词，独立向量 (Independent words, independent vectors)
什么是单词？我无法为你给出一个绝对的定义，但我可以给出几个英语中的例子：*tea*, *coffee*, *abbreviate*, *gumption*。在这里，我特地将单词 *antiridate* 定义为：“看着一个不可食用的装饰品并希望它像看起来那样美味的动作”。如果我使用这个记号成功与他人交流了我的渴望，那么对我而言，它就已经足够被称作一个单词了。

最简单的词表示方法可能就是将它们视作独立、互不相关的实体。你可以将其想象为一个集合：
$$\{\dots, \text{tea}, \dots, \text{coffee}, \dots, \text{antiridate}\}$$

这里我们引入一些术语：
> **词型 (Word Type)**：有限词汇表中的一个元素，与该词在上下文中的实际观察无关（即抽象的词）。上述写出的集合即为一组词型。
> 
> **词符 (Word Token)**：词型的一个具体实例，例如在某个上下文中观察到的单词。

我们目前的词表示方法为每个词型提供了一个单一的表示，并且在上下文中，我们会对该词型的任何词符出现都使用这同一个表示。

在本课程中，我们通常会使用向量来工作。表示独立成分的传统向量表示法是**独热向量（1-hot vectors）**或标准基向量集合。例如：
$$v_{\text{tea}} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad v_{\text{coffee}} = \begin{bmatrix} \vdots \\ 0 \\ 0 \\ 1 \\ \vdots \end{bmatrix} \tag{1}$$
其中 $v_{\text{tea}} = e_3$（第 3 个标准基向量），而 $v_{\text{coffee}} = e_j$（第 $j$ 个标准基向量）。

我们为什么要将单词表示为向量？答案是为了更好地利用它们进行计算。当我们使用独热向量进行计算时，我们确实能够区分不同的单词，但遗憾的是，**我们没有编码任何有意义的相似度或其他关系**。这是因为，如果我们使用点积（Dot Product）来定义相似度（或者使用 $L_2$ 距离、$L_1$ 距离等）：
$$v_{\text{tea}}^\top v_{\text{coffee}} = v_{\text{tea}}^\top v_{\text{the}} = 0 \tag{2}$$
所有的单词之间的相似度都是 0，它们都是同等程度不相似的。同时请注意，在这种表示中，单词并没有以任何方式（例如按字母顺序）排序。这是一个重要的细节：除了严格的同一性（即“这个单词的字符/字节序列是否与另一个单词完全相同？如果是，则它们具有相同的向量；如果不是，则它们具有完全独立的向量”）之外，这些字符串中没有任何显式的字符级信息。

由于在实际中显然并不是所有的单词都同等程度地互不相似，因此我们需要寻找其他的替代方案。

### 2.3 来自标注离散属性的向量 (Vectors from annotated discrete properties)
我们是否可以不用独热向量，而是将词义表示为一组特征，以及它们与语言学类别和其他单词之间的关系？

对于任何一个单词，比如 *runners*（跑步者们），我们可以标注大量的相关信息。这里有语法信息（如“复数”），有派生信息（如单词 *runners* 是由动词 *to run* 加上表示“施事者”或主体的后缀 *-er* 构成），还有语义信息（如 *runners* 可以是 *humans*、*animals* 或 *entities* 的下位词。下位词是 is-a 关系中的一员；例如，runner 是一种 human）。

目前，英语和少数其他语言中已经存在了大量用于各类单词标注信息的现有资源。例如，**WordNet** [Miller, 1995] 标注了同义词、下位词和其他语义关系；**UniMorph** [Batsuren et al., 2022] 标注了多种语言的词法（子词结构）信息。利用这些资源，我们可以构建如下所示的词向量：
$$v_{\text{tea}} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ \vdots \\ 1 \end{bmatrix} \begin{matrix} \text{(复数名词)} \\ \text{(单数第三人称动词)} \\ \text{(饮料的下位词)} \\ \vdots \\ \text{(chai 的同义词)} \end{matrix} \tag{3}$$

但在当今（2023年及以后），由这些方法得到的词向量并不是主流，它们也不会是本课程的重点。其主要缺陷如下：
1. **人工标注资源的词汇量总是匮乏的**。与能够从自然文本源中自动提取词汇的方法相比，人工标注资源的更新成本极其高昂，且它们永远是不完整的。
2. **向量维度与嵌入的实用性之间存在权衡**。表示所有这些类别需要一个极其高维的向量（维度要远远大于词汇表大小），而倾向于在稠密向量（dense vectors）上运行的现代神经网络方法在面对此类向量时表现并不好。
3. **数据驱动优于人工设计**。本课程将贯穿一个持续的主题：当有大量的数据可供学习时，人类关于“文本应该具有什么样表示”的设计，往往不如允许数据来决定更多属性的方法表现得好。

---

## 3 分布语义学与 Word2vec (Distributional Semantics and Word2vec)

深度学习的一个承诺是从数据中学习复杂对象的丰富表示。在 NLP 中，越来越重要的思想是我们可以**无监督（unsupervised）地从数据中学习丰富的表示**。无监督（或者近来被称为“自监督”，self-supervised）学习利用数据并尝试学习其中元素的属性，这通常是通过获取数据的一部分（例如句子中的一个词）并试图用它来预测数据的其他部分（其他的词）来实现的。

在语言学中，这一想法在多年前就被 Firth [Firth, 1957] 很好地捕捉到了，他有一句名言：
> *You shall know a word by the company it keeps.*  
> （你应当通过一个词的邻居来了解它。）

从高层面上看，你可以将出现在单词 *tea* 周围的单词分布视作定义该词含义的方法。例如，*tea* 经常与 *drank*, *the*, *pot*, *kettle*, *bag*, *delicious*, *oolong*, *hot*, *steam* 等词一起出现。显然，与 *tea* 相似的单词（如 *coffee*）将具有相似的上下文单词分布。虽然这个想法很简单，但它是所有现代 NLP 中最重要、最成功的思想之一，且其类似物已经在无数与机器学习相关的领域中扎根。

> **分布式假说 (Distributional Hypothesis)**：单词的含义可以从它所出现的上下文分布中推导出来。

以上是高层面的理念。但一如既往，细节决定成败。单词“在另一个单词附近”具体是什么意思？（是紧挨着它？距离两个词？在同一个文档中？）我们该如何表示这种编码并进行学习？让我们来探讨一些方案。

### 3.1 共现矩阵与文档上下文 (Co-occurrence matrices and document contexts)
如果你被要求写代码来实现“通过单词出现的上下文单词分布来表示它”这一想法，你可能会立即想到以下步骤：
1. 确定一个词汇表 $V$。
2. 创建一个大小为 $|V| \times |V|$ 的全零矩阵。
3. 遍历一组文档。对于每个文档中的每个单词 $w$，将文档中其他单词 $w'$ 的出现次数累加到对应 $w$ 的行、对应 $w'$ 的列上。
4. 对行进行归一化（使每一行的和为 1）。

这样你就为你的词汇表构建了一个文档级别的**共现矩阵（Co-occurrence Matrix）**！我们将该矩阵记为 $X$。词嵌入 $X_{\text{tea}} \in \mathbb{R}^{|V|}$（$X$ 的一行）显然比我们之前所拥有的独热向量 $e_{\text{tea}}$ 要有用得多。

这里我们做出了一个决定：文档级别的共现。但我们也可以说，只有当 $w'$ 出现得距离 $w$ 非常近（比如在几个单词以内）时，才认为 $w'$ 与 $w$ 共现。下面是一个标记了几个相关共现窗口的例子：

> **共现范围的影响**：更大范围的共现（例如大窗口或整个文档）会导致表示更偏向于编码语义甚至主题（Topic）信息；而更短的窗口则会导致表示更偏向于编码语法（Syntax）信息。

$$\text{[It's hot and delicious. [I poured } \underbrace{\text{[the tea]}}_{\text{中心词}} \text{ for]}_1 \text{ my uncle]}_3\text{. ]}_{\text{document}}$$

简而言之：
- **较短的窗口**（如上面标记为 $1$ 的单侧单词窗口）似乎更多地编码了**语法属性**。例如，名词倾向于紧挨着 *the* 或 *is* 出现，而复数名词则不会紧挨着 *a* 出现。
- **较大的窗口**倾向于编码更多的**语义属性**（在极端情况下，是类似于主题的属性）。注意像 *poured* 或 *delicious* 可能会出现在距离 *tea* 较远的位置，但它们依然与茶密切相关。对于长文档（数千字），文档级窗口归纳上是通过单词出现在什么类型的文档（如体育、法律、医学等）中来表示单词。

我们的另一个设计决定是在 $|V|$ 维向量中表示单词的显式计数。事实证明这通常是个很大的错误。
我们之前已经提到过，高维向量在今天的神经系统中往往难以处理。另一个问题是，单词的原始计数会过度强调像 *the* 这样非常常见单词的重要性。使用**对数词频（log token frequency）**被证明要有用得多。

关于原始共现计数方法存在什么缺陷，一篇极具影响力的论文通过引入 **GloVe** [Pennington et al., 2014] 给了我们很多启示。GloVe 是一种基于共现的词表示算法，其效果与我们将在下一节介绍的 Word2Vec 一样好。然而，由于 Word2Vec 的许多细节在课程后续讨论的各种方法中仍然适用，我们将把时间主要集中在 Word2Vec 上。

### 3.2 Word2vec 模型与目标函数 (Word2vec model and objective)
Word2Vec 模型将固定词汇表中的每个单词表示为一个低维（维度远远小于词汇表大小）向量。它通过一个简单的函数，基于通常很短的上下文（例如 2-4 个单词）中的单词分布，来学习每个单词的向量值，使其具有预测性。我们在这里描述的模型称为 **Skip-gram Word2Vec** 算法。

#### Skip-gram Word2Vec
通常，我们有一个有限的词汇表 $V$。设 $C, O$ 是随机变量，代表一对未知的 $C \in V$（中心词，Center word）和 $O \in V$（外部词，Outside word，出现在中心词的上下文中）。我们使用 $c, o$ 来指指这两个随机变量的具体取值。

设 $U \in \mathbb{R}^{|V| \times d}$ 和 $V \in \mathbb{R}^{|V| \times d}$。请注意，词汇表 $V$ 中的每个单词都与 $U$ 的一行以及 $V$ 的一列（或行，这取决于具体的矩阵排列）相关联。我们可以将这视作对 $V$ 进行任意排序的结果。Word2Vec 概率模型具体定义如下，其中 $u_w$ 代表矩阵 $U$ 中对应单词 $w \in V$ 的行向量，而 $v_c$ 代表矩阵 $V$ 中对应单词 $c \in V$ 的列/行向量：
$$p_{U,V}(o|c) = \frac{\exp(u_o^\top v_c)}{\sum_{w \in V} \exp(u_w^\top v_c)} \tag{4}$$

你可能对这个函数很熟悉，它就是 **Softmax 函数**。它接收任意的分数（这里，分数为词汇表中每个单词与中心词向量的点积），并生成一个概率分布，其中分数较高的单词获得更高的概率。请注意，在给定中心词的情况下，所有单词的概率向量 $p_{U,V}(\cdot | c) \in \mathbb{R}^{|V|}$ 与我们旧有的共现矩阵中的一行 $X_c$ 非常相似。

这只是一个模型。我们该如何估计参数 $U$ 和 $V$ 的值呢？我们的学习目标是最小化与真实分布 $P^*(O|C)$ 的**交叉熵损失（Cross-Entropy Loss）**：
$$\min_{U,V} \mathbb{E}_{o,c} \left[ -\log p_{U,V}(o | c) \right] \tag{5}$$

这个方程应当读作：“在参数 $U$ 和 $V$ 上，最小化当 $o$ 和 $c$ 从真实分布 $O$ 和 $C$ 中抽取时，在 $(U, V)$ 模型下给定 $c$ 时 $o$ 的负对数似然概率的期望值。”

这里有许多丰富的细节需要深入：
- 我们如何执行最小化（min）操作？
- 我们如何“获取”随机变量 $O$ 和 $C$？
- 为什么要取概率的负对数？
- 为什么这比共现计数要好得多？
- 你能看出为什么并非所有给定 $c$ 时 $o$ 的分布都能被这个模型表示吗？（这应该是好事？坏事？令人惊讶？还是显而易见？）

现在，让我们来看看在实践中如何实现这一点的几个细节。

### 3.3 从语料库估计 Word2vec 模型 (Estimating a word2vec model from a corpus)
在实践中我们如何训练 Word2Vec 呢？从上面给出的数学公式中定义 Word2Vec 模型是相对清晰的：我们构建矩阵 $U$ 和 $V$，并能写出概率的数学表达式。然而，如何估计参数可能还不够明显：（1）对于给定的 $U$ 和 $V$ 值，如何计算公式 (5) 中的期望值；（2）如何执行最小化操作。让我们从第一点开始。

#### Word2vec 经验损失 (Word2vec empirical loss)
设 $D$ 为一个文档集 $\{d\}$，其中每个文档都是一个单词序列 $w_1^{(d)}, \dots, w_m^{(d)}$，且所有的 $w \in V$。设 $k \in \mathbb{N}_{++}$ 为正整数窗口大小。让我们来定义我们的中心词随机变量 $C$ 和外部词随机变量 $O$ 如何与这个具体的显式数据集相联系。

$C$ 依次取每个文档中的每一个单词 $w_i$ 的值，而对于每一个这样的 $w_i$，外部词的集合为 $\{w_{i-k}, \dots, w_{i-1}, w_{i+1}, \dots, w_{i+k}\}$。因此，我们公式 (5) 的目标函数在经验上变为：
$$\mathcal{L}(U, V) = \sum_{d \in D} \sum_{i=1}^{m} \sum_{1 \le |j| \le k, 1 \le i+j \le m} -\log p_{U,V}(w_{i+j}^{(d)} | w_i^{(d)}) \tag{6}$$
请注意，这里我们是在对（1）所有文档，求和（2）文档中的所有单词，求和（3）出现在窗口内的所有外部词在给定中心词时的负对数似然。

现在，我们如何进行最小化？

#### 基于梯度的估计 (Gradient-based estimation)
在高层面上，我们通过从一个相对无信息的初始猜测开始，并迭代地朝着局部最能改善猜测的方向移动，来寻找使我们指定的目标函数“好”的 $U$ 和 $V$。这是通过基于梯度的方法完成的。简而言之，标量函数 $f$ 关于参数矩阵 $U$ 的梯度（可以理解为导数）$\nabla_U f$ 代表了为了最大化增加 $f$ 的值而对 $U$ 进行局部移动的方向。

因此，在实践中，我们首先随机初始化 $U^{(0)}$ 和 $V^{(0)}$，使 $U, V \sim \mathcal{N}(0, 0.001)^{|V| \times d}$（从均值为 0、方差很小的正态分布中进行独立采样的矩阵），然后执行若干次以下过程的迭代：
$$U^{(t+1)} = U^{(t)} - \alpha \nabla_U \mathcal{L}(U^{(t)}, V^{(t)}) \tag{7}$$
这代表将迭代 $t+1$ 时的 $U$ 值设为前一次迭代的值，加上在能够局部最能改善目标函数 $\mathcal{L}(U, V)$ 的反方向上迈出的一个小步（$\alpha$ 为较小的步长/学习率）。

#### 随机梯度 (Stochastic gradients)
这里还有一个关键的细节（除了如何计算梯度函数 $\nabla_U(\cdot)$ 之外）：计算 $\mathcal{L}(U, V)$ 极其昂贵，因为它需要遍历整个数据集。我们不精确计算目标函数，而是进行**随机梯度下降（Stochastic Gradient Descent, SGD）优化**，在公式 (7) 的每一步中使用少量的样本来近似 $\mathcal{L}(U, V)$。我们可能会通过采样文档 $d_1, \dots, d_\ell \sim D$ 并计算近似损失：
$$\hat{\mathcal{L}}(U, V) = \sum_{d \in \{d_1, \dots, d_\ell\}} \sum_{i=1}^{m} \sum_{1 \le |j| \le k, 1 \le i+j \le m} -\log p_{U,V}(w_{i+j}^{(d)} | w_i^{(d)}) \tag{8}$$

### 3.4 梯度推导与直观理解 (Working through a gradient)
Word2Vec 的梯度更新步骤是如何影响参数的？让我们推导一下数学并建立一些直观理解。

特别地，我们将写出对于单个中心词 $c$ 和单个外部词 $o$，损失关于参数 $v_c$ 的偏梯度。首先，我们将梯度算子穿过求和号：
$$\nabla_{v_c} \hat{\mathcal{L}}(U, V) = \sum_{d \in D} \sum_{i=1}^m \sum_{1 \le |j| \le k} -\nabla_{v_c} \log p_{U,V}(w_{i+j}^{(d)} | w_i^{(d)}) \tag{9}$$
直观上，求和中所有这些项的梯度就是它们各自梯度的和。现在，让我们计算单个项概率的梯度。为了符号简明，我们再次将 $w_{i+j}^{(d)}$ 记为 $o$，将 $w_i^{(d)}$ 记为 $c$。

因此，我们对求和中的单个项使用对数规则进行展开：
$$\nabla_{v_c} \log p_{U,V}(o | c) = \nabla_{v_c} \log \left( \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^n \exp(u_w^\top v_c)} \right) \tag{10}$$
$$= \underbrace{\nabla_{v_c} \log \exp(u_o^\top v_c)}_{\text{Part A}} - \underbrace{\nabla_{v_c} \log \sum_{w=1}^n \exp(u_w^\top v_c)}_{\text{Part B}} \tag{11}$$

#### Part A 求解
我们先对 Part A 进行求导，因为它更简单：
$$\nabla_{v_c} \log \exp(u_o^\top v_c) = \nabla_{v_c} (u_o^\top v_c) \quad \text{（互逆操作）} \tag{12}$$
$$= u_o \quad \text{（为什么？）} \tag{13}$$

为了看清为什么最后一个等号成立，我们来考虑 $v_c$ 的每个单独维度。$u_o^\top v_c$ 输出的偏导数为：
$$\frac{\partial}{\partial v_{c,i}} (u_o^\top v_c) = \frac{\partial}{\partial v_{c,i}} \sum_{j} u_{o,j} v_{c,j} = u_{o,i}$$
这是因为求和中只有一项与 $v_{c,i}$ 相关，根据单变量微积分，其导数为 $u_{o,i}$。当我们将这组单变量导数重新拼回向量时，我们得到了 $[u_{o,1}, \dots, u_{o,d}] = u_o$。根据惯例，梯度的形状与被求导对象的形状相同，因而梯度的形状是 $u_o$ 而不是 $u_o^\top$。

#### Part B 求解
现在我们对 Part B 进行求导：
$$-\nabla_{v_c} \log \sum_{w=1}^n \exp(u_w^\top v_c) = -\frac{1}{\sum_{w=1}^n \exp(u_w^\top v_c)} \nabla_{v_c} \sum_{x=1}^n \exp(u_x^\top v_c) \quad \text{（对数求导与链式法则）}$$
$$= -\frac{1}{\sum_{w=1}^n \exp(u_w^\top v_c)} \sum_{x=1}^n \nabla_{v_c} \exp(u_x^\top v_c) \quad \text{（梯度的线性性质）}$$
$$= -\frac{1}{\sum_{w=1}^n \exp(u_w^\top v_c)} \sum_{x=1}^n \exp(u_x^\top v_c) \nabla_{v_c} (u_x^\top v_c) \quad \text{（指数求导与链式法则）}$$
$$= -\frac{1}{\sum_{w=1}^n \exp(u_w^\top v_c)} \sum_{x=1}^n \exp(u_x^\top v_c) u_x \quad \text{（点积求导）}$$

#### 结合 Part A 和 Part B
现在我们将 Part A 和 Part B 组合起来，并进行一些代数变形：
$$\nabla_{v_c} \log p_{U,V}(o | c) = u_o - \sum_{x=1}^n \left( \frac{\exp(u_x^\top v_c)}{\sum_{w=1}^n \exp(u_w^\top v_c)} \right) u_x$$
注意括号中的项正是模型预测的条件概率 $p_{U,V}(x | c)$。因此：
$$\nabla_{v_c} \log p_{U,V}(o | c) = u_o - \sum_{x=1}^n p_{U,V}(x | c) u_x \tag{14}$$
$$= u_o - \mathbb{E}[u_x]$$
$$= \text{“观察值”} - \text{“期望值”}$$

直观地，一切都归结为上面这行最后的公式：我们拥有实际观察到的外部词向量 $u_o$，从中我们**减去模型所期望的外部词向量**——即对词汇表中所有单词的向量乘以模型分配给该单词的概率后求和（即期望向量）。

因此，梯度更新使得 $v_c$ 向量发生改变，使其朝向“与实际观察到的外部词向量 $u_o$ 更相似”的方向移动，并远离“模型预测期望的词向量”的方向。

如果你没有完全跟上上面的数学推导，请不要担心；我建议你多读几遍，不要着急。如果你因为能迅速理解而觉得这枯燥无味，那么请用你多出来的空闲时间去帮助并指导其他人！

### 3.5 负采样 Skip-gram (Skip-gram Negative Sampling, SGNS)
既然我们已经在对梯度进行随机估计，那么在计算 $\hat{\mathcal{L}}(U, V)$ 时，估计 Word2Vec 模型剩下的一个效率瓶颈就是计算精确的模型概率 $-\log p_{U,V}(o | c)$。对于给定的单词对，计算未归一化的分数 $\exp(u_o^\top v_c)$ 是很廉价的。然而，计算分母中的配分函数（Partition function，即所有单词分数的求和）是极其昂贵的，因为词汇表中的每个单词都需要计算一项。

直观地，配分函数在起什么作用，我们又该如何规避它？让我们再次写出 Softmax 公式：
$$p_{U,V}(o|c) = \frac{\overbrace{\exp(u_o^\top v_c)}^{\text{单词 } c \text{ 与上下文 } o \text{ 的亲和度}}}{\underbrace{\sum_{w \in V} \exp(u_w^\top v_c)}_{\text{配分函数 / 归一化项}}} \tag{15}$$

从概率的角度来看，配分函数通过归一化分数使其和为 1，从而保证其为一个合理的概率分布（指数函数保证分数为非负值）。从学习的角度来看，配分函数在“压低”除观察到的词以外的所有其他单词的概率。换句话说，该公式的分子鼓励模型使 $u_o$ 与 $v_c$ 更相似；而分母则鼓励所有其他的 $u_w$（其中 $w \neq o$）与 $v_c$ 更不相似。

**负采样（Negative Sampling）**的直观想法是：我们不需要在每一步都去压低词汇表中的*所有* $u_w$，因为这正是绝大多数计算开销的来源。

在实际应用中，负采样 Skip-gram（SGNS）的目标函数会有所不同，我们将其写在下面：
$$\log \sigma(u_o^\top v_c) + \sum_{\ell=1}^k \log \sigma(-u_\ell^\top v_c) \tag{16}$$
其中 $\sigma$ 是逻辑（Logistic / Sigmoid）函数：
$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$
...
而 $u_\ell \sim p_{\text{neg}}$ 代表从负样本分布 $p_{\text{neg}}$ 中抽取的噪声词向量（目前你可以将其简单理解为在 $V$ 上的均匀分布）。

这个目标函数在做什么？它包含两项，这与我们描述的原始 Skip-gram 类似：
- 第一项（$\log \sigma(u_o^\top v_c)$）鼓励 $v_c$ 和 $u_o$ 彼此更相似（即最大化它们共现的概率）。
- 第二项（$\sum_{\ell=1}^k \log \sigma(-u_\ell^\top v_c)$）则鼓励 $v_c$ 与从词汇表中随机抽取的 $k$ 个负样本单词 $u_\ell$ 彼此更不相似。

这背后的直观解释是：如果我们每一步都随机压低少数几个词，那么平均而言，最终的效果会与我们每一步都去压低词汇表中的所有词非常接近。

---

## 附录 A 补充笔记 (Extra Notes)

### A.1 连续词袋模型 (Continuous Bag-of-Words, CBOW)
*(注：原笔记此处未展开，主要指与 Skip-gram 相反，CBOW 是通过上下文的词向量之和/平均值来预测中心词的词嵌入模型。)*

### A.2 奇异值分解 (Singular Value Decomposition, SVD)
*(注：原笔记此处未展开，主要指另一种获取词嵌入的经典代数方法：先构建全局共现矩阵，然后对其进行奇异值分解 $X = U \Sigma V^\top$，取 $U$ 的前 $d$ 列作为降维后的稠密词向量表示。)*

---

## 参考文献 (References)

- [Batsuren et al., 2022] Batsuren, K., et al. (2022). UniMorph 4.0: Universal Morphology. *LREC 2022*.
- [Baum and Petrie, 1966] Baum, L. E. and Petrie, T. (1966). Statistical inference for probabilistic functions of finite state markov chains. *The Annals of Mathematical Statistics*.
- [Bengio et al., 2003] Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A neural probabilistic language model. *JMLR*.
- [Collobert et al., 2011] Collobert, R., et al. (2011). Natural language processing (almost) from scratch. *CoRR*.
- [Firth, 1957] Firth, J. R. (1957). Applications of general linguistics. *Transactions of the Philological Society*.
- [Manning, 2022] Manning, C. D. (2022). Human Language Understanding & Reasoning. *Daedalus*.
- [Mikolov et al., 2013] Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *CoRR*.
- [Miller, 1995] Miller, G. A. (1995). WordNet: a lexical database for english. *CACM*.
- [Rong, 2014] Rong, X. (2014). word2vec parameter learning explained. *CoRR*.
- [Rumelhart et al., 1988] Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1988). Learning Representations by Back-propagating Errors. *MIT Press*.
