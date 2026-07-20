## 页面 1: 自然语言处理历史

自然语言处理历史 (History of Natural Language Processing)
CS224N/Ling284
Christopher Manning
第 1 讲

## 页面 2: 人类语言理解与推理

Christopher D. Manning: 人类语言理解与推理 (Human Language Understanding & Reasoning)

**NLP 的四个时代**
* **1940–1969：早期探索 (Early Explorations)**
* **1970–1992：手工构建的符号化 NLP 系统 (Hand-built symbolic NLP systems)**，形式化程度逐渐提高
* **1993–2012：统计或概率 NLP (Statistical or Probabilistic NLP)**，以及更通用的**监督机器学习 (Supervised ML) 用于 NLP**
* **2013–至今：深度学习或人工神经网络 (Deep Learning or Artificial Neural Networks)**；无监督或自监督 NLP (Unsupervised or Self-Supervised NLP)；强化学习 (Reinforcement Learning)

**仅有部分交织的领域**
* NLP / 计算语言学 (Computational Linguistics)
* AI / 神经网络 (Neural Networks)

*文献来源：Dædalus 151(2): 127–138 (Spring 2022) [链接](https://www.amacad.org/publication/human-language-understanding-reasoning)*

## 页面 3: 早期探索 (1940–1969)

## 页面 4: 机器翻译：NLP 与计算语言学的起源

“同样，虽然对官方消息一无所知，但对密码学中强大的新型机械化方法进行了相当多的猜测和推断——我相信即使在不知道被编码的是什么语言的情况下，这些方法也能成功——人们自然会想，翻译问题是否可以被视为密码学中的问题。当我看到一篇俄语文章时，我说：‘这其实是用英语写的，只是被编码成了一些奇怪的符号。我现在开始解码。’”
—— Warren Weaver（1955:18，引用他于 1947 年写的一封信）

## 页面 5: 两种视角的博弈

* “当我看到一篇俄语文章时，我说：‘这其实是用英语写的，只是被编码成了一些奇怪的符号。我现在开始解码。’” —— Warren Weaver, 1947 年 3 月
* “……至于机械翻译问题，坦率地说，我担心不同语言中词汇的 [语义] 边界过于模糊……以至于任何半机械化的翻译方案都很难看到希望。” —— Norbert Wiener, 1947 年 4 月

* **Wiener**：麻省理工学院 (MIT) 控制论 (cybernetics) 的创始人，致力于将生物体和计算机中的通信、控制与反馈结合在一起。
* **Weaver**：数学家和工程师，因在洛克斐勒基金会和 OSR&D（美国政府二战科学资助机构）担任科学资助人，以及与香农 (Shannon) 合著通俗易懂的《信息论》入门书而闻名。

## 页面 6: [无内容]

## 页面 7: [无内容]

## 页面 8: [无内容]

## 页面 9: NLP 的早期历史

**20 世纪 50 年代的机器翻译 (MT)**

## 页面 10: 人工智能的开端

**达特茅斯夏季研究项目 (The Dartmouth Summer Research Project 1956)**

## 页面 11: 神经网络

**从神经元知识中获得的启发**

## 页面 12: 神经网络的先驱：McCulloch & Pitts

* **Warren S. McCulloch** (1898–1969)：神经生理学家 (医学博士)、精神病学家和控制论学者，1951 年进入 MIT，但维纳 (Wiener) 在 1952 年与他和皮茨 (Pitts) 分道扬镳。
* **Walter Pitts** (1923–1969)：自学成才的逻辑学家和计算神经科学家，曾在芝加哥和 MIT 过着无家可归的生活；死于酒精中毒。

1943 年，发表了《神经活动中内在思想的逻辑演算》 (A Logical Calculus of Ideas Immanent in Nervous Activity)，这是**第一个神经网络数学模型**，证明了它可以实现图灵机。
*[了解更多](https://nautil.us/the-man-who-tried-to-redeem-the-world-with-logic-235253/)*

## 页面 13: McCulloch-Pitts 神经元

原始 McCulloch & Pitts (1943) 阈值单元：
$$1(Wx > \theta) = 1(Wx - \theta > 0)$$

**该函数没有斜率 (slope)，因此无法进行基于梯度的学习 (no gradient-based learning)。**

## 页面 14: Frank Rosenblatt 与 (Mark I) 感知机

* Frank Rosenblatt: (Mark I) 感知机 (Perceptron)
*[了解更多](https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon)*

## 页面 15: 早期的人工智能炒作！《纽约时报》 1958 年 7 月 8 日

**海军新设备通过实践进行学习**
**心理学家展示旨在阅读并变得更聪明的计算机雏形**

海军今天展示了一台电子计算机的雏形，预计它将能够行走、说话、看见、书写、自我复制并预测自身的真实存在。
该雏形——气象局耗资 2,000,000 美元的 “704” 计算机——在海军为新闻记者举行的演示中，经过 50 次尝试后学会了区分左右。

## 页面 16: 20 世纪 50 年代人工智能的两种图景

* **控制论 (Cybernetics)**：Wiener, Rosenblatt
* **（符号）人工智能 (Symbolic Artificial Intelligence)**：Minsky, McCarthy, Simon, Newell

## 页面 17: 信息检索的构想：Vannevar Bush

**Bush (1945): 《如我们所想》 (As We May Think)**
“设想一个供个人使用的未来设备，它是一种机械化的私密文件库和图书馆。它需要一个名字，随便编一个，就叫 ‘memex’ 吧。memex 是一个设备，个人可以在其中存储他所有的书籍、记录和通信，并且它是机械化的，以便能以极高的速度和灵活性进行查阅。它是对其记忆力的大幅扩展和亲密补充。”
*[阅读原文](https://www.theatlantic.com/magazine/archive/1945/07/as-we-may-think/303881/)*

* **Vannevar Bush**：曾任 MIT 工程学院院长、华盛顿卡内基研究所所长，二战期间担任美国国家防御研究委员会主席（该委员会拥有巨额资金，指导了二战期间的所有科学研究，包括曼哈顿计划）。

## 页面 18: Cyril Cleverdon 引入基准测试 (Benchmarking)

* **Cranfield 测试 (1957–1967)**
* 定义了语言基准测试的概念，包含：文档集合、查询以及正确答案。
* 他在小型语料库上对文档相关性提供了详尽的答案！

## 页面 19: 20 世纪 50 年代 NLP/计算语言学的起源

*(时间线引用自 Ruth Camburn 的《计算语言学简史》。她是 2013 年左右加州州立大学弗雷斯诺分校的语言学研究生)*
* 自动机 (automata)、形式语言 (formal languages)、概率建模 (probabilistic modeling) 和信息论 (information theory) 的奠基性工作。
* 第一个语音系统 (Davis et al.)。
* 机器翻译 (MT) 获得军方的重金资助 —— 极其过度自信。
* 但使用的机器比现在的口袋计算器还要笨拙。
* 对句法 (syntax)、语义 (semantics)、语用 (pragmatics) 缺乏理解。

## 页面 20: 20 世纪 60 年代：David Glenn Hays

* ACL 创始人与依存句法分析 (dependency parsing) 的先驱。

## 页面 21: Raj Reddy

* Raj Reddy（1937–，斯坦福博士生 1963–1966；斯坦福教师 1966–1969；后至 CMU；图灵奖获得者）
* **部分学生**：James Gosling, 李开复 (Kai-Fu Lee), Roni Rosenfeld, Alex Waibel

## 页面 22: Joyce Friedman

* Joyce Friedman（1928–2018；在斯坦福 1965–1968）
* 培养了 David Scott Warren, C. Ray Perrault，其后是 James Allen。

## 页面 23: 20 世纪 60 年代的 NLP

* **ALPAC 报告 (1966)**：糟糕，这真的很困难！

## 页面 24: 手工构建的演示性 NLP 系统及形式化的提高 (1970–1992)

## 页面 25: Terry Winograd

* Terry Winograd（1946–，斯坦福教师 1973–2014；AI/NLP 方向 1973–1984）
* 导师 Seymour Papert，学生 Stuart Shieber。

## 页面 26: 形式语法与基于合一的语法

* **用于 NLP 的形式和基于合一的语法 (Formal and Unification-based Grammars)**：Martin Kay

## 页面 27: 先驱人物

* Fernando Pereira, Stuart Shieber

## 页面 28: 人类语言语义构建（1967–2017）

* 将句子分词为单词：*The red apple is on the table*
* 将其解析为树或图数据结构（使用上下文无关语法 [Context-Free Grammars] 及其扩展形式）
* 构建语义步骤：
  1. 词典查找 (lexical lookup)
  2. 语义组合 (semantic composition)：使用沿着语法树自底向上运行的“规则到规则 (rule-to-rule)”方法（例如：$PP: \alpha(\beta) \to P: \alpha \quad NP: \beta$）
* 组合结果示例：
  $$\text{on}(\iota(\lambda x(\text{apple}(x) \wedge \text{red}(x))), \iota(\lambda y. \text{table}(y)))$$

## 页面 29: NLP 与 知识表示：Norvig (1986) 博士论文

* Peter Norvig 的博士论文：《文本理解的统一推理理论》 (A Unified Theory of Inference for Text Understanding, 1986)
* 分析的语言片段：
  * *“在一个离中国海岸不远的岛上建起的一个贫穷的渔村里，一个名叫 Chang Lee 的年轻男孩和他的寡母住在一起。每天，小 Chang 都勇敢地带着他的网出发，希望从海里捕获几条鱼，他们可以把鱼卖掉，换点钱买面包。”*
* 推断内容：
  * (a) 存在一片围绕该岛的海洋，被村民用于捕鱼，并构成了中国海岸的一部分。
  * (b) Chang 打算用网来网鱼，该网是渔网。
  * (c) 单词 *which* 指代鱼。
  * (d) 单词 *they* 指代 Chang 和他的母亲。

## 页面 30: 统一推理理论

“正如我们刚刚看到的，合适的知识库是做出正确推理的前提条件”（第 4 页）。它的构建是为了实现推理。

**系统有 6 种通用的推理形式；构成 2 对，因此有 4 种基本类型**：
1. **细节阐述 (Elaboration)**：填补槽位 (slot) 以连接两个实体
   * 约翰得到储蓄罐的“原因”是攒钱，“原因”是买礼物。
2. **指代消解 (Reference Resolution)**：这正是共指消解 (coreference)！
3. **视角应用 (View Application)**：*The Red Sox killed the Yankees*（红袜队干掉了洋基队）
   * *KILLED* 并非指动物死亡；*KILLING* 被视为“彻底击败 (DEFEAT-CONVINCINGLY)”。
4. **具体化 (Concretization)**：推断更具体的内容
   * *TRAVELLING in an AUTOMOBILE*（坐汽车旅行）是 *DRIVING*（开车）的一个实例。

## 页面 31: 20 世纪 70 年代和 80 年代的 NLP

* 语音识别的奠基性工作：随机建模 (stochastic modeling)、隐马尔可夫模型 (Hidden Markov Models, HMM)、“信道噪声模型/信道预测模型 (noisy channel)”。
  * 这里的许多思想随后彻底改变了 NLP！
* 逻辑编程、规则驱动的 AI，以及用于句法分析 (syntactic parsing) 的确定性算法（例如 LFG）。
* 对自然语言理解 (Natural Language Understanding, NLU) 的兴趣日益增加：SHRDLU, LUNAR, CHAT-80。
* 但符号化 AI 遇到了瓶颈：“AI 寒冬” (AI winter)。

## 页面 32: 统计或概率 NLP (1993–2012)

统计或概率 NLP (“StatNLP”)，以及随后用于 NLP 的更通用的监督机器学习 (Supervised ML)。

## 页面 33: 奠基人

* Claude Shannon（香农）

## 页面 34: 通信的数学理论 (1948)

* 《通信的数学理论》 (A Mathematical Theory of Communication, 1948)
* 信道噪声模型 (The noisy channel model)
*[了解更多](https://www.quantamagazine.org/how-claude-shannons-information-theory-invented-the-future-20201222/)*

## 页面 35: 统计革命：20 世纪 90 年代

* 引入来自电子工程 (EE) 和自动语音识别 (ASR) 的新思想：概率建模、语料库统计、监督学习、经验性评估。
* 新的数据源：机器可读文本的爆炸式增长；人工标注的训练数据（例如宾州树库 [Penn Treebank]）。
* 标注数据 + 算法 + 概率预测。
* 降低预期：忘掉完全的语义理解，让我们来做文本分类 (text cat)、词性标注 (part-of-speech tagging)、命名实体识别 (NER) 和句法分析 (parsing)！
* 工具：朴素贝叶斯分类器 (Naïve Bayes)、隐马尔可夫模型 (HMM)、概率上下文无关语法 (PCFG)、条件随机场 (CRF)（见 CS228 课程！）。
* 代表人物：Fred Jelinek

## 页面 36: 概率拼写纠错

* 输入句子：*“She is a stellar and versatile acress whose combination of sass and glamour attracts”*
* 让我们估计候选词相继出现的条件概率：
  * $P(\text{actress}\|\text{versatile}) = 0.0019 \quad P(\text{whose}\|\text{actress}) = 0.0043$
  * $P(\text{across}\|\text{versatile}) = 0.000092 \quad P(\text{whose}\|\text{across}) = 0.000026$
* 计算组合概率：
  * $P(\text{“versatile actress whose”}) = 0.0019 \times 0.0043 = 817,000 \times 10^{-10}$
  * $P(\text{“versatile across whose”}) = 0.000092 \times 0.000026 = 239 \times 10^{-10}$

## 页面 37: 机器的崛起：21 世纪初

* 计算机算力大幅提升。
* 巩固了统计革命取得的成果。
* 更多复杂的统计建模和机器学习算法：最大熵模型 (MaxEnt)、支持向量机 (SVM)、贝叶斯网络 (Bayes Nets)、潜在狄利克雷分配 (LDA) 等。
* 浩瀚的“大数据”：网络体量增长了 100 倍，出现大规模服务器集群。
* 焦点从监督学习转向**无监督学习**。
* 对高层语义应用（High-level semantic applications）重拾兴趣。

## 页面 38: 深度学习与人工神经网络用于 NLP (2013–2021)

**4b. 神经自然语言处理 (Neural NLP)**

## 页面 39: 符号化 AI 与“控制论”的对立

**斯坦福：“符号系统 (Symbolic Systems)”的故乡**
* “符号系统”研究代表我们周围世界的有意义符号系统——例如人类语言、逻辑和编程语言——以及处理这些符号的系统——例如大脑、计算机和复杂的社会系统。
* 虽然“认知科学”将心灵和智能视作自然现象来研究，但符号系统同样关注人类构建的使用符号进行交流和表示信息的系统。
* 代表人物：Jon Barwise (1942–2000)

## 页面 40: 符号系统与其处理器

* 语言是典型的符号系统；我们应该研究并利用其符号结构。
* 但这并不表明处理这些符号的主要处理器——人类大脑——是以物理符号系统 (physical symbol system) 的形式实现的。
* 我们不需要将 NLP 系统设计为物理符号系统。
* 大脑更像是一个神经网络模型。
* 人工神经网络模型具有更好的扩展性，并且可以捕获由符号表示的世界。

## 页面 41: Hinton 的深度学习突破

* 多伦多大学，2009 – 2012 年。

## 页面 42: 深度学习在语音识别中的应用

* 语音识别最先展示了概率方法（如 HMM 和高斯混合模型 [GMM]）的突破性成功。
* “深度学习”在大数据集上的首个突破性成果也发生在语音识别领域。
* George Dahl 等人 (2010/2012)：《用于大词汇量语音识别的上下文相关预训练深层神经网络》 (Context-Dependent Pre-trained Deep Neural Networks for Large Vocabulary Speech Recognition)。

**字错误率 (WER) 性能对比**：

| 声学模型 \ 数据集 | RT03S | FSH Hub5 SWB |
| :--- | :--- | :--- |
| 传统 GMM (Dahl et al. 2012) | 27.4 | 23.6 |
| 深度学习 (Dahl et al. 2012) | 18.5 (-33%) | 16.1 (-32%) |
| 深度学习 (Saon et al. 2017) | 8.0 (-71%) | 5.5 (-77%) |

## 页面 43: 词义的（人工）神经网络或深度学习模型

我们将每个单词学习并表示为一个实数向量。
$$\text{versatile} = \begin{bmatrix} 0.286 \\ 0.792 \\ -0.177 \\ -0.107 \\ 0.109 \\ -0.542 \\ 0.349 \\ 0.271 \end{bmatrix}$$

**相似的向量 $\approx$ 相似的含义**。

## 页面 44: 通过分布相似性学习词向量

我们如何学习这些词向量？
“你应该通过一个词所处的环境（上下文）来认识它” —— (J. R. Firth 1957: 11)

例如：
* "...any devices with a web browser, from **laptops** and tablets to smart phones..."
* "...Users can download it for home computers or **laptops** from Microsoft Update website..."
* *（周围的这些上下文词汇将用来表示 laptops 的语义）*

**通过文本中的上下文分布定义相似性是现代计算语言学最成功的思想之一。**

## 页面 45: 循环神经网络 (RNN) 编码器-解码器网络

**输入**：*I am a student <EOS>* $\to$ **输出**：*Je suis étudiant <EOS>*
* 包含编码器 (Encoder) 和解码器 (Decoder)。
* 状态更新方程：$$h_t = \tanh(W[x_t] + Uh_{t-1} + b)$$

## 页面 46: LSTM 编码器-解码器网络

*(Sutskever et al. 2014)*
* 德语源句：*Die Proteste waren am Wochenende eskaliert <EOS>*
* 英语翻译：*The protests escalated over the weekend <EOS>*
* **编码器 (Encoder)**：逐步构建源句语义。
* **解码器 (Decoder)**：生成翻译。
* **瓶颈 (Bottleneck)**：编码器到解码器之间的中间连接状态。

## 页面 47: 机器翻译 (MT) 随时间的发展进程

机器翻译的性能显著提升 (BLEU 得分)：
* 2013 年：6.8（基于短语的统计机器翻译 SMT）
* 2014 年：13.5（基于句法的统计机器翻译 SMT）
* 2015 年：20.3（神经网络机器翻译 NMT，来自蒙特利尔大学）
* 2016 年：27.0

*[文献来源](http://www.meta-net.eu/events/meta-forum-2016/slides/09_sennrich.pdf) 与 [谷歌 NMT 报道](https://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html)*

## 页面 48: 深度学习与人工神经网络用于 NLP (2022–至今)

**4b. 大语言模型 (Large Language Models, LLMs)**

## 页面 49: 生成式 AI 的崛起：讨论 “AI” 的标普 500 指数成分股公司占比

* ChatGPT 发布后，讨论 AI 的标普 500 指数成分股公司占比急剧飙升，2024-2025 年间达到 40%-50% 以上。
*[文献来源](https://insight.factset.com/more-than-40-of-sp-500-companies-have-cited-ai-on-earnings-calls-for-5th-straight-quarter)*

## 页面 50: 大语言模型示意

*(由 GPT-4o 生成的一幅蜡笔漫画风格抽象图，描绘了大语言模型就像一台巨大的机器，能够吞入文本并在无尽的卷轴上输出崭新的文章，这台机器由无数的科学家与工程师共同维护。)*

## 页面 51: 语言模型 (LM) 发展史

* **1913 年**：Andrey A. Markov 探索了亚历山大·普希金的小说《叶甫盖尼·奥涅金》中的辅音-元音转移概率，从而开发了马尔可夫模型 (Markov Models)。
* **1948 年**：Claude E. Shannon 发表了《通信的数学理论》；探索了字符/词级别的 n-gram 模型、熵以及文本生成。
* **1975 年**：IBM 的 Frederick Jelinek 小组定义并命名了用于预测下一个词的现代（概率）“语言模型”概念。用于拼写纠错、语音识别、机器翻译等。

**但当时该技术并不被视为实现人工智能的途径**：
为了实现 AI，人们当时认为还需要记忆模型、知识表示、规划系统以及基于抽象概念的推理。

## 页面 52: 大语言模型 (LLM) 发展史

* **1998 年**：CPAT-Tree-Based Language Models with an Application for Text Verification in Chinese (ROCLing 1998)。据我所知，这是第一次使用 “LLM” 三元组；使用了 2 亿词级别的语料库。
* **2000 年**：A Neural Probabilistic Language Model (Bengio, Ducharme & Vincent, NIPS 2000)。首个在 3200 万词元 (tokens) 语料库、3.1 万词表上构建的神经语言模型。
* **2007 年**：Large Language Models in Machine Translation (Brants, Popat, Xu, Och and Dean, EMNLP 2007)。包含高达 5-gram 的 2 万亿词元语料库的 n-gram 模型。
* **2018 年**：GPT (Radford et al.) 和 BERT (Devlin et al.)。拥有 33 亿词元语料库。
* **2020 年–至今**：在大于 1 万亿词元上训练的、拥有千亿级参数的神经语言模型：GPT-3, GPT-4, PaLM 2, Llama 3, Nemotron-4 等。

## 页面 53: 大语言模型发展史（续）

*(从 1998 年到 2020 年的演变历史)*
* **过去**：算力不足！模型灵活性不足！数据不足！
* **现在**：大模型高歌猛进 (LLMs go brrr)！

## 页面 54: [无内容]

## 页面 55: 使用 GPT-4 的咨询顾问表现超越未使用者

*(任务输出质量分布。蓝色组未使用 GPT-4；绿色和红色组使用 GPT-4；红色组接受了有关如何使用 AI 的额外培训。)*
* 来自波士顿咨询集团 (BCG) 的咨询顾问在使用 GPT-4 时，平均完成的任务数量增加了 12.2%，完成速度提高了 25.1%，且产出的结果质量比未使用 AI 的顾问高出 40%。
* 大语言模型 (LLM) 的使用特别显著地改善了原本绩效较低的人员的表现。
* 结果会根据具体的任务而有所不同。
—— Dell’Acqua et al. 2023; Mollick 2023

## 页面 56: GPT-4 能写出与《纽约客》质量相媲美的小说吗？

* **好消息！不能！**（至少在 2023 年是这样……）
* GPT-4 在创意写作上的表现仍然比人类作家差 3–10 倍！
*[学术论文链接](https://arxiv.org/abs/2309.14556)*

## 页面 57: 自然语言处理历史（总结）

自然语言处理历史 (History of Natural Language Processing)
CS224N/Ling284
Christopher Manning
第 1 讲
