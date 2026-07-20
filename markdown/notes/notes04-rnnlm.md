# CS224n 课程笔记 4：语言模型、RNN、GRU 与 LSTM

**课程讲师**：Christopher Manning, Richard Socher  
**作者**：Milad Mohammadi, Rohit Mundra, Richard Socher, Lisa Wang, Amita Kamath  
**授课学期**：2019 冬季  

---

### 摘要
本篇笔记首先介绍了语言模型（Language Models）的基本概念。随后，我们深入探讨了循环神经网络（Recurrent Neural Networks, RNN）的架构，包括双向 RNN（Bidirectional RNN）和深层 RNN（Deep RNN）。接着，我们详细分析了 RNN 训练中经典的梯度消失（Vanishing Gradient）与梯度爆炸（Gradient Explosion）问题，并给出了相应的数学证明和解决方案。最后，我们详细介绍了两种为了解决梯度消失、捕捉长距离依赖而设计的复杂门控激活单元：门控循环单元（GRU）与长短期记忆网络（LSTM）。

---

## 1 语言模型 (Language Models)

### 1.1 什么是语言模型？ (Introduction)
**语言模型（Language Model）**用于计算一组特定单词序列发生的概率。

对于由 $m$ 个单词组成的序列 $\{w_1, \dots, w_m\}$，其联合概率记作 $P(w_1, \dots, w_m)$。利用概率的乘法公式，我们可以将其写为条件概率的连乘形式。由于每个单词前面的历史长度是可变的，在实际计算时，我们通常使用滑动窗口，假设当前词的发生仅依赖于其前面的 $n$ 个词（即 $n$ 阶马尔可夫假设），而不是依赖于所有的历史单词：
$$P(w_1, \dots, w_m) = \prod_{i=1}^m P(w_i | w_1, \dots, w_{i-1}) \approx \prod_{i=1}^m P(w_i | w_{i-n}, \dots, w_{i-1}) \tag{1}$$

语言模型在机器翻译和语音识别中扮演着重要角色。例如，在机器翻译系统中，对于一句源语言输入，翻译软件会生成多个候选译文序列（例如：*{I have, I had, I has, me have, me had}*）。系统通过语言模型对这些候选序列进行评分，从而选出概率最高、最符合人类表达习惯的译文序列。
- 语言模型会给 *"the cat is small"* 评定比 *"small the is cat"* 更高的概率。
- 同样，*"walking home after school"* 的概率也会高于 *"walking house after school"*。

### 1.2 n-gram 语言模型 (n-gram Language Models)
为了计算上述条件概率，最直接的方法是在大规模语料库中统计各单词组合的频数，这就是 **$n$-gram 语言模型**。
以二元语法（Bigram）和三元语法（Trigram）为例，其条件概率计算如下：
$$P(w_2 | w_1) = \frac{\operatorname{count}(w_1, w_2)}{\operatorname{count}(w_1)} \tag{2}$$
$$P(w_3 | w_1, w_2) = \frac{\operatorname{count}(w_1, w_2, w_3)}{\operatorname{count}(w_1, w_2)} \tag{3}$$

然而，仅依赖于固定大小上下文窗口的 $n$-gram 模型在捕获长距离语义依赖时面临瓶颈。
> **局限性示例**：
> 考虑句子 *"As the proctor started the clock, the students opened their _____"*（当监考人员按下计时器，学生们打开了他们的_____）。
> - 如果条件窗口仅包含前三个词 *"the students opened their"*，根据常识统计，下一个词很可能是 *"books"*（书本）。
> - 但如果窗口能够延伸到前面的 *"proctor"*（监考人），那么下一个词是 *"exam"*（试卷）的概率就会显著提升。

这导出了 $n$-gram 语言模型面临的两个致命问题：
1. **稀疏性（Sparsity）**：
   - 如果 $w_1, w_2, w_3$ 在语料库中从未同时出现，则分子为 0，导致概率为 0。解决方法是在分子上加一个微小的常量 $\delta$，这称为**加法平滑（Smoothing）**。
   - 如果历史 $w_1, w_2$ 从未在语料中出现，则分母为 0 无法计算。解决方法是舍弃最远的词 $w_1$，仅在 $w_2$ 上计算条件概率，这称为**回退（Backoff）**。
   - 随着 $n$ 的增加，稀疏问题会呈指数级加剧。在实践中，通常 $n \le 5$。
2. **存储（Storage）**：
   - 我们必须存储语料库中所有出现过的 $n$-gram 的频数。随着 $n$ 的增大，模型大小会急剧膨胀，存储开销极大。

### 1.3 基于窗口的神经网络语言模型 (Window-based Neural Language Model)
为了解决上述“维度灾难（Curse of Dimensionality）”，Bengio 等人在 2003 年发表了经典论文《A Neural Probabilistic Language Model》，首次将深度学习引入 NLP 语言模型中。该模型学习了单词的**分布式表示（词向量）**，并在此基础上构建语言模型预测概率。

```
   输入词向量 c(w_t-1), c(w_t-2), ...  ---> [ 拼接拼接 ] ---> 隐层 (tanh) ---\
                                           \--------------------------> [ Softmax 归一化 ] ---> 预测 y_hat
```

模型的计算流程可以简化表示为：
$$\hat{y} = \operatorname{softmax}\left(W^{(2)} \tanh(W^{(1)} x + b^{(1)}) + W^{(3)} x + b^{(3)}\right) \tag{4}$$
其中：
- $x$ 是前 $n$ 个历史单词的词向量拼接而成的长向量。
- $W^{(1)}$ 是隐藏层的权重，$\tanh$ 提供非线性变换。
- $W^{(3)}$ 是直接从输入向量到输出的线性跳跃连接（Skip-Connection）。
- 最终通过 $\operatorname{softmax}$ 函数输出在整个词汇表 $V$ 上的条件概率分布。

---

## 2 循环神经网络 (Recurrent Neural Networks, RNN)

与上述基于固定大小窗口的神经网络不同，**循环神经网络（RNN）**具有循环反馈结构，理论上能够保留并利用自序列起点以来的所有历史信息。

### 2.1 结构与计算流程
如图 3 所示，RNN 在时间轴上展开后，在每一个时间步 $t$，隐藏层状态 $h_t$ 不仅依赖于当前步的输入 $x_t$，还依赖于前一时间步的隐藏状态 $h_{t-1}$。

$$\begin{aligned}
h_t &= \sigma\left(W^{(hh)} h_{t-1} + W^{(hx)} x_t\right) \tag{5} \\
\hat{y}_t &= \operatorname{softmax}\left(W^{(S)} h_t\right) \tag{6}
\end{aligned}$$

- $x_t \in \mathbb{R}^d$：时间步 $t$ 处的输入词向量。
- $W^{(hx)} \in \mathbb{R}^{D_h \times d}$：输入向量的权重矩阵。
- $h_{t-1} \in \mathbb{R}^{D_h}$：上一时间步的隐藏层输出，携带了历史上下文信息。$h_0$ 是初始化的隐藏状态向量。
- $W^{(hh)} \in \mathbb{R}^{D_h \times D_h}$：前一隐藏状态的循环权重矩阵。
- $\sigma()$：非线性激活函数（如 Sigmoid 或 Tanh）。
- $W^{(S)} \in \mathbb{R}^{|V| \times D_h}$：输出权重矩阵，将隐藏状态映射到词汇表大小的得分空间。
- $\hat{y}_t \in \mathbb{R}^{|V|}$：在整个词汇表上的预测概率分布。

在 RNN 中，**参数 matrices $W^{(hh)}$ 和 $W^{(hx)}$ 在所有时间步上是完全共享的**。这使得模型的参数量大为减少，且不随输入序列的长度而增长，完美避开了维度灾难。

### 2.2 损失函数与困惑度 (RNN Loss and Perplexity)
RNN 语言模型的每个时间步都会产生预测，我们使用交叉熵损失。在时间步 $t$ 处的损失定义为：
$$J^{(t)}(\theta) = - \sum_{j=1}^{|V|} y_{t,j} \log \hat{y}_{t,j} \tag{7}$$
其中 $y_t$ 是真实单词的独热编码向量。对于长度为 $T$ 的整个语料库，平均交叉熵损失为：
$$J = - \frac{1}{T} \sum_{t=1}^T \sum_{j=1}^{|V|} y_{t,j} \log \hat{y}_{t,j} \tag{8}$$

评估语言模型表现的常用指标是**困惑度（Perplexity）**。它定义为平均交叉熵损失的指数形式：
$$\text{Perplexity} = 2^J \tag{9}$$
困惑度越小，代表模型在预测下一个词时“越不困惑”，预测精度越高。

### 2.3 RNN 的优缺点
- **优点**：
  1. 能处理任意长度的序列；
  2. 参数量不随输入序列长度增加而增加；
  3. 理论上可以利用无限步之前的历史上下文信息；
  4. 权值共享带来了对时间维度的平移对称性。
- **缺点**：
  1. **计算慢**：由于前后步存在时间依赖，RNN 的前向和反向计算是串行的，**无法在时间轴上进行并行化加速**。
  2. **长距依赖捕捉困难**：在实践中，由于反向传播中的梯度消失或梯度爆炸问题，RNN 很难有效利用 5-10 步之前的历史信息。

---

## 2.4 梯度消失与梯度爆炸证明 (Vanishing & Exploding Gradients)

为什么 RNN 难以捕捉长距离依赖？我们从数学角度来进行分析。
为了计算总损失关于权重矩阵 $W$ 的梯度，我们需要在每个时间步累加偏导数：
$$\frac{\partial E}{\partial W} = \sum_{t=1}^T \frac{\partial E_t}{\partial W} \tag{10}$$

根据链式法则，时间步 $t$ 处的损失对 $W$ 的梯度可以展开为：
$$\frac{\partial E_t}{\partial W} = \sum_{k=1}^t \frac{\partial E_t}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W} \tag{11}$$

这里核心项是隐藏状态对前一隐藏状态的偏导 $\frac{\partial h_t}{\partial h_k}$。根据多步链式法则，它是由相连隐层之间的雅可比矩阵（Jacobian matrices）连乘得到的：
$$\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^t W^{(hh)\top} \operatorname{diag}\left[ f'(z_{j-1}) \right] \tag{12}$$

由于状态向量 $h \in \mathbb{R}^{D_h}$，$\frac{\partial h_j}{\partial h_{j-1}}$ 是一个 $D_h \times D_h$ 的雅可比矩阵。我们对其取 $L_2$ 范数进行上界分析：
$$\left\| \frac{\partial h_j}{\partial h_{j-1}} \right\| \le \left\| W^{(hh)\top} \right\| \left\| \operatorname{diag}\left[ f'(z_{j-1}) \right] \right\| \le \beta_W \beta_h \tag{15}$$
对于 Sigmoid 激活函数，$\beta_h \le 1$。由此可以得出长距离的导数范数界限：
$$\left\| \frac{\partial h_t}{\partial h_k} \right\| = \left\| \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}} \right\| \le (\beta_W \beta_h)^{t-k} \tag{16}$$

分析式 (16) 可知，若连乘因子 $\beta_W \beta_h$ 的值不等于 $1$，当时间步长 $(t - k)$ 足够大时：
- **若 $\beta_W \beta_h > 1$**：连乘项将呈指数级膨胀，造成**梯度爆炸（Gradient Explosion）**。这会在运行时导致数值溢出（出现 `NaN`），极易被察觉。
- **若 $\beta_W \beta_h < 1$**：连乘项将指数级缩减至接近 0，造成**梯度消失（Vanishing Gradient）**。梯度一旦消失，模型在时间步 $t$ 处的误差将无法回传到遥远的步 $k$，使模型无法通过学习来建立长距离的依赖关系。

### 2.5 梯度爆炸与消失的解决方案
1. **针对梯度爆炸**：
   - **梯度裁剪（Gradient Clipping）**：Thomas Mikolov 提出，当梯度的范数超过设定的阈值时，直接将其强行缩放到阈值界限内：
     $$\text{if } \|\hat{g}\| \ge \text{threshold} \text{ then } \hat{g} \leftarrow \text{threshold} \frac{\hat{g}}{\|\hat{g}\|}$$
     *图 7 展示了梯度裁剪的效果。当 SGD 遇到由于梯度激增而形成的“陡峭悬崖”时，如果不裁剪，参数会一步跳出可行域导致训练发散；裁剪后能限制步长，使模型能沿着壁面平滑探索。*
   - 对循环权重施加 $L_1$ 或 $L_2$ 正则化惩罚，强制限制其谱半径。
2. **针对梯度消失**：
   - **单位矩阵初始化**：将循环权重矩阵 $W^{(hh)}$ 初始化为单位矩阵 $I$，而不是随机初始化。
   - **使用 ReLU 激活函数**：因为 ReLU 的导数在正区间恒为 $1$，梯度在通过激活单元回传时不会像 Sigmoid/Tanh 那样受到饱和区（导数趋近于 0）的严重衰减。
   - **门控结构（GRU/LSTM）**：通过建立“线性相加的直通通道”，直接从根本上改变了梯度的连乘形式。

---

## 2.6 双向 RNN 与深层 RNN (Bidirectional & Deep RNNs)

### 双向 RNN (Bidirectional RNN)
在很多文本序列处理任务（如命名实体识别）中，我们在当前时间步不仅希望能利用历史信息，也希望能利用未来的上下文信息。**双向 RNN** 分别对序列进行正向和反向两次循环扫描，并拼接两者的隐藏状态：

$$\begin{aligned}
\vec{h}_t &= f\left( \vec{W} x_t + \vec{V} \vec{h}_{t-1} + \vec{b} \right) \tag{17} \\
\overleftarrow{h}_t &= f\left( \overleftarrow{W} x_t + \overleftarrow{V} \overleftarrow{h}_{t+1} + \overleftarrow{b} \right) \tag{18} \\
\hat{y}_t &= g\left( U [\vec{h}_t; \overleftarrow{h}_t] + c \right) \tag{19}
\end{aligned}$$

拼接后的状态 $h_t = [\vec{h}_t; \overleftarrow{h}_t]$ 同时概括了当前词前后的双向上下文信息。

### 深层 RNN (Deep RNN)
为了提高模型的学习能力，我们可以像普通神经网络一样叠加多层 RNN 结构：
$$\begin{aligned}
\vec{h}^{(i)}_t &= f\left( \vec{W}^{(i)} h^{(i-1)}_t + \vec{V}^{(i)} \vec{h}^{(i)}_{t-1} + \vec{b}^{(i)} \right) \tag{20} \\
\overleftarrow{h}^{(i)}_t &= f\left( \overleftarrow{W}^{(i)} h^{(i-1)}_t + \overleftarrow{V}^{(i)} \overleftarrow{h}^{(i)}_{t+1} + \overleftarrow{b}^{(i)} \right) \tag{21} \\
\hat{y}_t &= g\left( U [\vec{h}^{(L)}_t; \overleftarrow{h}^{(L)}_t] + c \right) \tag{22}
\end{aligned}$$
其中，第 $i$ 层的输入是第 $i-1$ 层对应时间步的隐藏状态。

---

## 2.7 应用：RNN 序列翻译模型 (Seq2Seq Model)

传统的翻译模型是一个由多个独立统计模块拼接成的复杂流水线。我们可以使用端到端的 RNN 编码器-解码器（Seq2Seq）模型来进行替换。
例如，将德语 *"Echt dicke Kiste"* 翻译为英语 *"Awesome sauce"*：

```
   [ 德语输入 ] Echt --> dicke --> Kiste (编码器) ---> 最终状态 h_3 (语义向量)
                                                          |
                                                          v
   [ 英语输出 ]                       Awesome <--- sauce (解码器)
```

- **编码器（Encoder）**：
  $$h_t = f\left(W^{(hh)} h_{t-1} + W^{(hx)} x_t\right) \tag{23}$$
- **解码器（Decoder）**：
  $$\begin{aligned}
  h_t &= f\left(W^{(hh)} h_{t-1}\right) \tag{24} \\
  \hat{y}_t &= \operatorname{softmax}\left(W^{(S)} h_t\right) \tag{25}
  \end{aligned}$$

为了获得更高质量的翻译，业界引入了以下几项关键扩展：
1. **解耦权重**：编码器与解码器使用完全独立的循环权重矩阵。
2. **多信息桥接**：解码器在计算隐藏状态时，同时结合当前隐状态 $h_{t-1}$、编码器最后输出的句向量 $c$、以及上一步预测输出的单词 $y_{t-1}$。
   $$h_t = \phi(h_{t-1}, c, y_{t-1}) \tag{27}$$
3. **输入序列颠倒（Reversing Input）**：训练时将输入序列倒序输入（例如，将 A B C $\to$ X Y 改为 C B A $\to$ X Y）。这样，句首词 A 与首个翻译目标词 X 之间的物理距离被缩短，可以极大缓解句首信息的梯度消失，让翻译起步更加准确。

---

## 3 门控循环单元 (Gated Recurrent Units, GRU)

为了让循环神经网络拥有更长久的记忆，Cho 等人提出了**门控循环单元（GRU）**。它通过引入门机制来控制信息的流动和遗忘。

GRU 的前向计算公式如下：
$$\begin{aligned}
z_t &= \sigma\left(W^{(z)} x_t + U^{(z)} h_{t-1}\right) \quad \text{（更新门，Update Gate）} \\
r_t &= \sigma\left(W^{(r)} x_t + U^{(r)} h_{t-1}\right) \quad \text{（重置门，Reset Gate）} \\
\tilde{h}_t &= \tanh\left(r_t \circ (U h_{t-1}) + W x_t\right) \quad \text{（候选记忆，New Memory）} \\
h_t &= (1 - z_t) \circ \tilde{h}_t + z_t \circ h_{t-1} \quad \text{（隐藏状态，Hidden State）}
\end{aligned}$$

```
   x_t -------> [ 重置门 r_t ] ---> 门控历史 r_t * h_t-1 ---\
          \---------------------------------------------> [ tanh( 新记忆 h_tilde ) ] ---\
          \===> [ 更新门 z_t ] =========================================================+==> 新状态 h_t
   h_t-1 -------------------------------------------------------------------------------/
```

#### 四个基本操作阶段的直观物理意义：
1. **候选记忆生成 (New Memory Generation)**：
   利用当前输入 $x_t$ 与被重置门门控后的前一状态进行组合，通过 $\tanh$ 函数产生一个新的候选状态 $\tilde{h}_t$。它指示了将新词融入上下文之后的最新记忆候选。
2. **重置门 (Reset Gate)**：
   $r_t$ 负责控制前一状态 $h_{t-1}$ 应该保留多少用于当前候选记忆的计算。如果 $r_t \approx 0$，代表历史信息与当前输入完全无关，在计算新记忆时将历史状态清零抹去。
3. **更新门 (Update Gate)**：
   $z_t$ 决定我们需要将多少历史信息**直通**传递给下一步隐藏状态。若 $z_t \approx 1$，则上一状态 $h_{t-1}$ 被原封不动地复制保留到 $h_t$ 中；若 $z_t \approx 0$，则将当前新鲜候选记忆 $\tilde{h}_t$ 写入隐藏状态。
4. **隐藏状态 (Hidden State)**：
   根据更新门 $z_t$ 的指示，在旧状态 $h_{t-1}$ 与新记忆 $\tilde{h}_t$ 之间进行加权线性求和，生成最终的隐藏输出 $h_t$。

由于更新门 $z_t$ 可以完全为 $1$，因此 GRU 可以在极长的时间内让状态保持不变，从而使梯度可以直接穿透多个时间步进行原样回传，消除了梯度衰减，能够完美捕获极长距离的依赖关系。

---

## 4 长短期记忆网络 (Long Short-Term Memory, LSTM)

**长短期记忆网络（LSTM）**是 Hochreiter 等人提出的一种更为经典且鲁棒的门控循环网络。相比于 GRU，它进一步将内部的**记忆细胞状态（Cell State, $c_t$）**与外部的**隐藏状态（Hidden State, $h_t$）**进行了解耦。

LSTM 的前向计算公式如下：
$$\begin{aligned}
i_t &= \sigma\left(W^{(i)} x_t + U^{(i)} h_{t-1}\right) \quad \text{（输入门，Input Gate）} \\
f_t &= \sigma\left(W^{(f)} x_t + U^{(f)} h_{t-1}\right) \quad \text{（遗忘门，Forget Gate）} \\
o_t &= \sigma\left(W^{(o)} x_t + U^{(o)} h_{t-1}\right) \quad \text{（输出/曝光门，Output Gate）} \\
\tilde{c}_t &= \tanh\left(W^{(c)} x_t + U^{(c)} h_{t-1}\right) \quad \text{（候选记忆细胞）} \\
c_t &= f_t \circ c_{t-1} + i_t \circ \tilde{c}_t \quad \text{（最终记忆细胞，Final Cell State）} \\
h_t &= o_t \circ \tanh(c_t) \quad \text{（最终隐藏状态，Hidden State）}
\end{aligned}$$

```
                   c_t-1 ----> [ 遗忘门 f_t 过滤 ] ---> ( + ) --------------------------> 记忆细胞 c_t
                                                         ^                               |
   x_t , h_t-1 --> [ 输入门 i_t 过滤 ] ---> i_t * c_tilde /                               |
                                                                                         v
                                                       [ 输出门 o_t 过滤 ] ---> [ tanh ] ==> 隐状态 h_t
```

#### 各模块物理意义：
1. **候选记忆细胞生成**：
   使用当前输入 $x_t$ 和先前隐状态 $h_{t-1}$ 来准备一个新的临时记忆细胞候选 $\tilde{c}_t$。
2. **输入门 (Input Gate)**：
   $i_t$ 决定当前输入的新内容是否足够重要，是否需要写入到长久记忆细胞中。它用于门控 $\tilde{c}_t$。
3. **遗忘门 (Forget Gate)**：
   $f_t$ 评估历史记忆细胞 $c_{t-1}$ 对于当前步骤是否仍然有用。若不再有用，则允许遗忘门将其权重设为趋近于 $0$ 以释放内存空间。
4. **最终记忆细胞计算**：
   通过输入门与遗忘门联合控制，将历史记忆 $c_{t-1}$ 与当前新知识 $\tilde{c}_t$ **线性相加**得到最新的记忆细胞 $c_t$。由于这是加法操作，梯度在时间上传播时可以直接跨越时间步流动，完美解决了梯度消失问题。
5. **输出/曝光门 (Output Gate)**：
   $o_t$ 决定了当前的记忆细胞 $c_t$ 中有哪一部分需要输出并展示给外部隐藏状态 $h_t$。隐藏状态会被后续时间步的各个门控模块所使用，而记忆细胞则只在内部直通传递。
