# Lecture 8: Post-training

## 页面 1: 标题
* **课程名称**：基于深度学习的自然语言处理 (Natural Language Processing with Deep Learning)
* **课程编号**：CS224N / Ling284
* **主讲人**：Diyi Yang
* **主题**：第 8 讲：后训练 (Lecture 8: Post-training)
* *注：原课件中 Slide 1 标题写为了 Lecture 7: Post-training，实为第 8 讲。*

---

## 页面 2: 课程计划
1. **指令微调 (Instruction fine-tuning)**：(15分钟)
2. **人类反馈强化学习 (RLHF)**：(20分钟)
3. **InstructGPT 与 ChatGPT**：(5分钟)
4. **强化学习与奖励建模的局限性**：(5分钟)
5. **直接偏好优化 (DPO) 介绍**：(25分钟)
6. **人类偏好数据：人类反馈 vs. AI 反馈**：(5分钟)

*课程提示：期末项目相关细则今天发布，请尽快组建团队；作业 3 将于 2 月 5 日截止。*

---

## 页面 3 - 5: 背景介绍：语言模型的演进
* **大模型趋势**：大语言模型参数量越来越庞大，训练文本数据量呈指数级上升。
* **语言模型作为世界模型？**：大语言模型（如 Bing, ChatGPT, Claude）在代码编写 (Copilot)、学术考试等领域展现出了极其出色的通用泛化能力。

---

## 页面 6: 语言模型作为多任务助手？
* **核心差距**：我们如何从简单的“续写文本”模型（例如输入 `Stanford University is located in __________`），转变为能够听懂人类指令并提供多任务协助的“智能助手”？
* **解决方案**：引入后训练 (Post-training) 技术，进行微调与对齐。

---

## 页面 7: 课程计划大纲

---

## 页面 8 - 9: 语言建模 $\neq$ 协助用户
* **对齐失配 (Misalignment)**：单纯依靠下一个词预测训练出来的基座语言模型，与人类想要的用户意图并不一致 [Ouyang et al., 2022]。
* **基座模型的续写倾向**：
  * **人类输入**：一段关于宇航员登月的探险故事……
  * **基座模型**：可能只是一味续写更多类似的句子，而不是回答用户关于故事的提问。
* **出路**：通过微调 (Finetuning) 进行修正。

---

## 页面 10 - 11: 预训练到微调的范式演变
* **以往做法**：在无标注海量文本上预训练模型（作为参数初始化），随后在特定下游任务的少量标注数据上进行微调。
* **升级做法：多任务微调**：在数十上百个不同任务的数据集上同时微调模型，赋予其通用的多任务解决能力。

---

## 页面 12 - 13: 指令微调 (Instruction Fine-Tuning)
* **核心流程**：收集大量形如 `(指令, 期望输出)` 的配对样本，在基座模型上做常规的监督微调 (SFT) [Chung et al., 2022]。
* **规模化效应 (Scale)**：数据量与模型规模是指令微调成功的关键。例如，Super-NaturalInstructions 数据集包含了超过 1600 个任务、300 万以上的样本，涵盖了分类、序列标注、改写、翻译和问答等。

---

## 页面 14 - 17: 多任务评估与基准评测
* **MMLU (Massive Multitask Language Understanding)** [Hendrycks et al., 2021]：包含 57 个不同领域的知识密集型任务，是衡量大模型常识与推理能力的主流基准。
* **BIG-Bench** [Srivastava et al., 2022]：由谷歌发起的开源评估套件，包含 200 多个高难度推理、数学、逻辑等任务。
* *近年各大模型在 MMLU 等高难度 benchmarks 上取得了极为迅速和瞩目的进展。*

---

## 页面 18 - 20: 指令微调带来的性能增益
* **FLAN-T5** [Chung et al., 2022]：在 T5 [Raffel et al., 2018]（基于 Span 破坏预训练）的基础上，进行了 1800 个指令任务的微调。
* **重要结论**：**模型规模越大，指令微调带来的 Δ 提升越显著**。指令微调使得原本无法直接回答问题的基座模型，在微调后能极为精准地理解人类指令并给出答案。

---

## 页面 21 - 22: 开源社区的指令微调浪潮
* 随着 Meta 发布 LLaMA 模型，开源社区涌现了大量构建指令微调数据集的尝试：
  * **Alpaca** [Wang et al., 2022]：利用 GPT 模型通过 Self-Instruct 方式生成了 5.2 万条指令数据，在 LLaMA-7B 上微调取得了极佳的对话体验。
  * **LIMA (Less Is More for Alignment)** [Zhou et al., 2023]：指出只要指令质量极高，仅需 1000 个精选的高质量样本即可完成对齐，即“少即是多”假说。

---

## 页面 23: 课程计划大纲

---

## 页面 24: 指令微调的局限性
1. **收集标注成本昂贵**：获取完美的人工标签极为耗时耗力。
2. **生成任务无唯一标准答案**：例如“写一个关于狗和草蜢的小故事”，无法用固定的监督样本进行完美惩罚。
3. **标记级别的惩罚失衡**：在自回归交叉熵损失中，每个 token 的错误惩罚权重相同。但实际上，犯某些常识性错误或有害言论比漏掉一个标点符号要糟糕得多。
* **核心失配**：监督微调 (SFT) 的优化目标（交叉熵）与“满足人类偏好”的终极目标之间存在偏差。我们是否能直接对人类偏好进行端到端优化？

---

## 页面 25: 偏好奖励优化 (Optimizing for Human Preferences)
* 假定对于语言模型生成的任意一个文本样本 $s$，我们都可以获得一个人类的奖励评分 $R(s) \in \mathbb{R}$，分值越高表示人类越满意。
* **优化目标**：我们的目标是优化模型参数 $\theta$，最大化模型生成样本的期望奖励：
  $$\mathbb{E}_{\hat{s} \sim p_\theta(s)} [R(\hat{s})]$$

> **[学习注释：关于期望奖励的数学符号与强化学习映射]**
> * **符号物理含义**：大模型参数为 $\theta$，策略可记为 $p_\theta(s)$。生成一整段动作（文本轨迹 $s = [y_1, \dots, y_L]$）的概率为各步预测 Token 的条件概率连乘：$p_\theta(s) = \prod_{t=1}^L p_\theta(y_t \mid y_{<t})$。$\hat{s} \sim p_\theta(s)$ 代表从模型中采样生成的文本。$R(\hat{s})$ 是奖励模型给该文本的评分。期望奖励 $\mathbb{E}_{\hat{s} \sim p_\theta(s)} [R(\hat{s})] = \sum_{s} p_\theta(s) R(s)$ 表示所有可能生成回答的概率加权平均奖赏。
> * **强化学习对齐**：大语言模型即策略（Policy），输入 prompt 为环境状态（State），一步步输出 token 是动作序列（Action sequence），奖励模型输出的分数是奖励（Reward）。目标是找到最优策略以最大化累积期望奖励。

---

## 页面 26: 强化学习 (Reinforcement Learning) 的引入
* 针对非微分的奖励函数，强化学习 (RL) 提供了绝佳的工具。
* **PPO 算法** [Schulman et al., 2017]：作为一种近端策略优化（Proximal Policy Optimization）方法，因其在超大网络结构中的平稳收敛性，成为了大模型偏好优化的主流强化学习基础。

---

## 页面 27 - 29: 策略梯度与 REINFORCE 算法推导
为了最大化期望奖励 $\mathbb{E}_{s \sim p_\theta(s)} [R(s)]$，我们需要计算关于参数 $\theta$ 的梯度：
* **对数求导技巧 (Log-derivative trick)**：
  $$\nabla_\theta p_\theta(s) = p_\theta(s) \nabla_\theta \log p_\theta(s)$$
* **公式推导**：
  $$\nabla_\theta \mathbb{E}_{s \sim p_\theta(s)} [R(s)] = \sum_{s} R(s) \nabla_\theta p_\theta(s) = \sum_{s} p_\theta(s) R(s) \nabla_\theta \log p_\theta(s) = \mathbb{E}_{s \sim p_\theta(s)} [R(s) \nabla_\theta \log p_\theta(s)]$$
* **参数更新（Monte Carlo 估计）**：
  $$\theta_{t+1} \leftarrow \theta_t + \alpha \frac{1}{m} \sum_{i=1}^{m} R(s_i) \nabla_{\theta_t} \log p_{\theta_t}(s_i)$$
* **直观理解**：如果采样样本的奖励 $R(s_i)$ 为正，我们走正向梯度更新，**增加**模型以后生成该样本的概率；若 $R(s_i)$ 为负值，则走负向梯度更新，**减少**其概率。这就是“强化”的物理含义。

> **[学习注释：对数求导技巧的推导及 Reward 改变梯度的物理直观]**
> * **为什么需要对数求导？**：
>   * **数值防下溢**：各步 Token 的条件概率连乘极其微小，在计算机中直接连乘会发生数值下溢变为 0。使用对数后，乘法转为累加：$\log p_\theta(s) = \sum_{t=1}^L \log p_\theta(y_t \mid y_{<t})$，在大模型一次前向传播中把每一步真实 token 对应的 Log-Softmax 概率加和即可，数值非常稳定。
>   * **期望的可采样化**：最原始对目标函数求导得到的 $\sum_s R(s) \nabla_\theta p_\theta(s)$ 无法利用蒙特卡洛采样来计算，因为对参数 $\theta$ 的梯度分布无法采样。使用恒等式 $\nabla_\theta p_\theta(s) = p_\theta(s) \nabla_\theta \log p_\theta(s)$，可将梯度展开转换为包含 $p_\theta(s)$ 的期望表达：$\mathbb{E}_{s \sim p_\theta(s)} [R(s) \nabla_\theta \log p_\theta(s)]$，从而可以通过采样并用经验平均 $\frac{1}{m} \sum_{i=1}^m R(s_i) \nabla_\theta \log p_\theta(s_i)$ 来估计梯度。
> * **Reward 与 DL 参数更新梯度（Loss）的关系**：
>   * 在自动求导梯度下降的框架中，我们定义虚拟损失函数（Surrogate Loss）：$\text{Loss}(\theta) = -\frac{1}{m} \sum_{i=1}^m R(s_i) \log p_\theta(s_i)$。对该损失使用梯度下降更新参数，正好等价于最大化期望奖励的梯度上升。
>   * **Reward 充当了“梯度更新的系数/杠杆”**：
>     * 若生成句子 $s_i$ 的 $R(s_i) > 0$，参数朝向增加 $p_\theta(s_i)$ 概率的方向更新（正向强化）；
>     * 若 $R(s_i) < 0$，参数朝向减少其概率的方向更新（反向惩罚）。

---

## 页面 30: 引入 PPO 算法
* Vanilla REINFORCE 存在梯度估计方差过大、训练极度不稳定、数据利用率低等重大缺陷。
* **PPO 算法**限制了每一步更新中新策略偏离旧策略的幅度（通过截断 clip 机制），从而保证了训练的稳定运行。

> **[学习注释：关于此处课件内容与完整 PPO 算法的澄清]**
> * **内容说明**：需要注意的是，此处课件接下来介绍与推导的其实是简化版的 **REINFORCE（策略梯度）算法**，而非完整的 PPO 算法。
> * **完整 PPO 算法位置**：由于 PPO 引入了更复杂的 Actor-Critic 架构和 Clipping 机制，为了保证学习逻辑的流畅性，关于完整 PPO 算法的核心要点与公式详解已被整理补充在 **页面 34-35** 之后的专门注释中，建议读者阅读完整个 RLHF 全流程后再一并学习。

---

## 页面 31 - 33: 如何进行人类偏好建模？ (Reward Modeling)
* **痛点**：在强化学习训练过程中，如果每生成一个词都让人类实时打分，成本极度昂贵，无法实现。
* **对策**：从标注的偏好数据集中学习一个**奖励模型 (Reward Model, RM)**，用来自动化替代人类评判。
* **配对偏好数据**：
  * ... 人类在打分时容易出现主观差异（例如对于样本 1，一人打 4.1 分，另一人可能打 6.6 分，评分基线极不稳定）。
  * 相比于绝对评分，**二分类两两偏好比较**更具可信度：给定输入 prompt $x$，模型生成两个样本 $s_w$（人类更偏好的 winning 样本）和 $s_l$（losing 样本）。
* **奖励模型损失函数（Bradley-Terry 模型）**：
  $$J_{RM}(\phi) = -\mathbb{E}_{(s_w, s_l) \sim D} \left[ \log \sigma(RM_\phi(s_w) - RM_\phi(s_l)) \right]$$
  我们训练一个网络参数为 $\phi$ 的奖励模型，使得更受人类偏好的样本 $s_w$ 的奖励得分显著高于不好的样本 $s_l$。

> **[学习注释：奖励模型有监督训练中的输入、输出与梯度求导本质]**
> * **何为模型预测，何为人工标注？**：
>   * **模型预测值**：将同一个 Prompt 的两个回答 $s_w$ 和 $s_l$ 喂给网络，前向传播输出的两个实数得分 $RM_\phi(s_w)$ 和 $RM_\phi(s_l)$。
>   * **人工标注（Ground Truth）**：体现在数据集 $D$ 的**配对关系本身**。人工做的是二分类抉择，我们将受偏好的回答排在前面记为 $s_w$，不喜欢的记为 $s_l$。标签隐含在位置上，即目标是令 $RM_\phi(s_w) > RM_\phi(s_l)$。
> * **梯度求导反向传播本质**：
>   * 奖励模型训练是**纯粹的有监督深度学习**，不涉及强化学习。计算损失 $L(\phi) = -\log \sigma(RM_\phi(s_w) - RM_\phi(s_l))$ 关于网络权重 $\phi$ 的梯度：
>     $$\nabla_\phi L(\phi) = -\sigma\Big(RM_\phi(s_l) - RM_\phi(s_w)\Big) \cdot \Big( \nabla_\phi RM_\phi(s_w) - \nabla_\phi RM_\phi(s_l) \Big)$$
>   * **物理直观**：若好回答得分远高于坏回答，梯度前的系数趋近于 0，模型微弱更新；若预测相反或接近，梯度前面的负系数会促使反向传播**拉高 $RM_\phi(s_w)$ 并压低 $RM_\phi(s_l)$ 的权重**。

---

## 页面 34 - 35: 完整的 RLHF 后训练流程
最终的 RLHF 优化包含以下几个步骤：
1. 从 SFT 模型初始化一个强化学习 Policy 模型 $p^{RL}_\theta(s)$。
2. 约束参数更新，防止其与初始 SFT 模型 $p^{PT}(s)$ 偏离过远。这是通过引入 **KL 散度惩罚项**实现的，防止网络寻找偏门漏洞欺骗奖励模型。
3. 优化的最终奖励形式：
   $$R(s) = RM_\phi(s) - \beta \log \frac{p^{RL}_\theta(s)}{p^{PT}(s)}$$
   其中第二项即为 KL 散度惩罚，$\beta$ 是控制惩罚强度的超参数。

> **[学习注释：RLHF 流程中的模型符号含义与 Loss 计算中的数学本质]**
> * **训练阶段的关系澄清（偏好对比 vs 策略优化）**：
>   * **第一阶段：训练奖励模型（页面 31-33）**：此时大模型参数被冻结，只通过输出两个回答好坏的对比（$RM(s_w) - RM(s_l)$）来训练一个静态的、代表人类品味的“AI 裁判” $RM_\phi$。
>   * **第二阶段：优化策略模型（页面 34-35）**：利用上一步已训练完毕且参数固定的裁判模型 $RM_\phi$ 来优化大模型本身 $p^{RL}_\theta$。此阶段不进行两两对比，大模型生成单个回答 $s$ 并由 $RM_\phi$ 直接打分结合 KL 散度得出奖励。这两个阶段属于上下游的关系。
> * **符号的双重代指与实际生成计算先后逻辑**：
>   * $p^{RL}_\theta$ 与 $p^{PT}$ 既代表模型本身，但在奖励公式 $R(s) = RM_\phi(s) - \beta \log \frac{p^{RL}_\theta(s)}{p^{PT}(s)}$ 中代表**两个模型对相同生成句子 $s$ 计算得到的概率数值**。
>   * **计算先后逻辑**：在每次迭代中，先由当前训练的 Policy 模型 $p^{RL}_\theta$ 采样生成回答 $s$，并计算出其生成的对数概率 $\log p^{RL}_\theta(s)$；随后，**将这个完全相同的生成回答 $s$** 喂给被冻结的初始 SFT 模型 $p^{PT}$ 跑一次前向传播，计算出其对应的对数概率 $\log p^{PT}(s)$。将这两个 Log-Softmax 概率值累加实数相减，即得到了 KL 散度惩罚。
> * **为什么“最小化 Loss”与“最大化 Reward”在数学上不矛盾？**：
>   * **自变量与常数系数**：在计算对 $m$ 个样本进行采样均值后的虚拟损失函数 $\text{Loss}(\theta) = - \frac{1}{m} \sum_{i=1}^m R(s_i) \log p^{RL}_\theta(s_i)$ 时，大模型在求导的那个微观瞬间无法更改已经计算完成的奖励值 $R(s)$，因此 **$R(s)$ 充当的是常数系数**，唯一的**优化自变量**是概率 $p^{RL}_\theta(s_i)$。
>   * **梯度的流向**：
>     * 当生成的句子奖励为正（$R(s_i) > 0$）时，为了减小 Loss，优化迫使 $\log p^{RL}_\theta(s_i)$ 增大（即**提高**该好回答的生成概率）；
>     * 当生成的句子奖励为负（$R(s_i) < 0$）时，为了减小 Loss，优化迫使 $\log p^{RL}_\theta(s_i)$ 减小（即**降低**该坏回答的生成概率）。
>   * **宏观表现**：由此更新参数 $\theta$，便能最终重塑模型的概率分布，获得更符合人类偏好的模型，从而在宏观上实现了期望奖励的稳步上升。

> **[学习注释：深入理解完整的 PPO 算法 —— 解决 REINFORCE 的发散痛点]**
> 为了克服 REINFORCE 仅使用绝对奖励 $R(s)$ 导致梯度更新方差过大、极易发散 of 缺陷，PPO 算法在实际实现中引入了 **Critic（价值模型）** 与 **Clipping（裁剪机制）** 两大武器：
> 
> * **一、 为什么 REINFORCE 极易导致模型发散？（高方差痛点）**
>   * **绝对值导致过度偏移**：若模型生成的回答整体不错（如 $99, 98, 97$ 分），模型会一味地大幅度拉高这所有回答的概率，导致大模型参数在单步更新中过度偏移（发散），且无法对细微的好坏差异做出区分。
>   * **解决方案**：工业界大模型对齐统一使用 **PPO (Proximal Policy Optimization)** 算法，它由 **Critic（价值模型）** 与 **Clipping（裁剪机制）** 两大武器组成，以控制参数的平稳更新。
> 
> * **二、 第一重保险：Critic 估值模型与优势函数（Advantage）**
>   * **什么是 Critic？**：它是一个随 Actor 同步训练、实时更新参数的“估值模型” $V_\phi$。其任务是仅根据 Prompt $x$ 预测 Actor 以后能拿到的期望平均分，扮演“随身军师”的角色。
>   * **Critic 如何更新？**：它是常规的有监督均方误差训练。每次迭代中，Actor 生成回答 $s_i$，已训练好的奖励模型打出最终分数 $R(s_i)$（作为 Critic 的 Ground Truth）。Critic 通过最小化 MSE Loss 进行梯度更新：
>     $$\text{Loss}_{\text{Critic}}(\phi) = \frac{1}{m} \sum_{i=1}^m \left( V_\phi(x_i) - R(s_i) \right)^2$$
>   * **如何控制方差？**：在更新 Actor 时，PPO 用优势值 $A(s_i) = R(s_i) - V_\phi(x_i)$（实际表现相对预估水平的相对优劣）来代替绝对奖励。
>     * *注：此处的数学表达式为全句维度的概念级简化，将原本需要在各个 Token（时间步）上进行的时序递推，简化为了“整句奖励值减去初始 Critic 估值”的形式。在真实的自回归长序列训练中，这种全局均摊的做法会导致严重的信用分配和时间因果性缺失。真实的 Token 级优势值计算采用 GAE（广义优势估计）算法。关于大模型 GAE 对每个 Token 局部梯度的精准控制机制、以及 Critic 价值网络在此上面临的收敛性痛点，请跳转参考 [第12讲: 页面40-42 的详细注释](file:///wsl.localhost/Ubuntu/home/blybq/code-project/cs224n/markdown/slides/lecture12-reasoning-part1.md#L341-L368)。*
>     * 若 $A(s_i) > 0$（表现好于预期），正向微调概率；
>     * 若 $A(s_i) < 0$（表现差于预期），反向压低概率；
>     * 若 $A(s_i) \approx 0$（符合预期），参数保持稳定。由此将梯度更新的方差降到最低。
> * **三、 第二重保险：新旧策略比值裁剪机制（Clipping）**
>   * **策略比例**：计算新策略与旧策略的概率比值 $r(\theta) = p_\theta(s) / p_{\theta_{\text{old}}}(s)$。
>   * **裁剪损失**：Actor 的损失函数为 $\text{Loss}_{\text{CLIP}}(\theta) = - \frac{1}{m} \sum_{i=1}^{m} \min \left( r_i(\theta) A(s_i), \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A(s_i) \right)$，其中限制系数 $\epsilon \approx 0.2$。
>   * **经验直觉**：
>     * **表现好时（$A > 0$）**：梯度上升促使 $r(\theta)$ 变大（大于 1）。一旦概率比值 $r(\theta)$ 突破 $1 + \epsilon$（如 1.2），`clip` 将其截断，梯度强行归零，**防止模型过度强化导致发散**。
>     * **表现差时（$A < 0$）**：梯度下降促使 $r(\theta)$ 变小（小于 1）。一旦概率比值 $r(\theta)$ 跌破 $1 - \epsilon$（如 0.8），`clip` 将其截断，梯度强行归零，**防止模型过度惩罚导致分布崩溃**。

---

## 页面 36 - 37: RLHF 流水线
* SFT (指令微调) $\rightarrow$ 训练奖励模型 $\rightarrow$ 基于 PPO 的强化学习优化。
* 实验证明，经过 RLHF 优化后的模型在生成质量和人类满意度上全面优于仅做预训练和 SFT 的模型。

---

## 页面 38 - 42: InstructGPT 与 ChatGPT
* **InstructGPT** [Ouyang et al., 2022]：将 RLHF 成功规模化应用到了数万个不同指令任务上，使得 1.3B 的模型在回答满意度上超越了 175B 的基座 GPT-3 模型。
* **ChatGPT**：将指令微调与 RLHF 技术拓展应用到了专门的对话智能体中，通过迭代式人类对齐训练，开创了生成式 AI 时代。

---

## 页面 43 - 47: RLHF 带来的表现与风格变化
* 经过对齐后的模型，其输出风格会发生显著变化：往往会给出**字数更长、排版更精致（使用大量 bullet points 列出要点）、态度极度客气且有条理**的回答。
* 许多工作利用 GPT-4 作为裁判 (GPT-4 as a judge) 代替人类进行自动评估。

---

## 页面 48 - 51: 强化学习与奖励建模的局限性
1. **奖励模型欺骗 (Reward Hacking)**：随着强化学习更新，Policy 模型会发现奖励模型设计中的“漏洞”（比如只要一味道歉或者多写废话就能刷高奖励得分），导致输出崩坏（奖励值高但人类实际觉得很蠢）。
2. **真假失衡**：如果人类标注员容易被模型生成的天花乱坠、看起来“权威自信”但实则编造的事实所欺骗，奖励模型便会倾向于奖励这些充斥着**幻觉 (Hallucinations)** 且伪造事实的回答。

---

## 页面 52 - 53: 强化学习 PPO 算法自身的痛点
* **实现与工程极其复杂**：
  * 在线生成耗时极长。
  * 需要同时在内存中加载 Actor、Reference model、Critic 以及 Reward Model，显存占用极大。
  * 对超参数极其敏感，极其容易发散，极难复现。

---

## 页面 54 - 55: 移出 RLHF 中的强化学习：直接偏好优化 (Direct Preference Optimization, DPO)
* **数学洞察**：在带 KL 散度约束的偏好最大化问题中，其最优解存在**闭式解 (Closed-form solution)**：
  $$p^*(y \mid x) = \frac{1}{Z(x)} p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)$$
* 将上式重新排列并进行对数变换，我们可以用模型在 $y$ 上的概率 $p_\theta(y \mid x)$ 以及预训练基线概率 $p^{PT}(y \mid x)$ 来**隐式地表征奖励值**，从而省去显式训练奖励模型的步骤：
  $$RM_\theta(x, y) = \beta \log \frac{p_\theta(y \mid x)}{p^{PT}(y \mid x)} + \beta \log Z(x)$$
* **DPO 损失函数** [Rafailov et al., 2023]：
  将代入后的隐式奖励带入 Bradley-Terry 偏好模型中，常数归一化项 $Z(x)$ 在两项相减时抵消，从而得到一个**极其简单、无需强化学习、仅需常规监督训练 (MLE) 的目标函数**：
  $$J_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{p_\theta(y_w \mid x)}{p^{PT}(y_w \mid x)} - \beta \log \frac{p_\theta(y_l \mid x)}{p^{PT}(y_l \mid x)} \right) \right]$$

  > **[学习注释：DPO 中的监督学习含义与符号定义]**
  > * **$w$ 与 $l$ 的英文缩写**：$w$ 代表 **winning（胜出、更受人类偏好的回答）**，$l$ 代表 **losing（落败、人类不喜欢的回答）**。在有些偏好对齐文献中也写作 $y_w$ 为 preferred (winner)，$y_l$ 为 dispreferred (loser)。
  > * **何为监督学习（MLE）？**：在 DPO 的训练数据中，**人类标注的 Ground Truth（监督标签）仅仅是每对样本中的胜负关系本身，而不需要任何打分数值。** 监督优化的目标非常简单：通过最大似然估计，促使模型 $p_\theta$ 在好回答 $y_w$ 上的生成概率相对基座 $p^{PT}$ 的提升，显著大于其在坏回答 $y_l$ 上的概率提升。大模型可以通过计算各自的 forward 概率值直接进行反向传播更新参数，完全无需强化学习框架（不需要采样探索、不需要 Critic 价值网络、不需要 Reward 模型）。

> **[学习注释：DPO 的数学灵魂 —— 求解带约束的偏好最大化问题]**
> 
> * **统一的数学问题定义**：
>   DPO 的核心数学基础是求解一个纯粹的带约束优化问题。在给定输入 Prompt $x$ 时，优化目标是调整模型的概率分布自变量 $p(y \mid x)$，使得如下目标函数 $E$ 最大化：
>   $$\max_{p} E = \sum_y p(y \mid x) \left( RM(x, y) - \beta \log \frac{p(y \mid x)}{p^{PT}(y \mid x)} \right)$$
>   * **硬性约束条件**：$\sum_y p(y \mid x) = 1$。
>   * **已知量（求导常数项）**：对于优化自变量 $p(y \mid x)$ 而言，已冻结的裁判评分 $RM(x, y)$ 和基座模型概率 $p^{PT}(y \mid x)$ 均被视为常数。
>   
>   以下是求解该问题的三种经典数学方法（其中方法二为方法一在数学底层上的本源理解）：
> 
> ---
> 
> * **方法一：基于信息论的 KL 散度配方法**：
>   * **KL 散度（Kullback-Leibler Divergence）的数学定义**：
>     用于衡量两个概率分布 $P$ 和 $Q$ 之间的差异度。其离散形式定义为：
>     $$\mathbb{D}_{\text{KL}}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$
>     * **核心结论（吉布斯不等式）**：$\mathbb{D}_{\text{KL}}(P \parallel Q) \ge 0$，且当且仅当 $P = Q$ 时取得最小值 0。
>   
>   * **求解过程（配分函数法）**：
>     1. 我们的目标是最大化 $E$。为了拼凑出 KL 散度的对数除法结构，我们先将括号内第一项进行“对数配方”：$RM(x,y) = \beta \log \exp\left(\frac{1}{\beta} RM(x,y)\right)$。代入目标函数 $E$ 中：
>        $$E = \sum_y p(y \mid x) \left( - \beta \log p(y \mid x) + \beta \log \left( p^{PT}(y \mid x) \exp\left( \frac{1}{\beta} RM(x, y) \right) \right) \right)$$
>     2. 为了使括号右项成为合法的概率分布，我们引入全局归一化常数（配分函数） $Z(x) = \sum_y p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)$，并定义一个合法的理想分布：
>        $$q(y \mid x) = \frac{1}{Z(x)} p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)$$
>     3. 将其代回 $E$ 的公式，合并对数项，展开常数：
>        $$E = \sum_y p(y \mid x) \left( \beta \log Z(x) + \beta \log \frac{q(y \mid x)}{p(y \mid x)} \right) = \beta \log Z(x) - \beta \mathbb{D}_{\text{KL}}\left(p(y \mid x) \parallel q(y \mid x)\right)$$
>     4. 由于 $\beta \log Z(x)$ 是不含优化自变量 $p$ 的固定常数，要使 $E$ 最大，必须令负项的 KL 散度达到最小值 0。
>     5. 根据上述 KL 散度非负性结论，当且仅当 $p(y \mid x) = q(y \mid x)$ 时散度为 0。由此一步得出最优概率分布闭式解：
>        $$p^*(y \mid x) = \frac{1}{Z(x)} p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)$$
> 
> ---
> 
> * **方法二：基于严格凹函数对数放缩的加权琴生不等式法（方法一的本源理解）**：
>   * **代数重组**：
>     我们直接对原目标函数 $E$ 的代数结构进行变形，将第一项 $RM(x,y)$ 收入对数中：
>     $$E = \sum_y \beta \cdot p(y \mid x) \log \frac{p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)}{p(y \mid x)}$$
>   * **琴生不等式放缩**：
>     由于 $p(y \mid x) \ge 0$ 与 $\sum_y p(y \mid x) = 1$，我们可以将外部的 $p(y \mid x)$ 看作加权琴生不等式中的权重 $w_i$，将对数内的项 $\frac{p^{PT} \exp(RM/\beta)}{p}$ 看作自变量 $x_i$。由于对数函数 $\log x$ 是严格凹函数（即 $\sum_i w_i \log x_i \le \log \sum_i w_i x_i$）：
>     $$E \le \beta \log \left( \sum_y p(y \mid x) \cdot \frac{p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)}{p(y \mid x)} \right)$$
>     在右侧的累加括号内部，**自变量大模型生成概率 $p(y \mid x)$ 在分子分母的乘法中被直接约去**！不等式大幅简化为：
>     $$E \le \beta \log \left( \sum_y p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right) \right)$$
>     这括号内正是配分函数 $Z(x)$ 的表达式。因此，我们通过一步放缩，直接推导出了目标函数 $E$ 的理论极大值上限：
>     $$E \le \beta \log Z(x)$$
>   * **取等条件**：
>     根据严格凹/凸函数的性质，琴生不等式取得等号（即目标函数 $E$ 达到理论上限 $\beta \log Z(x)$）的充要条件是对于所有的 $y$，对数内的项必须全部相等且退化为同一个与 $y$ 无关的常数 $C$：
>     $$\frac{p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)}{p(y \mid x)} = C \implies p(y \mid x) = \frac{1}{C} p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)$$
>     为满足概率分布的硬性归一化条件 $\sum_y p(y \mid x) = 1$，常数 $C$ 必须等于 $Z(x)$。
>     由此极其自然地导出了最优解分布的表达式：
>     $$p^*(y \mid x) = \frac{1}{Z(x)} p^{PT}(y \mid x) \exp \left( \frac{1}{\beta} RM(x, y) \right)$$
>     * **注**：吉布斯不等式（KL 散度非负性）在底层的推导本源也是依靠琴生不等式放缩。因此本方法正是方法一在信息论概念背后的底层数学本源理解。
> 
> ---
> 
> * **方法三：基于微积分的拉格朗日乘子法求偏导**：
>   * **求解过程（一阶导数法）**：
>     1. 我们将每个可能的回答 $y_i$ 的概率记为 $p_i$，将常数奖励 $RM(x, y_i)$ 记为 $RM_i$。引入拉格朗日乘子 $\lambda$ 约束 $\sum_i p_i = 1$，构建拉格朗日目标函数：
>        $$\mathcal{L}(p, \lambda) = \sum_i p_i \left( RM_i - \beta \log \frac{p_i}{p^{PT}_i} \right) + \lambda \left( \sum_i p_i - 1 \right)$$
>     2. 展开对数项以方便求导：
>        $$\mathcal{L}(p, \lambda) = \sum_i \left( p_i RM_i - \beta p_i \log p_i + \beta p_i \log p^{PT}_i + \lambda p_i \right) - \lambda$$
>     3. 对每一个独立的概率分量 $p_i$ 求偏导数，并令偏导数等于 0：
>        $$\frac{\partial \mathcal{L}}{\partial p_i} = RM_i - \beta \log p_i - \beta + \beta \log p^{PT}_i + \lambda = 0 \implies RM_i - \beta \log \frac{p_i}{p^{PT}_i} - \beta + \lambda = 0$$
>     4. 整理方程，将对数项孤立在等式一侧：
>        $$\log \frac{p_i}{p^{PT}_i} = \frac{RM_i + \lambda - \beta}{\beta} = \frac{1}{\beta} RM_i + \frac{\lambda - \beta}{\beta}$$
>     5. 对等式两边取以 $e$ 为底的自然指数，反解并消除对数：
>        $$p_i = p^{PT}_i \cdot \exp\left( \frac{\lambda - \beta}{\beta} \right) \cdot \exp\left( \frac{1}{\beta} RM_i \right)$$
>     6. 因为常数项 $\exp\left(\frac{\lambda-\beta}{\beta}\right)$ 与具体的句子 $i$ 无关，我们将其设为归一化常数 $C = 1/Z$。为满足 $\sum_i p_i = 1$ 的硬性约束：
>        $$\sum_i p_i = \frac{1}{Z} \sum_i p^{PT}_i \exp\left( \frac{1}{\beta} RM_i \right) = 1 \implies Z = \sum_i p^{PT}_i \exp\left( \frac{1}{\beta} RM_i \right)$$
>     7. 代入消去拉格朗日乘子项，同样导出与方法一、方法二完全相同的最优分布解：
>        $$p_i^* = \frac{1}{Z} p^{PT}_i \exp\left( \frac{1}{\beta} RM_i \right)$$
> 
> * **思路上的哲学转换总结**：
>   * 这三种方法在数学上殊途同归，共同揭示了 DPO 核心的**“代数转换”**灵魂：即最大化期望奖励指标（最大化 $E$），本质上可以直接等价转化为**“寻找模型实际概率分布 $p_\theta(y \mid x)$ 所需要契合的理想分布特征”**。
>   * 在方法三中，因为目标函数中含有 $\mathbb{D}_{\text{KL}}$ 带来的熵项 $p \log p$，**求导反解必产生指数 $\exp$**，自然决定了最优分布的指数族特征。
>   * 我们利用这种等价性，通过将 $RM(x,y)$ 反向用最优分布 $p^*$ 来隐式表达，在 Bradley-Terry 分类差值中刚好**完全抵消了全局归一化常数 $Z(x)$**。从而将极其复杂的 RL 训练直接降维成了在好坏回答上对大模型条件概率概率比值的有监督二分类微调。

---

## 页面 56 - 57: DPO 的优异表现
* **优势**：消除了复杂的 PPO 强化学习组件，训练不仅快且极度平稳，且效果在很多评测（如总结任务）中优于传统 PPO RLHF。
* 当前开源社区（如 Hugging Face TRL 框架下的主流大模型后训练）几乎全盘转向了使用 DPO 或其变体（如 KTO, IPO）。

---

## 页面 58: 强化学习算法的改进：GRPO
* **组相对策略优化 (GRPO)** [DeepSeek Math; Shao et al., 2024]：
  * 通过从同一 prompt 中采样一组（Group）候选回答进行相对对比，从而**取消了显式计算价值函数 (Value Network/Critic) 的网络开销**，极大地节省了显存开销，并提升了数学与逻辑推理任务的后训练效果。

> **[学习注释：DeepSeek 的 GRPO 算法 —— 彻底取消 Critic 的轻量化对齐]**
> 
> * **一、 传统 PPO 算法的痛点**：
>   * **显存开销倍增**：传统 PPO 需要实时维护一个与 Actor 同样庞大的 Critic（价值估值）模型。在显存中同时加载并更新这两个巨型网络，导致硬件开销翻倍。
>   * **双模型训练不稳定**：Actor 策略网络与 Critic 估值网络同步更新，由于彼此拉扯，极易因预测偏差而发生训练发散。
> 
> * **二、 GRPO 的完整操作步骤**：
>   对于输入的题目/Prompt $q$：
>   1. **组内采样 (Group Sampling)**：大模型 $p_\theta$ 对同一提示词 $q$ 并行采样生成一组（共 $G$ 个）候选回答，记为 $\{s_1, s_2, \dots, s_G\}$（通常 $G \approx 8$）。
>   2. **裁判打分 (Reward Scoring)**：将这组回答喂给冻结的奖励模型，得到每个回答的绝对得分 $\{r_1, r_2, \dots, r_G\}$。
>   3. **计算组内相对优势 (Group Advantage)**：直接计算这组得分的均值 $\mu$ 和标准差 $\sigma$，对每个回答进行组内 Z-score 归一化，得到组内相对优势值：
>      $$A_i = \frac{r_i - \mu}{\sigma}$$
>      * **数学直觉**：高出组内平均分的回答其优势 $A_i > 0$，低于平均分的回答其优势 $A_i < 0$。用这组样本自身的分布特征**彻底替代了 Critic 模型预测的基准分**，从而完全省去了 Critic 网络的显存占用与训练开销。
> 
> * **三、 GRPO 损失函数表达式（基于整句概率比的简化写法）**：
>   定义新策略与旧策略的整句概率比值为 $r_i(\theta) = \frac{p_\theta(s_i)}{p_{\theta_{\text{old}}}(s_i)}$，GRPO 的优化损失函数为：
>   $$\text{Loss}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left( \min \left[ r_i(\theta) A_i, \text{clip}\left(r_i(\theta), 1-\epsilon, 1+\epsilon\right) A_i \right] - \beta \mathbb{D}_{\text{KL}}\left(p_\theta \parallel p^{ref}\right) \right)$$
>   * **反向传播自变量**：要求导更新的唯一自变量为当前大模型参数 $\theta$；在求导瞬间，旧策略概率 $p_{\theta_{\text{old}}}$、基座模型 $p^{ref}$ 以及标准优势分数 $A_i$ 均被视为常数项，优势值 $A_i$ 充当梯度更新的杠杆系数。
> 
> ---
> 
> > **[学习注释：DPO 与 GRPO 的本质对比与工业界选型逻辑]**
> > 
> > * **核心本质区别（“模仿”与“探索”的对立）**：
> >   * **DPO 的本质是“拟合与模仿”（离线静态）**：DPO 是一种**离线式 (Offline)** 方法，完全在人类标注好的静态偏好数据对 $(y_w, y_l)$ 上进行拟合。在训练过程中，大模型**自身从不主动生成任何新答案，也不存在“根据评分优化自身新生成结果”的循环**。它仅能被动地调整对已知好坏回答的概率映射。
> >   * **GRPO 的本质是“探索与顿悟”（在线动态）**：GRPO 是一种**在线式 (Online)** 强化学习方法。大模型在训练时**必须亲自生成新文本，实时针对同一个 Prompt 并行探索出一组不同的答案**，并通过裁判（奖励模型或规则）对这组“自己刚刚写出的草稿”进行优劣对比和动态打分，从而实现自我迭代。
> > 
> > * **不同工业场景下的技术选型法则**：
> >   1. **数学推理、代码编写、逻辑解谜（具有客观标准的场景） $\rightarrow$ 必须选择 GRPO (或 PPO)**：
> >      * **为什么 DPO 在此类场景效果很差？**：客观题有严密的对错标准且解题路径多元。由于 DPO 无法在训练中实时探索新解法，它只认识静态数据集里已有的标准步骤。一旦模型在考试中自己写出了一种静态数据集里没有的“天才新解法 C”，因为离线状态下没有打分器实时支持，DPO 无法判断其正确性，反而会强行压低其概率；此外，模型在离线状态下也无法学会在“中途写错后如何自我反思纠正（Self-Correction）”。
> >      * **GRPO 的神效**：通过在训练中实时采样一组回答并根据客观规则（如编译器或数学计算器）动态打分，模型能够在一组解法对比中自己**顿悟**出正确的逻辑推理路径，甚至学会写出“Wait, I made a mistake, let me recalculate...”这样的反思和自我纠正过程（如 DeepSeek-R1 超长思维链的涌现）。
> >   2. **日常闲聊、文学写作、安全红队对齐（偏向主观偏好的场景） $\rightarrow$ 优先选择 DPO**：
> >      * **为什么 DPO 在此类场景效率极高？**：主观偏好任务（如“写一首委婉的道歉信”）没有客观的物理对错，只有人类喜好（如格式排版、礼貌客气）。研发团队手里往往积攒了海量静态对比数据，此时使用 DPO 可以直接避开昂贵且不稳定的强化学习探索，极快、极稳地让模型学会人类说话的语气和安全规范。
> > 
> > * **总结**：
> >   DPO 适合作为大模型对齐的“终点线修剪器”（教模型学会客气、安全与排版格式），而 GRPO 则是激发模型硬核推理与思考能力的“发动机”。

---

## 页面 59 - 62: 数据与标注偏见问题
* **标注劳工偏见**：大模型的后训练标注工作大多外包给低收入地区员工，这会带来偏见传递。
* **意识形态与国别偏见**：人类标注员的个人信仰、社会阶层以及价值偏见会直接固化在大模型内部，导致模型回答偏向某些特定文化和价值观。

---

## 页面 63 - 64: 总结与未来展望
* RLHF 比简单的指令微调走得更远，但对偏好数据的依赖依然昂贵。
* **前沿趋势**：
  * **AI 反馈强化学习 (RLAIF / RL from AI feedback)** [Bai et al., 2022]：利用强大的大语言模型作为偏好裁判去训练其他模型，省去人工偏好标注。
  * **自我博弈与自我反思微调**：大模型基于自身生成的文本进行自我纠错 and 对齐 [Huang et al., 2022]。
* 虽然对齐策略很成功，但大模型本身的架构性限制（如大体积、事实性幻觉问题）可能仅靠 RLHF 或后训练手段依然无法完全根治。
