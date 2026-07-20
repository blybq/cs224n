## 页面 1: 深度学习自然语言处理

深度学习自然语言处理 (Natural Language Processing with Deep Learning)

CS224N/Ling284
Diyi Yang & Yejin Choi
第 1 讲：自然语言处理导论与历史回顾 (Lecture 1: Introduction and the History of NLP)

## 页面 2: 课程大纲

第 1 讲：自然语言处理导论与历史回顾
1. 课程介绍 (10 分钟)
2. NLP 历史：我们是如何走到今天的 (65 分钟)

**今日核心学习目标**：理解 NLP 中的范式转移 (paradigm shifts)，以及我们对语言的假设是如何塑造每个时代的技术可能性的。

## 页面 3: 课程教务简述

* **授课教师**：Diyi Yang, Yejin Choi
* **课程负责人 (Head TA)**：Julie Kallini
* **课程管理员**：John Cho
* **助教团队**：很多优秀的助教！请参见网站
* **上课时间**：周二/周四 4:30–5:50 太平洋时间，Nvidia Aud. (→ 录像)
* **联络邮箱**：cs224n-win2526-staff@lists.stanford.edu
* **其他重要信息**：我们已在课程主页上放置了许多其他重要信息。请务必阅读！
  * http://cs224n.stanford.edu/ 亦即 http://www.stanford.edu/class/cs224n/
  * 包含：助教信息、课程大纲、答疑/办公时间 (Office Hours)、Ed 论坛（用于所有课程疑问与讨论）
* **办公时间 (Office hours)** 自本周四开始！
* **Python/NumPy 和 PyTorch 教程**：前两个周五举行。
* **幻灯片 PDF** 将在每节课前上传。

## 页面 4: [无内容]

## 页面 5: 我们的教学期望是什么？（即“学习目标”）

1. **应用于 NLP 的现代深度学习高效方法的基石**
   * **基础内容**：词向量 (word vectors)、循环神经网络 (recurrent networks)、注意力机制 (attention)、Transformer
   * **NLP 2025 年的关键方法**：预训练 (pretraining)、后训练 (post-training)、高效适配 (efficient adaptation)、智能体 (agents)、推理 (reasoning)、多语言 (multilinguality)、多模态 (multimodality)、可解释性 (interpretability) 等。
2. **对人类语言的宏观理解，以及通过计算机理解和生成语言的难点**
3. **理解并有能力为语言与计算涉及的一些主要问题构建系统**：
   * 词表示 (word representations)、问答 (question answering)、大语言模型微调 (fine-tuning LLMs)、检索增强生成 (RAG)、智能体系统与工具使用 (agentic systems and tool use) 以及大语言模型评估 (LLMs evaluation)

## 页面 6: 课程作业与评分政策

* **4 次作业（每次耗时约 1.5 周）**：6% + 3 × 14% = 48%
  * HW1 今日发布！下周二下午 4:30 截止！
  * 提交至 Canvas 中的 Gradescope（即使用 @stanford.edu 邮箱注册您的 Gradescope 账号）
* **期末项目（默认或自定义项目，1–3 人）**：49%
  * 项目提案 (Project proposal): 8%，里程碑 (milestone): 6%，海报或网页总结 (poster or web summary): 3%，期末报告 (report): 32%
* **课堂参与度**：3%
  * 特邀讲座反馈、Ed 社区表现、课程评估、印象分 (karma) —— 详见网站！
* **迟交政策 (Late day policy)**
  * 拥有 6 天免费迟交额度；此后每迟交一天扣除课程总评成绩的 1%
  * 除非提前获得批准，否则每份作业迟交超过 3 天将不予接收

## 页面 7: 课程作业与评分政策

* **协作政策**：
  * 请阅读网站和学术诚信守则 (Honor Code)！理解被允许的协作范围以及如何进行记录：不要从网上抄袭代码；承认与其他同学的合作；独立撰写作业的解答。
  * 学生必须独立提交各自的 CS224N 作业。
* **AI 工具使用政策**：
  * 大语言模型虽然好用，但我们不希望在作业中看到由 ChatGPT 生成的解答。
  * 允许与 AI 工具协作，但严禁直接向其索要作业答案。
  * 利用 AI 工具实质性地完成作业将被视为违反学术诚信守则（更多详情请参见此处生成式 AI 政策指南 [Generative AI Policy Guidance]）。

## 页面 8: 作业总体规划（需独立完成！）

* **HW1**：希望这是一个简单的入门作业 —— 一个 Jupyter/IPython Notebook。
* **HW2**：涵盖神经网络基础 and 张量导数计算，您将围绕依存句法分析 (dependency parsing) 构建一个用于 NLP 任务的小型网络。
* **HW3**：从零编写 Transformer 并理解注意力机制 (attention)。
* **HW4**：专注于大语言模型评估 (LLMs evaluation) 和红队测试 (redteaming)。
* **期末项目 (Final Project)**：后续会介绍更多细节，您可以选择：
  * **做默认项目**：
    * 您将实现一个 GPT，然后对其进行微调和适配以用于下游任务。
    * 开放式但起步较容易；对许多人来说是一个好选择。
  * **提出自定义期末项目**：需经我们批准。
    * 您将获得导师（助教/教授/博士后/博士生）的反馈。
    * 可以 1–3 人的团队形式开展工作；可以使用任何语言/工具包。

## 页面 9: 课程大纲

第 1 讲：自然语言处理导论与词向量
1. 课程介绍 (10 分钟)
2. NLP 历史：我们是如何走到今天的 (65 分钟)
   * 审视 NLP 历史的一种极具创意的方式
     * *鸣谢斯坦福 NLP 小组；请前往 Gates 3B 展区参观展览*

**今日核心学习目标**：理解 NLP 中的范式转移，以及我们对语言的假设是如何塑造每个时代的技术可能性的。

## 页面 10

## 页面 11

## 页面 12

## 页面 13

## 页面 14

## 页面 15

## 页面 16

## 页面 17

## 页面 18

## 页面 19: 课程大纲

第 1 讲：自然语言处理导论与词向量
1. 课程介绍 (10 分钟)
2. NLP 历史：我们是如何走到今天的 (65 分钟)
   * 审视 NLP 历史的一种创意方式 (5 分钟)
     * *鸣谢斯坦福 NLP 小组；请前往 Gates 3B 展区参观展览*
   * 审视 NLP 历史的一种科学方式 (60 分钟)

**今日核心学习目标**：理解 NLP 中的范式转移，以及我们对语言的假设是如何塑造每个时代的技术可能性的。

## 页面 20

## 页面 21: 特邀讲座

CS224N 创始人 Chris Manning 教授特邀讲座

**NLP 的历史**
