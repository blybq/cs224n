# 🤗 Hugging Face Transformers Tutorial

## 页面 1: 标题
* **主题**：Hugging Face Transformers 实践教程
* **课程名称**：CS224N：基于深度学习的自然语言处理 (Natural Language Processing with Deep Learning)
* **时间**：2026年冬季学期
* **课件主讲人**：Minsik Oh

---

## 页面 2: 课程大纲
互动式的动手实践教程安排：
1. **环境准备与介绍**：安装依赖包，概述 Hugging Face 生态系统。
2. **深入分词器 (Tokenizers)**：深入探索分词流水线、特殊标记、数据填充 (Padding) 与批量编码。
3. **模型与推理 (Models & Inference)**：加载预训练模型、AutoModel 通用类、执行前向传播、注意力机制可视化。
4. **在 IMDB 数据集上进行微调**：加载数据集、手写训练循环、使用 HF Trainer 接口、使用回调函数 (Callbacks) 与模型评估。
5. **文本生成与总结**：利用 GPT-2 进行文本生成、处理自定义数据集、快捷推理流水线 (Pipelines)。

---

## 页面 3: 什么是 Hugging Face Transformers？
Hugging Face 生态包含以下几个核心模块：
* **Models (模型库)**：提供各种大模型的预训练参数权重与加载代码（如 LLaMA 3, DBRX, BERT, GPT-2 等）。
* **Tokenizers (分词器)**：用 Python 和 Rust 实现的针对具体模型的文本预处理工具，将文本转换为数字。
* **Pipelines (流水线)**：仅需一行代码，即可快速部署执行常见的 NLP 任务推理。
* **Datasets (数据集)**：提供标准公开数据集的便捷加载与预处理接口。
* **Trainer (训练器)**：高度封装的训练循环抽象类，内置日志记录与保存检查点 (Checkpoint) 功能。
* **支持的深度学习框架**：PyTorch（本教程使用）、TensorFlow、Flax / JAX。
* **适合的期末项目类型**：
  * 将预训练模型应用到全新任务。
  * 深度探究并分析大模型的行为特征与内部表征。

---

## 页面 4: 通用工作模式
Hugging Face 的工作流遵循简单的三步走模式：
1. **寻找模型**：在 Hugging Face Hub (社区官网) 上寻找适合的模型，任何开发者都可以上传并托管模型。
2. **初始化**：加载对应模型的分词器和模型本身。
3. **预测**：使用分词器处理文本，将其输入模型执行前向传播，得到最终预测值。
```python
# 示例代码：
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model     = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
outputs   = model(**tokenizer("Hugging Face is great!", return_tensors="pt"))
```

---

## 页面 5: 实践部分：环境配置与首次预测
1. 打开课程页面上链接的 Google Colab 笔记本。
2. 运行环境配置单元格： `pip install transformers datasets accelerate`
3. 运行 Part 0 单元格：加载情感分类模型和分词器。
4. 尝试更改输入的字符串，并重新执行预测。
* *💡 预期输出：输入默认文本，应该会打印输出预测结果："The prediction is POSITIVE"。*

---

## 页面 6: 分词器：从纯文本到数值编码
* **分词逻辑**：
  * **原始文本**：`"Hugging Face is great!"`
  * **分词列表 (Tokens)**：`["Hu", "##gging", "Face", "is", "great", "!"]` （通过子词切分，减小词表大小）
  * **Token ID 映射**：`[20164, 10932, 10289, 1110, 1632, 106]`
  * **张量化模型输入**：`tensor([[101, 20164, ...]])`
* **加载分词器的三种方式**：
  * `DistilBertTokenizer`：特定于模型的 Python 分词器。
  * `DistilBertTokenizerFast`：使用 Rust 开发的高速分词器。
  * `AutoTokenizer`：推荐做法！自动检测并加载对应模型的适配分词器。

---

## 页面 7: 分词器的具体步骤演示
```python
input_tokens = tokenizer.tokenize(input_str)
# → ["Hu", "##gging", "Face", "Transformers", "is", "great", "!"]

input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
# → [20164, 10932, 10289, 25267, 1110, 1632, 106]

input_ids_special = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
# → [101, 20164, ..., 106, 102] （加上了 [CLS] 开头符和 [SEP] 结尾符）

decoded = tokenizer.decode(input_ids_special)
# → "[CLS] Hugging Face Transformers is great! [SEP]"
```
* **特殊标记 (Special Tokens)**：如开头 `[CLS]`，结尾 `[SEP]`，这依具体模型而异。
* **子词分词 (Subword Tokenization)**：`"Hugging"` 被拆为 `"Hu"` + `"##gging"`，有效解决未登录词 (OOV) 的编码。
* **注意力掩码 (Attention Mask)**：常规 token 设为 1，填充 token 设为 0。指导模型哪些词需要关注，哪些忽略。

---

## 页面 8: 填充、截断与分批
```python
model_inputs = tokenizer(
    ["Short text", "A much longer text..."],
    return_tensors="pt", padding=True, truncation=True
)
```
* **处理过程图解**：
  * 第一句（较短）：`[CLS] Short text [SEP] [PAD] [PAD] [PAD]`
  * 第二句（较长）：`[CLS] A much longer text ... [SEP]`
  * 第一句的注意力掩码：`1 1 1 1 0 0 0` （末尾三个 `[PAD]` 的 mask 值为 0）。

---

## 页面 9: 实践部分：玩转分词器
1. 运行 Section 0.1，体验并对比通用分词器与特定 Rust 分词器的加载与分词差异。
2. 尝试对自己手写的长短句进行 tokenize 并重新解码（decode）。
3. 传入不同长度句子的列表，实验 `padding=True` 的填充效果。
4. 使用 `char_to_token()` 查询某个字符归属于哪一个 subwordpiece。
5. 体验 `batch_decode` 时，设置 `skip_special_tokens=True` 或 `False` 的区别。

---

## 页面 10: 模型：架构与模型头部 (Model Heads)
* **Transformer 模型的三大主要类型**：
  1. **编码器 (Encoders)**（如 BERT, DistilBERT）：适合做分类任务、命名实体识别 (NER)、特征探针分析。
  2. **解码器 (Decoders)**（如 GPT-2, LLaMA）：适合自回归文本生成。
  3. **编解码器 (Encoder-Decoder)**（如 BART, T5）：适合做文本摘要和机器翻译。
* **面向特定任务的模型头部 (Model Heads)**：
  在同一个 Base 模型表征上，可以套接不同的输出头来处理不同任务：
  * `*ForMaskedLM`：完形填空任务。
  * `*ForSequenceClassification`：情感分类、主题分类。
  * `*ForTokenClassification`：命名实体识别 (NER)、词性标注 (POS)。
  * `*ForQuestionAnswering`：抽取式问答。
  * `*ForCausalLM`：文本生成任务。

---

## 页面 11: 模型推理：它完全就是 PyTorch！
```python
# 数据处理
model_inputs = tokenizer(text, return_tensors="pt")
# 前向传播
model_outputs = model(**model_inputs)
# 计算 loss 并求反向传播
loss = F.cross_entropy(model_outputs.logits, labels)
loss.backward()
```
* **标准 PyTorch 模型**：你可以直接使用任何标准的 PyTorch 优化器 (Optimizer)、学习率调度器 (Scheduler) 或自定义损失函数。
* **模型自带 Loss 计算**：直接在参数中传入 `labels=labels`，模型内部会依据架构自动计算并返回对应的损失值 `outputs.loss`。
* **获取内部状态**：可传入 `output_attentions=True` 以及 `output_hidden_states=True` 来提取注意力权重和每层隐藏层向量。

---

## 页面 12: 实践部分：模型推理深度探索
1. 运行 Section 0.2，加载 DistilBERT 分类模型。
2. 对比 base 基础模型输出的隐藏层表征与分类模型输出的 logits。
3. 体验将 `labels` 传给模型，直接获取 `outputs.loss`。
4. 运行注意力可视化代码，查看特征热图。
5. 观察哪些注意力头最关注 `[SEP]` 标记，哪些更关注局部上下文。

---

## 页面 13: 细致微调：加载与清洗数据集
* **IMDB 数据集**：包含 25,000 条训练评论和 25,000 条测试评论，旨在进行二分类情感分析（正向/负向）。
* **加载与 Map 数据操作**：
```python
from datasets import load_dataset
ds = load_dataset("stanfordnlp/imdb")

# 批量分词映射
tokenized = ds.map(
  lambda ex: tokenizer(ex['text'], padding=True, truncation=True),
  batched=True
)
```
* **数据集预处理管道的四个步骤**：
  1. **分词 (Tokenize)**：在文本字段上应用 tokenizer，执行填充和截断。
  2. **删除无用字段**：调用 `remove_columns(["text"])`，去除原始字符串字段，只留下 ID。
  3. **重命名字段**：将 `"label"` 更改为 `"labels"`（这是 HF Trainer 所期待的关键字参数）。
  4. **类型转换**：调用 `set_format("torch")` 将数据直接转换为 PyTorch 格式的张量。

---

## 页面 14: 模型微调的两种写法
* **方法 A：手写原生 PyTorch 训练循环**：
  * 手写两层 for 循环（epochs, batches），显式调用 `optimizer.step()`, `optimizer.zero_grad()` 等，能获得最清晰的细粒度控制。
* **方法 B：使用 HF Trainer 工具 (推荐)**：
```python
args = TrainingArguments(
  output_dir="checkpoints",
  per_device_train_batch_size=16,
  num_train_epochs=2,
  eval_strategy="epoch",
  learning_rate=2e-5
)

trainer = Trainer(
  model=model,
  args=args,
  train_dataset=train_ds,
  eval_dataset=val_ds,
  compute_metrics=compute_metrics_fn
)
trainer.train()
```

---

## 页面 15: Trainer 特性：回调函数 (Callbacks) 与模型评估
* **常用的内置回调函数**：
  * `EarlyStoppingCallback`：在验证集 loss 不再下降（进入瓶颈）时，自动提前终止训练，防止过拟合。
  * 自定义 `LoggingCallback`：例如将每一次评测的指标实时打印并保存到 JSONL 文件中。
  * 回调函数可以灵活挂载在 `on_log`, `on_epoch_end`, `on_evaluate`, `on_save` 等多个时间节点。
* **模型评估**：
  * 调用 `trainer.evaluate()` 会直接运行评测指标并返回结果。
  * 调用 `trainer.predict(test_ds)` 提取测试集的模型预测概率和最终指标。
  * `compute_metrics` 是一个用于计算自定义评价指标（如 Accuracy, F1 值等）的函数。
* > [!TIP]
  > **微调起手超参数推荐**：
  > * **Epochs**: 2 至 4 次即可。
  > * **学习率 (Learning Rate)**: $2 \times 10^{-5}$ 或 $5 \times 10^{-5}$。
  > * **Batch Size**: 设为你的 GPU 显存能容纳的最大值。
  > * **权重衰减 (Weight Decay)**: 0, 0.01 或 0.1。
  > * **预热步数 (Warmup Steps)**: 0, 100 或 500。

---

## 页面 16: 实践部分：在 IMDB 上进行微调
1. 运行 Section 2.1：加载并清洗 IMDB 数据集。
2. 运行 Section 2.2：使用原生的 PyTorch 训练循环微调模型，观察 loss 的下降过程。
3. 使用封装好的 HF Trainer 版本再次微调模型，对比两者的便捷性与输出差异。
4. 添加早停回调 (`EarlyStoppingCallback`) 和日志回调。
5. 运行 `trainer.predict()` 查看测试集上的准确率。
6. 加载保存下来的 checkpoints 权重，对自己手动输入的自定义影评进行情感倾向测试。

---

## 页面 17: 利用 GPT-2 进行文本生成
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

prompt = "Once upon a time"
tokenized = tokenizer(prompt, return_tensors="pt")
output = model.generate(**tokenized, max_length=50, do_sample=True, top_p=0.9)
```
* **控制生成的核心参数**：
  * `max_length`：最大生成的 token 长度。
  * `do_sample`：设为 True 代表启用概率采样，False 则代表使用贪婪解码 (Greedy decoding)。
  * `top_p` (核采样/Nucleus sampling)：仅在累积概率之和达到指定概率 $p$ 的候选词集合中进行采样。
  * `temperature` (温度值)：控制生成的随机性。越低生成越确定（通常不易跑题但相对单一），越高生成越发散、越富有多样性。
  * `top_k`：仅在概率最高的前 $k$ 个词中进行随机采样。

---

## 页面 18: 快捷 Pipeline 与完形填空任务
* **一行代码极速推理 Pipeline**：
```python
from transformers import pipeline

sa = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
print(sa("This movie is great!"))
```
* 支持的快捷任务包括：情感分析、Mask 填充（完形填空）、文本生成、问答、摘要生成、翻译、命名实体识别等。
* **Mask 填充示例 (fill-mask)**：
```python
mlm = pipeline("fill-mask", model="bert-base-cased")
print(mlm("I am [MASK] to learn!"))
```
* 预测出的 Mask 候选词概率：`excited` (35.5%), `going` (15.6%), `eager` (7.9%), `here` (3.6%)。
* **自定义数据集**：可继承 PyTorch 的 `Dataset` 并让 `__getitem__` 返回分词器映射后的 dictionary 供 Trainer 使用。详情参阅笔记本附录 1 和 2 的 seq2seq 数据处理细节。

---

## 页面 19: 实践部分：生成、流水线与总结
1. 运行 Appendix 0，用轻量级的 DistilGPT-2 模型进行自回归文本生成。
2. 调整 prompt 和超参（top_p, temperature 等），观察生成文本的变化。
3. 运行 Appendix 2，用 pipeline 对特定文本直接进行情感倾向测试。
4. 运行 Appendix 4，用 BERT 跑完形填空，设计你自己的 `[MASK]` 题目并检查模型的常识与语法匹配度。
* *🎯 挑战任务：你能通过仔细调整温度 (temperature) 和采样超参，让 GPT-2 生成一篇逻辑相对连贯的段落吗？*

---

## 页面 20: 学习资源与下一步行动
* **官方文档 (HF Docs)**：最权威的 API 指南与模型手册。huggingface.co/docs
* **免费课程 (HF Course)**：由浅入深的 NLP 互动视频与实践教程。huggingface.co/course
* **任务代码模板 (HF Examples)**：包含丰富的脚本，是开启 CS224N 项目的极佳范本。github.com/huggingface/transformers/examples
* **项目实战建议**：做 CS224N 期末项目时，推荐先基于预训练模型使用 HF Trainer 跑通基本 Baseline，随后调整超参数、尝试不同 prompt、监测验证集 loss。祝大家项目取得优秀成果！
