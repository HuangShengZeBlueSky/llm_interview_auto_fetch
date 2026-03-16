# LLMs_interview_notes 提取：一、大模型（LLMs）基础面

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
本专题是针对大语言模型（LLMs）的“地基”测试，主要考察候选人对大模型底层架构、数学原理、演进路线以及基础工程能力的掌握程度。核心考点包括：
- **宏观架构认知**：主流开源模型的派系划分及底层架构差异（如不同 Decoder 变体的本质区别）。
- **微观数学推导**：核心组件（Norm、FFN、激活函数）的计算公式，以及演进原因（如为什么用 RMSNorm 替代 LayerNorm）。
- **性能瓶颈与优化**：传统 Attention 的痛点（复杂度、显存占用）及前沿优化方案（算法层与系统层）。
- **工程实践能力**：基于主流框架（如 HuggingFace `transformers`）的模型加载与中间层特征提取。

## 答题主线
面试答题时，建议遵循**“演进逻辑 + 核心公式 + 工程实现”**的三维主线：
1. **谈架构讲 Mask**：区分不同架构（Causal/Prefix/Encoder-Decoder）时，一针见血地指出其本质是 **Attention Mask 矩阵的形状不同**。
2. **谈组件讲效率**：回答公式不仅要写对，还要点出“为什么现在的大模型都这么改”。例如 RMSNorm 去掉均值平移是为了提速，GeLU/Swish 是为了更平滑的梯度。
3. **谈优化讲瓶颈**：分析 Attention 优化时，严格区分**计算复杂度瓶颈**（用 Sparse/FlashAttention 解决）和**显存带宽瓶颈**（用 MQA/GQA/PagedAttention 解决）。
4. **谈工程讲 API**：代码题直接给出最简洁、最 native 的框架调用方法，体现日常开发的熟练度。

## 代表题答法

### 问题1：主流开源模型体系有哪些？不同 Decoder 架构（Prefix/Causal/Encoder-Decoder）的区别是什么？
**答题要点：**
- **主流体系**：目前以 LLaMA 系（Causal Decoder代表）、Qwen 系、GLM 系（Prefix Decoder代表）、Mistral 系为主。
- **架构区别（核心在于 Attention Mask）**：
  - **Causal Decoder (如 LLaMA, GPT)**：严格的下三角 Mask。每个 token 只能看到自己及之前的 token。训练效率高，是目前绝对的主流。
  - **Prefix Decoder (如 ChatGLM)**：前缀部分（Prompt）是全双工可见的（全1 Mask），生成部分是单向可见的（下三角 Mask）。结合了双向理解和单向生成的优势。
  - **Encoder-Decoder (如 T5, BART)**：Encoder 端全双工，Decoder 端单向，且 Decoder 会对 Encoder 的输出做 Cross-Attention。参数量分配分散，目前在千亿级 LLM 中较少使用。

### 问题2：请写出 Layer Norm、RMS Norm、FFN 和 GeLU 的计算公式，并说明演进关系。
**答题要点：**
- **Layer Norm**：计算均值 $\mu$ 和方差 $\sigma^2$。公式：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
- **RMS Norm**：舍弃了中心化（均值 $\mu$），只计算均方根。公式：$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$。**加分项**：指出 RMSNorm 假设均值接近0，减少了计算量，在保持模型效果的同时提升了约 10%-50% 的计算效率，被 LLaMA 等广泛采用。
- **FFN (Feed-Forward Network)**：传统公式为 $y = W_2(\sigma(W_1 x + b_1)) + b_2$。**加分项**：指出目前主流 LLM 已将其升级为 SwiGLU 结构（无偏置，且引入门控机制）。
- **GeLU**：结合了 Dropout 和 ReLU 的思想，公式为 $x \cdot \Phi(x)$，其中 $\Phi(x)$ 是标准正态分布的累积分布函数。常使用近似公式：$0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$。

### 问题3：传统 Attention 存在哪些问题？有哪些优化方向？
**答题要点：**
- **存在的痛点**：
  1. **计算与空间复杂度高**：序列长度 $N$ 增加时，Attention 矩阵的计算量和显存占用呈 $O(N^2)$ 爆炸。
  2. **推理期显存瓶颈（Memory Bound）**：自回归生成时，需保存历史的 KV Cache，导致显存占用大，且受限于显存带宽，推理速度慢。
- **优化方向**：
  - **算法结构层**：引入 MQA (Multi-Query Attention) 或 GQA (Grouped-Query Attention) 共享 KV 头，大幅降低 KV Cache 显存占用；引入 Sparse Attention 或 Sliding Window Attention (如 Mistral) 降低长文本计算复杂度。
  - **系统工程层**：使用 FlashAttention（利用 GPU SRAM 硬件特性，通过 Tiling 切块融合算子，减少 HBM 读写，实现 IO 优化）；使用 PagedAttention（vLLM 核心，解决 KV Cache 显存碎片化问题）。

### 问题4：如何利用 transformers 库加载 Bert 模型并输出指定的 hidden_state？
**答题要点：**
直接给出标准代码实现，强调 `output_hidden_states=True` 参数。
```python
from transformers import AutoModel, AutoTokenizer

# 1. 加载模型和分词器
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 关键点：配置 output_hidden_states=True
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# 2. 前向传播
outputs = model(**inputs)

# 3. 获取 hidden_states
# outputs.hidden_states 是一个 tuple，包含 Embedding 层输出 + 所有 Transformer 层的输出
# 长度为 13 (1层 emb + 12层 encoder)
all_hidden_states = outputs.hidden_states

# 获取指定层（例如最后一层）的 hidden_state
last_layer_hidden = all_hidden_states[-1] 
# 获取第一层（索引为1，0是embedding）的 hidden_state
first_layer_hidden = all_hidden_states[1] 
```

## 大模型（LLMs）基础面

- 来源链接：https://articles.zsxq.com/id_mw52p1pfbzql.html
- 题目数：12

### 原始题目

1. 1 目前 主流的开源模型体系 有哪些？
2. 2 prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？
3. 3 大模型LLM的 训练目标 是什么？
4. 4 涌现能力是啥原因？
5. 5 为何现在的大模型大部分是Decoder only结构？
6. 6 简单 介绍一下 大模型【LLMs】？
7. 7 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？
8. 8 大模型【LLMs】具有什么优点？
9. 9 大模型【LLMs】具有什么缺点？
10. 10 encoder-only, decoder-only, encoder-decoder的区别?
11. 11 BART、llama、gpt、t5、palm等主流模型异同点?
12. 12 prefix LM 和 causal LM 区别是什么?

### 可直接复用的答案

#### 示例 1. 5 为何现在的大模型大部分是Decoder only结构？

大模型从模型架构上主要分为三种：Only-encoder, Only-Decoder, Encoder-Decoder三种模型架构

- Only-encoder：例如BERT，通过在大规模无标签文本上进行预训练，然后在下游任务上进行微调，具有强大的语言理解能力和表征能力。

- Only-Decoder: 例如GPT，通过在大规模无标签文本上进行预训练，然后在特定任务上进行微调，具有很强的生成能力和语言理解能力。

- Encoder-Decoder：例如T5（Text-to-Text Transfer Transformer）可以用于多种自然语言处理任务，如文本分类、机器翻译、问答等。

而LLM之所以主要都用Decoder-only架构，除了训练效率和工程实现上的优势外，在理论上是因为Encoder的双向注意力会存在低秩问题，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数。所以，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。

> 匹配来源：DeepLearing-Interview-Awesome-2024_LLMs专题:LLMs/Reference.md | 匹配分数：0.96

## Layer normalization 篇

- 来源链接：https://articles.zsxq.com/id_pzcgd4ovk098.html
- 题目数：8

### 原始题目

1. Layer Norm 的计算公式写一下？
2. RMS Norm 的计算公式写一下？
3. RMS Norm 相比于 Layer Norm 有什么特点？
4. Deep Norm 思路？
5. 写一下 Deep Norm 代码实现？
6. Deep Norm 有什么优点？
7. 1 LN 在 LLMs 中的不同位置 有什么区别么？如果有，能介绍一下区别么？
8. LLMs 各模型分别用了 哪种 Layer normalization？

## LLMs 激活函数篇

- 来源链接：https://articles.zsxq.com/id_6xm3wzzice2s.html
- 题目数：8

### 原始题目

1. 1 介绍一下 FFN 块 计算公式？
2. 2 介绍一下 GeLU 计算公式？
3. 3 介绍一下 Swish 计算公式？
4. 4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？
5. 5 介绍一下 使用 GeLU 的 GLU 块 计算公式？
6. 6 介绍一下 使用 Swish 的 GLU 块 计算公式？
7. 7 各LLMs 都使用哪种激活函数？
8. 8 Adam优化器和SGD的区别？

## Attention 升级面

- 来源链接：https://articles.zsxq.com/id_u67us9zex93d.html
- 题目数：26

### 原始题目

1. 1 传统 Attention 存在哪些问题？
2. 2 Attention 有哪些 优化方向？
3. 3 Attention 变体有哪些？
4. 4.1 Multi-head Attention 存在什么问题？
5. 4.2 介绍一下 Multi-Query Attention？
6. 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？
7. 4.4 Multi-Query Attention 这样做的好处是什么？
8. 4.5 有 哪些模型 是 使用 Multi-Query Attention？
9. 5.1 什么是 Grouped-query Attention？
10. 5.2 有哪些大模型使用 Grouped-query Attention？
11. 6.1 为什么需要 FlashAttention？
12. 6.2 简单介绍一下 FlashAttention？
13. 6.3 简单介绍一下 FlashAttention 核心？
14. 6.4 介绍一下 FlashAttention 优点？
15. 6.5 介绍一下 FlashAttention 代表模型？
16. 8 attention计算复杂度以及如何改进？
17. 9.1 简单介绍一下 Paged Attention？
18. 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?
19. 一、为什么需要 跨注意力机制（Cross-Attention）？
20. 二、介绍一些 跨注意力机制（Cross-Attention）？
21. 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？
22. 3.2 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么不同点？
23. 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？
24. 五、Cross Attention 代码实现
25. 六、Cross Attention 应用场景
26. 七、Cross Attention 的优势和挑战？

### 可直接复用的答案

#### 示例 1. 6.2 简单介绍一下 FlashAttention？

**1. 题目描述**
详细阐述 FlashAttention 算法的核心设计思想以及具体的底层实现机制。

**2. 前置基础知识**
*   **标准 Attention 的计算复杂度**：时间复杂度与空间复杂度均为 $O(N^2)$（$N$ 为序列长度）。
*   **GPU 存储层级（Memory Hierarchy）**：GPU 拥有容量大但速度慢的 HBM（高带宽内存/主存），以及容量极小但速度极快的 SRAM（片上共享内存）。
*   **访存密集型算子（Memory-bound）**：标准 Attention 中包含大量如 Softmax、Dropout 等 element-wise 操作，导致频繁读写 HBM，计算速度被访存速度拖累。

**3. 核心解答**
*   **核心思想（IO-Awareness）**：
    FlashAttention 的根本目的不是减少计算量（FLOPs），而是**减少对 HBM 的读写次数（Memory Accesses）**。它通过将 Attention 计算的中间结果保留在 SRAM 中，避免了实例化庞大的 $N \times N$ 注意力分数矩阵（Attention Matrix），从而打破显存墙，实现加速并节省显存。
*   **具体做法**：
    1.  **分块计算（Tiling）**：将输入的 $Q, K, V$ 矩阵从 HBM 切分成小块（Blocks），分批加载到 SRAM 中进行计算。
    2.  **在线 Softmax（Online Softmax）**：这是 Tiling 能否实现的关键…

> 匹配来源：本仓库沉淀:字节跳动/LLM基础/20260312_101317_算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版.md | 匹配分数：0.71

#### 示例 2. 6.3 简单介绍一下 FlashAttention 核心？

**1. 题目描述**
详细阐述 FlashAttention 算法的核心设计思想以及具体的底层实现机制。

**2. 前置基础知识**
*   **标准 Attention 的计算复杂度**：时间复杂度与空间复杂度均为 $O(N^2)$（$N$ 为序列长度）。
*   **GPU 存储层级（Memory Hierarchy）**：GPU 拥有容量大但速度慢的 HBM（高带宽内存/主存），以及容量极小但速度极快的 SRAM（片上共享内存）。
*   **访存密集型算子（Memory-bound）**：标准 Attention 中包含大量如 Softmax、Dropout 等 element-wise 操作，导致频繁读写 HBM，计算速度被访存速度拖累。

**3. 核心解答**
*   **核心思想（IO-Awareness）**：
    FlashAttention 的根本目的不是减少计算量（FLOPs），而是**减少对 HBM 的读写次数（Memory Accesses）**。它通过将 Attention 计算的中间结果保留在 SRAM 中，避免了实例化庞大的 $N \times N$ 注意力分数矩阵（Attention Matrix），从而打破显存墙，实现加速并节省显存。
*   **具体做法**：
    1.  **分块计算（Tiling）**：将输入的 $Q, K, V$ 矩阵从 HBM 切分成小块（Blocks），分批加载到 SRAM 中进行计算。
    2.  **在线 Softmax（Online Softmax）**：这是 Tiling 能否实现的关键…

> 匹配来源：本仓库沉淀:字节跳动/LLM基础/20260312_101317_算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版.md | 匹配分数：0.68

## transformers 操作篇

- 来源链接：https://articles.zsxq.com/id_rsll7gsd8va5.html
- 题目数：2

### 原始题目

1. 1. 如何 利用 transformers 加载 Bert 模型？
2. 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？

## LLMs 损失函数篇

- 来源链接：https://articles.zsxq.com/id_q0ajjlbc8493.html
- 题目数：9

### 原始题目

1. 一、介绍一下 KL 散度？
2. 二、交叉熵损失函数写一下，物理意义是什么？
3. 三、KL 散度与交叉熵的区别？
4. 四、多任务学习各loss差异过大怎样处理？
5. 五、分类问题为什么用交叉熵损失函数不用均方误差（MSE）？
6. 六、什么是信息增益？
7. 七、多分类的分类损失函数(Softmax)？
8. 八、softmax和交叉熵损失怎么计算，二值交叉熵呢？
9. 九、如果softmax的e次方超过float的值了怎么办？

## 相似度函数篇

- 来源链接：https://articles.zsxq.com/id_wp25j5xr8ocw.html
- 题目数：3

### 原始题目

1. 一、除了cosin还有哪些算相似度的方法
2. 二、了解对比学习嘛？
3. 三、对比学习负样本是否重要？负样本构造成本过高应该怎么解决？


