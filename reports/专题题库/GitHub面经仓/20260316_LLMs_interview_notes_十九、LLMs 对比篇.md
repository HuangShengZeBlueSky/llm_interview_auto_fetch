# LLMs_interview_notes 提取：十九、LLMs 对比篇

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
- **宏观视野与模型演进**：考察对当前主流开源/闭源大模型（如 LLaMA 系列、ChatGLM 系列、Baichuan 等）发展脉络、核心差异及优缺点的整体认知。
- **训练范式理解**：深刻理解预训练（Base）与对齐微调（Chat/Instruct）在目标函数、数据构造及模型能力上的本质区别。
- **底层架构与机制细节**：精准掌握不同模型的 Attention Mask 设计、位置编码（RoPE/ALiBi）、注意力机制变体（MQA/GQA）以及高效微调（如 Prefix-tuning）的底层逻辑。
- **推理工程与源码级认知**：考察对自回归生成过程的源码级理解，特别是 KV Cache (`past_key_value`) 的运作机制及逐层前向传播过程。

## 答题主线
- **宏观对比抓核心**：回答模型对比时，按照“架构改进（如 Attention 变体） -> 上下文长度（位置编码） -> 训练数据/对齐策略”的维度展开。
- **范式差异抓目标**：Base 模型强调“涌现与知识压缩”（Next Token Prediction），Chat 模型强调“指令遵循与人类价值观对齐”（SFT + RLHF/DPO）。
- **底层机制抓效率**：无论是 GQA、MQA 还是 KV Cache，本质都是为了解决大模型推理时的显存瓶颈和计算效率问题，答题时需点出“性能与效果的 Trade-off”。

## 代表题答法

### 问题1：谈谈对当前各种大模型的见解，以及 Base 模型与 Chat 模型训练方式的区别？
- **大模型见解**：当前大模型趋于“架构同质化”（多采用 Decoder-only、RMSNorm、SwiGLU、RoPE），核心壁垒在于**高质量数据**与**工程训练稳定性**。开源生态以 LLaMA 系列为基石，国内模型（如 Baichuan、Qwen、ChatGLM）在中文词表优化、长文本支持及本土化对齐上表现更优。
- **Base 与 Chat 的区别**：
  - **Base 模型（预训练）**：使用海量无标签语料，目标是 Next Token Prediction。主要学习语言规律和世界知识，表现为“文本续写”能力。
  - **Chat 模型（对齐训练）**：在 Base 基础上经过 SFT（监督微调）和 RLHF（基于人类反馈的强化学习）。SFT 激发指令遵循能力，RLHF 解决幻觉、安全性及价值观对齐问题。训练数据格式需严格遵循特定的 Chat Template（如 `<|user|>`、`<|assistant|>`）。

### 问题2：相比较于 LLaMA，LLaMA2 有哪些改进？应该如何 Finetune？
- **核心改进**：
  - **数据与上下文**：训练数据量增加 40%（达到 2T Tokens），上下文长度从 2K 扩展到 4K。
  - **架构优化**：34B 和 70B 模型引入了 **GQA（Grouped-Query Attention）**，大幅降低推理时的 KV Cache 显存占用，提升推理速度。
  - **安全性与对齐**：引入了 Ghost Attention (GAtt) 提升多轮对话中对 System Prompt 的控制力，并进行了更严格的 RLHF 训练。
- **Finetune 建议**：
  - 必须严格复用 LLaMA2 官方的 Prompt Template（包含 `[INST]` 和 `<&lt;SYS&gt;>` 标签），否则会破坏预训练的对齐效果。
  - 针对长文本任务，可结合 RoPE 线性插值或 NTK-Aware 缩放技术进行微调。

### 问题3：ChatGLM1 和 ChatGLM2 的 Attention Mask 是怎么样的？Prefix-tuning 的 Prefix Tokens 是双向注意力吗？
- **ChatGLM 系列 Mask 对比**：
  - **ChatGLM1**：采用 Prefix-LM 架构。Mask 分为两部分：Context（输入）部分是**双向注意力**（全 1 矩阵），Generation（输出）部分是**单向因果注意力**（下三角矩阵）。同时使用了独特的 2D RoPE。
  - **ChatGLM2**：回归了标准的 Decoder-only 架构，采用**全局单向因果注意力**（Causal Mask），废弃了 2D RoPE 改用标准 RoPE，并引入了 MQA（Multi-Query Attention）以提升推理速度。
- **Prefix-tuning 的注意力机制**：
  - Prefix tokens 之间是**双向注意力**（互相可见），但对于后续的实际文本 tokens，依然遵循自回归的单向注意力（只能看到 Prefix 和当前 token 之前的 tokens）。

### 问题4：Baichuan-7B 的架构解构及训练数据构建方式？
- **架构解构**：整体借鉴 LLaMA 架构（Decoder-only）。核心特点：采用 **RoPE** 位置编码，激活函数为 **SwiGLU**，归一化采用 **RMSNorm**。针对中文优化了词表（Vocabulary Size 达 64K），提升了中文编解码效率。
- **数据构建**：
  - **收集**：抓取海量中英文网页、开源数据集、书籍、论文等。
  - **清洗与构建**：经过严格的启发式规则过滤（剔除乱码、敏感词）、局部敏感哈希（MinHash LSH）进行文档级去重、以及基于质量模型的数据打分过滤，最终构建出高质量的预训练语料。

### 问题5：GPT 源码中的 `past_key_value` 是干啥的？每一层怎么输入输出？
- **`past_key_value` 的作用**：
  - 它是 **KV Cache** 的实现。在自回归生成（One-by-One）时，当前 Token 的生成只依赖前面的 Tokens。为了避免每次生成新 Token 时重复计算历史 Token 的 Key 和 Value 矩阵，将其缓存下来（即 `past_key_value`）。新 Token 只需计算自己的 K、V，并与缓存拼接，极大降低了计算量（从 $O(N^2)$ 降为 $O(N)$）。
- **逐层输入输出（One-by-One 生成过程）**：
  1. **输入**：将上一步生成的单个 Token 转换为 Embedding，加上位置编码。
  2. **Transformer 层**：
     - 进入 Self-Attention：计算当前 Token 的 Q、K、V。将 K、V 追加到 `past_key_value` 中。用当前的 Q 与全量的 K、V 计算注意力权重，得到 Attention 输出。
     - 经过残差连接、LayerNorm 和 FFN（前馈神经网络）。
  3. **输出**：经过 $N$ 层堆叠后，最后一层的隐状态输入到 LM Head（线性层），通过 Softmax 得到词表大小的概率分布，采样或 Argmax 得到下一个 Token。

## LLMs 对比篇

- 来源链接：https://articles.zsxq.com/id_fsq8czgwjxse.html
- 题目数：30

### 原始题目

1. 一、谈谈你对当前出现的各种大模型的见解？
2. 二、目前大模型常见的 base 模型训练和 chat 模型训练 方式 的区别么？
3. 3.1.1.1 llama 训练数据 介绍
4. 3.1.1.2 llama 模型参数量 介绍
5. 3.1.1.3 llama 模型结构 介绍
6. 3.1.1.4 llama 训练目标 介绍
7. 3.1.1.5 llama tokenizer 介绍
8. 3.1.1.6 llama 衍生模型 介绍
9. 3.2.1 llama2 系列 数据预处理方式？
10. 3.2.2 llama2 系列 Tokenizer 处理方式？
11. 3.2.3 llama2 系列 Architectural？
12. 3.2.4 llama2 系列 content长度？
13. 3.2.1 Mistral 7B Architectural？
14. 3.3.1 Qwen 系列 数据预处理方式？
15. 3.3.2 Qwen 系列 Tokenizer 处理方式？
16. 3.3.3 Qwen 系列 ARCHITECTURE？
17. 3.4.1.1 Baichuan2 系列 数据预处理方式？
18. 3.4.1.2 Baichuan2 系列 Tokenizer 处理方式？
19. 3.4.1.2 Baichuan2 系列 Architecture ？
20. 3.5.1.1 ChatGLM-6B 结构特点？
21. 3.5.1.2 ChatGLM-6B 训练目标？
22. 3.5.1.3 ChatGLM-6B tokenizer？
23. 3.6.1.1 BLOOM 训练数据构建？
24. 3.6.1.2 BLOOM 模型参数量？
25. 3.6.1.3 BLOOM 模型结构？
26. 3.6.1.4 BLOOM 训练目标？
27. 3.6.1.5 BLOOM tokenizer?
28. 四、分析与总结？
29. 4.1 大模型训练共同点？
30. 4.2 大模型训练不同点？

## LLMs 对比篇

- 来源链接：https://articles.zsxq.com/id_0j7k3gxa5hpm.html
- 题目数：3

### 原始题目

1. 1、prefix-tuning的prefix tokens是双向注意力吗？
2. 2、chatglm1和chatglm2的attention mask是怎么样的？
3. 3、llama的attention mask是怎么样的？

## 百川智能baichuan7B、13B、53B、baichuan2 总结篇

- 来源链接：https://articles.zsxq.com/id_ma6pw7v2g9pi.html
- 题目数：3

### 原始题目

1. 1. 你了解baichuan-7B解构么？介绍一下？
2. 2. baichuan-7B 如何 收集原始数据并 构建 训练数据？
3. 3. baichuan-7B 如何 提高 训练稳定性和吞吐？

## LLaMa 篇

- 来源链接：https://articles.zsxq.com/id_9ba6a72wan2w.html
- 题目数：1

### 原始题目

1. 一、相比较于llama而言，llama2有哪些改进，对于llama2是应该如何finetune？

## GPT 经验篇

- 来源链接：https://articles.zsxq.com/id_r46k6bqu34xh.html
- 题目数：7

### 原始题目

1. 一、gpt源码past\_key\_value是干啥的？
2. 二、gpt onebyone 每一层怎么输入输出？
3. 三、bert和gpt有什么区别
4. 四、文本生成的几大预训练任务？
5. 五、讲讲T5和Bart的区别，讲讲bart的DAE任务？
6. 六、讲讲Bart和Bert的区别？
7. 七、gpt3和gpt2的区别？


