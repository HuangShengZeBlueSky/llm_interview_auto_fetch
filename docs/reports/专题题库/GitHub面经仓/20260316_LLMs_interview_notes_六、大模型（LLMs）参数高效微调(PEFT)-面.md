# LLMs_interview_notes 提取：六、大模型（LLMs）参数高效微调(PEFT) 面

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
- **微调基础理论**：全量微调（Full Fine-Tuning）的痛点与参数高效微调（PEFT）的定义及核心优势。
- **主流 PEFT 算法演进与原理**：对 Adapter-tuning、Prompting（提示学习）以及 LoRA 等经典算法的底层逻辑、网络结构变化及优缺点的深刻理解。
- **LoRA 深度剖析**：作为目前工业界绝对主流的方案，需熟练掌握 LoRA 的低秩分解数学原理、前向/反向传播过程及推理无延迟的特性。
- **工程落地与调优经验**：Hugging Face `peft` 库的实际应用（如 `LoraConfig` 核心参数解析），以及结合量化（如 QLoRA）的模型加载与显存优化策略。

## 答题主线
- **背景引入**：从大模型时代“参数量爆炸”导致全量微调显存耗尽、存储成本极高的痛点切入，引出 PEFT“冻结主体，只训局部”的破局思路。
- **技术对比**：按演进路线梳理——早期 **Adapter**（结构修改，引入推理延迟） $\rightarrow$ **Prompting**（输入层修改，占用上下文且难收敛） $\rightarrow$ **LoRA**（权重旁路更新，可合并权重无延迟），突出 LoRA 的代差优势。
- **核心深挖（LoRA）**：重点阐述 $\Delta W = B \times A$ 的低秩矩阵分解思想，解释其为何能在极少参数量下保持与全量微调相近的效果。
- **工程实践**：结合实际代码（如 `LoraConfig` 中的 $r$、$\alpha$、`target_modules`）和显存优化技巧（4-bit/8-bit 量化加载、梯度检查点）展现动手能力。

## 代表题答法

### 问题1：什么是微调？为什么大模型时代必须使用 PEFT？
**答题要点：**
1. **微调定义**：在预训练模型的基础上，使用特定下游任务的数据进行继续训练，使模型适应特定场景（如对话对齐、垂直领域知识注入）。
2. **全量微调的痛点**：
   - **显存墙**：需要保存模型权重、梯度、优化器状态（如 Adam 需要额外两倍模型参数量的显存），千亿参数模型微调成本极高。
   - **存储灾难**：每个下游任务都需要保存一份完整的庞大模型权重。
   - **灾难性遗忘**：全量更新容易破坏预训练阶段学到的通用泛化能力。
3. **PEFT 的优势**：冻结预训练模型的大部分参数，仅训练极少量的额外参数（通常 $<1\%$）。大幅降低显存和算力需求，且不同任务只需切换极小的参数模块（如 LoRA 权重），部署极其灵活。

### 问题2：简述 Adapter-tuning 和 Prompting 的核心思路及其局限性。
**答题要点：**
1. **Adapter-tuning（适配器微调）**：
   - **思路**：在 Transformer 的每个 Block 中（通常在多头注意力或 FFN 之后）插入小型的瓶颈结构（Bottleneck MLP，先降维后升维）。训练时冻结原模型，只更新 Adapter。
   - **局限**：改变了模型架构，增加了额外的网络层，导致**推理延迟（Inference Latency）**增加。
2. **Prompting（提示学习，如 Prompt Tuning / Prefix Tuning）**：
   - **思路**：不修改模型结构，而是在输入层（或每一层的隐状态）拼接一段可训练的连续向量（Continuous Embeddings）作为“软提示”。
   - **局限**：占用了宝贵的上下文窗口长度（Context Length）；且由于直接优化输入层的连续向量，优化难度较大，容易陷入局部最优。

### 问题3：详细讲讲 LoRA 的原理？它为什么比 Adapter 更好？
**答题要点：**
1. **核心思想**：LoRA（Low-Rank Adaptation）基于“模型在特定任务上的权重更新具有极低的内在秩”这一假设。它冻结预训练权重矩阵 $W \in \mathbb{R}^{d \times k}$，通过旁路引入两个低秩矩阵 $A \in \mathbb{R}^{r \times k}$ 和 $B \in \mathbb{R}^{d \times r}$（其中秩 $r \ll \min(d, k)$）来模拟权重的变化量 $\Delta W$。
2. **前向传播**：$h = Wx + \Delta Wx = Wx + BAx$。初始化时，$A$ 为高斯分布，$B$ 为全零矩阵，保证初始状态下旁路输出为 0，不影响原模型。
3. **为什么更好（核心优势）**：
   - **推理零延迟**：在部署阶段，可以直接将训练好的低秩矩阵乘积加回原权重中（$W_{new} = W + BA$），彻底消除类似 Adapter 的推理延迟。
   - **效果好且参数少**：在极低的参数量下（如 0.1%），能达到媲美全量微调的效果。

### 问题4：在工程实践中，如何配置 LoraConfig？模型加载策略有哪些？
**答题要点：**
1. **LoraConfig 核心参数**：
   - `r`（秩）：决定了更新矩阵的表达能力和参数量。通常设为 8、16 或 64。任务越复杂，需要的 $r$ 越大。
   - `lora_alpha`：缩放因子。LoRA 的实际更新量会乘以 $\frac{\alpha}{r}$。通常经验值设定为 $\alpha = 2r$（如 $r=8, \alpha=16$），以保证学习率的稳定性。
   - `target_modules`：指定注入 LoRA 的层。早期多只作用于 Attention 的 $Q, V$ 矩阵，现在为了更好效果，通常作用于所有线性层（`all-linear`）。
   - `lora_dropout`：防止过拟合，通常设为 0.05 或 0.1。
2. **模型加载与显存优化策略**：
   - **半精度加载**：使用 `torch.float16` 或 `bfloat16` 加载基础模型。
   - **量化加载（QLoRA 策略）**：结合 `bitsandbytes` 库，将基础模型以 8-bit 或 4-bit（NF4 数据类型）加载并冻结，只用高精度（BF16）训练 LoRA 权重，极大降低显存门槛（如单卡 24G 即可微调 7B/13B 模型）。
   - **Gradient Checkpointing（梯度检查点）**：用计算时间换显存空间，丢弃前向传播的部分中间激活值，在反向传播时重新计算，进一步节省显存。

## 大模型（LLMs）参数高效微调(PEFT) 面

- 来源链接：https://articles.zsxq.com/id_ipkod91a939n.html
- 题目数：4

### 原始题目

1. 1. 微调方法是啥？如何微调？
2. 2. 为什么需要 PEFT？
3. 3. 介绍一下 PEFT？
4. 4. PEFT 有什么优点？

## 配器微调（Adapter-tuning）篇

- 来源链接：https://articles.zsxq.com/id_0n6pfw0wz3xb.html
- 题目数：4

### 原始题目

1. 一、为什么 需要 适配器微调（Adapter-tuning）？
2. 二、适配器微调（Adapter-tuning）思路？
3. 三、 适配器微调（Adapter-tuning）特点是什么？
4. 四、AdapterFusion 思路 是什么？

## 提示学习（Prompting）

- 来源链接：https://articles.zsxq.com/id_662wpbw47gtj.html
- 题目数：8

### 原始题目

1. 一、为什么需要 提示学习（Prompting）？
2. 二、什么是 提示学习（Prompting）？
3. 三、提示学习（Prompting） 有什么优点？
4. 四、提示学习（Prompting）有哪些方法，能不能稍微介绍一下它们间？
5. 4.1.1 为什么需要 前缀微调（Prefix-tining）？
6. 4.1.2 前缀微调（Prefix-tining）思路是什么？
7. 4.1.3 前缀微调（Prefix-tining）的优点是什么？
8. 4.1.4 前缀微调（Prefix-tining）的缺点是什么？

## LoRA 系列篇

- 来源链接：https://articles.zsxq.com/id_gjkhd8xn4pvt.html
- 题目数：13

### 原始题目

1. 1.1 什么是 LoRA？
2. 1.2 LoRA 的思路是什么？
3. 1.3 LoRA 的特点是什么？
4. 1.4 简单描述一下 LoRA?
5. 1.5 解释一下 LORA 微调的原理和计算流程？
6. 2.1.1 QLoRA 的思路是怎么样的？
7. 2.1.2 QLoRA 的特点是什么？
8. 2.1.3 QLORA相比LORA做了哪些改进?
9. .2.1 AdaLoRA 的思路是怎么样的？
10. 2.3.1 为什么需要 LongLoRA？
11. 2.3.2 LongLoRA 思路是什么？
12. 2.3.3 介绍一下 shift short attention？
13. 三、Lora的矩阵怎么初始化？为什么要初始化为全0？

### 可直接复用的答案

#### 示例 1. 三、Lora的矩阵怎么初始化？为什么要初始化为全0？

[大模型实战：使用 LoRA（低阶适应）微调 LLM](https://zhuanlan.zhihu.com/p/672999750)

> 匹配来源：DeepLearing-Interview-Awesome-2024_LLMs专题:LLMs/Reference.md | 匹配分数：0.95

## 如何使用 PEFT库 中 LoRA？

- 来源链接：https://articles.zsxq.com/id_8lx1t1t3w4qf.html
- 题目数：7

### 原始题目

1. 二、如何 配置 LoraConfig？
2. 3.1 模型加载 策略有哪些？
3. 3.2 模型显存占用的部分有哪些？
4. 3.3 模型显存占用 优化策略？
5. 3.3.1 8bit量化 优化策略？
6. 3.3.2 梯度检查 优化策略？
7. 3.4 如何 向 模型 加入PEFT策略？

## 大模型 SFT 方式对比篇

- 来源链接：https://articles.zsxq.com/id_e2piver2uzei.html
- 题目数：11

### 原始题目

1. 一、SFT 微调方案如何选择？
2. 3.1 介绍一下 Full Fine Tuning？
3. 3.2 介绍一下 Full Fine Tuning 优点？
4. 3.3 介绍一下 Full Fine Tuning 缺点？
5. 4.1 介绍一下 Parameter-Efficient Fine-Tuning？
6. 5.1 介绍一下 LoRA？
7. 5.2 介绍一下 LoRA 流程？
8. 5.3 介绍一下 LoRA 优点？
9. 5.4 介绍一下 LoRA 缺点？
10. 6.1 介绍一下 QLoRA？
11. 6.2 介绍一下 QLoRA 流程？

