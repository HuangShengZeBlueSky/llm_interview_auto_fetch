# LLMs_interview_notes 提取：十七、大模型（LLMs）加速篇

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
大模型推理加速是当前 AI 工程化落地的核心痛点，面试官主要通过该专题考察候选人的**底层工程视野与性能调优能力**。核心考点包括：
- **底层机制理解**：是否清晰掌握 LLM 自回归生成的两个阶段（Prefill 与 Decode）及其不同的硬件瓶颈（算力墙 vs 内存墙）。
- **性能评估体系**：能否准确使用 TTFT、TPOT、吞吐量等指标来衡量推理性能，而不是笼统地说“快慢”。
- **前沿优化技术**：对当前主流的加速手段（如 KV Cache 优化、投机解码、量化、FlashAttention）的原理是否有深度认知。
- **主流框架底层原理**：特别是对 vLLM（PagedAttention）和 TensorRT-LLM 等工业界标配框架的核心创新点是否吃透。

## 答题主线
面对推理加速问题，建议采用 **“剖析瓶颈 -> 确立指标 -> 分层优化”** 的逻辑主线：
1. **剖析瓶颈**：明确指出大模型推理慢的根源在于**自回归（Auto-regressive）机制**和 **KV Cache 带来的显存带宽瓶颈（Memory-bound）**。
2. **确立指标**：区分面向用户的**延迟（Latency/TTFT）**和面向系统的**吞吐量（Throughput）**。
3. **分层优化**：
   - **模型/算法层**：MQA/GQA（减少 KV Cache 体积）、量化（降低显存占用与带宽）、投机解码（打破串行生成瓶颈）、FlashAttention（减少访存开销）。
   - **系统/框架层**：Continuous Batching（提升 GPU 利用率）、PagedAttention（解决显存碎片化）、张量并行（TP）。

## 代表题答法

### 问题1：介绍一下 LLMs 的推理过程？为什么需要对大模型推理进行加速？
**答题要点：**
大模型的推理过程（文本生成）主要分为两个截然不同的阶段：
1. **Prefill（预填充阶段）**：
   - **过程**：一次性处理用户的输入 Prompt，计算并保存所有输入 token 的 KV Cache，同时生成第一个输出 token。
   - **特征**：矩阵乘法密集型，属于**算力瓶颈（Compute-bound）**。
2. **Decoding（解码阶段）**：
   - **过程**：自回归生成，每次利用历史的 KV Cache 和当前生成的 token，预测下一个 token，并更新 KV Cache。
   - **特征**：逐个 token 串行生成，频繁读取庞大的 KV Cache，属于**显存带宽瓶颈（Memory-bound）**，即“内存墙”。

**需要加速的原因**：
- **自回归的串行本质**导致无法像训练时那样高度并行。
- **KV Cache 显存占用随序列长度线性增长**，极易耗尽显存，且传统的连续显存分配会导致严重的显存碎片化，限制了 Batch Size 的提升，进而拉低了整体吞吐量。

### 问题2：如何准确衡量大模型的推理速度？
**答题要点：**
衡量推理速度不能单一而论，需区分场景（ToC 体验 vs ToB 成本），核心指标包括：
1. **TTFT (Time To First Token，首字延迟)**：从发送请求到收到第一个 token 的时间。直接决定用户的等待体验，主要受 Prefill 阶段速度影响。
2. **TPOT (Time Per Output Token，每个输出 token 的延迟)**：生成每个 token 的平均时间。决定了模型“说话”的语速，主要受 Decoding 阶段速度影响。
3. **Throughput (吞吐量)**：系统在单位时间内处理的 Token 数量（Tokens/s）或请求数量（Requests/s）。决定了服务器的并发能力和算力成本。
4. **Latency (端到端延迟)**：完成整个请求的总时间（TTFT + TPOT * 生成长度）。

### 问题3：当前优化模型最主要的技术手段有哪些？主流推理框架有什么特点？
**答题要点：**
**主要技术手段（分层回答）：**
1. **显存与带宽优化**：KV Cache 优化（MQA/GQA 减少头数）、模型量化（PTQ/QAT，如 INT8/INT4、AWQ、GPTQ，大幅降低显存读取压力）。
2. **计算与 IO 优化**：FlashAttention（通过 Tiling 机制在 SRAM 内完成注意力计算，避免 HBM 频繁读写）。
3. **打破串行瓶颈**：投机解码（Speculative Decoding，小模型快速草拟多个 token，大模型一次性并行验证，用算力换时间）。
4. **调度优化**：Continuous Batching（细粒度到 iteration 级别的动态批处理，解决请求长短不一导致的算力闲置）。

**主流推理框架及特点：**
- **vLLM**：主打高吞吐量，核心是 PagedAttention，适合高并发的线上服务。
- **TensorRT-LLM**：NVIDIA 官方出品，极致的 C++ 和 CUDA 算子优化，在 N 卡上单机/单卡性能天花板，但闭源且编译复杂。
- **TGI (Text Generation Inference)**：HuggingFace 出品，生态兼容性极好，开箱即用，支持多种开源模型。
- **LMDeploy**：商汤开源，TurboMind 引擎对量化（W4A16）和多模态支持极佳，性能比肩甚至部分超越 vLLM。

### 问题4：为什么需要 vLLM？它解决了什么问题，又是如何优化的？
**答题要点：**
**存在的问题（为什么需要 vLLM）：**
在传统推理中，KV Cache 是在连续的显存空间中分配的。由于大模型生成的序列长度是不可预测的，系统往往会预先分配最大可能的显存（如 2048 tokens）。这导致了极大的**内部碎片**（预分配但未使用的空间）和**外部碎片**。据统计，传统方式下 KV Cache 的显存浪费率高达 60%-80%，严重限制了 Batch Size。

**vLLM 的优化机制（PagedAttention）：**
vLLM 借鉴了操作系统的**虚拟内存分页管理**思想，提出了 PagedAttention：
1. **显存分页**：将 KV Cache 划分为固定大小的块（Blocks/Pages），每个块包含固定数量 token 的 KV 值。
2. **非连续存储**：逻辑上连续的 KV Cache，在物理显存上可以是非连续的块。
3. **动态分配**：生成新 token 时，按需动态分配物理块，通过一张“页表（Block Table）”记录逻辑块到物理块的映射。

**优化效果**：
- 显存浪费降至 4% 以下（仅剩最后一个块的微小内部碎片）。
- 释放的显存允许系统接入更大的 Batch Size，使**吞吐量提升 2-4 倍**。
- 天然支持内存共享（如 Beam Search 或并行采样时，多个序列可以共享同一个 Prompt 的物理块，进一步节省显存）。

## 大模型(LLM)部署框架对比篇

- 来源链接：https://articles.zsxq.com/id_7d31dgh26fcp.html
- 题目数：1

### 原始题目

1. 一、为什么需要对大模型推理加速？

## 大模型（LLMs）推理加速篇

- 来源链接：https://articles.zsxq.com/id_kgzsxgro8cee.html
- 题目数：4

### 原始题目

1. 一、 推理过程 分哪些阶段？
2. 1.2 Decoding（递归推理与解码输出）阶段
3. 二、 推理性能的评价指标？
4. 三、 当前优化模型最主要技术手段有哪些？

## 大模型（LLMs）加速篇

- 来源链接：https://articles.zsxq.com/id_w9wewc152eux.html
- 题目数：3

### 原始题目

1. 1 当前优化模型最主要技术手段有哪些？
2. 2 推理加速框架有哪一些？都有什么特点？
3. 3.1 vLLM 的 功能有哪些？

## LLMs 推理性能面

- 来源链接：https://articles.zsxq.com/id_jwd03u0l7feo.html
- 题目数：3

### 原始题目

1. 一、介绍一下 LLMs 的文本生成过程？
2. 二、如何准确衡量模型的推理速度呢？
3. 三、如果对整体推理时延有具体目标，有哪些有效的启发式方法来评估模型？

## LLM（大语言模型）部署加速方法——PagedAttention篇

- 来源链接：https://articles.zsxq.com/id_p22mjq881n3n.html
- 题目数：3

### 原始题目

1. 一、vLLM 用于大模型并行推理加速 存在什么问题？
2. 二、vLLM 如何 优化 大模型并行推理加速？
3. 三、什么是 PagedAttention？

## 大模型推理加速工具 —— vLLM

- 来源链接：https://articles.zsxq.com/id_zw5h9ogvac2w.html
- 题目数：4

### 原始题目

1. 1.2 为什么 需要 vLLM ?
2. 1.3 vLLM 具有哪些特点 ?
3. 1.4 vLLM 支持哪些 Huggingface 模型 ?
4. 二、vLLM 性能如何？

## LLM（大语言模型）部署加速方法——Faster Transformer篇

- 来源链接：https://articles.zsxq.com/id_dd2gowztxtfg.html
- 题目数：3

### 原始题目

1. 一、为什么需要 FasterTransformer？
2. 二、FasterTransformer 介绍一下？
3. 三、FasterTransformer 核心是什么？

## 纯Python超轻量高性能LLM推理框架 —— LightLLM

- 来源链接：https://articles.zsxq.com/id_9a643feq2b0b.html
- 题目数：7

### 原始题目

1. 1.2 为什么 需要 LightLLM ?
2. 1.3 目前 LLM推理框架 有 哪些?
3. 二、LightLLM 介绍一下？
4. 2.1 什么是 LightLLM ？
5. 2.2 Token Attention 介绍？
6. 2.3 Efficient Router 介绍？
7. 三、LightLLM 性能表现 介绍？

## LLM推理技术之StreamingLLM：如何拥有无限长生成能力

- 来源链接：https://articles.zsxq.com/id_w1gwi9z7qm5s.html
- 题目数：4

### 原始题目

1. 1.1 大型语言模型（LLM）存在什么问题？
2. 1.2 StreamingLLM 背景介绍
3. 1.3 StreamingLLM 核心问题？
4. 二、StreamingLLM 的思路是什么？

## SwiftInfer —— 大模型无限流式输入推理飙升46%，打破多轮对话长度限制

- 来源链接：https://articles.zsxq.com/id_0rpua5fejfwc.html
- 题目数：4

### 原始题目

1. 一、为什么需要 StreamingLLM？
2. 二、StreamingLLM 思路是什么？
3. 三、StreamingLLM 优点是什么？
4. SwiftInfer 篇：基于TensorRT的StreamingLLM实现
