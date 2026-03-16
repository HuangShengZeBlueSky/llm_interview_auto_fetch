# LLMs_interview_notes 提取：二十九、大模型推理加速——KV Cache篇

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
- **核心机制与原理**：自回归生成中 Attention 计算的重复性问题，以及 KV Cache 如何通过“空间换时间”解决该问题。
- **性能瓶颈分析**：理解 LLM 推理中的 Prefill（预填充，Compute-bound）与 Decode（解码，Memory-bound）阶段的差异。
- **显存占用估算**：能够准确推导并计算特定模型参数下，KV Cache 所需的物理显存大小。
- **前沿优化技术**：掌握工业界针对 KV Cache 显存占用大、碎片化严重等痛点的解决方案（如 MQA/GQA、PagedAttention、KV 量化等）。

## 答题主线
- **痛点引入**：说明大模型自回归（Auto-Regressive）生成的特点，每次生成新 Token 都会重复计算历史 Token 的特征，导致算力浪费。
- **原理解析**：点出“空间换时间”的核心思想，详细说明在 Attention 计算中，只需保留历史的 Key 和 Value 矩阵，当前 Query 只需与缓存的 KV 进行计算。
- **代价与瓶颈**：指出 KV Cache 虽然降低了计算量，但极大地增加了显存占用和访存带宽压力，使推理进入 Memory-bound 状态。
- **进阶优化（加分项）**：从“减少显存占用”（模型架构层面的 MQA/GQA、算法层面的 KV 量化/Token 淘汰）和“提高显存利用率”（系统层面的 PagedAttention）两个维度展开。

## 代表题答法

### 问题1：介绍一下 KV Cache 是啥？
**答题要点：**
1. **定义与目的**：KV Cache 是大模型推理中一种“空间换时间”的加速技术。在自回归生成中，为了避免每次生成新 Token 时重新计算前面所有历史 Token 的注意力，将历史 Token 的 Key (K) 和 Value (V) 向量缓存到显存中。
2. **工作流程（分阶段）**：
   - **Prefill（预填充）阶段**：处理用户输入的 Prompt。此时没有缓存，模型并行计算所有输入 Token 的 K 和 V，并将它们存入 KV Cache。此阶段是计算密集型（Compute-bound）。
   - **Decode（解码）阶段**：逐个生成新 Token。模型只需计算当前最新 Token 的 Q、K、V，然后将新的 KV 追加到缓存中，当前 Q 与完整的 KV Cache 进行 Attention 计算。此阶段是访存密集型（Memory-bound）。
3. **显存占用估算（核心考点）**：
   - 每个 Token 的 KV Cache 大小 = `2 (K和V) × 层数 (num_layers) × 隐藏层维度 (hidden_size) × 数据类型字节数 (如FP16为2)`。
   - *举例*：LLaMA-7B（32层，维度4096，FP16），单 Token 的 KV Cache 占用约为 `2 * 32 * 4096 * 2 = 512 KB`。如果上下文长度为 2048，单条请求的 KV Cache 就高达 1GB。

### 问题2：针对 KV Cache 的推理加速与优化技术有哪些？
**答题要点：**
面试官主要考察你对工业界落地方案的了解深度，可从以下三个维度展开：
1. **模型架构层（减少 KV 数量）**：
   - **MQA (Multi-Query Attention)**：所有 Attention Head 共享同一组 K 和 V。极大地减少了 KV Cache 体积，但可能带来轻微性能下降。
   - **GQA (Grouped-Query Attention)**：MQA 和 MHA 的折中方案（如 LLaMA-2/3 采用）。将 Head 分组，组内共享 KV，既大幅降低了 KV Cache 占用，又保持了接近 MHA 的模型能力。
2. **系统显存管理层（提高利用率）**：
   - **PagedAttention (vLLM 核心技术)**：传统 KV Cache 采用连续显存分配，导致严重的内部和外部显存碎片（利用率常低于 50%）。PagedAttention 借鉴操作系统的虚拟内存分页机制，将 KV Cache 划分为固定大小的 Block（如 16 个 Token），实现非连续存储，将显存利用率提升至 90% 以上，并支持高效的 Beam Search 显存共享。
3. **算法与压缩层（降低精度或长度）**：
   - **KV Cache 量化**：将 FP16 的 KV Cache 压缩为 INT8、INT4 甚至 FP8 格式（如 KVCache-FP8），直接将显存占用减半或更多，同时缓解访存带宽瓶颈。
   - **Token 淘汰/滑动窗口**：如 Window Attention（只缓存最近 N 个 Token 的 KV）或 H2O (Heavy Hitter Oracle，只保留注意力分数最高的关键 Token 和局部 Token 的 KV)，适用于超长上下文推理。

## 大模型推理加速——KV Cache篇

- 来源链接：https://articles.zsxq.com/id_swmfcls3sp1j.html
- 题目数：6

### 原始题目

1. 大模型推理加速——KV Cache篇
2. 一、介绍一下 KV Cache是啥？
3. 二、为什么要进行 KV Cache？
4. 2.1 不使用 KV Cache 场景
5. 2.2 使用 KV Cache 场景
6. 三、说一下 KV Cache 在 大模型中的应用？


