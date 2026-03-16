# LLMs_interview_notes 提取：十三、大模型（LLMs）分布式训练面

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
本专题主要考察候选人对**大模型分布式训练底层原理与工程实现**的掌握程度。面试官旨在评估你是否真正理解“为什么需要分布式”以及“如何高效进行分布式训练”。核心考点包括：
- **痛点认知**：大模型训练面临的显存墙、计算墙与通信墙。
- **数据并行演进**：从 `DP` 到 `DDP` 的演进逻辑，以及核心通信算法（Ring-AllReduce）的数学与工程意义。
- **模型并行基础**：流水线并行（Pipeline Parallelism）的切分逻辑与“气泡（Bubble）”优化。
- **PyTorch 底层机制**：GPU 异步操作、点对点通信与多进程（Multiprocessing）在分布式中的应用。

## 答题主线
回答分布式训练问题，建议遵循**“遇到什么瓶颈 -> 引入什么并行策略 -> 策略带来什么新问题 -> 如何在工程上优化”**的逻辑主线：
1. **起因（瓶颈）**：单卡装不下模型（显存瓶颈）或训练太慢（计算瓶颈），引出分布式需求。
2. **数据并行（DP -> DDP）**：模型装得下但数据多时，讲清 DP 的单节点/多线程/单卡通信瓶颈，顺势引出 DDP 的多进程/Ring-AllReduce 优势。
3. **模型并行（Pipeline）**：模型单卡装不下时，按层切分引出流水线并行，重点阐述如何通过微批次（Micro-batch）调度来压缩“气泡”。
4. **工程落地（PyTorch机制）**：结合 PyTorch 的异步执行特性和 `torch.multiprocessing`，说明分布式脚本是如何被拉起和执行的。

## 代表题答法

### 问题1：训练大语言模型存在哪些核心问题？
**答题要点：**
主要面临三大“墙”：
1. **显存墙（Memory Wall）**：单卡显存（如 80G）无法容纳 LLM 训练所需的全部显存。显存占用不仅包含模型参数，还包括梯度、优化器状态（如 Adam 的动量和方差，占大头）以及前向激活值（Activations）。
2. **计算墙（Compute Wall）**：LLM 训练需要的 FLOPs 极高，单卡算力需要数年才能训完，必须进行大规模集群扩展。
3. **通信墙（Communication Wall）**：多卡/多机协同训练时，节点间的网络带宽（如 NVLink, InfiniBand）往往成为木桶效应的最短板，导致 GPU 算力闲置等待数据传输。

### 问题2：为什么弃用 nn.DataParallel (DP) 而全面转向 nn.parallel.DistributedDataParallel (DDP)？其核心 Ring-AllReduce 是什么？
**答题要点：**
- **DP 的致命缺陷**：基于单进程多线程，受 Python GIL（全局解释器锁）限制；采用 Parameter Server 架构，所有 GPU 计算完梯度后都要发给 GPU 0 聚合，**GPU 0 成为通信和显存的绝对瓶颈**；且仅支持单机。
- **DDP 的优势**：采用多进程（Multiprocessing），绕过 GIL；支持多机多卡；最重要的是采用了 **Ring-AllReduce** 通信架构，消除了中心节点瓶颈。
- **Ring-AllReduce 核心机制**：将所有 GPU 连成一个环。分为两步：
  1. **Scatter-Reduce**：每个 GPU 只负责一部分梯度的规约，环形传递后，每个 GPU 得到一部分完整的梯度。
  2. **All-Gather**：将各自完整的局部梯度再次环形广播，最终所有 GPU 拥有完全一致的完整梯度。
  *亮点：* 这种方式让通信时间与 GPU 数量基本无关，极大提升了扩展性。

### 问题3：为什么需要流水线并行（Pipeline Parallelism）？其优化目标是什么？
**答题要点：**
- **为什么需要**：当模型参数量过大，单卡连模型的一份副本都装不下时（例如千亿参数模型），数据并行失效。此时需要将模型的不同层（Layers）切分到不同的 GPU 上，即流水线并行。
- **存在的问题**：朴素的层间切分会导致严重的**“气泡（Bubble）”**问题，即 GPU A 在计算时，GPU B 只能空等 A 的输出，导致设备利用率极低。
- **优化目标**：**最小化气泡时间，最大化系统吞吐量**。通常通过将一个 Batch 划分为多个 Micro-batch，并采用特定的调度策略（如 GPipe 的前向-后向交替，或 1F1B 策略）让不同 GPU 尽可能同时处于工作状态。

### 问题4：PyTorch 分布式训练的底层通信（点对点）与多进程机制是怎样的？
**答题要点：**
- **GPU 默认操作**：PyTorch 中的 GPU 计算默认是**异步（Asynchronous）**的。CPU 下发指令到 GPU Stream 后立即返回，这为计算与通信的重叠（Overlap）提供了基础。
- **点对点通信（P2P）**：指明确的发送端和接收端（如 `send` 和 `recv`）。在流水线并行中，相邻层所在的 GPU 之间传递激活值和梯度，使用的就是 P2P 通信。
- **多进程（torch.multiprocessing）**：为了配合 DDP 绕过 GIL，PyTorch 使用 `multiprocessing.spawn` 或 `torchrun` 启动多个独立的 Python 进程。每个进程绑定一张 GPU（Local Rank），独立执行前向和反向传播，仅在梯度同步时通过底层通信后端（如 NCCL）进行交互。

## 大模型（LLMs）分布式训练面

- 来源链接：https://articles.zsxq.com/id_ah2ibj3z22c7.html
- 题目数：25

### 原始题目

1. 1.1 训练 大语言模型 存在问题？
2. 1.2 什么是 点对点通信？
3. 1.3 什么是 集体通信？
4. 1.4 什么是 数据并行？
5. 1.5 数据并行 如何 提升效率？
6. 1.6 什么是 流水线并行？
7. 1.7 什么是 张量并行 (intra-layer)？
8. 1.8 数据并行 vs 张量并行 vs 流水线并行?
9. 1.9 什么是 3D并行？
10. 1.10 想要训练1个LLM，如果只想用1张显卡，那么对显卡的要求是什么？
11. 1.11 如果有N张显存足够大的显卡，怎么加速训练？
12. 1.12 如果显卡的显存不够装下一个完整的模型呢？
13. 1.13 PP推理时，是一个串行的过程，1个GPU计算，其他空闲，有没有其他方式？
14. 1.14 3种并行方式可以叠加吗？
15. 1.15 Colossal-AI 有1D/2D/2.5D/3D，是什么情况？
16. 1.16 除了3D并行有没有其他方式大规模训练？
17. 1.17 有了ZeRO系列，为什么还需要3D并行？
18. 1.18 平民适不适合玩3D并行？
19. 1.19 平民适不适合直接上多机多卡的ZeRO3（万兆网）？
20. 1.20 分布式并行及显存优化技术并行技术有哪一些，都有什么特点？
21. 1.21 显存优化技术有哪一些，都有什么特点？
22. 1.22 常见的分布式训练框架哪一些，都有什么特点？
23. 2.1 假如有超多的8卡A100节点（DGX A100），如何应用3D并行策略？
24. 2.2 如果想构这样一个大规模并行训练系统，训练框架如何选？
25. 2.3 训练框架如何选？

## 图解分布式训练（一） —— 流水线并行（Pipeline Parallelism）面

- 来源链接：https://articles.zsxq.com/id_wre1eni0oq7d.html
- 题目数：2

### 原始题目

1. 为什么需要流水线并行（Pipeline Parallelism）？
2. 一、流水线并行（Pipeline Parallelism） 优化目标是什么？

## 图解分布式训练（二） —— nn.DataParallel面

- 来源链接：https://articles.zsxq.com/id_9dfwi0ooio2z.html
- 题目数：4

### 原始题目

1. 为什么需要nn.DataParallel？
2. 一、pytorch中的GPU操作默认是什么样？
3. 二、介绍一下 nn.DataParallel 函数？
4. 三、nn.DataParallel 函数 处理逻辑 介绍一下？

## 图解分布式训练（三） —— nn.parallel.DistributedDataParallel

- 来源链接：https://articles.zsxq.com/id_i4s3ia057rmh.html
- 题目数：4

### 原始题目

1. 为什么需要 nn.parallel.DistributedDataParallel ？
2. 一、什么是 DistributedDataParallel 核心 —— Ring-AllReduce？
3. 二、nn.parallel.DistributedDataParallel 函数 介绍一下？
4. 三、nn.parallel.DistributedDataParallel 函数 如何多卡加速训练？

## 图解分布式训练（四） —— torch.multiprocessing 详细解析

- 来源链接：https://articles.zsxq.com/id_gu9smpbn510e.html
- 题目数：2

### 原始题目

1. 一、torch.multiprocessing 函数介绍一下？
2. 二、torch.multiprocessing 函数如何使用？

## 图解分布式训练（五） —— AMP混合精度训练 详细解析

- 来源链接：https://articles.zsxq.com/id_0slrgoti6gvb.html
- 题目数：4

### 原始题目

1. 为什么需要 AMP混合精度训练？
2. 一、什么是自动混合精度训练(AMP)
3. 二、为什么需要自动混合精度？
4. 三、混合精度训练的优点是什么？

## 图解分布式训练（六） —— Pytorch的 DeepSpeed 详细解析

- 来源链接：https://articles.zsxq.com/id_kmq9rn2vo4kz.html
- 题目数：5

### 原始题目

1. 一、为什么需要 Deepspeed？
2. 二、DeepSpeed 基本概念 介绍一下？
3. 2.1 DeepSpeed 介绍
4. 三、DeepSpeed 通信策略 介绍一下？
5. 四、DeepSpeed 如何使用？

## 图解分布式训练（七）—— accelerate 分布式训练 详细解析

- 来源链接：https://articles.zsxq.com/id_o5wkeionnqr7.html
- 题目数：2

### 原始题目

1. 一、为什么需要 accelerate 分布式训练？
2. 二、什么是 accelerate 分布式训练?

## 图解分布式训练（八）—— ZeRO 学习

- 来源链接：https://articles.zsxq.com/id_grv7uddls2g1.html
- 题目数：3

### 原始题目

1. 一、什么是 3D 并行？
2. 二、3D 并行 策略有哪些？
3. 三、为什么需要 ZeRO？

## 大模型分布式训练故障恢复篇

- 来源链接：https://articles.zsxq.com/id_zspm2q33tckx.html
- 题目数：3

### 原始题目

1. 一、为什么 大模型分布式训练 需要 故障恢复？
2. 二、如何获取最优的ckpt存储间隔？
3. 三、ckpt存储能否实现异步或者部分掩盖？

## 图解分布式训练（九）—— Megatron-LM 篇

- 来源链接：https://articles.zsxq.com/id_o4qtcspmuwqv.html
- 题目数：3

### 原始题目

1. 1、Activation Recomputation是怎么实现的?
2. 2、Megatron中的OverlappedDistributed Optimizer 是如何实现的?
3. 3.1 介绍一下 Megatron-LM 中 Context Parallel 实现原理？

## pytorch 分布式计算 坑/bug 梳理篇

- 来源链接：https://articles.zsxq.com/id_onztfzwdckom.html
- 题目数：1

### 原始题目

1. 二、如果是用pytorch实现同步梯度更新，自研 数据接口，出现 第一个epoch结尾处程序卡死问题
