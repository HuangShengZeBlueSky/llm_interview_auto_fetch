# LLMs_interview_notes 提取：二十三、大模型蒸馏篇

> 来源仓库：https://github.com/km1994/LLMs_interview_notes
> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。

## 专题总结

## 本专题考什么
- **大模型轻量化与加速技术**：虽然专题名为“蒸馏”，但实际考点涵盖了模型压缩的两大核心支柱——知识蒸馏（Knowledge Distillation）与底层精度/量化技术（Mixed Precision & Quantization）。
- **知识蒸馏的原理与进阶**：Teacher-Student 架构的底层逻辑、Soft Label 的作用、特征层蒸馏，以及如何利用无监督数据进行伪标签蒸馏。
- **底层数据格式与混合精度**：FP32 与 FP16/BF16 的本质区别，混合精度训练（Mixed Precision Training）中“权重备份”与“Loss Scaling”的工程实现。
- **极致量化工具的工程实践**：`bitsandbytes` 库的底层作用（8-bit/4-bit 量化、NF4 数据格式），以及其在 QLoRA 等高效微调中的具体使用方法。

## 答题主线
- **蒸馏线（算法层面）**：从“Teacher指导Student”的基础概念出发，讲清 Logits 蒸馏（温度系数 $T$）与特征蒸馏的区别。结合业务场景，说明如何用 Teacher 模型在无监督数据上打伪标签来扩充训练集。最后补充大模型时代的特有改进（如指令蒸馏、CoT 逻辑链蒸馏）。
- **精度线（底层硬件层面）**：从显存和算力瓶颈切入，对比 FP32（单精度）与 FP16（半精度）在指数位和尾数位上的差异，指出 FP16 易发生“数值溢出”的问题。顺势引出混合精度训练的解决方案（FP16 前向/反向 + FP32 主权重更新 + Loss Scaling）。
- **量化线（工程落地层面）**：以大模型微调的显存痛点为背景，介绍 `bitsandbytes` 作为量化核心依赖库的作用。强调其提供的 LLM.int8() 和 4-bit (NF4) 量化技术，并简述在 Hugging Face 生态中的调用配置。

## 代表题答法

### 问题1：知识蒸馏的基本原理与无监督样本训练？
**答题要点：**
1. **基本原理**：知识蒸馏通过引入一个能力强的大模型（Teacher）来指导一个小模型（Student）训练。核心是利用 Teacher 输出的 Soft Labels（带有温度系数 $T$ 的概率分布），让 Student 学习到类别间的“暗知识”（Dark Knowledge），而不仅仅是 Hard Label 的非黑即白。
2. **无监督样本训练**：在缺乏标注数据时，可以将海量无监督数据输入给 Teacher 模型，生成预测结果（Soft Labels 或直接采样生成 Hard Labels/文本）。然后将这些带有“伪标签”的数据作为训练集去训练 Student 模型。这种方法极大缓解了数据稀缺问题，是大模型时代“弱模型超越强模型”或“小模型逼近大模型”的常用手段。

### 问题2：对知识蒸馏知道多少，有哪些改进用到了？
**答题要点：**
1. **基础蒸馏**：Hinton 提出的基于 Logits 的蒸馏（KL 散度损失）。
2. **特征层蒸馏**：不仅对齐最终输出，还对齐中间隐藏层（Hidden States）或注意力矩阵（Attention Maps），如 TinyBERT 的做法。
3. **大模型时代的改进（重点）**：
   - **指令/生成式蒸馏**：Teacher（如 GPT-4）生成高质量的指令遵循数据，Student（如 LLaMA）进行 SFT（如 Alpaca 模型）。
   - **Step-by-Step / CoT 蒸馏**：不仅蒸馏最终答案，还要求 Student 学习 Teacher 的思维链（Chain of Thought）推理过程。
   - **MiniLLM / 逆向 KL 散度**：针对生成任务，传统 KL 散度容易导致小模型生成长文本时出现模式崩溃，改用 Reverse KL 或其他强化学习手段进行序列级蒸馏。

### 问题3：FP32和FP16的区别，混合精度的原理？（含半精度是什么）
**答题要点：**
1. **FP32 与 FP16 的区别**：
   - **FP32（单精度）**：32位（1位符号，8位指数，23位尾数），动态范围大，精度高，但占用显存大，计算慢。
   - **FP16（半精度）**：16位（1位符号，5位指数，10位尾数），显存占用减半，计算速度翻倍（Tensor Core 加成），但动态范围小，极易发生**下溢出（Underflow）**或**上溢出（Overflow）**。
2. **混合精度训练（AMP）原理**：
   - **FP16 计算**：前向传播（Forward）和反向传播（Backward）使用 FP16 进行，加速计算并节省显存。
   - **FP32 主权重备份**：在优化器中保留一份 FP32 的模型权重（Master Weights）。每次更新时，将 FP16 的梯度应用到 FP32 的权重上，防止微小梯度在 FP16 下变成 0。
   - **Loss Scaling（损失缩放）**：为了防止反向传播时梯度过小导致 FP16 下溢出，在计算 Loss 后将其乘以一个大常数（Scale），放大梯度；在更新权重前，再将梯度除以该常数缩放回来。

### 问题4：什么是 bitsandbytes，如何才能使用它？
**答题要点：**
1. **什么是 bitsandbytes**：
   - 它是一个轻量级的 Python 封装库，专门针对 CUDA 自定义函数进行了优化。
   - **核心功能**：在大模型领域，它主要提供**8-bit 优化器**和**极低精度量化**（如 LLM.int8() 和 4-bit NormalFloat/NF4 量化）。它是实现 QLoRA（Quantized LoRA）等单卡微调大模型技术的底层核心依赖。
2. **如何使用**：
   - **安装**：通过 `pip install bitsandbytes` 安装。
   - **代码调用**：在 Hugging Face `transformers` 库中已深度集成。加载模型时，通过传入 `BitsAndBytesConfig` 对象，设置 `load_in_8bit=True` 或 `load_in_4bit=True` 即可实现模型的量化加载。
   - **配合 PEFT**：通常加载完量化底座模型后，会配合 `peft` 库插入 LoRA 适配器，从而在极低显存下完成大模型的微调。

## 大模型蒸馏篇

- 来源链接：https://articles.zsxq.com/id_jkiw9vhzopgv.html
- 题目数：3

### 原始题目

1. 一、知识蒸馏和无监督样本训练？
2. 二、对知识蒸馏知道多少，有哪些改进用到了？
3. 三、谈一下对模型量化的了解？

## LLMs 浮点数篇

- 来源链接：https://articles.zsxq.com/id_vu744g6jklli.html
- 题目数：3

### 原始题目

1. 一、fp32和fp16的区别，混合精度的原理
2. 二、半精度是什么？
3. 三、半精度的理论原理是什么？

## 自定义 CUDA 函数的轻量级包装器 —— bitsandbytes篇

- 来源链接：https://articles.zsxq.com/id_2nwi4napgvlh.html
- 题目数：3

### 原始题目

1. 一、什么是 bitsandbytes?
2. 二、如何才能使用 bitsandbytes？
3. 三、如何使用 bitsandbytes？


