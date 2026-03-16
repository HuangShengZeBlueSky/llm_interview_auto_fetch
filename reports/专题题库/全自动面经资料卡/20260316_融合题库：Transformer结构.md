# 融合题库：Transformer结构

> 已经把 GitHub 题库和真实面经合并去重。
> 本页共 78 道题，按同题合并后的题卡展示。

### 1. 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？

- 主标签：Transformer结构
- 来源条数：2
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 核心数学公式一致，均采用缩放点积注意力计算权重。
- 均依赖Q、K、V三个矩阵进行特征空间的线性映射。
- 均支持多头机制以捕捉不同子空间的丰富语义特征。

### 详细解答

Cross Attention与Self Attention在底层数学逻辑和计算框架上具有高度的一致性。结论上，两者的相同点主要体现在计算公式、参数结构以及对多头机制的支持上。在原理层面，它们都使用缩放点积注意力，即通过计算Q和K的点积来衡量相似度，经过Softmax归一化后作为权重对V进行加权求和。此外，两者在输入阶段都需要通过可学习的权重矩阵将输入特征映射到特定的隐空间。在工程实现上，两者的底层算子是可以通用的，都面临着序列长度带来的二次复杂度问题，且都可以通过多头机制并行计算来增强模型对不同表征子空间的捕捉能力，在代码层面往往复用同一个注意力模块。

### 案例模拟

面试官追问：“既然公式一样，在代码实现时能否复用同一个模块？”回答：“完全可以。在主流深度学习框架中，多头注意力模块同时支持这两种注意力。当传入的Query、Key、Value是同一个张量时，它就是自注意力；当Query来自一个张量，而Key和Value来自另一个张量时，它就变成了跨注意力。底层算子调用是完全相同的。”

### 2. Attention计算复杂度以及如何改进

- 主标签：Transformer结构
- 来源条数：2
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 标准自注意力机制的时间和空间复杂度均为序列长度的平方。
- 局部注意力通过限制感受野只计算相邻位置交互来降低复杂度。
- 线性注意力或低秩近似通过核函数或降维将复杂度降为线性。

### 详细解答

结论：标准Transformer中Attention的计算复杂度为O(N^2)，N为序列长度，这在处理长文本时会带来巨大的计算和显存瓶颈。 原理与改进：Attention需要计算Q和K的内积，生成N×N的注意力矩阵。为了降低复杂度，工程上常采用以下策略：1. 局部注意力（如Longformer），只计算当前位置与窗口内元素的交互，复杂度降为O(N×W)；2. 稀疏注意力，结合局部和全局节点减少计算量；3. 线性注意力（如Performer），利用核函数近似或低秩分解，避免显式计算N×N矩阵；4. 硬件级优化，如FlashAttention，通过分块计算和减少显存读写（IO感知）来大幅提升实际运行速度。 工程权衡：近似方法虽然降低了理论复杂度，但往往会损失一定的精度或全局感受野。FlashAttention等系统级优化则在不损失精度的前提下提升了效率，是目前大模型训练的标配。

### 案例模拟

面试追问：“在实际大模型推理中，除了Attention复杂度，还有什么瓶颈？” 回答示例：“推理时的主要瓶颈往往是显存带宽（Memory Bound）而非纯算力。特别是自回归生成阶段，每次生成一个Token都需要读取整个模型的权重和KV Cache。为了优化，我们会采用KV Cache量化、PagedAttention来优化显存碎片，或者使用MQA（多查询注意力）和GQA（分组查询注意力）来大幅减少KV Cache的显存占用，从而提升并发吞吐量。”

### 3. LayerNorm:

- 主标签：Transformer结构
- 来源条数：2
- 答案生成方式：近似题匹配：DeepLearing-Interview-Awesome-2024_LLMs专题:LLMs/Reference.md
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 早期模型探索过Post-Norm以期获得更好的泛化性能。
- LLaMA等现代大模型全面转向Pre-Norm或其变体。
- 归一化位置的选择本质是训练稳定性与表达能力的博弈。

### 详细解答

结论：在LLM的发展历程中，归一化层的位置经历了从Post-Norm到Pre-Norm的演进，核心驱动力是大规模分布式训练对稳定性的极致渴求。原理解释：参考资料中提到GPT-3采用Post-Norm（注：实际上GPT-2/3已开始使用Pre-Norm，但早期Transformer确实使用Post-Norm）。Post-Norm先计算Attention/FFN再归一化，这有助于约束每一层的输出尺度，理论上泛化能力强，但深层梯度回传受阻，极易崩溃。LLaMA等现代模型明确采用Pre-Norm（具体为Pre-RMSNorm），先归一化再进入计算模块。工程权衡：Pre-Norm构建了一条直达底层的“梯度高速公路”，使得千亿参数模型在数百张GPU上训练时，能够有效抵抗数值溢出和梯度消失。虽然牺牲了极少量的理论泛化上限，但换来了工程上至关重要的“一次训练成功率”。

### 案例模拟

面试官追问：“如果我非要用Post-Norm训练一个深层大模型，你会怎么做？”回答示例：“如果必须使用Post-Norm，工程上需要极其精细的初始化和学习率调度。首先必须引入较长的Learning Rate Warm-up阶段，让模型在初期以极小的步长探索；其次，可以采用Admin（Adaptive Model Initialization）等特殊的初始化方法，动态调整各层的方差；或者直接改用DeepNorm架构从根本上缓解梯度缩放问题。”

### 4. 1 传统 Attention 存在哪些问题？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 计算复杂度呈序列长度的平方级增长，长文本显存易OOM。
- 缺乏对相对位置信息的感知，需额外引入位置编码机制。
- 显存访问开销大，Memory Bound问题限制了推理生成速度。

### 详细解答

传统标准 Attention（即 Vanilla Self-Attention）在推动 NLP 发展的同时，也暴露出了几个致命的架构与工程瓶颈。首先是平方级复杂度问题：其时间复杂度和空间复杂度均与序列长度 $N$ 呈 $O(N^2)$ 关系。当处理长上下文时，计算量和显存占用会呈指数级爆炸，极易导致 OOM。其次是显存墙（Memory Bound）问题：在自回归推理阶段，每次生成一个 Token 都需要加载所有历史的 KV Cache，导致极高的显存带宽压力，计算单元大量时间在等待数据搬运。最后是位置感知缺失：纯粹的 Attention 具有排列不变性，无法区分 Token 的先后顺序，必须依赖绝对或相对位置编码（如 RoPE、ALiBi）来补充位置信息。工程权衡上，为了解决这些问题，业界不得不引入复杂的系统级优化（如 FlashAttention、MQA/GQA）来换取可用性。

### 案例模拟

面试官追问：“针对传统Attention的OOM问题，在不改变模型结构的前提下，有哪些系统级优化方案？” 回答：“在不改变模型数学等价性的前提下，最核心的系统级优化是FlashAttention。它利用GPU的SRAM和HBM层级存储特性，通过Tiling（分块计算）和Recomputation（重计算）技术，将Attention的显存复杂度从O(N^2)降到了O(N)，同时大幅减少了显存读写次数，缓解了Memory Bound问题。此外，还可以结合PagedAttention技术消除KV Cache的显存碎片。”

### 5. 1. 如何 利用 transformers 加载 Bert 模型？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / transformers 操作篇 / 未知](https://articles.zsxq.com/id_rsll7gsd8va5.html)

### 基础知识补充

- 使用HuggingFace提供的AutoModel类进行实例化。
- 通过from_pretrained方法加载预训练权重与配置。
- 需配套使用AutoTokenizer处理文本分词与编码。

### 详细解答

利用HuggingFace的transformers库加载Bert模型，最标准且推荐的做法是使用AutoModel和AutoTokenizer类。结论是：通过调用from_pretrained()方法并传入模型名称或本地路径，即可自动解析配置文件并加载权重。原理上，该方法会读取config.json构建模型计算图，并将pytorch_model.bin或safetensors中的参数映射到网络层中。在工程实践中，为了适配不同下游任务，通常会使用带有特定Head的派生类（如AutoModelForSequenceClassification）。此外，加载时可通过device_map="auto"或torch_dtype参数实现多卡自动分配与半精度加载，从而优化显存占用。

### 案例模拟

面试官追问：“如果内网环境无法连接HuggingFace Hub，如何加载模型？” 回答：“在离线业务场景中，我会先在外网通过huggingface-cli或代码将模型权重、配置文件和词表下载到本地目录。然后在内网代码中，将from_pretrained()的参数替换为该本地绝对路径。同时，为了避免代码在离线时仍尝试联网检查更新，可以设置环境变量HF_HUB_OFFLINE=1或在参数中指定local_files_only=True。”

### 6. 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- MHA：多头注意力，每个Q头对应独立的K和V头。
- MQA：多查询注意力，所有Q头共享唯一的一组K和V头。
- GQA：分组查询注意力，将Q头分组，每组共享一组KV。

### 详细解答

MHA、MQA和GQA是Transformer中三种不同的注意力机制，核心区别在于K和V头的数量，本质是推理速度与模型精度的权衡。MHA（Multi-Head Attention）中，Q、K、V的头数相同，精度最高，但推理时KV Cache占用极大，容易遇到内存带宽瓶颈。MQA（Multi-Query Attention）极端优化了这一点，所有的Q头共享同一个K头和V头，极大压缩了KV Cache的体积，显著提升了推理速度，但会导致一定的模型性能下降。GQA（Grouped-Query Attention）则是两者的折中方案，将Q头划分为G个组，每个组内的Q头共享一组K和V。GQA在保留了接近MQA的推理加速效果和低显存占用的同时，维持了与MHA相当的模型精度，目前已成为LLaMA-2/3等主流大模型的标配。

### 案例模拟

面试官追问：“如果我要把一个已有的MHA模型转换成MQA或GQA，应该怎么做？” 回答：“可以通过Mean Pooling（均值池化）来实现。将MHA中同一组的多个K头和V头的权重矩阵进行平均，合并成一个K头和V头，从而初始化GQA或MQA的权重。之后，只需要在原始训练数据上进行少量的微调（Uptraining），模型就能快速恢复性能。这种方法比从头训练一个GQA模型成本低得多。”

### 7. 2 Attention 有哪些 优化方向？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 稀疏注意力机制：如Longformer限制局部与全局注意力。
- 架构级降本：引入MQA或GQA减少KV Cache显存占用。
- 硬件感知优化：FlashAttention通过分块减少显存读写。

### 详细解答

针对传统 Attention 的瓶颈，业界的优化方向主要集中在算法架构改进和硬件感知优化两个维度。结论上，这些优化旨在打破 $O(N^2)$ 的复杂度魔咒并缓解显存带宽压力。算法架构层面：1) 稀疏化/线性化：如 Sparse Attention、Linformer 或 Linear Attention，将复杂度降至 $O(N)$，但通常会损失部分精度；2) 共享机制：多查询注意力（MQA）和分组查询注意力（GQA）通过让多个 Query 共享同一组 KV 头，成倍降低推理时的 KV Cache 显存占用，是目前 LLaMA 3 等大模型的标配。硬件感知层面：FlashAttention 及其迭代版本通过 Tiling（分块）技术优化 GPU 显存层级访问，在数学等价的前提下实现了提速和降显存。工程权衡上，目前的趋势是“GQA + FlashAttention + RoPE”的组合，最大化了长文本的吞吐量。

### 案例模拟

面试官追问：“MQA和GQA相比标准的MHA（多头注意力），在训练和推理阶段分别有什么影响？” 回答：“在训练阶段，MQA/GQA由于减少了KV的投影参数，计算量略微下降，但对整体训练速度提升不明显。在推理阶段（特别是Decode阶段），影响是颠覆性的。标准MHA的KV Cache占用极大，极易触发显存墙导致Batch Size无法开大。MQA/GQA大幅减少了KV Cache的显存占用和访存带宽需求，使得推理系统能支持更大的并发量，显著提升了吞吐量（Throughput）。”

### 8. 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / transformers 操作篇 / 未知](https://articles.zsxq.com/id_rsll7gsd8va5.html)

### 基础知识补充

- 前向传播时需设置output_hidden_states为True。
- 返回的元组中hidden_states包含所有层的输出张量。
- 索引0为词嵌入输出，后续对应各Transformer层特征。

### 详细解答

要输出Bert指定的hidden_state，关键在于配置模型返回所有隐藏层特征，并通过索引提取目标层。结论是：在调用模型前向传播时，传入参数output_hidden_states=True，或者在初始化Config时全局设置该属性。模型返回的输出对象中会包含一个hidden_states元组。其原理是，Bert由Embedding层和多层Transformer Block组成，开启该选项后，框架会缓存并返回每一层的激活值。元组长度为层数加一（例如12层Bert返回13个张量），第0个元素是Embedding输出，第i个元素是第i层的输出。工程应用中，常提取最后几层（如倒数第二层）的特征进行拼接或池化，因为顶层特征可能过于拟合预训练任务，而浅层特征缺乏高级语义。

### 案例模拟

面试官追问：“在做文本分类时，为什么有时用倒数第二层的hidden_state比最后一层效果好？” 回答：“在实际项目中，Bert的最后一层特征往往高度拟合其预训练任务（如MLM和NSP）。倒数第二层的特征保留了更丰富的通用语义信息，尚未被预训练任务的特定目标过度偏置。因此，在微调数据量较少的下游分类任务中，提取倒数第二层特征或将最后四层特征进行均值池化，通常能获得更好的泛化能力和分类指标。”

### 9. 2.2 Token Attention 介绍？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 纯Python超轻量高性能LLM推理框架 —— LightLLM / 未知](https://articles.zsxq.com/id_9a643feq2b0b.html)

### 基础知识补充

- 1. 显存管理机制：以单个Token为粒度分配和管理KV Cache
- 2. 零显存碎片：彻底解决按块分配带来的内部显存浪费问题
- 3. 动态连续性：物理显存不连续，通过索引表实现逻辑连续计算

### 详细解答

Token Attention是LightLLM框架中提出的一种创新的KV Cache显存管理机制。结论上，它通过将显存分配粒度细化到单个Token，实现了真正的零显存浪费。原理方面，传统的PagedAttention（如vLLM）以Block（通常包含16或32个Token）为单位分配显存，当请求长度不是Block大小的整数倍时，仍会产生内部显存碎片。Token Attention则预先分配一块巨大的连续显存池，每个Token生成时动态申请一个单位的显存槽位。在计算Attention时，通过维护一个Token到物理显存地址的映射索引表，利用Triton编写的自定义算子，直接在物理上不连续的Token显存间完成高效的注意力计算。工程权衡上，这种极致的细粒度管理最大化了显存利用率，支持更高的并发Batch Size；但代价是索引表的维护成本略有增加，且对底层算子的访存优化提出了更高要求。

### 案例模拟

面试官追问：“Token Attention和vLLM的PagedAttention有什么本质区别？”回答：“本质区别在于显存管理的粒度。PagedAttention借鉴了操作系统的虚拟内存分页，以Block为单位（如16个Token），虽然解决了外部碎片，但最后一个Block往往填不满，存在内部碎片。Token Attention则将粒度降到了极致的1个Token，彻底消除了内部碎片。在处理大量长度参差不齐的短请求时，Token Attention的显存利用率优势会更加明显。”

### 10. 2、chatglm1和chatglm2的attention mask是怎么样的？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 对比篇 / 未知](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

### 基础知识补充

- GLM1掩码设计：采用二维RoPE与分段式前缀双向注意力掩码。
- GLM2掩码演进：回归标准的单向因果注意力掩码（Causal Mask）。
- 架构范式转变：从Prefix-LM架构向纯粹的Decoder-only架构转变。

### 详细解答

结论：ChatGLM-6B（一代）采用了独特的二维注意力掩码（Prefix LM），而 ChatGLM2-6B 则回归了标准的单向因果掩码（Causal Mask）。原理上，一代基于 GLM 架构，输入部分（Context）采用双向注意力（互相可见），生成部分采用单向注意力，并配合二维 RoPE 位置编码来区分输入和生成的相对位置。这种设计在理解任务上表现较好，但增加了工程复杂度和推理优化的难度。为了提升训练效率和长文本推理性能，ChatGLM2 放弃了前缀双向掩码，全面转向了主流的纯 Decoder-only 架构，使用标准的下三角因果掩码，即每个 Token 只能看到自己及之前的 Token。这一改变不仅简化了模型结构，还完美适配了 FlashAttention 等加速技术，大幅提升了推理速度。

### 案例模拟

面试官追问：“ChatGLM2 放弃双向注意力掩码后，理解能力会下降吗？如何弥补？” 回答：“理论上纯单向掩码在双向上下文理解上弱于 Prefix LM，但 ChatGLM2 通过扩大预训练数据量、引入更充分的指令微调，以及采用更先进的 GQA（分组查询注意力）机制，不仅弥补了这一理论上的损失，反而在各项评测指标上全面超越了一代。这也证明了在足够的数据规模下，标准 Decoder-only 架构足以涌现出强大的理解能力。”

### 11. 3 Attention 变体有哪些？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 稀疏注意力机制（如Sparse Transformer）降低计算复杂度。
- 线性注意力（如Linformer）通过核函数近似实现线性复杂度。
- 共享键值注意力（如MQA、GQA）优化推理阶段的显存占用。

### 详细解答

Attention变体主要为了解决标准自注意力机制计算复杂度高（$O(N^2)$）和推理显存占用大的问题。主要分为三大类：第一类是稀疏注意力，如Longformer和BigBird，通过引入局部窗口、全局节点或随机稀疏连接，将复杂度降至$O(N \log N)$或$O(N)$，适合长文本处理；第二类是线性注意力，如Performer和Linear Attention，利用核函数技巧分解Softmax，实现真正的线性复杂度，但在常规长度下效果可能受损；第三类是针对推理优化的变体，如Multi-Query Attention (MQA) 和 Grouped-Query Attention (GQA)，通过共享KV头来大幅减少KV Cache的显存占用，提升生成速度。工程权衡上，目前大模型最常用的是GQA，因为它在性能和显存之间取得了最佳平衡。

### 案例模拟

面试官追问：“在处理超长上下文时，你会优先选择哪种Attention变体？” 回答：“如果是万字以内的长文本，我倾向于使用FlashAttention结合GQA，因为硬件级优化比算法近似更无损且高效。如果是十万字级别，我会考虑使用Ring Attention或者稀疏注意力（如Longformer的滑动窗口机制），配合RoPE的外推技术。在实际工程中，无损的系统级优化通常优先于有损的算法变体。”

### 12. 3、llama的attention mask是怎么样的？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 对比篇 / 未知](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

### 基础知识补充

- 掩码基础结构：采用标准的下三角因果注意力掩码机制。
- 自回归特性：确保当前Token只能关注自身及历史Token。
- 结合位置编码：与RoPE旋转位置编码结合，处理相对位置信息。

### 详细解答

结论：LLaMA 采用了标准的单向因果注意力掩码（Causal Attention Mask），即典型的下三角矩阵掩码。原理上，LLaMA 是纯粹的 Decoder-only 架构，为了保证文本生成的自回归特性（即预测下一个词时不能提前看到未来的词），其 Attention Mask 会将矩阵的上三角部分（未来信息）置为负无穷大（在 Softmax 后变为 0）。这样，序列中的每个 Token 在计算注意力权重时，只能与自身以及它之前的 Token 发生交互。在工程实现中，LLaMA 的这种标准掩码结构能够无缝集成 FlashAttention 等硬件级加速算子，极大地提升了长上下文的训练和推理效率。同时，该掩码机制与 LLaMA 使用的 RoPE（旋转位置编码）完美契合，共同构建了强大的序列建模能力。

### 案例模拟

面试官追问：“在进行 SFT（指令微调）时，LLaMA 的 Attention Mask 会发生变化吗？” 回答：“Attention Mask 的下三角因果结构本身不会变。但在 SFT 构建数据时，我们通常会使用 Loss Mask（损失掩码）。即对于输入序列中的 Prompt（用户指令）部分，我们会将其对应的 Loss 权重设为 0，只对模型生成的 Answer（回复）部分计算交叉熵损失。这确保了模型专注于学习如何回答，而不是去学习预测用户的提问。”

### 13. 4.1 Multi-head Attention 存在什么问题？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 推理阶段KV Cache占用大量显存，限制最大并发数。
- 访存计算比（Memory-bound）极低，导致带宽成为瓶颈。
- 多头独立计算带来冗余，部分注意力头学习到的特征高度相似。

### 详细解答

Multi-head Attention (MHA) 的核心问题在于推理阶段极高的显存占用和极低的访存效率。在自回归生成（Decoding）阶段，每次生成新词都需要读取历史所有的Key和Value向量（即KV Cache）。由于MHA中每个Query头都有独立的K和V头，导致KV Cache的体积随序列长度和Batch Size呈线性爆炸式增长。这不仅极大地消耗了GPU显存，限制了单卡能支持的最大Batch Size，还导致计算过程严重受限于显存带宽（Memory-bound），计算单元（如Tensor Core）大量时间在等待数据搬运，无法发挥算力。此外，研究表明MHA中不同头之间存在信息冗余，独立保存所有头的KV向量在工程上是不经济的。

### 案例模拟

面试官追问：“你能具体估算一下MHA的KV Cache显存占用吗？” 回答：“可以。假设模型维度为4096，层数为32，使用FP16精度。每个Token的KV Cache大小为 $2 \times 2 \times 32 \times 4096 = 512$ KB。如果序列长度为2048，Batch Size为16，那么单次推理仅KV Cache就会占用约 $512\text{KB} \times 2048 \times 16 \approx 16$ GB的显存。这在7B模型中占比极大，严重挤压了模型权重的显存空间，因此必须优化。”

### 14. 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 维度不同：多头是结构优化，跨注意力是数据流向。
- 相同点：底层均依赖缩放点积和Softmax归一化。
- 差异点：多头可用于自注意力，跨注意力强调异源。

### 详细解答

探讨两者的异同点，本质上是区分注意力机制的计算拓扑与数据来源。结论上，相同点在于它们都建立在标准的缩放点积注意力基础之上，共享相同的数学内核；不同点在于它们解决的问题维度完全不同。在原理上，多头注意力是一种计算范式，它既可以作用于同源数据构成多头自注意力，也可以作用于异源数据构成多头跨注意力，核心目的是增强特征表达的多样性。而Cross Attention特指Q与K/V来源不同的注意力模式，核心目的是实现跨序列的信息融合。在工程实践中，我们几乎不会使用单头的跨注意力，而是默认采用多头跨注意力结构，以平衡计算复杂度与特征提取的丰富度。

### 案例模拟

面试官追问：“如果我把Cross Attention的Q、K、V都设为同一个输入，它和多头注意力有什么关系？”回答：“如果Q、K、V同源，它就退化成了自注意力。如果此时你使用了多个注意力头并行计算，那么它就是一个多头自注意力。这说明多头是一种可插拔的结构特性，而Cross或Self定义的是输入数据的绑定关系，两者是正交的概念。”

### 15. 4.2 介绍一下 Multi-Query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- MQA所有Query头共享同一组Key和Value投影矩阵。
- 极大地压缩了KV Cache的体积，提升推理吞吐量。
- 训练阶段仍可并行计算，但模型表达能力会有轻微下降。

### 详细解答

Multi-Query Attention (MQA) 是一种专为加速大模型推理而设计的注意力机制变体。其核心原理是：保持Query的多头结构不变，但让所有的Query头共享唯一的一个Key头和一个Value头。在标准MHA中，如果有$H$个头，就会有$H$组K和V；而在MQA中，K和V的头数被强制设为1。这种设计的直接结论是，推理时的KV Cache显存占用降低到了原来的$1/H$。在工程实现上，这大幅缓解了Decoding阶段的Memory-bound问题，使得显存带宽不再是绝对瓶颈，从而可以显著增大Batch Size，成倍提升系统的吞吐量（Throughput）。不过，由于K和V的表达空间被大幅压缩，MQA在某些复杂推理任务上的模型性能会比MHA有轻微下降。

### 案例模拟

业务案例模拟：在部署一个用于高并发客服系统的对话模型时，我们发现使用标准MHA的7B模型在并发量达到32时就出现了OOM。为了解决这个问题，我们引入了采用MQA架构的模型（如Falcon）。替换后，KV Cache显存占用降低了近90%，使得单卡最大并发量提升到了128以上，首字延迟（TTFT）和生成速度（TPS）均有显著改善，完美满足了业务的高吞吐需求。

### 16. 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 结构差异：MHA多头独立KV，MQA所有Q头共享单组KV。
- 性能对比：MHA模型表达能力强，MQA推理吞吐量极高。
- 显存占用：MQA的KV Cache大小仅为MHA的头数分之一。

### 详细解答

MHA和MQA是模型表达能力与工程推理效率两个极端的对比。结论上，MHA注重效果，MQA注重速度和并发。原理上，MHA为每个Query头配备独立的Key和Value头，能够捕捉丰富的多子空间特征，但代价是推理时KV Cache庞大，导致严重的显存带宽瓶颈；MQA则让所有Query头共享同一组KV，极大地压缩了KV Cache体积（通常缩小32或64倍）。在工程权衡中，MHA适合对生成质量要求极高且并发要求不大的场景；而MQA通过牺牲约1-2%的下游任务指标，换取了推理阶段3-4倍的吞吐量提升和极低的显存占用。MQA的缺点是训练时容易出现梯度不稳定，且在复杂推理任务上容易出现性能衰减。

### 案例模拟

面试官追问：“如果我已经训练好了一个MHA模型，能直接转成MQA吗？” 回答：“不能直接无损转换，但可以通过微调来实现。一种常见的工程做法是：将MHA中多个K和V头的权重进行平均池化（Mean Pooling）或者保留其中一个头，作为MQA的初始KV权重，然后使用少量高质量数据进行继续预训练（Continue Pre-training）或微调。这样可以在节省大量从头训练算力的同时，让模型快速适应MQA结构并恢复大部分性能。”

### 17. 4.4 Multi-Query Attention 这样做的好处是什么？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 显著降低推理阶段的KV Cache显存占用，防OOM。
- 缓解Memory-bound问题，提高显存带宽利用率。
- 允许更大的Batch Size，大幅提升系统整体吞吐量。

### 详细解答

MQA带来的核心好处集中在推理阶段的性能飞跃，结论是它能以极小的精度损失换取数倍的吞吐量提升。首先，最直观的好处是大幅降低了KV Cache的显存占用，通常能减少到原来的几十分之一，这使得在相同显存容量下，可以支持更长的上下文或者更大的Batch Size。其次，它解决了自回归生成时的显存带宽瓶颈（Memory-bound）。在MHA中，读取庞大KV数据的耗时远大于计算耗时；MQA由于KV数据极小，数据搬运时间锐减，使得GPU的计算单元（ALU）能保持高负载运行。工程上，这意味着单台服务器可以处理更多的并发请求，极大地降低了大模型部署的单次调用成本（Cost per Token）。

### 案例模拟

面试官追问：“MQA对Prefill（预填充）阶段和Decode（解码）阶段的加速效果一样吗？” 回答：“不一样。MQA主要加速的是Decode阶段。在Prefill阶段，由于是并行计算所有Token的Attention，计算本身是Compute-bound（算力瓶颈）的，MQA和MHA的耗时差异不大。但在Decode阶段，每次只生成一个Token，属于Memory-bound（显存带宽瓶颈），MQA极大地减少了读取KV Cache的访存量，因此能显著提升Decode阶段的生成速度。”

### 18. 4.5 有 哪些模型 是 使用 Multi-Query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- Google的PaLM系列模型是早期采用MQA的代表作。
- 开源模型Falcon系列广泛使用了MQA架构提升推理速度。
- 智谱的ChatGLM2和ChatGLM3也采用了MQA机制。

### 详细解答

采用Multi-Query Attention (MQA) 的模型主要集中在追求极致推理效率的工业级大模型中。结论上，Google是MQA的早期倡导者，随后被多个知名开源模型采纳。具体来说，Google的PaLM（540B）和PaLM 2系列模型全面使用了MQA，以解决超大参数量下的推理延迟问题。在开源社区，TII推出的Falcon系列（如Falcon-7B, Falcon-40B）是典型的MQA代表，凭借极高的吞吐量在开源界引起广泛关注。国内方面，智谱AI的ChatGLM2-6B和ChatGLM3-6B也采用了MQA技术，这使得它们在消费级显卡上不仅能跑得起来，而且能实现非常流畅的生成速度。此外，StarCoder等代码生成模型也使用了MQA以支持更长的代码上下文。

### 案例模拟

面试官追问：“为什么现在很多最新的模型（如Llama 3）没有用MQA，而是用了GQA？” 回答：“因为MQA虽然速度快，但将所有头压缩到1个KV头会导致模型表达能力受损，特别是在复杂的逻辑推理和多轮对话中容易出现性能下降。GQA（Grouped-Query Attention）作为MHA和MQA的折中方案，将Q头分组共享KV，既保留了接近MQA的推理速度，又维持了与MHA相当的模型性能。因此，Llama 2/3等较新模型更倾向于使用GQA来实现效能的完美平衡。”

### 19. 5.1 什么是 Grouped-query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- GQA是MHA和MQA的折中方案，将Query头进行分组。
- 每组Query共享一个Key和一个Value头，平衡性能与速度。
- 既大幅降低了KV Cache占用，又保持了接近MHA的精度。

### 详细解答

Grouped-Query Attention (GQA) 是一种兼顾模型表达能力和推理效率的注意力机制。结论上，它是标准多头注意力（MHA）和多查询注意力（MQA）的完美折中。原理上，GQA将所有的Query头划分为$G$个组（Group），在同一个组内的Query头共享同一组Key和Value头。当$G$等于Query头数时，GQA就退化为MHA；当$G=1$时，GQA就变成了MQA。在工程权衡上，MQA虽然推理极快但会导致一定程度的性能下降，而GQA通过保留多个（通常是8个或4个）KV头，成功捕捉了不同的子空间特征，使得模型性能几乎与MHA持平。同时，相比于MHA，GQA依然将KV Cache的显存占用降低了数倍，极大地提升了推理吞吐量。

### 案例模拟

业务案例模拟：在自研百亿参数模型时，我们需要在推理成本和模型能力间做取舍。测试发现，使用MQA会导致模型在数学推理任务上准确率下降约3%，而使用MHA则导致单卡并发量不足。最终我们采用了GQA（将32个Q头分为8组，每组4个Q头共享1个KV）。压测结果显示，GQA的推理吞吐量达到了MQA的90%，而下游任务的准确率与MHA的差距缩小到了0.5%以内，成功实现了降本增效。

### 20. 5.2 有哪些大模型使用 Grouped-query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- Meta的Llama 2 (70B) 和 Llama 3 系列全面采用GQA。
- 法国开源模型Mistral及其混合专家模型Mixtral使用GQA。
- 国内的Qwen系列（通义千问）和DeepSeek系列也广泛使用。

### 详细解答

Grouped-Query Attention (GQA) 目前已经成为业界主流大语言模型（LLM）的标配架构。结论上，几乎所有2023年下半年及之后发布的高性能开源大模型都采用了GQA。最著名的代表是Meta的Llama系列，从Llama 2的34B/70B版本开始引入GQA，到Llama 3系列则是全参数规模（8B, 70B, 400B+）全面标配GQA。此外，欧洲明星初创公司Mistral AI发布的Mistral-7B以及Mixtral 8x7B MoE模型也采用了GQA，以支持长上下文和快速推理。在国内，阿里云的Qwen（通义千问）系列、深度求索的DeepSeek系列、百川智能的Baichuan2等主流模型，均将GQA作为底层架构的核心组件，以在部署时获得最佳的性价比。

### 案例模拟

面试官追问：“在Llama 3 8B模型中，GQA的具体配置是怎样的？这对部署有什么影响？” 回答：“Llama 3 8B模型配置了32个Query头和8个KV头，这意味着每4个Q头共享1组KV（即Group数为8）。在部署时，这种配置使得它的KV Cache大小只有同等规模MHA模型的四分之一。这不仅让8B模型能够轻松塞进单张24G显存的消费级显卡（如RTX 4090）中，还能在支持8K甚至更长上下文的同时，保持较高的并发处理能力，极大地降低了端侧和边缘侧的部署门槛。”

### 21. 6.1 为什么需要 FlashAttention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 标准Attention的时间和空间复杂度均为序列长度的平方。
- GPU显存读写（SRAM与HBM之间）是限制计算速度的核心瓶颈。
- 长序列场景下，传统注意力机制会导致显存溢出（OOM）问题。

### 详细解答

需要FlashAttention的核心原因是解决标准Transformer在处理长序列时面临的显存墙和计算效率瓶颈。在标准Attention中，计算复杂度为$O(N^2)$，且需要将庞大的注意力分数矩阵（$N \times N$）实例化并写入GPU的HBM（高带宽内存）中，然后再读出进行Softmax计算。这种频繁的HBM读写（Memory Bound）极大地拖慢了计算速度，并导致显存占用随序列长度呈平方级爆炸。FlashAttention通过引入Tiling（分块计算）和Recomputation（重计算）技术，将计算过程保留在速度更快的SRAM中完成，避免了中间矩阵的HBM读写。这不仅将显存复杂度从$O(N^2)$降低到了$O(N)$，还大幅提升了实际运行速度，使得大模型能够高效处理几十K甚至上百K的超长上下文。

### 案例模拟

面试官追问：“在实际工程中，除了长序列，FlashAttention对短序列有帮助吗？” 回答：“也有帮助。虽然短序列下$O(N^2)$的计算量不大，但标准Attention的HBM访存开销依然存在。FlashAttention通过减少内存读写次数，在短序列下同样能带来一定的加速比。不过，其最大的业务价值还是体现在长文本问答、代码级生成等需要极大上下文窗口的场景，能直接避免OOM并成倍提升吞吐量。”

### 22. 6.2 简单介绍一下 FlashAttention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- FlashAttention是一种IO感知的精确注意力算法。
- 核心目标是减少GPU高带宽内存（HBM）的读写次数。
- 能够在不牺牲模型精度的前提下实现显著的加速与省显存。

### 详细解答

FlashAttention是一种硬件（IO）感知的注意力机制计算算法，旨在打破GPU内存读写瓶颈。它的核心结论是：通过优化GPU显存层级之间的数据传输，可以实现比标准Attention更快且显存占用更低的精确计算。传统方法在计算Attention时，会生成庞大的中间矩阵并频繁在HBM和SRAM之间搬运数据。FlashAttention巧妙地利用了GPU的SRAM（容量小但速度极快）和HBM（容量大但速度慢）的特性，将输入矩阵分块（Tiling），在SRAM中直接完成局部Attention计算和Softmax的缩放，最后只将最终结果写回HBM。同时，在反向传播时，它不保存前向的中间激活值，而是通过重计算（Recomputation）在SRAM中快速恢复数据。这种设计使其成为大模型长文本处理的标配。

### 案例模拟

面试官追问：“FlashAttention是近似算法吗？会影响模型效果吗？” 回答：“FlashAttention是精确的注意力算法，不是近似算法（如稀疏注意力或线性注意力）。它在数学上与标准Attention完全等价，只是改变了计算顺序和内存访问模式。因此，它完全不会影响模型的精度和最终效果。在我们的长文本大模型训练项目中，替换为FlashAttention后，Loss曲线与基线完全一致，但训练速度提升了近3倍。”

### 23. 6.3 简单介绍一下 FlashAttention 核心？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- Tiling（分块）：将大矩阵切分为适合SRAM大小的小块。
- Recomputation（重计算）：反向传播时不存中间激活值。
- Online Softmax：通过维护局部最大值实现分块Softmax。

### 详细解答

FlashAttention的核心在于三大技术：Tiling（分块计算）、Online Softmax和Recomputation（重计算）。首先，Tiling技术将Q、K、V矩阵切分成能够装入GPU SRAM的小块，在SRAM内完成矩阵乘法，避免了全局大矩阵在HBM中的读写。其次，为了在分块的情况下正确计算Softmax（因为Softmax需要全局分母），引入了Online Softmax技巧，通过维护局部的最大值和指数和，在遍历各个分块时逐步更新全局Softmax结果。最后，在反向传播阶段，传统方法需要读取前向保存的巨大中间注意力矩阵，而FlashAttention采用Recomputation策略，利用保存的局部Softmax统计量在SRAM中快速重新计算中间值，从而将显存复杂度从$O(N^2)$降至$O(N)$。

### 案例模拟

面试官追问：“Online Softmax在分块计算时是如何保证数学等价性的？” 回答：“Online Softmax的核心是维护两个标量：当前块的最大值和指数和。当计算新块时，如果发现更大的最大值，会对之前累积的指数和进行缩放（乘以旧最大值与新最大值差值的指数），然后再加入新块的指数和。这样即使数据是分块流入的，最终计算出的Softmax分母也与全局计算完全一致，保证了数学上的精确等价。”

### 24. 6.4 介绍一下 FlashAttention 优点？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 显著提升计算速度，训练和推理吞吐量大幅增加。
- 极大降低显存占用，显存复杂度从平方级降至线性。
- 完美支持超长上下文窗口，且不损失任何模型精度。

### 详细解答

FlashAttention的优点主要体现在速度、显存和精度三个维度。首先是速度极快，通过大幅减少GPU HBM的读写次数（克服Memory Bound），其前向和反向计算速度通常比标准PyTorch实现快2到4倍。其次是极度节省显存，得益于分块计算和重计算机制，它不需要实例化$N \times N$的注意力分数矩阵，将显存复杂度从$O(N^2)$降低到了$O(N)$。这使得在相同硬件下，可以训练或推理更长序列的模型。最后是精确无损，与Linformer等近似注意力机制不同，FlashAttention在数学上与标准Attention完全等价，没有任何精度损失。这些优点使其成为目前所有主流大模型（如LLaMA、Qwen等）底层加速的标配组件。

### 案例模拟

面试官追问：“在你的项目中，引入FlashAttention带来了哪些具体收益？” 回答：“在我们的百亿参数模型微调项目中，原本使用标准Attention时，80G A100最多只能支持4K的上下文长度，再长就会OOM。引入FlashAttention-2后，我们不仅将上下文窗口成功扩展到了32K，而且整体训练吞吐量提升了约2.5倍。这让我们能够以更低的算力成本完成长文本阅读理解任务的模型迭代。”

### 25. 6.5 介绍一下 FlashAttention 代表模型？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- LLaMA系列：广泛采用FlashAttention提升长文本能力。
- Qwen/ChatGLM：国内主流开源模型均内置该加速算子。
- Falcon/Mistral：新一代大模型标配FlashAttention。

### 详细解答

FlashAttention自提出后，迅速成为了大语言模型（LLM）领域的底层基础设施，几乎所有现代主流大模型都将其作为标配。代表性模型包括Meta的LLaMA系列（LLaMA-2、LLaMA-3），它们在预训练和微调阶段均深度集成了FlashAttention，从而实现了对长上下文的高效支持。此外，Mistral和Mixtral模型也依赖该技术实现了极高的推理吞吐量。在国内，阿里云的Qwen系列、智谱的ChatGLM系列、百川智能的Baichuan等开源模型，其底层代码库（如基于Megatron-LM或HuggingFace Transformers）均默认开启FlashAttention。可以说，任何需要处理超过4K上下文的现代Transformer模型，都是FlashAttention的代表模型。

### 案例模拟

面试官追问：“HuggingFace中如何快速启用FlashAttention？” 回答：“在HuggingFace的Transformers库中，对于支持的模型（如LLaMA），只需在from_pretrained加载模型时，添加参数attn_implementation="flash_attention_2"即可快速启用。前提是环境里已经安装了flash-attn包，并且硬件是支持的NVIDIA GPU（如Ampere架构及以上）。这在工程上极大简化了长文本模型的部署。”

### 26. 7 并行 transformer block

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 将Attention和FFN层从串行计算改为并行计算。
- 核心公式变为：x = x + Attention(x) + FFN(x)。
- 能够减少通信开销，提升大规模分布式训练的吞吐量。

### 详细解答

并行Transformer Block（Parallel Transformer Block）是对标准Transformer结构的工程优化。结论是：它通过将自注意力层（Self-Attention）和前馈神经网络层（FFN）并行执行，显著提升了训练效率。在标准结构中，数据是串行流动的：先经过Attention，再经过FFN，即$x_1 = x + \text{Attn}(x)$，然后$x_2 = x_1 + \text{FFN}(x_1)$。而在并行结构中，输入$x$同时送入Attention和FFN，最后将两者的结果相加：$x_{out} = x + \text{Attn}(x) + \text{FFN}(x)$。这种设计的最大优势在于模型并行（Tensor Parallelism）场景下，可以将Attention和FFN的All-Reduce通信合并为一次，大幅减少了跨GPU的通信延迟，从而提升整体计算吞吐量，且对模型最终精度的影响微乎其微。

### 案例模拟

面试官追问：“并行Transformer结构最早是哪个模型提出的？有什么潜在缺点？” 回答：“最早由GPT-J和PaLM等模型广泛采用。其主要缺点是改变了原始的数学等价性，因为FFN不再接收Attention的输出作为输入，这在某些极深的网络或特定任务中可能会导致轻微的性能下降。但在千亿参数规模下，这种并行化带来的通信开销减半（All-Reduce次数减半）的收益远大于微小的精度波动，因此在工程实践中非常受欢迎。”

### 27. 9.1 简单介绍一下 Paged Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 借鉴操作系统虚拟内存分页思想管理KV Cache。
- 将连续的KV Cache切分为固定大小的物理块（Block）。
- 解决大模型推理时显存碎片化和内存浪费严重的痛点。

### 详细解答

Paged Attention是vLLM框架提出的一种用于大模型推理加速的核心显存管理技术。它的核心结论是：通过引入操作系统中虚拟内存和分页管理的思想，彻底解决了LLM推理过程中KV Cache显存碎片化的问题。在传统推理中，系统会为每个请求预先分配最大可能长度的连续显存来存储KV Cache，这导致了严重的内部碎片和外部碎片，显存浪费率高达60%-80%。Paged Attention将每个序列的KV Cache划分为固定大小的块（Blocks），这些物理块在显存中不需要连续。通过维护一张逻辑块到物理块的映射表（Block Table），模型可以在生成时动态按需分配显存。这不仅将显存浪费降至极低（不到4%），还天然支持了Beam Search等算法中的内存共享，极大提升了推理并发量。

### 案例模拟

面试官追问：“Paged Attention在处理Beam Search时有什么特殊优势？” 回答：“在Beam Search或并行采样中，同一个Prompt会生成多个不同的候选分支。传统方法需要为每个分支复制一份完整的KV Cache。而Paged Attention通过Block Table实现了Copy-on-Write（写时复制）机制。多个分支可以共享Prompt阶段的物理块，只有在生成不同Token时才分配新的物理块，这大幅降低了显存占用并提升了吞吐量。”

### 28. BEVFormer中的Spatial Cross-Attention的步骤？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/IndustryAlgorithm/Reference.md)

### 基础知识补充

- 将BEV Query提升为3D空间中的柱状区域Pillar。
- 将Pillar内3D参考点投影到多个相机视图的2D平面。
- 在命中的相机视图中采样特征，并通过注意力权重进行融合。

### 详细解答

BEVFormer中的Spatial Cross-Attention主要包含四个关键步骤。首先，将每个BEV Query提升为3D空间中的柱状区域（Pillar），即在Z轴上采样一系列3D参考点。其次，利用相机的内外参矩阵，将这些3D参考点投影到各个环视相机的2D图像平面上，确定它们在不同视图中的对应位置。接着，在那些被成功投影（命中）的相机视图中，围绕投影的2D点局部区域进行特征采样，通常借助Deformable Attention机制来高效提取多尺度图像特征。最后，根据网络学习到的注意力权重，对来自不同视图和不同参考点的采样特征进行加权融合，从而更新BEV Query的特征表示。这种机制有效建立了2D图像与3D BEV空间的几何映射。

### 案例模拟

面试官追问：为什么采用Deformable Attention而不是全局Attention？ 回答：全局Attention计算复杂度与图像分辨率的平方成正比，在自动驾驶高分辨率环视图像中计算量过大。Deformable Attention仅在投影的2D参考点附近采样少量关键点进行注意力计算，大幅降低了计算复杂度，同时能有效聚焦于对当前BEV Query最有价值的局部图像特征，提升了推理速度与显存利用率。

### 29. DiT (Diffusion Transformer): 25次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- DiT具备良好的扩展性规律，模型参数越大生成质量越高。
- Patch化处理将空间维度的去噪转化为序列维度的特征预测。
- 相比U-Net，DiT在处理高分辨率和长视频生成时更具优势。

### 详细解答

结论：DiT（Diffusion Transformer）通过引入Transformer架构，成功将大语言模型的Scaling Law复刻到了视觉生成领域，是目前主流的图像/视频生成范式。原理：DiT首先利用VAE将高维像素空间的图像压缩到低维潜空间，然后在潜空间中进行Patch切分。这些Patch序列经过多层Transformer Block处理，预测出每一步的噪声。DiT的设计极简，去除了U-Net中复杂的下采样、上采样和跳跃连接，完全依赖全局自注意力机制捕捉空间依赖关系。对比与工程权衡：U-Net在小数据量和低分辨率下表现优异，且卷积操作对局部纹理捕捉较好；而DiT在参数量扩展到Billion级别时展现出压倒性优势，特别是在Sora等视频生成任务中，Transformer能更自然地统一空间（图像）和时间（帧）维度的Token，但代价是极高的算力消耗和显存墙问题。

### 案例模拟

业务案例模拟：在研发企业级文生图大模型时，我们从Latent Diffusion的U-Net架构迁移到了DiT架构。初期遇到了训练极难收敛的问题，我们通过引入adaLN-Zero初始化和调整Patch Size（从8x8改为2x2以保留更多空间信息），最终在千万级图文对上跑通了模型。随着参数量从400M扩展到3B，模型的FID指标呈现出完美的线性下降趋势，验证了其扩展性。

### 30. GQA (Grouped-Query Attention): 31次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- GQA将多个Query头划分为若干组，每组共享一组Key和Value头。
- 相比MHA，GQA显著减少了推理阶段KV Cache的显存占用量。
- GQA有效缓解了生成阶段的显存带宽瓶颈，提升了系统吞吐量。

### 详细解答

结论：GQA（分组查询注意力）是目前大语言模型（如LLaMA-2/3、ChatGLM）中最主流的注意力机制，它通过分组共享KV头来优化推理效率。原理：假设模型有H个Query头，GQA将其分为G组，每组包含H/G个Query头，并且这H/G个Query头只对应1个Key头和1个Value头。因此，总共只有G个KV头。当G=H时退化为MHA，当G=1时退化为MQA。工程权衡：在长文本生成或大并发推理场景下，KV Cache的显存占用会呈线性增长，极大地限制了单卡能支持的最大Batch Size。GQA将KV Cache的显存需求降低到了MHA的G/H，使得显存可以容纳更多的并发请求。虽然在训练阶段GQA的计算量减少并不明显，但在部署推理时，它极大地降低了访存开销，是提升大模型服务性价比的关键技术。

### 案例模拟

面试官追问：“在实现GQA的推理算子时，如何处理Query和KV头数量不一致的问题？” 回答示例：“在底层算子（如FlashAttention或vLLM的PagedAttention）实现中，通常不需要在显存中物理复制KV Cache。相反，我们会利用CUDA的线程块索引映射，让同一组内的不同Query头在计算注意力分数时，通过广播机制读取同一份KV数据，从而节省显存并提高缓存命中率。”

### 31. KV Cache与PagedAttention

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- KV Cache缓存历史Token的键值向量，避免自回归重复计算。
- 传统预分配机制存在内部碎片和外部碎片，显存利用率极低。
- PagedAttention将KV Cache划分为固定大小物理块进行分页管理。

### 详细解答

结论：PagedAttention是解决大模型推理中KV Cache显存碎片化问题的革命性技术，极大提升了系统的并发吞吐量。原理：在LLM自回归生成中，KV Cache随序列长度动态增长。传统框架（如FasterTransformer）需按最大可能长度预分配连续显存，导致严重的内部碎片（预留但未生成的Token空间）和外部碎片。PagedAttention借鉴操作系统的虚拟内存分页机制，将KV Cache切分为固定大小的Block（如包含16个Token）。工程权衡：通过维护一张Block Table，逻辑上连续的KV Cache可以映射到物理上不连续的显存块中。这不仅将显存浪费降至极低，还天然支持了Beam Search等算法中不同序列间的内存共享（Copy-on-Write），大幅降低了显存占用，使得单卡能支撑的Batch Size成倍增加。

### 案例模拟

面试官追问：“PagedAttention中Block Size设置过大或过小有什么影响？”回答：“Block Size过大（如128）会导致每个Block内部仍存在一定的内部碎片，降低显存利用率；Block Size过小（如1）虽然碎片率极低，但会导致Block Table变得非常庞大，增加寻址开销，且在CUDA Kernel计算时无法充分利用内存合并访问（Memory Coalescing），降低访存效率。通常设为16或32是较好的折中。”

### 32. LayerNorm: 22次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- LayerNorm在特征维度上进行均值和方差的归一化。
- 解决BatchNorm在动态序列长度下表现不佳的问题。
- 包含可学习的缩放参数gamma和平移参数beta。

### 详细解答

结论：LayerNorm（层归一化）是Transformer架构中稳定深层网络训练的核心组件，尤其适用于处理变长序列的NLP任务。原理解释：与BatchNorm在Batch维度上统计不同，LayerNorm是在单个样本的特征维度（Hidden Dimension）上计算均值和方差。这意味着它的计算完全独立于Batch Size，对长短不一的文本序列非常友好。计算公式为 $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$。工程权衡：LayerNorm通过限制激活值的分布范围，有效缓解了深层网络中的梯度消失和爆炸问题，使得模型可以使用更大的学习率。然而，由于其需要同时计算均值和方差，在极大规模模型中，其计算和显存访问开销逐渐成为瓶颈，这也是后来RMSNorm崛起的原因。

### 案例模拟

面试官追问：“为什么Transformer不用BatchNorm而用LayerNorm？”回答示例：“NLP任务中句子长度差异大，如果用BatchNorm，在Padding位置会引入大量无意义的零值参与统计，导致均值和方差估算不准；且推理时如果遇到比训练时更长的序列，BN的全局统计量会失效。LayerNorm在特征维度归一化，每个Token独立计算，完美避开了序列长度变化和Batch Size大小的限制。”

### 33. MHA (Multi-Head Attention): 25次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- MHA(多头注意力)是Transformer架构中最经典的注意力机制。
- 每个Query头都有专属的Key和Value头，特征表达能力最强。
- MHA在长文本推理时会产生庞大的KV Cache，导致显存瓶颈。

### 详细解答

结论：MHA（Multi-Head Attention）是Transformer模型的基石，通过将特征空间映射到多个独立的子空间来捕捉丰富的上下文信息，但其高昂的推理成本促使了后续变体的诞生。原理：在MHA中，输入序列被并行地投影到多个不同的Query、Key和Value空间中。每个注意力头独立计算注意力权重并输出结果，最后将所有头的输出拼接并进行线性变换。这种设计允许模型同时关注不同位置、不同表征维度的信息。工程权衡：MHA的最大优势在于其强大的表征能力和模型性能，因此在早期的BERT、GPT-2中被广泛使用。然而，在工程部署时，MHA要求为每个头缓存完整的KV张量。随着序列长度和Batch Size的增加，KV Cache的显存占用急剧膨胀，导致GPU显存耗尽或受限于显存带宽。因此在百亿参数以上的现代大模型中，MHA正逐渐被GQA取代。

### 案例模拟

面试官追问：“既然MHA推理成本高，为什么在编码器模型（如BERT）或视觉Transformer中仍然广泛使用MHA？” 回答示例：“这是因为BERT和ViT主要用于理解任务，通常只需要进行一次前向传播（Prefill阶段），不需要像生成式大模型那样逐字自回归生成。在Prefill阶段，所有的QKV都是同时计算的，属于计算密集型任务，不会产生需要长期驻留显存的KV Cache，因此没有显存带宽瓶颈。”

### 34. MQA (Multi-Query Attention): 19次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 多查询注意力机制让所有Query共享一组KV头。
- 核心优势在于大幅降低推理时的KV Cache显存占用。
- 相比标准多头注意力，MQA会带来一定程度的模型性能下降。

### 详细解答

结论：MQA（Multi-Query Attention）是一种专为大模型推理加速设计的注意力变体，通过让所有Query头共享唯一的一组Key和Value头，极大缓解了显存带宽瓶颈。原理上，标准MHA中每个Query头都有独立的KV头，导致推理生成阶段KV Cache体积庞大，访存成为主要瓶颈（Memory-bound）。MQA将KV头数量降为1，使得KV Cache的显存占用和读取数据量成比例下降，从而显著提升生成速度（Decode阶段的吞吐量）。工程权衡方面，虽然MQA能带来极高的推理效率，但由于参数量和表达能力的削弱，通常会导致模型在复杂任务上的效果有所下降，因此目前更多被折中的GQA方案所取代。

### 案例模拟

面试官追问：“MQA在训练和推理阶段的计算量有变化吗？”回答示例：“在训练阶段（Prefill），由于是并行计算，MQA的计算量和MHA差异不大，主要节省的是显存；但在推理生成阶段（Decode），由于KV Cache变小，访存量大幅降低，虽然FLOPs减少不多，但因为打破了访存带宽瓶颈，实际推理延迟显著降低，吞吐量大幅提升。”

### 35. Megatron-LM 的张量并行（TP）逻辑是考察重点，特别是涉及到 Transformer 层内部的具体切分方式。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 自注意力层对QKV权重矩阵进行列切分以独立计算
- 输出投影矩阵采用行切分，后接All-Reduce同步
- MLP层采用列切分与行切分结合，消除中间通信

### 详细解答

结论：Megatron-LM 的张量并行（TP）通过巧妙的行列切分组合，最大化了 Transformer 层内的计算独立性，将通信开销降至最低。原理：在 Self-Attention 模块中，Q、K、V 的权重矩阵被按列切分（Column Parallelism），每张卡独立计算出部分注意力头（Attention Heads），随后在输出投影矩阵（Output Projection）采用按行切分（Row Parallelism），计算后通过一次 All-Reduce 得到完整输出。在 MLP 模块中，第一个全连接层按列切分，激活函数在各卡独立执行；第二个全连接层按行切分，同样在末尾进行一次 All-Reduce。工程权衡：这种设计的精妙之处在于，无论是 Attention 还是 MLP 模块，都只需要在模块末尾进行一次 All-Reduce 通信，完全消除了两次矩阵乘法之间的通信需求，极大地提升了并行效率。

### 案例模拟

面试官追问：“在 Megatron-LM 的张量并行中，多头注意力（MHA）是如何切分的？如果头数不能被 GPU 数整除怎么办？” 回答：“MHA 的切分是在‘头（Head）’的维度上进行的。例如有 16 个头，4 张卡，每张卡负责计算 4 个头，各卡独立完成 QK^T 和 Softmax 计算。如果头数不能被 GPU 数量整除，通常会报错，因为 Megatron 的基础实现要求均匀切分以保证负载均衡。工程上遇到这种情况，要么调整模型结构的头数，要么调整张量并行的 GPU 数量，确保其能整除。”

### 36. PagedAttention

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 借鉴操作系统虚拟内存分页，将KV缓存划分为固定大小块。
- 解决传统推理中预分配导致的显存碎片化和严重浪费问题。
- 允许非连续物理显存存储连续逻辑Token，提升吞吐量。

### 详细解答

PagedAttention是vLLM框架提出的革命性显存管理算法。其核心结论是：通过分页管理彻底消除了KV Cache的内部和外部显存碎片，将GPU显存利用率从不足50%提升至近乎100%。原理上，传统推理系统在请求到达时会按最大可能长度预分配连续显存，导致大量显存闲置（内部碎片）或因长度不一产生缝隙（外部碎片）。PagedAttention借鉴OS的虚拟内存机制，将KV Cache切分为固定大小的Block。逻辑上连续的Token在物理显存中可以是非连续分布的，通过Block Table进行地址映射。工程权衡上，这种设计虽然引入了极小的地址转换开销，但使得系统能够支持成倍增加的Batch Size，极大提升了服务吞吐量。

### 案例模拟

面试官追问：“PagedAttention如何加速Beam Search或并行采样？” 回答示例：“在Beam Search或多候选采样中，同一个Prompt会生成多个不同的输出分支。PagedAttention通过Block Table实现了内存共享（Copy-on-Write机制）。多个分支可以共享Prompt阶段的物理Block，只有在生成不同Token时才分配新的物理块。这不仅节省了大量显存，还减少了冗余计算。”

### 37. PagedAttention (vLLM): 33次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- vLLM是基于PagedAttention构建的高吞吐量开源推理引擎。
- 虚拟内存机制解耦了KV Cache的逻辑连续性与物理连续性。
- 支持写时复制机制，在复杂解码算法中实现显存高效共享。

### 详细解答

结论：vLLM框架凭借PagedAttention技术，解决了大模型推理中的显存碎片痛点，成为目前工业界广泛采用的高并发推理基座。原理：在vLLM中，请求的KV Cache被划分为逻辑块，通过Block Table映射到GPU显存池中的物理块。当新请求到来时，调度器会检查显存池是否有足够的空闲物理块，如果有则分配并更新映射表。对比与工程权衡：相比于HuggingFace原生实现或早期的FasterTransformer，vLLM在处理高并发、长文本请求时，吞吐量可提升2-4倍。其核心优势不仅在于减少碎片，更在于其灵活的内存共享能力。例如在Parallel Sampling或Beam Search中，多个候选序列可以共享同一个Prompt的物理块，直到它们生成不同的Token时才触发写时复制（Copy-on-Write），极大地节省了显存，提升了并发上限。

### 案例模拟

面试官追问：“vLLM在处理超长文本（如100K上下文）时可能会遇到什么瓶颈？如何解决？”回答：“超长文本会导致单个请求的KV Cache极大，可能耗尽单卡显存，且PagedAttention的Block Table也会变大。解决方案：1. 引入张量并行（TP）将KV Cache分布到多卡；2. 采用RingAttention等技术处理超长上下文；3. 开启KV Cache量化（如FP8/INT4）压缩缓存体积；4. 将部分不常用的KV Cache卸载（Offload）到CPU内存中。”

### 38. PagedAttention原理：这是vLLM库的核心。面试官要求类比操作系统。传统KV Cache预分配显存导致碎片化和浪费。PagedAttention将KV Cache切分为固定大小的块，通过一张“页表”将逻辑上的连续token映射到物理上不连续的显存块。这使得显存利用率接近100%，从而支持极大的Batch Size。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 借鉴操作系统虚拟内存分页机制管理显存
- 将KV Cache切分为固定大小的物理显存块
- 通过页表映射逻辑连续Token与物理不连续块

### 详细解答

结论：PagedAttention是vLLM的核心技术，通过引入操作系统中的虚拟内存分页思想，彻底解决了大模型推理中KV Cache显存碎片化的问题。原理：传统推理时，系统会为每个请求预分配最大可能长度的连续显存，导致大量内部碎片和显存浪费。PagedAttention将KV Cache划分为固定大小的块（Block），并在逻辑空间和物理空间之间建立“页表”映射。逻辑上连续的Token可以映射到物理上不连续的显存块中。工程权衡：这种设计使得显存利用率接近100%，极大提升了系统的Batch Size上限和吞吐量，但引入了页表维护的微小计算开销。

### 案例模拟

面试官追问：“PagedAttention在处理Beam Search等复杂解码策略时有什么优势？”回答示例：“在Beam Search或并行采样中，多个候选序列会共享相同的前缀。PagedAttention允许不同逻辑序列通过页表指向相同的物理KV块，实现内存级别的零拷贝共享，大幅降低了显存占用并提升了生成效率。”

### 39. RoPE (旋转位置编码)

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 核心思想是通过绝对位置的旋转变换实现相对位置编码效果。
- 借助欧拉公式，将位置信息映射为复平面上的特定旋转角度。
- 具有良好的长度外推性，常配合动态NTK或YaRN等插值技术。

### 详细解答

结论：旋转位置编码（RoPE）是目前大语言模型（如LLaMA、Qwen）标配的位置编码方案。其数学原理在于，将Query和Key向量的特征维度两两分组，视为复数。在计算Attention前，根据Token的绝对位置m和n，在复平面上对Q和K进行相应角度的旋转。当Q和K进行点积时，复数内积的特性使得结果中绝对位置项被抵消，仅保留了相对位置（m-n）的夹角信息。相比于传统的相对位置编码，RoPE无需修改Attention的计算图，直接在输入向量上做轻量级的逐元素操作，计算效率极高。同时，RoPE天然具备一定的长度外推潜力，结合位置插值（PI）或NTK-aware缩放技术，能以极低成本扩展模型的上下文窗口。

### 案例模拟

面试官追问：“如果想把基于RoPE的模型上下文从4K扩展到32K，你会怎么做？”回答：“直接外推会导致高频特征混乱，Attention分数崩塌。在项目中，我会采用线性位置插值（PI）或NTK-aware缩放。NTK方法通过改变不同维度的旋转基数，对高频特征保持不变，对低频特征进行插值缩放。这样只需用少量长文本数据微调几百步，就能让模型平滑适应32K长窗口，且不损失短文本能力。”

### 40. Transformer/CNN/RNN的时间复杂度对比

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/DeepLearning/Reference.md)

### 基础知识补充

- Transformer复杂度：与序列长度的平方成正比，全局感受野。
- RNN复杂度：与序列长度呈线性关系，但存在时序依赖无法并行。
- CNN复杂度：与卷积核大小和序列长度相关，支持高度并行计算。

### 详细解答

结论：在处理序列数据时，Transformer的时间复杂度随序列长度呈平方增长，而RNN和CNN随序列长度呈线性增长。 原理与对比：设序列长度为N，特征维度为D。Transformer的自注意力机制需要计算每两个Token之间的相似度，复杂度为$O(N^2 \cdot D)$，当N很大时计算极慢，但其优势在于高度并行化和全局感受野。RNN每步计算涉及隐藏状态的矩阵乘法，复杂度为$O(N \cdot D^2)$，虽然对N是线性的，但由于时序依赖无法并行计算。CNN通过滑动窗口卷积，复杂度为$O(N \cdot K \cdot D^2)$（K为核大小），支持高度并行且捕捉局部特征。 工程权衡：在短序列下，Transformer速度极快且效果好；但在长文本或高分辨率图像中，$N^2$的复杂度成为显存和算力的瓶颈。因此工程上常采用FlashAttention、稀疏注意力或线性Transformer来降低长序列的计算开销。

### 案例模拟

面试官追问：既然Transformer处理长序列复杂度高，目前业界有哪些主流的优化方案？ 回答示例：业界主要从系统和算法两方面优化。系统层面，最著名的是FlashAttention，通过分块计算和减少HBM读写，在不损失精度的前提下大幅提升速度并降低显存占用。算法层面，有采用滑动窗口或稀疏注意力的模型（如Longformer），将复杂度降至$O(N \cdot \log N)$或$O(N)$；还有近期热门的Mamba等状态空间模型，试图在保持全局感受野的同时实现线性复杂度。

### 41. Transformer中的Attention计算复杂度以及如何改进？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 复杂度瓶颈：标准自注意力计算的时间与空间复杂度为平方级。
- 局部注意力：通过滑动窗口机制，仅计算相邻词元的注意力。
- 线性注意力：利用核函数或低秩近似，将计算复杂度降至线性。

### 详细解答

结论：标准Transformer的自注意力机制计算复杂度为O(N^2)（N为序列长度），这限制了其处理长文本的能力。改进方向主要包括稀疏化、近似计算和降维。 原理与改进：在标准Attention中，每个Token都需要与所有Token计算点积，导致平方级复杂度。改进方法包括：1. 局部注意力：限制感受野，只计算当前位置附近子序列的交互。2. 稀疏注意力：结合局部与全局节点，减少不必要的计算。3. 近似方法：使用局部敏感哈希或随机采样来近似全量计算。4. 线性注意力：通过核函数展开或低秩映射，将输入映射到低维空间，从而将复杂度降至O(N)。 工程权衡：虽然改进方法降低了复杂度，但往往会牺牲一定的精度或全局信息捕捉能力。目前主流大模型多采用FlashAttention，通过底层硬件显存读写优化，在不损失精度的前提下大幅提升了计算速度。

### 案例模拟

面试官追问：你提到了FlashAttention，它和局部注意力有什么本质区别？ 回答示例：本质区别在于是否产生精度损失。局部注意力（如滑动窗口）在算法层面上改变了计算逻辑，丢弃了部分全局连接，属于近似计算，会带来精度损失。而FlashAttention是精确计算，它没有改变Attention的数学结果，而是通过分块和重计算技术，优化了GPU的SRAM和HBM之间的内存读写，从而在硬件层面实现了加速并降低了显存占用。

### 42. Transformer为何使用多头注意力机制

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 多头机制允许模型关注不同表示子空间的信息。
- 各个注意力头独立计算并拼接以捕捉丰富特征。
- 类似于卷积神经网络中的多通道特征提取机制。

### 详细解答

结论：Transformer使用多头注意力（Multi-Head Attention）是为了让模型能够同时从不同的表示子空间中捕捉多样化的语义特征，从而增强模型的表达能力。 原理：单头注意力机制在计算时，所有的Query、Key、Value都映射在同一个高维空间中，容易导致注意力分布过于集中，忽略了句子中其他潜在的关联信息。多头机制通过将输入线性投影到多个低维的子空间中，在每个子空间内独立执行注意力计算，最后将所有头的输出拼接并进行线性变换。这使得模型可以同时关注语法结构、指代关系、情感色彩等不同层面的信息。 工程权衡：多头机制在不增加总参数量（通过降维拆分）的前提下，显著提升了特征提取的丰富度。但在长文本推理时，多头带来的KV Cache显存压力较大，因此现代大模型常采用MQA或GQA来优化。

### 案例模拟

面试追问：“多头注意力在推理时会导致显存占用过大，有什么优化方案？” 回答示例：“在标准多头注意力（MHA）中，每个头都有独立的Key和Value，导致KV Cache占用极大。在我们的百亿参数模型部署中，我们采用了分组查询注意力（GQA）技术。GQA将多个Query头共享同一组Key和Value头，比如将32个Q头分为8组，每组共享1个K和V。这样在几乎不损失模型生成质量的前提下，将KV Cache的显存占用降低了四分之三，大幅提升了推理并发量。”

### 43. Transformer架构作为现代大模型的基石，其考察深度在2025年达到了前所未有的水平。面试官默认候选人已经熟悉基本结构，转而主要攻击架构中的具体组件选型及其背后的数学原理。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 考察重点转向底层算子优化、显存管理及分布式训练的协同设计。
- 数学原理深挖包括位置编码的推导、注意力机制的核函数近似等。
- 组件选型需结合硬件特性，如FlashAttention对SRAM的极致利用。

### 详细解答

结论：2025年的大模型面试不再停留在Transformer的宏观结构，而是极度聚焦于组件选型的数学本质及其在GPU硬件上的工程实现。原理解释：面试官会深挖为什么选择特定的组件。例如，不仅要懂RoPE的公式，还要能推导其旋转矩阵的复数形式及外推性原理；不仅要知道GQA，还要能计算不同配置下的KV Cache显存占用量。对比与权衡：在架构设计上，候选人需要理解算法与硬件的协同（Hardware-aware）。例如，标准Attention受限于HBM内存带宽（Memory-bound），而FlashAttention通过分块计算（Tiling）和重计算（Recomputation）将访存转移到高速SRAM中，虽然增加了FLOPs，但大幅降低了IO耗时，实现了整体加速。这种在算力与访存间权衡的思维是高级岗位的核心要求。

### 案例模拟

面试官追问：如果让你设计一个支持100k长上下文的模型，你会从哪些架构组件上进行优化？ 回答示例：首先，位置编码会采用RoPE并结合YaRN或NTK-Aware插值算法，提升长文本外推能力；其次，注意力机制必须使用GQA以降低KV Cache显存，并结合Ring Attention或DeepSpeed Ulysses解决单卡显存不足的并行问题；最后，可以引入稀疏注意力或滑动窗口注意力（如Mistral的SWA），在保证局部上下文的同时降低全局计算复杂度。

### 44. Transformer的注意力机制常用softmax函数，可以使用sigmoid代替吗？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/VisionPerception/Reference.md)

### 基础知识补充

- Softmax确保注意力权重之和为1，具有归一化效果。
- Sigmoid独立计算每个元素的概率，缺乏全局竞争机制。
- Softmax能产生强制稀疏化，突出最相关的特征。

### 详细解答

结论：可以使用Sigmoid代替，但通常不建议，因为Softmax的归一化特性更符合“注意力”的物理直觉，而Sigmoid会导致注意力机制的性质发生改变。 原理与对比：Softmax函数会将所有Token的注意力得分映射到0-1之间，且总和强制为1。这种机制引入了全局竞争，具有强制稀疏化的效果，迫使模型将有限的“注意力”集中在少数最相关的Token上，起到正则化和特征筛选的作用。相反，Sigmoid对每个Token的得分进行独立映射，互不干扰。这使得Sigmoid的灵活度更高，极端情况下可能导致模型对所有Token都赋予高权重（什么都注意）或都赋予低权重（什么都不注意）。在工程实践中，如果任务需要多标签式的特征聚合，Sigmoid可能有奇效；但在标准的Transformer中，Softmax的竞争机制能带来更稳定、更具解释性的特征表达。

### 案例模拟

面试官追问：在什么特定场景下，使用Sigmoid代替Softmax可能会取得更好的效果？ 回答：在处理多模态任务或多标签分类时，Sigmoid可能更具优势。例如在视觉-语言模型中，图像的一个区域可能同时对应文本中的多个独立概念。如果使用Softmax，权重会被强制分散，导致每个概念的响应变弱；而使用Sigmoid，各个概念的注意力得分独立计算，允许模型同时对多个特征保持高度关注。此外，在某些线性Transformer变体中，也会用Sigmoid替代Softmax以实现核函数的分解。

### 45. U-Net vs Transformer：传统的Stable Diffusion使用U-Net作为去噪骨干。DiT将其替换为Transformer。面试题：“为什么DiT比U-Net扩展性更好？”答：Transformer架构对Patch数量不敏感，且具有明确的Scaling Law，可以通过堆叠层数和增加宽度持续提升性能，而U-Net的卷积结构在高分辨率下感受野受限且计算复杂度难以优化。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- U-Net依赖卷积结构，高分辨率下全局感受野受限。
- Transformer对Patch数量不敏感，自注意力具备全局视野。
- DiT具有明确的Scaling Law，可通过堆叠层数提升性能。

### 详细解答

结论：DiT比U-Net扩展性更好，核心在于Transformer架构的全局感受野和对计算资源的高效利用，使其完美契合Scaling Law。原理与对比：传统的U-Net基于CNN结构，通过下采样和上采样提取特征，其感受野是局部的，在处理极高分辨率或长视频时，难以捕捉远距离的复杂依赖关系；同时，CNN的计算复杂度优化存在瓶颈。相反，Transformer将数据转化为Patch序列，自注意力机制天生具备全局视野，对Patch数量不敏感。工程权衡上，DiT可以通过简单地增加网络宽度（Hidden Size）和深度（层数）来持续提升生成质量（FID下降），这种线性扩展能力在U-Net中很难实现。此外，Transformer架构高度统一，便于利用现有的硬件加速算子进行极致的并行优化。

### 案例模拟

面试官追问：“既然Transformer这么好，为什么早期Stable Diffusion不用？”回答示例：“早期不用主要受限于算力和数据规模。Transformer缺乏CNN的归纳偏置（如平移不变性），需要海量数据和算力才能收敛。当时U-Net在小规模数据上更容易训练且显存占用相对可控。随着算力提升和FlashAttention等显存优化技术的成熟，Transformer在长序列上的计算瓶颈被打破，其扩展性优势才得以全面释放。”

### 46. 一、vLLM 用于大模型并行推理加速 存在什么问题？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——PagedAttention篇 / 未知](https://articles.zsxq.com/id_p22mjq881n3n.html)

### 基础知识补充

- 静态显存分配导致严重的内部碎片与外部碎片问题。
- 预留显存无法根据实际生成长度动态调整造成浪费。
- 传统KV Cache在多请求共享前缀时存在冗余存储。

### 详细解答

在vLLM提出之前，传统的大模型并行推理加速（如FasterTransformer）存在严重的显存管理问题。首先是显存碎片化：由于无法预知每个请求最终生成的Token数量，系统通常会按照最大可能长度（如2048）为每个请求预先分配连续的显存空间。这导致了严重的内部碎片（实际生成较短，剩余空间闲置）和外部碎片（显存块不连续，无法分配给新请求）。据统计，传统方法中只有约20%到40%的KV Cache显存被有效利用。其次是缺乏内存共享机制：在Beam Search或多轮对话等场景中，多个请求往往共享相同的Prompt前缀，但传统框架会为每个请求单独存储一份前缀的KV Cache，造成极大的显存冗余，严重制约了系统的最大并发量。

### 案例模拟

面试官追问：“你提到的内部碎片和外部碎片，在LLM推理中具体是怎么产生的？” 回答：“内部碎片是因为我们按最大长度（比如预设生成1000个token）分配了显存，但模型实际只生成了100个token就输出了EOS停止，剩下的900个token空间就被浪费了。外部碎片则是由于不同请求的生命周期不同，释放显存后留下许多不连续的小块显存，导致无法为新的长序列请求分配连续空间。”

### 47. 一、为什么需要 FasterTransformer？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——Faster Transformer篇 / 未知](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### 基础知识补充

- 原生深度学习框架在Transformer推理时存在大量算子开销
- 动态图机制导致显存分配和计算图调度效率低下
- 缺乏针对特定GPU架构的底层CUDA算子深度融合与优化

### 详细解答

结论：需要FasterTransformer（FT）是为了突破PyTorch等通用深度学习框架在Transformer模型推理时的性能瓶颈。原理：原生框架通常将Transformer层拆分为多个细粒度的基本算子（如矩阵乘、加法、LayerNorm、激活函数），这会导致频繁的GPU显存读写（Memory Bound）和内核启动开销。FT通过算子融合（Operator Fusion）技术，将多个小算子合并为一个定制的CUDA Kernel，大幅减少显存访问次数。对比与工程权衡：相比于原生PyTorch，FT能显著降低推理延迟并提升吞吐量，尤其在低Batch Size下优势明显。但其代价是极高的开发和维护成本，模型需要转换为特定的二进制格式，且对新模型架构的适配周期较长，灵活性不如基于Python的现代框架。

### 案例模拟

面试官追问：FasterTransformer的算子融合具体是怎么做的？回答：以Transformer的MLP层为例，原生框架会分别调用矩阵乘、偏置加法和GELU激活三个算子，中间结果需要写回全局显存。FT会将偏置加法和GELU融合到矩阵乘的Epilogue阶段，或者将LayerNorm与残差连接融合。在实际项目中，这种融合能将显存带宽占用降低一半，使推理延迟下降约30%。

### 48. 一、为什么需要 跨注意力机制（Cross-Attention）？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 解决多源信息融合问题，打破单一序列的特征孤岛。
- 在Seq2Seq任务中建立编码器与解码器的信息桥梁。
- 允许模型在生成目标时动态聚焦源序列的关键部分。

### 详细解答

需要跨注意力机制的核心原因在于解决多源数据之间的动态对齐与信息融合问题。结论上，它是Seq2Seq模型和多模态模型中不可或缺的信息桥梁。在原理层面，如果仅使用自注意力，模型只能理解单一序列内部的依赖关系；而跨注意力允许解码器在生成每个Token时，通过Q去源序列的K和V中检索最相关的上下文信息。例如在机器翻译中，生成目标词时需要精准定位源语言中的对应词汇。在工程实践中，跨注意力机制解耦了源序列和目标序列的长度限制，使得模型可以灵活处理输入输出长度不一致的任务。相比于简单的特征拼接或池化，它提供了更细粒度、可解释性更强的特征交互方式，显著提升了生成质量。

### 案例模拟

业务案例模拟：“在语音识别（ASR）项目中，输入是长音频特征，输出是文本。由于音频帧数远大于文本长度，直接拼接无法对齐。我们引入跨注意力机制，将文本解码器的隐状态作为Q，音频编码器的输出作为K和V。这使得模型在预测每个字时，能自动聚焦到对应的音频片段，有效解决了长语音的对齐难题。”

### 49. 七、Cross Attention 的优势和挑战？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 融合多模态或多源序列特征的核心注意力机制。
- 查询向量来自解码器，键值向量来自编码器输出。
- 计算复杂度随双序列长度乘积呈二次方增长。

### 详细解答

Cross Attention 的核心优势在于能够高效实现不同模态或不同来源序列之间的信息对齐与融合。在机器翻译或多模态任务中，它允许解码器在生成每个目标Token时，动态聚焦于源序列中最相关的部分，从而建立跨序列的强语义关联。然而，其面临的主要挑战在于计算与内存开销。由于需要计算两个序列长度乘积的注意力矩阵，当处理长文本或高分辨率图像特征时，显存占用呈二次方爆炸。工程权衡上，常采用降采样、局部注意力或Perceiver架构中的Latent Bottleneck机制来压缩键值序列长度，从而在保证跨模态融合效果的同时降低计算复杂度。

### 案例模拟

面试官追问：“在多模态大模型中，如何缓解Cross Attention处理高分辨率图像时的显存压力？” 回答：“在实际工程中，我们通常不会直接将原始图像Patch展平输入。可以采用类似Flamingo的Perceiver Resampler机制，先用少量可学习的Latent Queries对图像特征进行降维压缩，提取固定数量的视觉Token，然后再与文本进行Cross Attention计算，这样能大幅降低显存消耗并提升推理速度。”

### 50. 三、FasterTransformer 核心是什么？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——Faster Transformer篇 / 未知](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### 基础知识补充

- 深度算子融合技术减少GPU显存读写和内核启动开销
- 针对不同精度和硬件架构高度优化的定制化CUDA内核
- 高效的分布式推理支持包括张量并行和流水线并行机制

### 详细解答

结论：FasterTransformer的核心在于极致的底层算子优化（尤其是算子融合）以及高效的分布式并行策略。原理：在单卡层面，其核心是Operator Fusion。Transformer中的大量访存密集型操作（如LayerNorm、Add、Activation）被融合进计算密集型操作（如GEMM）中，极大缓解了显存带宽瓶颈。同时，它利用cuBLASLt等底层库针对不同矩阵尺寸自动寻找最优算法。在多卡层面，其核心是高效的通信掩盖与并行调度，通过张量并行切分权重，利用NVLink实现极低延迟的All-Reduce同步。对比与工程权衡：这种核心设计的优势是榨干了GPU的每一滴算力，但劣势是与硬件强绑定，且丧失了动态图的灵活性。任何对模型结构的微小修改都需要深入修改C++和CUDA源码，工程维护成本极高。

### 案例模拟

面试官追问：你提到算子融合，能具体说说哪些算子不能被融合吗？回答：通常计算密集型算子（如两个大型矩阵乘法）之间很难直接融合，因为它们都需要占用大量的寄存器和共享内存，强行融合会导致寄存器溢出（Register Spilling），反而降低性能。FT的核心智慧在于将访存密集型的小算子依附到计算密集型算子的前后，实现计算与访存的平衡。

### 51. 三、什么是 PagedAttention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——PagedAttention篇 / 未知](https://articles.zsxq.com/id_p22mjq881n3n.html)

### 基础知识补充

- 借鉴操作系统虚拟内存分页思想的注意力机制。
- 将连续的KV Cache切分为固定大小的非连续物理块。
- 通过Block Table维护逻辑Token到物理块的映射。

### 详细解答

PagedAttention是vLLM框架提出的一种创新的注意力计算机制，旨在解决大模型推理中的显存碎片问题。传统的Attention要求KV Cache在物理显存中是连续存储的，这导致了严重的显存浪费。PagedAttention借鉴了操作系统中虚拟内存和分页管理的思想，将每个序列的KV Cache划分为固定大小的块（KV Blocks）。每个块包含固定数量Token的键值向量。在计算注意力时，PagedAttention内核通过一张块表（Block Table）将连续的逻辑Token索引映射到非连续的物理显存块上，按块进行读取和计算。这种机制使得显存可以按需动态分配，几乎消除了显存碎片，并将显存利用率提升至90%以上，是目前大模型高吞吐部署的基石技术。

### 案例模拟

面试官追问：“PagedAttention中的Block大小（Block Size）设置多大合适？有什么权衡？” 回答：“Block Size的设置是一个典型的工程权衡。如果Block Size过大（如128），会导致最后一个Block未填满时产生较大的内部碎片；如果Block Size过小（如1），虽然碎片极小，但会导致Block Table变得非常庞大，增加寻址开销，且无法充分利用显存带宽。实践中，通常设置为16或32，能在显存利用率和访存效率之间取得最佳平衡。”

### 52. 为什么transformer块使用LayerNorm而不是BatchNorm

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- BatchNorm在Batch维度计算均值方差依赖样本数。
- LayerNorm在单个样本的特征维度上进行归一化。
- NLP中序列长度多变导致BatchNorm难以对齐特征。

### 详细解答

结论：Transformer使用LayerNorm（LN）而非BatchNorm（BN），主要是因为NLP任务中序列长度可变且词向量特征不适合跨样本归一化。 原理与对比：BN是对一个Batch内所有样本的同一维度特征求均值和方差。在CV中，像素的通道特征具有一致性，BN效果好；但在NLP中，不同句子的长度差异大，Padding会导致BN计算出现严重偏差。此外，将不同句子的同一个位置的词向量进行归一化，违背了语言的语义独立性直觉。相反，LN是对单个样本内的所有特征维度（即词向量维度）进行归一化，完全不依赖Batch Size和序列长度。 工程权衡：LN在RNN和Transformer等处理变长序列的模型中表现稳定，支持Batch Size为1的在线推理。目前大模型常采用更高效的RMSNorm来替代标准LN，进一步降低计算开销。

### 案例模拟

面试追问：“你提到RMSNorm，它和LayerNorm有什么区别？为什么大模型爱用？” 回答示例：“RMSNorm是LayerNorm的简化版。LayerNorm在归一化时需要计算均值和方差，并进行减均值操作。而RMSNorm假设均值接近于0，直接计算均方根（RMS）来进行缩放，省去了计算均值的步骤。在我们的LLaMA微调项目中，使用RMSNorm不仅保持了与LayerNorm相当的模型收敛稳定性，还将归一化层的计算耗时降低了约10%到20%，对大规模训练的吞吐量提升很有帮助。”

### 53. 二、FasterTransformer 介绍一下？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——Faster Transformer篇 / 未知](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### 基础知识补充

- 由NVIDIA开源的针对Transformer架构的底层推理加速库
- 采用C++和CUDA编写以提供极致的硬件级性能优化
- 支持张量并行和流水线并行以实现超大模型的分布式推理

### 详细解答

结论：FasterTransformer是NVIDIA开发的高性能推理加速库，专门用于优化基于Transformer架构的神经网络模型。原理：它完全脱离了PyTorch等上层框架，底层采用C++和CUDA/cuBLAS重写了Transformer的核心组件。FT不仅实现了深度的算子融合，还针对不同架构的NVIDIA GPU（如Volta、Ampere）进行了指令级优化，支持FP16、BF16和INT8等低精度计算。此外，它内置了高效的分布式通信原语（NCCL），支持张量并行（TP）和流水线并行（PP）。对比与工程权衡：FT的性能在很长一段时间内是行业标杆，尤其适合对延迟要求极苛刻的工业级部署。但由于其高度定制化，代码门槛极高，现已被NVIDIA整合并演进为更易用的TensorRT-LLM，纯FT的直接使用在逐渐减少。

### 案例模拟

业务案例模拟：在早期的机器翻译项目中，我们需要将延迟控制在100ms以内。使用PyTorch推理耗时超过200ms，无法满足SLA。我们引入了FasterTransformer，将训练好的模型权重导出为FT所需的格式，并利用其高度优化的Encoder-Decoder C++ API进行部署。最终在T4显卡上将单次推理延迟压缩到了60ms，成功上线。

### 54. 二、vLLM 如何 优化 大模型并行推理加速？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——PagedAttention篇 / 未知](https://articles.zsxq.com/id_p22mjq881n3n.html)

### 基础知识补充

- 引入PagedAttention实现非连续显存的动态分配。
- 采用操作系统虚拟内存分页思想管理KV Cache。
- 支持Block级别的内存共享以优化复杂解码算法。

### 详细解答

vLLM通过引入PagedAttention技术彻底优化了显存管理，从而大幅提升并行推理加速效果。其核心思想借鉴了操作系统的虚拟内存分页机制，将KV Cache划分为固定大小的块（Block）。在生成过程中，vLLM不再预先分配连续的大块显存，而是按需动态分配Block，并通过一张映射表（Block Table）将逻辑上连续的Token映射到物理上非连续的显存块中。这种设计将显存浪费控制在最后一个未填满的Block内，并彻底消除了外部碎片。此外，基于Block的映射机制天然支持内存共享，对于Beam Search或共享系统Prompt的场景，不同请求可以指向相同的物理Block，极大节省了显存，使得Batch Size可以成倍增加，吞吐量显著提升。

### 案例模拟

面试官追问：“vLLM的这种分页机制会引入额外的计算开销吗？” 回答：“会引入少量的寻址开销，因为在计算Attention时，需要通过Block Table去查找非连续的物理显存块，这破坏了传统连续内存访问的局部性。但是，由于LLM推理的瓶颈在于显存容量限制了Batch Size，vLLM通过分页机制省下的显存可以用来成倍扩大Batch Size。这种吞吐量的巨大收益远远覆盖了寻址带来的微小计算开销。”

### 55. 二、介绍一些 跨注意力机制（Cross-Attention）？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 标准交叉注意力：Q来自解码器，K和V来自编码器。
- 多模态交叉注意力：文本作为Q，图像或音频作为KV。
- 线性交叉注意力：通过核函数近似降低二次计算复杂度。

### 详细解答

跨注意力机制在不同领域衍生出了多种变体，以适应特定的业务需求。结论上，常见的跨注意力包括标准编解码跨注意力、多模态跨注意力以及高效跨注意力变体。在原理上，标准变体广泛应用于Transformer机器翻译，Q来自Decoder，K/V来自Encoder；在多模态领域，文本特征作为K和V，图像的潜变量作为Q，从而指导图像生成；在高效变体方面，如Perceiver架构，使用固定长度的潜向量作为Q，去交叉关注超长输入序列，从而将复杂度从平方级降至线性级。工程权衡上，选择哪种变体主要取决于输入序列的长度差异以及跨模态对齐的细粒度要求，长序列通常必须采用降维或线性化变体。

### 案例模拟

面试官追问：“在Stable Diffusion中，Cross-Attention是如何工作的？”回答：“在SD的UNet中，图像的潜特征被展平后作为Query，而文本Prompt经过编码后的特征作为Key和Value。通过Cross-Attention，图像在去噪过程中能够不断从文本特征中提取语义指导信息，从而确保生成的图像与文本描述高度一致。”

### 56. 五、Cross Attention 代码实现

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 需分别定义Q的映射层与KV的映射层以对齐特征。
- 核心步骤包括张量维度变换、点积计算与掩码处理。
- 框架中可调用多头注意力API进行快速工程实现。

### 详细解答

Cross Attention的代码实现逻辑清晰，重点在于处理好不同输入序列的维度对齐。结论上，实现跨注意力的关键是分别对目标序列和源序列进行线性映射，并确保它们在特征维度上一致以便进行点积。在原理与代码映射上，首先初始化三个独立的线性层。前向传播时，目标序列通过Q映射层得到Query，源序列通过KV映射层得到Key和Value。接着计算Q与K转置的点积，除以缩放因子，应用掩码忽略无效填充，最后经过Softmax并与V相乘。工程权衡上，手写实现便于定制相对位置编码，但在生产环境中，强烈建议使用官方优化的点积注意力算子，它底层集成了FlashAttention，能大幅降低显存占用。

### 案例模拟

面试官追问：“在代码实现中，如果Query序列长度为L，KV序列长度为S，输出的形状是什么？”回答：“输出的形状将与Query序列保持一致。具体来说，Q的形状是(B, L, D)，K和V的形状是(B, S, D)。Q和K转置相乘得到注意力权重矩阵(B, L, S)，再与V相乘，最终输出形状为(B, L, D)。这体现了跨注意力以Query为主导提取信息的特性。”

### 57. 介绍transformer算法

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 采用编码器解码器架构，完全摒弃了传统的循环神经网络结构。
- 核心组件包括多头自注意力机制和前馈神经网络模块。
- 引入残差连接和层归一化来缓解深层网络训练中的梯度消失。

### 详细解答

结论：Transformer是一种基于注意力机制的序列到序列（Seq2Seq）模型，彻底改变了自然语言处理的范式。 原理：模型由Encoder和Decoder两部分组成。Encoder包含多个堆叠的Block，每个Block由多头自注意力（Multi-Head Self-Attention）和前馈神经网络（FFN）构成，用于提取输入序列的全局特征。Decoder的Block在此基础上增加了一个交叉注意力（Cross-Attention）模块，用于融合Encoder的输出和当前解码状态。为了保证深层网络的稳定训练，每个子模块后都引入了残差连接（Residual Connection）和层归一化（Layer Normalization）。此外，由于缺乏位置信息，模型还引入了位置编码。 工程权衡：Transformer支持高度并行的矩阵运算，训练效率远超RNN。但其全局注意力的平方级复杂度导致处理超长文本时显存开销巨大。

### 案例模拟

面试追问：“Transformer中的Layer Normalization为什么不用Batch Normalization？” 回答示例：“在NLP任务中，输入序列的长度往往是不固定的，如果使用BN，会对同一个Batch内不同样本的同一位置特征进行归一化，这在序列长度差异大时会导致统计量极不稳定，且Padding部分会引入噪声。而LN是对单个样本的所有特征维度进行归一化，不依赖于Batch大小，也不受序列长度变化的影响，因此更适合处理文本等变长序列数据。”

### 58. 代码复现：亲手用PyTorch从零实现一个包含RoPE、RMSNorm、GQA的小型Llama结构。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- RoPE通过复数乘法实现绝对位置编码与相对位置信息的融合。
- RMSNorm舍弃均值中心化操作，提升计算效率与训练稳定性。
- GQA通过分组共享键值头，在推理速度与模型性能间取得平衡。

### 详细解答

从零复现Llama核心结构是检验大模型底层代码能力的最佳方式。首先，RMSNorm相比传统的LayerNorm去除了均值计算，仅保留均方根进行缩放，这在数学上简化了梯度计算，同时在工程上减少了同步开销。其次，RoPE（旋转位置编码）的实现需要精通张量维度变换，通过构造预计算的正余弦矩阵，在Attention计算前对Query和Key的特征维度进行旋转，巧妙地将相对位置信息注入内积运算中。最后，GQA（分组查询注意力）是MQA和MHA的折中方案，代码实现中需要将Key和Value的头数减少，并在计算Attention时通过repeat_interleave或expand操作将其广播到与Query相同的头数，从而大幅降低推理时的KV Cache显存占用。

### 案例模拟

面试官追问：“在实现RoPE时，如何处理不同长度的输入序列以支持动态Batch？” 回答：“在PyTorch实现中，我们通常会预先计算一个足够长（如最大上下文长度）的频率矩阵。在实际前向传播时，根据当前输入序列的实际长度和位置索引（Position IDs），从预计算的矩阵中切片提取对应的正余弦值。对于动态Batch，如果序列经过了Padding，我们需要传入具体的Position IDs张量，而不是简单地使用arange，以确保每个Token都能匹配到正确的绝对位置编码，从而保证Attention计算的准确性。”

### 59. 位置编码

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：近似题匹配：本仓库沉淀:字节跳动/LLM基础/20260312_101445_算法面经：字节豆包大模型11.13_2_AI实战领航员_来自小红书网页版.md
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 绝对位置编码直接加在输入词向量上，如正余弦或可学习编码。
- 相对位置编码作用于注意力权重计算，如ALiBi或T5 Bias。
- RoPE通过绝对位置的计算方式，巧妙实现了相对位置的效果。

### 详细解答

结论：位置编码是Transformer打破排列不变性、感知序列顺序的核心机制。主流方案分为绝对位置编码和相对位置编码。绝对位置编码（如Sinusoidal、Learned PE）直接在输入层注入位置信息，实现简单但长度外推性较差。相对位置编码（如ALiBi）通过在Attention矩阵上施加距离衰减偏置，更符合语言的局部依赖特性。目前最流行的是旋转位置编码（RoPE），它借助欧拉公式，将Token位置映射为复平面上的旋转角度。在Q和K相乘前进行旋转，使得内积结果仅与相对位置相关。RoPE兼顾了绝对位置的全局感知与相对位置的局部优势，且无需修改Attention计算图，配合动态NTK等插值技术具有极强的长度外推能力。

### 案例模拟

面试官追问：“RoPE在计算时是如何保证效率的？”回答：“在实际工程中，我们不会真正去构建庞大的旋转矩阵进行矩阵乘法，因为这会带来巨大的显存和计算开销。相反，我们会利用复数乘法的性质，将其转化为两两维度的交错相乘与加减操作，即通过Hadamard积（逐元素乘法）来实现。这种方式将时间复杂度从O(d^2)降到了O(d)，在CUDA算子层面极其高效。”

### 60. 位置编码（Positional Encoding）：从绝对到相对的演变

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 绝对位置编码如正弦波或可学习参数，直接加在输入词向量上。
- 相对位置编码如ALiBi，通过在注意力分数矩阵上添加距离偏置。
- 旋转位置编码RoPE结合两者优势，通过绝对旋转实现相对位置表达。

### 详细解答

结论：位置编码经历了从绝对位置编码（Sinusoidal/Learned）到相对位置编码（Relative Bias/ALiBi），再到目前主流的旋转位置编码（RoPE）的演进。原理与对比：Transformer本身具有排列不变性，必须引入位置信息。早期的绝对位置编码直接将位置向量与Token Embedding相加，简单但缺乏对Token间相对距离的直接感知，且长度外推性差。相对位置编码（如T5、ALiBi）直接在Attention的QK点积结果上施加与距离相关的惩罚项，外推性好但计算较复杂。工程权衡：RoPE（旋转位置编码）巧妙地利用复数乘法，在Query和Key的向量空间中进行旋转。它在输入阶段施加绝对位置信息，但在点积计算时自然推导出了相对位置的内积形式。RoPE不仅计算高效，且通过修改旋转基数（Base）极易实现上下文长度的扩展（如线性插值），成为LLaMA等现代大模型的标配。

### 案例模拟

面试官追问：RoPE（旋转位置编码）是如何实现长度外推的？常见的插值方法有哪些？ 回答示例：RoPE通过高频维度捕捉局部关系，低频维度捕捉长距离关系。当输入长度超过训练长度时，低频维度的旋转角度会超出训练分布。常见的扩展方法是位置插值（PI），即将未见过的长位置映射回训练时的短位置范围内。进阶方法如NTK-Aware插值，它保持高频维度不变（保留高分辨率局部信息），仅对低频维度进行缩放，从而在不微调或少微调的情况下实现优秀的长度外推。

### 61. 六、Cross Attention 应用场景

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 机器翻译：解码器通过跨注意力对齐源语言词汇。
- 多模态生成：如扩散模型中的图文特征深度融合。
- 语音识别：将声学特征序列与文本解码序列进行对齐。

### 详细解答

Cross Attention在需要多源信息交互的AI任务中占据统治地位。结论上，其核心应用场景集中在序列到序列文本生成、多模态理解与生成以及跨模态对齐任务中。在原理应用上，机器翻译是最经典的场景，解码器在生成译文时通过跨注意力实时查阅编码器的源语言上下文。在多模态领域，如扩散模型，文本特征作为KV指导图像Q的去噪过程；在视觉问答中，问题文本作为Q去检索图像特征中的关键区域。工程权衡方面，在这些场景中，由于不同模态的序列长度差异巨大，直接计算跨注意力会导致显存爆炸，因此常采用重采样技术或将图像特征池化降维后再进行跨注意力计算，以保证推理的实时性。

### 案例模拟

业务案例模拟：“在开发视频字幕生成系统时，我们提取了视频的逐帧视觉特征和音频特征。为了生成准确的描述，我们在文本解码器中引入了双路跨注意力：一路用文本Q去查询视频帧KV，另一路查询音频KV。通过这种门控机制，模型能动态决定当前生成的词是更依赖画面动作还是背景声音，显著提升了字幕准确度。”

### 62. 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 多头机制是架构增强，跨注意力是信息来源的定义。
- 跨注意力通常以多头注意力的形式进行工程实现。
- 多头机制将特征切分到多个子空间并行计算注意力。

### 详细解答

Cross Attention与多头注意力并非对立概念，而是不同维度的机制设计。结论上，Cross Attention描述的是Q与K/V的数据来源不同，而多头注意力描述的是注意力计算的并行结构方式，两者在实际模型中通常是结合使用的。在原理上，多头机制通过将高维特征切分为多个低维子空间，让模型能够同时关注不同位置、不同维度的特征信息；而将这种多头机制应用于跨注意力时，就构成了多头跨注意力，使得解码器能够从编码器输出的多个不同语义维度提取信息。在工程权衡中，多头机制虽然增加了张量重塑的开销，但由于各个头可以高度并行计算，整体计算效率极高，是现代大模型特征融合的标配。

### 案例模拟

面试官追问：“多头机制在Cross Attention中有什么具体好处？”回答：“在机器翻译中，一个目标词可能与源句子的多个词相关，且相关的原因不同（如语法依赖、语义指代）。多头机制允许Cross Attention的不同头分别关注这些不同的方面。比如一个头关注动宾关系，另一个头关注时态修饰，从而实现更丰富、更准确的跨语言特征对齐。”

### 63. 复数域的旋转诠释：候选人需要展示如何利用复数乘法将位置信息注入。给定词向量 ，RoPE通过乘以一个旋转矩阵 来实现位置编码：。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 将二维向量映射为复数，利用欧拉公式进行旋转。
- 旋转矩阵是正交矩阵，保持了向量的模长不发生改变。
- 高维向量两两分组，在多个二维子空间内独立旋转。

### 详细解答

结论：RoPE的核心数学思想是将位置信息的注入转化为复数域上的旋转操作。 原理：给定一个词向量，RoPE首先将其按维度两两分组，每组视为复平面上的一个复数。对于位置为m的Token，RoPE通过乘以复数 $e^{im\theta}$ 来实现旋转，其中 $\theta$ 是预设的频率参数。在实数域中，这等价于乘以一个二维旋转矩阵。由于旋转操作不改变向量的模长，它只改变了向量的方向，从而将绝对位置信息优雅地编码进向量的相位中。 对比：与直接相加的绝对位置编码相比，复数旋转是一种乘法交互，它在后续计算内积时能够利用三角函数的和差化积公式，自然地消去绝对位置，提取出相对位置。 工程权衡：在代码实现中，为了避免复杂的复数运算和稠密矩阵乘法，通常将旋转矩阵展开为逐元素操作：将向量分为实部和虚部（或交错维度），分别与 $\cos(m\theta)$ 和 $\sin(m\theta)$ 进行哈达玛积并相加减，极大提升了计算效率。

### 案例模拟

面试官追问：请简述一下RoPE中频率参数 $\theta$ 是如何设置的？ 回答示例：在RoPE中，不同二维子空间的旋转频率 $\theta_i$ 是不同的，通常按照指数衰减的方式设置，公式为 $\theta_i = 10000^{-2i/d}$，其中 $d$ 是向量维度。这种设计使得低维度（高频）对局部相对位置非常敏感，而高维度（低频）旋转极慢，能够捕捉长距离的全局依赖关系，类似于多分辨率的位置感知机制。

### 64. 外推性与NTK-Aware Scaling：随着长上下文需求的爆发，RoPE的扩展性成为必考题。面试中常出现的情境是：“我们将模型上下文窗口从4k扩展到32k，直接线性插值会有什么问题？NTK-Aware Scaling是如何解决高频信息丢失问题的？”数据表明，能够解释清楚“高频分量旋转速度过快导致插值混叠，而NTK方法通过非线性调整基频来平衡高低频分量的分辨率”这一深层机制的候选人，获得P7+评级的概率显著增加。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 线性插值直接压缩位置索引会导致高频信息严重丢失。
- NTK-Aware Scaling通过非线性调整基频平衡分辨率。
- 高频分量负责局部特征，低频分量负责长距离依赖关系。

### 详细解答

结论：在长上下文扩展中，直接线性插值（PI）会导致高频信息混叠，而NTK-Aware Scaling通过非线性调整基频完美解决了这一问题。原理解释：RoPE的各个维度对应不同的旋转频率。低频分量旋转慢，负责捕捉长距离依赖；高频分量旋转快，负责局部精确匹配。当窗口从4k扩展到32k时，直接线性插值会将所有频率等比例压缩。这使得原本就旋转极快的高频分量变得更加密集，导致相邻位置的向量在空间中难以区分（高频混叠），模型丧失局部精准度。NTK方法基于神经正切核理论，对高频分量保持原有旋转速度（不插值或少插值），仅对低频分量进行插值扩展，从而在不微调或少微调的情况下，兼顾了局部高频分辨率与全局低频长程感知。

### 案例模拟

业务案例模拟：在开发长文本法律文档分析大模型时，我们发现将上下文强行扩展到32k后，模型虽然能回答全局性问题，但对具体条款的细节提取经常张冠李戴。排查发现是线性插值导致高频位置信息丢失。我们随后引入了NTK-Aware Scaling，通过动态调整Base值，使得模型在长文本下依然保持对相邻词汇的精确区分，细节提取准确率提升了40%。

### 65. 多头注意力机制MHA是Transformer模型中的核心组件, KV Cache和GQA优化的核心思想？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- KV Cache通过缓存历史Token的键值向量减少重复计算。
- GQA将查询头分组并共享键值头以降低显存访问压力。
- GQA是多头注意力与多查询注意力之间的有效折中方案。

### 详细解答

KV Cache和GQA的核心思想都是为了缓解大模型推理时的显存和访存瓶颈。结论上，KV Cache用空间换时间，GQA则通过结构优化减少空间占用。原理方面，在自回归生成中，每个新Token的计算都依赖历史Token，KV Cache将历史的Key和Value缓存下来，避免每次生成都重新计算整个序列，从而将时间复杂度从二次降为线性。然而，随着序列变长，KV Cache的显存占用急剧增加。为此，GQA（分组查询注意力）将多个Query头划分为若干组，每组共享一对Key和Value头。对比MHA（多头）和MQA（单键值头），GQA在大幅减少KV Cache显存占用和内存带宽压力的同时，保留了比MQA更强的模型表达能力，是当前大模型推理的标配。

### 案例模拟

面试官追问：在部署长文本模型时，如何进一步优化KV Cache的显存占用？ 回答：在实际业务中，面对超长上下文，单纯依赖GQA还不够。我们会采用PagedAttention技术，将KV Cache划分为固定大小的块，像操作系统管理虚拟内存一样进行非连续的显存分配，极大减少了显存碎片。此外，还可以结合KV Cache量化（如INT8格式）以及Token丢弃策略，在保证生成质量的前提下，将显存占用再降低一半以上。

### 66. 推理优化技术 Flash Attention 的作用是什么？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- Flash Attention是一种硬件感知的精确注意力计算算法。
- 通过分块计算减少GPU的SRAM与HBM之间的内存读写次数。
- 能够显著降低显存占用并提升长序列场景下的计算速度。

### 详细解答

结论：Flash Attention 是一种旨在解决 Transformer 长序列训练和推理瓶颈的高效注意力机制实现，其核心作用是大幅降低显存占用并提升计算速度。 原理与解释：标准注意力机制的计算复杂度与序列长度呈平方关系，且在计算过程中需要实例化庞大的注意力分数矩阵（S矩阵），导致频繁的GPU显存（HBM）读写，成为性能瓶颈。Flash Attention 引入了硬件感知设计，利用分块计算（Tiling）和重计算（Recomputation）技术，将计算过程融合在一个CUDA Kernel中。它将数据加载到高速的SRAM中完成计算，避免了中间结果写回HBM。 工程权衡：这种方法不仅将显存复杂度从平方级降至线性级，还大幅减少了内存带宽限制带来的延迟。它在不牺牲任何模型精度（精确计算而非近似）的前提下，使得大模型能够轻松处理几十K甚至上百K的超长上下文。

### 案例模拟

面试官追问：Flash Attention在推理阶段和训练阶段的作用侧重点有什么不同？ 回答：在训练阶段，Flash Attention的核心价值在于节省显存（避免存储庞大的注意力中间矩阵）和加速前向/反向传播，使得单卡可以塞下更长的序列。而在推理阶段（尤其是Prefill阶段），它的主要作用是缓解内存带宽瓶颈（Memory-bound），通过减少HBM访存来加速长Prompt的计算。对于Decode阶段，由于每次只生成一个Token，通常需要配合PagedAttention等技术来优化KV Cache。

### 67. 核心架构演进：Transformer组件的细粒度考察

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 归一化层从Post-LN演进为Pre-LN，再到目前主流的RMSNorm。
- 激活函数从ReLU演进为GeLU，再到目前大模型标配的SwiGLU。
- 注意力机制从标准MHA演进为MQA与GQA，以优化推理显存占用。

### 详细解答

结论：Transformer架构在演进中，其细粒度组件经历了从追求理论完备到追求工程效率与训练稳定性的转变。原理与对比：1. 归一化：早期Post-LN容易导致深层梯度消失，Pre-LN解决了训练稳定性但可能影响性能。如今LLaMA等模型广泛采用RMSNorm，去除了均值计算，仅保留方差，计算速度提升且效果相当。2. 激活函数：SwiGLU取代了GeLU，通过门控机制引入动态信息流，虽然增加了参数量，但在同等计算量下表现更好。3. 注意力：为了解决长文本推理时KV Cache显存爆炸的问题，GQA（分组查询注意力）成为折中方案，既保留了MHA的表达能力，又接近MQA的推理速度。工程权衡：这些组件的替换本质上都是在模型表达能力、训练稳定性与硬件执行效率（如内存带宽限制）之间寻找最优解。

### 案例模拟

面试官追问：为什么现在的大模型普遍采用RMSNorm而不是传统的LayerNorm？ 回答示例：LayerNorm需要计算输入的均值和方差，并进行减均值除方差的操作。研究发现，LayerNorm中对模型性能起决定性作用的是方差缩放（缩放不变性），而均值平移（平移不变性）作用微乎其微。RMSNorm去掉了均值计算，直接用均方根进行归一化，减少了计算开销，提升了约10%-50%的前向速度，且模型效果不降反升。

### 68. 模型问题：为什么现在的llm大模型主要都是用RoPE位置编码而非其他？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- RoPE通过旋转矩阵将绝对位置信息注入词嵌入向量中。
- 结合了绝对位置编码的实现与相对位置编码的优良特性。
- 具有良好的外推性，能够处理超过训练长度的文本序列。

### 详细解答

结论：RoPE（旋转位置编码）因其兼具绝对位置的计算效率和相对位置的表达能力，且具备优秀的长度外推性，成为当前LLM的主流选择。 原理与对比：RoPE的核心思想是将词嵌入向量的维度两两分组，视为复数平面上的向量，然后通过乘以一个与位置相关的旋转矩阵来编码位置信息。相比于传统的正弦绝对位置编码，RoPE在计算注意力内积时，自然地推导出了相对位置的衰减特性；相比于ALiBi等相对位置编码，RoPE不需要修改注意力矩阵，计算更高效。 工程权衡：RoPE不仅在理论上优雅，在工程上还可以通过NTK-Aware等插值方法轻松实现上下文窗口的扩展（如从8K扩展到32K），极大地降低了长文本大模型的训练成本。

### 案例模拟

面试官追问：RoPE在进行长文本外推时，常用的线性插值法有什么缺点？ 回答示例：线性插值法直接将未见过的长位置索引按比例缩放到训练时的短位置范围内。其缺点在于，它对所有频率的维度进行了同等程度的缩放，导致高频维度（负责局部细节）的相对位置分辨率下降，模型容易丢失局部精确信息。因此，现在更倾向于使用NTK-Aware或YaRN等方法，对高低频维度进行差异化处理。

### 69. 模型问题：如何理解DETR中的object query的概念，要为 cross attention 提供更好的位置先验该如何设计模型？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/VisionPerception/Reference.md)

### 基础知识补充

- 对象查询：一组可学习的嵌入向量，代表模型对目标的提问。
- 交叉注意力：查询向量与图像特征交互，动态聚合目标信息。
- 位置先验优化：将查询显式建模为锚框坐标，加速模型收敛。

### 详细解答

结论：DETR中的Object Query本质上是一组可学习的嵌入向量，充当了“目标探测器”的角色。为了提供更好的位置先验，可以通过显式引入Anchor Box坐标来重构Query。 原理与对比：在原始DETR中，Object Query是随机初始化的隐式向量，通过Cross Attention与全局图像特征交互来寻找目标。由于缺乏明确的物理意义和位置约束，模型需要漫长的训练才能学会关注特定区域。相比之下，DAB-DETR等改进模型将Query解耦为内容部分和位置部分，并将位置部分显式定义为4D的Anchor Box（中心点坐标和宽高）。 工程权衡：引入Anchor Box作为位置先验，使得Cross Attention能够直接利用坐标信息进行局部特征采样（如Deformable DETR），极大地缩小了搜索空间。这种设计不仅将DETR的收敛速度提升了数倍，还增强了模型对多尺度和小目标的检测能力，是目前Transformer检测模型的主流工程范式。

### 案例模拟

面试官追问：除了DAB-DETR引入Anchor Box，还有其他提供位置先验的方法吗？ 回答示例：有的。例如Conditional DETR通过将Query解耦为内容和空间两部分，利用空间Query生成条件交叉注意力权重，从而缩小搜索区域。另外，Two-Stage DETR（如Deformable DETR的变体）直接利用编码器输出的密集特征图生成高质量的初始Proposal，将其作为解码器Query的位置先验。这种基于图像特征动态生成先验的方法，比静态学习的Anchor更加精准。

### 70. 用户行为序列：DIN（Deep Interest Network）引入了Attention机制，计算用户历史行为与当前候选商品的关联度。DIEN（Deep Interest Evolution Network）进一步引入GRU来建模兴趣随时间的演化。面试官会要求手写Attention Unit的代码实现。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- DIN通过注意力机制计算历史行为与当前候选商品的关联度。
- DIEN引入GRU网络，用于建模用户兴趣随时间序列的演化。
- Attention单元包含全连接层，输出归一化或非归一化权重。

### 详细解答

用户行为序列建模是推荐系统精准捕捉用户意图的核心。结论上，DIN和DIEN通过引入注意力机制和时序网络，极大地提升了模型对用户动态兴趣的表达能力。DIN的原理在于打破了传统定长向量的限制，针对不同的候选商品，动态计算用户历史行为的权重。其Attention Unit通常将User Behavior和Candidate Item的Embedding进行拼接、相减和内积，再通过MLP输出权重。值得注意的是，DIN为了保留用户兴趣强度，通常不使用Softmax进行归一化。DIEN则更进一步，利用GRU提取行为序列的隐状态，并设计AUGRU（带有注意力机制的GRU）将DIN的注意力权重直接作用于GRU的更新门，从而精准刻画兴趣的演化轨迹。工程上，DIEN的串行计算会导致高延迟，常需进行序列截断或并行化改造。

### 案例模拟

面试官要求：“请手写DIN中Attention Unit的核心代码实现。”回答模拟：“可以使用PyTorch实现。首先将Query（候选商品）和Keys（历史行为）的维度对齐。然后将Query扩展到与Keys相同的序列长度。接着，将Query、Keys、Query与Keys的差值、Query与Keys的逐元素乘积拼接在一起，送入一个多层感知机（MLP）。MLP的最后一层输出维度为1，代表每个历史行为的权重。最后将权重与Keys相乘并求和，得到最终的序列特征表示。”

### 71. 绝对位置编码: 8次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 为每个位置生成固定的或可学习的独立位置向量。
- Transformer原论文采用正弦和余弦函数固定编码。
- 无法直接体现两个Token之间的相对距离关系。

### 详细解答

结论：绝对位置编码是Transformer早期最常用的位置信息注入方式，分为固定式和可学习式两种。 原理：由于自注意力机制本身是排列不变的，必须显式引入位置信息。固定式绝对位置编码利用不同频率的正余弦函数生成位置向量，直接与词嵌入相加；可学习式则将位置视作特殊的Token，通过Embedding层随机初始化并随模型一起训练更新。 对比：固定式无需训练，理论上支持任意长度，但实际外推效果差；可学习式在特定长度内表现更好，如BERT和GPT-2所采用，但完全无法处理超过训练最大长度的文本。两者共同的缺点是，在计算Attention时，无法自然地表达词与词之间的相对距离。 工程权衡：在短文本或固定长度任务中，可学习绝对位置编码简单高效；但在当前追求超长上下文的大模型时代，已逐渐被RoPE等替代。

### 案例模拟

面试官追问：为什么绝对位置编码直接与词向量相加，而不是拼接？ 回答示例：拼接会增加模型的维度，导致后续线性层的参数量翻倍，增加计算开销。相加操作在数学上可以近似看作是在高维空间中将词义信息与位置信息进行正交叠加。由于高维空间的稀疏性，词向量和位置向量近似正交，相加后神经网络依然能够通过线性变换将这两种信息有效解耦并分别处理。

### 72. 考察点：原理是小模型快速生成 个token，大模型并行验证这 个token。如果验证通过，则一次性接受多个token，从而打破Transformer解码的串行限制。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 草稿模型自回归生成K个候选Token，速度极快且开销低。
- 目标大模型通过一次前向传播并行验证这K个候选Token。
- 拒绝采样机制确保最终输出分布与目标大模型完全一致。

### 详细解答

结论：投机采样的核心机制是“小模型猜测，大模型验证”，通过并行化打破Transformer的串行解码限制。原理：在传统的自回归生成中，生成K个Token需要大模型进行K次前向传播，访存开销巨大。投机采样让小模型先快速生成K个Token（草稿），然后大模型将这K个Token作为输入，执行一次并行的前向传播，计算出每个位置的真实概率分布。通过对比小模型和大模型的概率分布，利用拒绝采样（Rejection Sampling）算法决定接受或拒绝。如果第i个Token被拒绝，则大模型根据自身分布重新采样第i个Token，并丢弃后面的草稿。工程权衡：K值的设定是关键，K过大导致验证失败率上升，浪费计算；K过小则无法充分掩盖大模型的访存延迟。

### 案例模拟

业务案例模拟：在部署千亿参数大模型提供实时对话服务时，首字延迟和吞吐量是核心指标。我们引入了投机采样，将K值动态设置为4到6。当用户询问常识性问题时，小模型预测准确率极高，大模型一次性接受5个Token，生成速度提升2.5倍；而在处理复杂代码生成时，接受率下降，系统会自动调低K值以减少无效计算，保障整体吞吐。

### 73. 表1：Transformer核心组件考察绝对频次

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 自注意力机制是Transformer考察频次最高的核心组件。
- 位置编码用于补充序列顺序信息，常见有绝对与相对位置编码。
- 残差连接与层归一化是保证深层网络稳定训练的关键组件。

### 详细解答

Transformer核心组件的考察频次直接反映了其在现代AI模型中的基石地位。结论上，Self-Attention、Position Encoding、LayerNorm和FFN是面试中最常被拆解提问的四大组件。自注意力机制（Self-Attention）是重中之重，面试官常要求推导其$O(N^2)$的时间复杂度，并解释除以$\sqrt{d_k}$防止梯度消失的原理。位置编码（PE）的考察重点在于为何需要它（因为Attention本身是排列不变的），以及RoPE（旋转位置编码）等现代大模型常用方案的数学推导。LayerNorm与BatchNorm的对比也是经典考点，主要涉及NLP任务中序列长度不一致导致的统计量波动问题。工程权衡上，面试官常会结合KV Cache机制，考察这些组件在推理阶段的显存占用和计算加速策略。

### 案例模拟

面试官追问：“为什么Transformer中要使用LayerNorm而不是BatchNorm？如果把LayerNorm放在Attention之前（Pre-LN）和之后（Post-LN）有什么区别？”回答：“BatchNorm在Batch维度计算，NLP中句子长度不一，Padding会导致统计量极不稳定；LayerNorm在特征维度计算，不受序列长度影响。Post-LN在早期模型中使用，但深层网络容易出现梯度消失，需要Warmup；Pre-LN将LN放在残差块内部，使得深层梯度更稳定，无需Warmup即可训练更深的网络。”

### 74. 跨注意力机制（Cross-Attention）篇

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 跨注意力机制用于融合两个不同来源的序列特征。
- 核心公式为Softmax(QK^T/sqrt(d))V。
- Q来自目标序列，而K和V来自源序列，实现信息对齐。

### 详细解答

跨注意力机制（Cross-Attention）是Transformer架构中用于处理多源信息融合的核心组件。结论上，它通过将一个序列作为查询（Query），另一个序列作为键（Key）和值（Value），实现了不同模态或不同语言序列之间的特征对齐与交互。在原理上，与自注意力机制中Q、K、V同源不同，跨注意力的Q通常来自解码器（如生成文本），而K和V来自编码器（如输入文本或图像）。这种非对称结构使得模型能够根据当前生成的目标上下文，动态地去源端提取相关信息。在工程权衡中，跨注意力的计算复杂度与两个序列长度的乘积成正比，因此在处理长序列或高分辨率图像时，常需要引入局部注意力或线性注意力来降低显存占用和计算开销，或者通过降维手段压缩KV的序列长度以提升推理速度。

### 案例模拟

面试官追问：“在多模态大模型中，Cross-Attention通常放在什么位置？”回答：“在多模态架构（如Flamingo）中，通常在LLM的自注意力层之间插入交叉注意力层。图像特征作为K和V，文本特征作为Q。这样可以保持预训练LLM的权重冻结，仅通过训练交叉注意力层来实现视觉信息的注入，既节省算力又能有效融合多模态特征。”

### 75. 这里的深层洞察是，Transformer的激活值分布通常具有各向同性，均值偏移并不携带核心语义信息，而缩放不变性才是归一化的核心贡献。省去计算均值和减均值的操作，在大规模张量运算中能带来显著的Kernel性能提升（约10%-40%的加速，取决于具体算子实现）。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 激活值分布的各向同性使得均值偏移不携带核心语义信息。
- 归一化层的核心贡献在于缩放不变性，而非严格的零均值。
- 简化均值计算可减少显存访问，带来10%到40%的算子加速。

### 详细解答

结论：在Transformer中省略归一化的均值计算不仅不影响性能，反而能大幅提升训练速度。原理层面，深层洞察表明Transformer的激活值分布通常具有各向同性，这意味着特征向量在各个方向上的分布相对均匀，均值偏移并不携带核心语义信息。相比之下，缩放不变性（即控制方差）才是防止梯度爆炸或消失、维持训练稳定性的核心贡献。工程权衡上，传统的LayerNorm需要两次遍历数据（一次算均值，一次算方差），而省略均值计算的RMSNorm只需一次遍历。这种简化在大规模张量运算中极大地减少了显存带宽占用，能够带来约10%-40%的Kernel性能加速，具体取决于算子实现。

### 案例模拟

业务案例模拟：在自研大模型训练框架时，发现LayerNorm算子耗时占比较高。优化方案是将所有LayerNorm替换为RMSNorm，并使用Triton编写融合算子（Fused RMSNorm）。通过省去均值计算和减少Global Memory的读写次数，该层的计算延迟降低了约30%，整体模型训练吞吐量提升了约3%，且在千亿参数规模下Loss收敛曲线与原版完全一致。

### 76. 连接层（Projector）：如何将ViT输出的图像特征对齐到LLM的文本空间？简单方案是Linear Projection（线性映射）。进阶方案是Q-Former（BLIP-2）或C-Abstractor（Honeybee），使用一组Learnable Queries通过Cross-Attention提取图像特征。阿里云面试官倾向于询问Qwen-VL的做法，即通过C-Abstractor压缩视觉token数量，以减少对LLM上下文窗口的占用。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 线性映射直接转换维度，但无法减少视觉Token的数量。
- Q-Former利用可学习Query通过交叉注意力提取核心视觉特征。
- Qwen-VL采用类似C-Abstractor的结构大幅压缩视觉上下文。

### 详细解答

结论：将ViT特征对齐到LLM文本空间，主流方案从简单的线性映射演进到了以Q-Former和C-Abstractor为代表的注意力压缩网络，核心目的是在保留视觉信息的同时减少对LLM上下文窗口的占用。原理解释：简单方案如Linear Projection直接对特征进行仿射变换，这会导致高分辨率图像产生海量Token，极大增加LLM的计算负担。进阶方案则引入了Learnable Queries。例如Qwen-VL采用的视觉接收器（类似C-Abstractor），通过单层交叉注意力机制，让固定数量的Query去主动提取ViT输出的网格特征。工程权衡：这种设计的优势在于实现了视觉Token的极致压缩（例如将数千个Token压缩至256个），显著降低了多模态对话时的显存占用和推理延迟。尽管引入复杂Projector会增加初期的对齐训练难度，但从长远来看，它是支持多图输入和长视频理解的必由之路。

### 案例模拟

面试官追问：“在训练Qwen-VL这种带有复杂Projector的模型时，通常分为几个阶段？”回答示例：“通常分为三个阶段。第一阶段是模态对齐，冻结ViT和LLM，仅使用海量图文对训练Projector；第二阶段是多任务预训练，解冻LLM和Projector，输入更高分辨率的图像和多样化数据；第三阶段是指令微调（SFT），全参数或部分参数微调，提升模型在VQA、多轮对话等具体任务上的指令遵循能力。”

### 77. 面试官会特别关注候选人是否理解RoPE设计的核心目标：通过旋转操作，使得两个token之间的注意力分数（Attention Score）仅依赖于它们的相对距离 ，而非绝对位置。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- 注意力分数由Query和Key向量的内积计算得出。
- 旋转矩阵的正交性使得内积运算可以转化为角度差。
- 绝对位置m和n的差值m-n直接决定了最终的内积结果。

### 详细解答

结论：RoPE通过巧妙的旋转矩阵设计，使得Query和Key在进行内积计算时，绝对位置变量被抵消，只留下相对位置变量。 原理：在自注意力机制中，注意力分数依赖于 $q_m$ 和 $k_n$ 的内积。在RoPE中，$q_m$ 是原始查询向量乘以旋转矩阵 $R_m$，$k_n$ 是原始键向量乘以 $R_n$。当计算内积 $\langle R_m q, R_n k \rangle$ 时，由于旋转矩阵的转置等于其逆矩阵，且 $R_m^T R_n = R_{n-m}$，内积结果最终化简为 $\langle q, R_{n-m} k \rangle$。 对比：这意味着，无论这两个Token在句子中的绝对位置是 (10, 15) 还是 (100, 105)，只要它们的相对距离都是 5，它们位置编码对注意力分数的贡献模式就是完全相同的。这完美契合了相对位置编码的初衷，同时又避免了构建庞大的相对位置偏置矩阵。 工程权衡：这种设计使得模型在训练时只需处理绝对位置索引，推理时却能享受相对位置平移不变性的红利，是目前兼顾计算效率与表达能力的最佳方案。

### 案例模拟

面试官追问：如果相对距离 $m-n$ 很大，RoPE的内积会有什么表现？ 回答示例：根据RoPE的数学推导和频率衰减特性，当相对距离 $m-n$ 增大时，由于各个维度的高频震荡相互叠加抵消，内积的期望值会呈现出一种长程衰减（Long-term Decay）的趋势。这意味着距离越远的Token，它们之间的注意力分数倾向于越小，这非常符合自然语言中局部上下文相关性更强的先验知识。

### 78. 面试官通常从“为什么RoPE比绝对位置编码更适合长文本？”这一问题切入。优秀的回答不能止步于定性描述，必须进入数学证明层面。

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里系大模型常见问题汇总 / 阿里巴巴](https://www.nowcoder.com/discuss/848942791164981248)

### 基础知识补充

- RoPE通过内积自然导出相对位置，符合语言规律。
- 绝对位置编码在超出训练长度时会遇到未定义向量。
- RoPE的频率衰减特性为长文本外推提供了数学基础。

### 详细解答

结论：RoPE比绝对位置编码更适合长文本，根本原因在于其通过绝对位置的旋转实现了相对位置的表达，且具备更好的外推潜力。 原理：在自然语言中，词与词之间的相对距离比它们在句子中的绝对位置更重要。绝对位置编码在训练时只见过有限长度的位置向量，当推理长度超过训练长度时，模型无法处理新的绝对位置索引。而RoPE在计算注意力分数时，两个Token的内积仅取决于它们的相对距离。 对比：虽然绝对位置编码可以通过正弦函数直接生成新位置的向量，但模型并未学习过这些新位置与旧位置的交互；RoPE则保证了无论绝对位置多大，只要相对距离在训练范围内，其注意力模式就是一致的。 工程权衡：为了进一步提升RoPE的长文本能力，工程上引入了位置插值（PI）和NTK感知插值。这些方法利用RoPE的数学形式，通过缩放位置索引或调整基频，以极低的微调成本实现了上下文窗口的数倍扩展。

### 案例模拟

面试官追问：既然RoPE依赖相对距离，那它和直接使用相对位置偏置（如ALiBi）相比有什么优劣？ 回答示例：ALiBi通过在注意力矩阵上直接加上与相对距离成正比的负偏置，外推性极强，且无需位置插值。但ALiBi强行假设距离越远注意力越弱，这在某些需要长距离精确检索的任务（如大海捞针）中表现不佳。RoPE保留了更灵活的注意力分布能力，虽然原生外推性不如ALiBi，但结合NTK插值后，综合表现更优。
