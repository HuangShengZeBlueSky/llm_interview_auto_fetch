# 融合题库：Transformer结构

> 已经把 GitHub 题库和真实面经合并去重。
> 本页共 51 道题，按同题合并后的题卡展示。

### 1. 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？

- 主标签：Transformer结构
- 来源条数：2
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 核心数学公式相同：均采用缩放点积注意力机制。
- 计算流程一致：都包含线性映射、归一化和加权求和。
- 均支持多头机制，以捕捉不同特征子空间的语义信息。

### 详细解答

结论：Cross Attention 和 Self Attention 在底层数学原理、计算图结构以及优化策略上具有高度的相似性。 原理与对比：从数学公式来看，两者都遵循 Attention(Q, K, V) = Softmax(QK^T / sqrt(d))V。它们都需要将输入通过权重矩阵映射为 Query、Key 和 Value 空间；都利用点积计算相似度，并通过 Softmax 转化为概率分布；最后对 Value 进行加权聚合。此外，两者都可以无缝扩展为多头注意力（Multi-Head Attention），并且都可以应用 FlashAttention 等底层算子优化技术来加速计算。 工程权衡：由于计算逻辑相同，在深度学习框架（如 PyTorch）中，这两种注意力通常可以复用同一套底层 API（如 nn.MultiheadAttention）。开发者只需通过传入不同的参数（Q 和 KV 是否来自同一张张量）即可实现两者的切换，极大地降低了工程实现成本。

### 案例模拟

面试官追问：既然公式一样，在 PyTorch 中如何用同一个接口实现这两种 Attention？ 回答示例：在 PyTorch 的 nn.MultiheadAttention 中，前向传播接收 query, key, value 三个参数。如果实现 Self Attention，我们传入的这三个参数是同一个张量（例如 mha(x, x, x)）；如果实现 Cross Attention，我们将目标序列作为 query，源序列作为 key 和 value（例如 mha(target, source, source)）。底层计算逻辑完全复用。

### 2. Attention计算复杂度以及如何改进

- 主标签：Transformer结构
- 来源条数：2
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

- 代码中的to_qkv()函数，即用于生成q、k、v三个特征向量 !Alt python self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False) self.to_out = nn.Linear(inner_dim, dim) - 在标准的Transformer中，Attention计算的时间复杂度为O(N^2)，其中N是输入序列的长度。为了降低计算复杂度，可以采用以下几种方法： - 使用自注意力机制，减少计算复杂度。自注意力机制不需要计算输入序列之间的交叉关系，而是计算每个输入向量与自身之间的关系，从而减少计算量。 - 使用局部注意力机制，只计算输入序列中与当前位置相关的子序列的交互，从而降低计算复杂度。 - 采用基于近似的方法，例如使用随机化和采样等方法来近似计算，从而降低计算复杂度。 - 使用压缩注意力机制，通过将输入向量映射到低维空间来减少计算量，例如使用哈希注意力机制和低秩注意力机制等。

### 案例模拟

面试表达可以这样组织：先用一句话回答“Attention计算复杂度以及如何改进”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 3. 1 传统 Attention 存在哪些问题？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 计算复杂度为序列长度的平方，长文本开销大。
- 显存占用随序列长度呈平方级增长，易 OOM。
- 缺乏对相对位置信息的原生感知，需位置编码。

### 详细解答

传统的多头自注意力机制（Multi-Head Attention）虽然在捕捉全局依赖方面表现出色，但在实际应用中面临着严峻的挑战。结论上，其最核心的问题是计算和显存复杂度随序列长度呈平方级增长（$O(N^2)$）。原理上，在计算注意力分数矩阵时，序列中的每个 Token 都需要与所有其他 Token 进行点积运算，这导致当上下文窗口扩展到 32K 甚至 128K 时，计算量和激活值的显存占用会爆炸式增加。此外，传统 Attention 在自回归解码阶段，每次生成新词都需要重新加载历史的 KV Cache，导致严重的显存带宽瓶颈（Memory-bound）。工程权衡上，为了处理长文本，必须在注意力机制的精确度与计算效率之间做出妥协，这也催生了后续众多针对 Attention 的优化。

### 案例模拟

面试官追问：除了平方复杂度，传统 Attention 在推理时还有什么痛点？ 回答：推理时的最大痛点是 KV Cache 的显存占用和访存带宽瓶颈。在生成阶段，虽然计算量只是矩阵向量乘法，但需要频繁从显存读取庞大的 KV Cache。对于大并发服务，这会导致极低的计算访存比，GPU 算力无法跑满。因此工程上必须引入 PagedAttention 或改用 MQA/GQA。

### 4. 1. 如何 利用 transformers 加载 Bert 模型？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / transformers 操作篇 / 未知](https://articles.zsxq.com/id_rsll7gsd8va5.html)

### 基础知识补充

- 使用 AutoModel 或 BertModel 类可以快速实例化预训练模型。
- 通过 from_pretrained 方法加载本地路径或 HuggingFace 权重。
- 需同步加载对应的 AutoTokenizer 以保证输入数据格式一致。

### 详细解答

结论：利用 HuggingFace 的 transformers 库，可以通过 from_pretrained 接口一键加载 BERT 模型及其分词器。 原理与工程实践：transformers 库对底层 PyTorch 或 TensorFlow 进行了高度封装。加载模型时，通常使用 AutoModel.from_pretrained("bert-base-uncased")，该方法会自动解析配置文件（config.json）并下载或加载对应的权重文件（pytorch_model.bin 或 safetensors）。为了将文本转换为模型可接受的张量，必须配套使用 AutoTokenizer.from_pretrained 加载分词器。在工程中，为了加速加载和节省内存，常结合 device_map="auto" 或 torch_dtype=torch.float16 参数进行半精度加载，特别是在资源受限的环境下。

### 案例模拟

面试官追问：如果内网环境无法连接 HuggingFace，如何加载模型？ 回答示例：在内网业务中，我们会提前在外网将模型权重（如 safetensors 文件）、配置文件（config.json）和分词器词表（vocab.txt）下载到本地目录。代码中直接将 from_pretrained 的参数替换为该本地目录的绝对路径即可。此外，为了避免代码尝试联网检查更新，可以设置环境变量 TRANSFORMERS_OFFLINE=1，确保加载过程完全离线，提升启动速度。

### 5. 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- MHA：每个 Query 头都有独立的 Key 和 Value 头。
- MQA：所有 Query 头共享唯一的一组 Key 和 Value 头。
- GQA：Query 头被分组，每组共享一对 Key 和 Value 头。

### 详细解答

结论：MHA、MQA 和 GQA 是三种不同的注意力机制，主要区别在于 Key 和 Value 头的数量，体现了模型效果与推理性能之间的权衡。 原理与对比：MHA 是标准设计，Q、K、V 头数相同，模型表达能力最强，但推理时 KV Cache 占用极大，容易遇到显存带宽瓶颈。MQA 走向极端，所有 Q 头共享 1 个 K 头和 1 个 V 头，KV Cache 极小，推理速度极快，但模型性能会有所下降。GQA 则是两者的折中方案，将 Q 头分成 G 组，每组共享一对 K、V 头。例如 LLaMA-2 采用 GQA，既大幅降低了 KV Cache 占用和访存压力，又保持了接近 MHA 的模型表现。 工程权衡：在实际工程中，GQA 已经成为当前大模型（如 LLaMA-3、Qwen）的主流选择。它完美平衡了训练效果和推理成本，特别是在长上下文场景下，GQA 对显存的节省至关重要。

### 案例模拟

面试官追问：如果要把一个预训练好的 MHA 模型转换为 MQA 或 GQA，应该怎么做？ 回答示例：可以通过“均值池化”或“保留特定头”的方法，将 MHA 的多个 K、V 头压缩为 1 个或 G 个头，然后进行少量的微调（Uptraining）让模型适应新的结构。研究表明，仅需使用原始训练数据的一小部分（如 5%）进行微调，转换后的 GQA 模型就能恢复到与原始 MHA 几乎相同的性能。

### 6. 2 Attention 有哪些 优化方向？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 稀疏注意力：限制注意力范围，如局部窗口。
- 线性注意力：利用核函数近似，降为线性复杂度。
- 硬件级优化：如 FlashAttention，减少显存读写。

### 详细解答

针对传统 Attention 的平方复杂度问题，学术界和工业界提出了多种优化方向。结论上，优化主要分为算法层面的结构改进和系统层面的硬件加速。原理上，算法层面的优化包括：1) 稀疏注意力，如 Longformer，只计算局部窗口和少量全局节点的注意力；2) 线性注意力，通过核函数展开改变矩阵乘法顺序，将复杂度降为 $O(N)$；3) 架构变体，如 MQA 和 GQA，通过共享 KV 头来减少推理时的显存占用。系统层面的优化则以 FlashAttention 为代表，它不改变数学等价性，而是通过分块计算（Tiling）和重计算技术，将中间结果保留在 SRAM 中，大幅减少了 HBM 的读写开销。工程权衡上，算法近似往往会损失一定精度，而硬件优化则能实现无损加速。

### 案例模拟

面试官追问：FlashAttention 是如何做到既快又省显存的？ 回答：FlashAttention 的核心思想是“访存感知”。传统 Attention 会将庞大的注意力分数矩阵写入显存再读出，极其耗时。FlashAttention 通过 Tiling 技术，将 Q、K、V 切块加载到速度极快的 SRAM 中完成局部计算，并利用在线 Softmax 技巧实时更新全局统计量，彻底避免了中间大矩阵的显存读写。

### 7. 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / transformers 操作篇 / 未知](https://articles.zsxq.com/id_rsll7gsd8va5.html)

### 基础知识补充

- 在 from_pretrained 或前向传播时设置 output_hidden_states=True。
- 模型返回结果中的 hidden_states 是一个包含各层输出的元组。
- 元组的第一个元素是 Embedding 层输出，后续为各 Transformer 层输出。

### 详细解答

结论：通过在模型配置或调用时显式指定 output_hidden_states=True，可以获取 BERT 所有隐藏层的输出张量。 原理与工程实践：默认情况下，BERT 模型只返回最后一层的隐藏状态（last_hidden_state）和池化输出（pooler_output）。当设置 output_hidden_states=True 后，模型输出对象会增加一个 hidden_states 字段。对于 12 层的 BERT-base，该字段是一个长度为 13 的元组，索引 0 为词嵌入层输出，索引 1 到 12 分别对应第 1 到 12 层的输出。在特征提取或知识蒸馏任务中，我们经常需要提取特定层（如倒数第二层）的特征，因为最后一层往往过于拟合预训练任务（如 MLM），而中间层保留了更丰富的通用语义信息。

### 案例模拟

面试官追问：为什么在做文本分类或特征提取时，常使用倒数第二层的 hidden_state 而不是最后一层？ 回答示例：在我们的文本匹配项目中发现，BERT 最后一层的特征高度偏向于其预训练任务（如掩码语言模型），包含了较多任务特定的噪声。而倒数第二层的特征更加通用，保留了丰富的句法和语义信息。因此，我们通过 outputs.hidden_states[-2] 提取该层特征，经过均值池化后作为句向量，在下游任务中取得了比直接使用最后一层更好的效果。

### 8. 2.2 Token Attention 介绍？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 纯Python超轻量高性能LLM推理框架 —— LightLLM / 未知](https://articles.zsxq.com/id_9a643feq2b0b.html)

### 基础知识补充

- 将KV Cache管理粒度细化到单个Token
- 彻底消除显存的内部和外部碎片
- 依赖Triton实现高效的非连续显存访问

### 详细解答

Token Attention是LightLLM框架中最核心的显存管理创新技术。结论上，它实现了显存的零浪费，特别适合处理超长文本和复杂的多轮对话。原理上，在传统的PagedAttention中，显存按固定大小的Block分配，容易产生内部碎片。Token Attention打破了这一限制，预先在显存中分配一个以Token为单位的一维数组。每个Token的KV数据独立存放在任意空闲位置，通过索引表记录物理位置。在计算Attention时，Triton编写的Kernel根据索引表直接从非连续地址拉取数据。工程权衡上，细粒度管理带来了零碎片的优势，但也意味着索引表变得更大，在极端长文本下读取开销略微增加，但总体收益远大于开销。

### 案例模拟

面试官追问：Token Attention中，非连续的显存访问会不会导致GPU访存效率下降？ 回答：GPU读取连续显存时效率最高。虽然Token Attention中不同Token的物理地址不连续，但同一个Token内部的特征维度在物理显存中是连续存储的。Kernel在加载数据时，以Token的特征维度为单位进行向量化读取，这依然能够保证较高的显存带宽利用率，弥补了Token间不连续带来的影响。

### 9. 2、chatglm1和chatglm2的attention mask是怎么样的？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 对比篇 / 未知](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

### 基础知识补充

- ChatGLM1采用二维的Attention Mask机制。
- ChatGLM2回归了标准的Causal Attention Mask。
- 掩码机制决定了模型在自回归生成时的可见上下文。

### 详细解答

ChatGLM1和ChatGLM2在Attention Mask的设计上存在显著差异。ChatGLM1为了兼顾双向上下文理解和单向自回归生成，采用了一种独特的二维Attention Mask（2D RoPE配合Prefix LM Mask）。在Prefix部分，Token之间可以双向互相看到，而在生成部分，Token只能看到Prefix和当前位置之前的Token。这种设计虽然增强了理解能力，但增加了工程实现的复杂度。 ChatGLM2则为了提升训练和推理效率，回归了标准的单向Causal Attention Mask（即下三角矩阵），配合标准的RoPE（旋转位置编码）。这种改变使得ChatGLM2能够更好地适配FlashAttention等计算加速技术，大幅提升了模型的吞吐量和上下文长度，同时在多轮对话中表现更加稳定。

### 案例模拟

面试官追问：ChatGLM2放弃二维Mask后，理解能力会下降吗？ 回答：虽然放弃了Prefix双向可见性，但ChatGLM2通过扩大模型规模、增加高质量训练数据以及引入多查询注意力（MQA），弥补了这一潜在损失。在实际业务中，标准的Causal Mask配合FlashAttention带来的推理加速和长上下文支持，对工程部署的收益远大于二维Mask带来的微弱理解提升。

### 10. 3 Attention 变体有哪些？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- Multi-Query Attention (MQA)：共享一组 KV 头。
- Grouped-Query Attention (GQA)：分组共享 KV 头。
- Sliding Window Attention：关注局部上下文窗口。

### 详细解答

为了解决标准多头注意力（MHA）在长文本和高效推理中的瓶颈，涌现了多种 Attention 变体。结论上，目前大模型中最主流的变体是 MQA 和 GQA，以及用于长文本的滑动窗口注意力。原理上，MQA 让所有的 Query 头共享唯一的一组 Key 和 Value 头，极大地压缩了 KV Cache 的体积，提升了推理吞吐量，但可能会带来轻微的性能下降。GQA 则是 MHA 和 MQA 的折中方案，将 Query 头分成若干组，每组共享一组 KV 头，在保持接近 MHA 模型精度的同时，获得了接近 MQA 的推理速度。此外，Mistral 等模型采用了 Sliding Window Attention，每个 Token 只与前 W 个 Token 计算注意力，有效控制了长序列的计算复杂度。工程上，选择哪种变体取决于对推理成本和模型能力的综合考量。

### 案例模拟

面试官追问：在实际业务中，如何选择 MHA、MQA 和 GQA？ 回答：如果是小规模模型或对推理成本极度敏感的场景，我会选择 MQA，它能最大化并发量。对于百亿参数以上的现代大模型，GQA 是目前的最佳实践（如 LLaMA-3）。GQA 通过分组完美平衡了模型表达能力和 KV Cache 显存占用，性价比最高。MHA 目前主要用于追求极致效果的离线任务。

### 11. 3、llama的attention mask是怎么样的？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 对比篇 / 未知](https://articles.zsxq.com/id_0j7k3gxa5hpm.html)

### 基础知识补充

- LLaMA采用标准的下三角因果注意力掩码。
- 掩码矩阵主对角线及以下为1，其余为负无穷。
- 确保自回归生成时当前Token无法看到未来信息。

### 详细解答

LLaMA模型采用的是标准的因果注意力掩码（Causal Attention Mask）。在自回归语言模型中，为了保证模型在预测下一个词时只能依赖当前及之前的上下文，LLaMA使用了一个下三角矩阵作为Attention Mask。具体来说，掩码矩阵的主对角线及其左下方的元素为0（或1，取决于实现），而右上方的元素被设置为负无穷大（在Softmax操作后变为0）。 这种标准的Mask设计使得LLaMA能够完美适配各种主流的硬件加速库，如FlashAttention和xFormers。相比于一些复杂的混合Mask设计，标准的下三角Mask在计算上更加高效，内存访问模式也更加规则，极大地提升了模型在大规模预训练和长文本推理时的吞吐量，是目前主流开源LLM的标配。

### 案例模拟

面试官追问：在微调LLaMA进行多轮对话时，Mask需要做特殊处理吗？ 回答：在多轮对话微调（如指令微调）时，通常会将用户的输入（Prompt）和模型的回复拼接在一起。为了提高训练效率，我们会对Prompt部分的Loss进行Mask（即不计算Prompt的损失），但Attention Mask依然保持全局的下三角因果掩码，确保模型在生成回复时能看到完整的历史对话上下文。

### 12. 4.1 Multi-head Attention 存在什么问题？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 推理阶段 KV Cache 占用大量显存，限制并发。
- 访存带宽成为瓶颈，导致自回归解码速度缓慢。
- 多头之间存在计算冗余，部分注意力头特征相似。

### 详细解答

多头注意力机制（MHA）虽然赋予了模型强大的多维度特征捕捉能力，但在工程落地特别是推理阶段存在严重缺陷。结论上，MHA 最大的问题是极高的 KV Cache 显存占用和访存带宽瓶颈。原理上，在 MHA 中，每个 Query 头都有自己独立的 Key 和 Value 头。在自回归生成时，为了避免重复计算，我们需要将历史所有 Token 的 KV 向量缓存下来。随着序列变长和并发 Batch Size 的增加，KV Cache 的体积会迅速膨胀，轻易耗尽 GPU 显存，导致无法支持高并发。同时，每次生成新词都需要从显存中读取庞大的 KV Cache 进行矩阵向量乘法，这属于典型的访存密集型操作，导致推理延迟居高不下。工程权衡上，必须通过牺牲一定的多头独立性来换取推理效率。

### 案例模拟

面试官追问：能具体算一下 7B 模型在 MHA 下的 KV Cache 占用吗？ 回答：假设模型维度 4096，32 个头，32 层，采用 FP16 精度。每个 Token 的 KV Cache 大小为：2×4096×32×2字节 = 512 KB。如果上下文长度是 4096，一个请求的 KV Cache 就是 2GB。如果 Batch Size 是 16，单单 KV Cache 就要占用 32GB 显存，严重制约了吞吐量。

### 13. 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 相同点：底层均基于缩放点积注意力，均可高度并行计算。
- 差异点：跨注意力指数据来源，多头注意力指计算结构。
- 结合点：跨注意力通常采用多头结构来增强特征表达能力。

### 详细解答

结论：两者的异同点主要体现在概念层级上：Cross Attention 定义了输入数据的关系，而 Multi-Head Attention 定义了计算的拓扑结构。 原理与对比：相同点在于，它们都依赖于 Q、K、V 矩阵的乘法运算，都使用 Softmax 进行归一化，且都可以利用相同的硬件加速算子。不同点在于，当我们讨论 Multi-Head Attention 时，通常默认是 Self-Attention（如 BERT 或 GPT 中），Q=K=V；而讨论 Cross Attention 时，强调的是 Q ≠ K=V。实际上，Multi-Head 是一种增强技术，既可以作用于 Self-Attention，也可以作用于 Cross Attention。 工程权衡：在模型架构设计时，我们通常会为 Cross Attention 配置与 Self Attention 相同数量的“头（Heads）”，以保持张量维度的对齐，方便残差连接（Residual Connection）和层归一化（LayerNorm）的计算，从而简化整体网络的代码实现。

### 案例模拟

面试官追问：如果一个 Cross Attention 只有一个头（Single-Head），会存在什么问题？ 回答示例：如果只有一个头，模型在融合两个序列时，只能学习到一种对齐模式。比如在图文匹配中，可能只能关注到颜色特征，而忽略了形状或位置特征。采用多头结构，可以让不同的头分别关注颜色、形状、动作等不同的语义维度，使得跨模态信息的融合更加丰富和鲁棒。

### 14. 4.2 介绍一下 Multi-Query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- MQA 是一种优化推理效率的注意力机制变体。
- 核心机制是所有 Query 头共享同一组 KV 头。
- 大幅减少了推理时的 KV Cache 显存占用和访存。

### 详细解答

Multi-Query Attention（MQA）是由 Google 提出的一种旨在加速 Transformer 推理的注意力架构。结论上，MQA 通过极端的参数共享策略，彻底解决了传统多头注意力在推理时的显存和访存瓶颈。原理上，在标准的 MHA 中，如果有 $H$ 个 Query 头，就会有 $H$ 个 Key 头和 $H$ 个 Value 头。而在 MQA 中，无论有多少个 Query 头，都只保留 1 个 Key 头和 1 个 Value 头。所有的 Query 头在计算注意力分数时，都会与这唯一的一组 KV 进行点积。这种设计使得推理阶段需要缓存的 KV Cache 体积缩小到了原来的 $1/H$。工程权衡上，虽然 MQA 极大地降低了显存占用，提高了内存读取效率，但由于 KV 表达能力的急剧压缩，模型的整体性能和泛化能力通常会有所下降，需要通过更长时间的训练来弥补。

### 案例模拟

面试官追问：MQA 在训练阶段能加速吗？ 回答：MQA 主要的加速收益体现在推理阶段（自回归解码）。在训练阶段（Prefill 阶段），由于是高度并行的矩阵乘法，计算主要受限于算力而非访存，因此 MQA 对训练速度的提升并不明显，甚至因为需要对共享的 KV 进行广播操作，可能还会带来微小的额外开销。它的核心价值在于大幅提升部署吞吐量。

### 15. 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 结构：MHA 有 H 组 KV 头，MQA 仅有 1 组 KV 头。
- 显存：MQA 的 KV Cache 显存占用是 MHA 的 1/H。
- 性能：MHA 模型表达能力更强，MQA 推理吞吐量高。

### 详细解答

Multi-head Attention（MHA）和 Multi-Query Attention（MQA）代表了模型表达能力与推理效率两个极端的权衡。结论上，MHA 追求极致的特征捕捉能力，而 MQA 追求极致的推理吞吐量。原理对比上，MHA 为每个 Query 头配备独立的 Key 和 Value 头，能够从多个不同的子空间独立提取信息，但代价是推理时 KV Cache 极其庞大，导致严重的访存瓶颈。相反，MQA 强制所有 Query 头共享唯一的一组 KV 头，这使得 KV Cache 的体积缩小了数十倍，极大地缓解了显存压力，使得 GPU 能够支持更大的 Batch Size。工程权衡方面，MQA 的极端共享会导致模型容量下降，在复杂推理任务上可能表现不如 MHA。因此，在实际应用中，往往需要根据业务场景对延迟和精度的要求做出选择。

### 案例模拟

面试官追问：如果我已经有一个训练好的 MHA 模型，能直接转成 MQA 吗？ 回答：不能直接无损转换，但可以通过 Upcycling 技术进行微调。具体做法是将 MHA 的多个 KV 头的权重进行平均，作为 MQA 的初始 KV 头权重，然后冻结大部分模型参数，仅对模型进行少量的继续预训练。这样可以在消耗极少算力的情况下，将 MHA 模型转化为高效的 MQA 模型。

### 16. 4.4 Multi-Query Attention 这样做的好处是什么？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 显著降低 KV Cache 显存占用，支持更大并发。
- 减少显存读写量，缓解 Memory-bound 瓶颈。
- 提升了系统的整体吞吐量，降低大模型部署成本。

### 详细解答

Multi-Query Attention（MQA）的核心优势在于对大模型推理性能的全面解放。结论上，MQA 带来的最大好处是极大地降低了推理成本并提升了服务吞吐量。原理上，由于所有 Query 头共享一组 KV 头，推理时需要保存的 KV Cache 体积成比例缩小。这带来了两个直接好处：第一，显存占用大幅降低，使得单张 GPU 能够同时处理更多的并发请求（即更大的 Batch Size）；第二，在自回归解码时，从显存读取 KV Cache 的数据量锐减，极大地缓解了显存带宽瓶颈（Memory-bound），使得计算单元能更高效地工作，从而降低了每个 Token 的生成延迟。工程权衡上，这种对推理极度友好的设计，使得企业在部署千亿级大模型时，可以显著减少所需的 GPU 数量，大幅降低运营成本。

### 案例模拟

面试官追问：MQA 减少了 KV Cache，对长文本处理有什么特别的意义吗？ 回答：意义非常重大。在长文本场景下，KV Cache 的膨胀是致命的。使用 MHA 时，长文本的 KV Cache 甚至会超过模型权重本身的显存占用，导致单卡根本无法运行。MQA 将 KV Cache 压缩了数十倍，使得在有限的显存下处理超长上下文成为可能，这也是 ChatGLM 等模型采用 MQA 的原因。

### 17. 4.5 有 哪些模型 是 使用 Multi-Query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 谷歌的 PaLM 和 PaLM 2 系列模型采用了 MQA。
- 智谱 AI 的 ChatGLM2 和 ChatGLM3 采用了 MQA。
- Falcon 系列模型（如 Falcon-40B）也使用了 MQA。

### 详细解答

Multi-Query Attention（MQA）因其卓越的推理效率，被多个知名的大语言模型所采用。结论上，谷歌是 MQA 的坚定拥趸，而国内外的开源社区也在特定模型中广泛实践了这一架构。具体来说，谷歌的 PaLM（540B）和后续的 PaLM 2 均采用了 MQA，这使得它们在提供庞大参数规模的同时，依然能保持可接受的推理延迟。在开源领域，阿联酋技术创新研究院发布的 Falcon 系列（如 Falcon-40B）采用了 MQA，以优化部署成本。国内方面，智谱 AI 的 ChatGLM2-6B 和 ChatGLM3-6B 也从第一代的 MHA 转向了 MQA，极大地提升了模型的推理速度和长文本处理能力，使其在消费级显卡上也能流畅运行。工程上，这些模型的成功证明了通过充分的训练，MQA 完全可以弥补结构简化带来的性能损失。

### 案例模拟

面试官追问：为什么 LLaMA 系列没有使用 MQA？ 回答：LLaMA-1 使用了传统的 MHA 以追求最佳的模型效果。在开发 LLaMA-2 时，Meta 团队发现 MQA 虽然极快，但在某些复杂任务上会导致性能下降和训练不稳定。因此，他们提出并采用了 GQA 作为折中方案。GQA 既保留了接近 MHA 的模型表现，又获得了接近 MQA 的推理速度，是更优的工程选择。

### 18. 5.1 什么是 Grouped-query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- GQA 是 MHA 和 MQA 之间的折中注意力机制。
- 将 Query 头分成 G 个组，每个组共享一组 KV 头。
- 兼顾了 MHA 的模型表达能力和 MQA 的推理效率。

### 详细解答

Grouped-query Attention（GQA）是一种旨在平衡模型性能与推理效率的先进注意力机制。结论上，GQA 成功地在传统多头注意力（MHA）和多查询注意力（MQA）之间找到了最优解。原理上，GQA 将所有的 Query 头划分为 $G$ 个组。在同一个组内的多个 Query 头共享同一组 Key 和 Value 头。当 $G=H$ 时，GQA 退化为 MHA；当 $G=1$ 时，GQA 退化为 MQA。通过设置合理的组数（例如 32 个 Q 头，8 个 KV 头），GQA 能够保留多个独立的 KV 表达空间，从而维持接近 MHA 的模型精度和泛化能力；同时，其 KV Cache 的体积仅为 MHA 的 $1/4$，大幅降低了显存占用和访存带宽压力，实现了接近 MQA 的推理速度。工程权衡上，GQA 是目前大模型兼顾效果与成本的最佳实践。

### 案例模拟

面试官追问：GQA 在底层算子实现上和 MHA 有什么不同？ 回答：在底层实现时，GQA 需要处理 Q 和 KV 维度不匹配的问题。通常的做法是在进行矩阵乘法前，将 KV 张量在头部维度上进行物理复制或逻辑广播，使其形状与 Q 对齐。优秀的推理框架（如 vLLM 或 FlashAttention-2）会在 Kernel 内部隐式地处理这种广播，避免额外的显存分配和数据搬运。

### 19. 5.2 有哪些大模型使用 Grouped-query Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- Meta 的 LLaMA-2 (70B) 和 LLaMA-3 全系列采用 GQA。
- Mistral-7B 和 Mixtral 8x7B 广泛使用了 GQA 架构。
- 阿里开源的 Qwen 系列模型也全面拥抱了 GQA。

### 详细解答

Grouped-query Attention（GQA）凭借其在性能和效率上的完美平衡，已经成为当前最先进大语言模型的标配架构。结论上，几乎所有新一代的主流开源大模型都采用了 GQA。具体来说，Meta 在 LLaMA-2 的 70B 版本中首次引入 GQA，并在最新的 LLaMA-3 全系列中全面采用，这使得 LLaMA-3 在保持极高推理吞吐量的同时，各项评测指标均达到 SOTA。此外，欧洲明星初创公司 Mistral AI 的 Mistral-7B 及其 MoE 版本 Mixtral 8x7B 也使用了 GQA，以支持更长的上下文和更快的生成速度。国内方面，阿里云开源的 Qwen 系列同样全面拥抱了 GQA 架构。工程上，GQA 的广泛普及也促使了各类推理加速引擎（如 TensorRT-LLM、vLLM）将其作为核心优化对象，形成了良好的生态闭环。

### 案例模拟

面试官追问：LLaMA-2 为什么只有 70B 版本用了 GQA，而 7B 和 13B 没用？ 回答：这主要是基于参数规模和显存瓶颈的权衡。对于 7B 和 13B 的小模型，其本身的 KV Cache 绝对体积相对较小，MHA 带来的显存压力在可接受范围内，为了保证小模型的表达能力，Meta 保留了 MHA。而对于 70B 的大模型，MHA 的 KV Cache 会极其庞大，严重限制部署，因此引入 GQA 解决显存瓶颈。

### 20. 6.1 为什么需要 FlashAttention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 标准注意力机制的时间和空间复杂度均为序列长度的平方。
- 显存读写是制约注意力计算速度的主要硬件瓶颈。
- 长序列场景下，标准注意力会导致显存溢出（OOM）问题。

### 详细解答

结论：需要 FlashAttention 是为了打破标准注意力机制在处理长序列时面临的显存墙和计算效率瓶颈。 原理与对比：标准 Attention 需要将中间结果（如注意力分数矩阵）写入 HBM（高带宽内存），然后再读出进行下一步计算，这种频繁的显存读写极大地拖慢了速度。随着序列长度增加，显存占用呈二次方增长，导致无法训练长文本模型。FlashAttention 通过 Tiling（分块）和 Recomputation（重计算）技术，将计算过程融合在一个 CUDA Kernel 中，避免了中间结果的 HBM 读写，直接在 SRAM 中完成计算。 工程权衡：虽然重计算增加了计算量（FLOPs），但由于大幅减少了耗时的显存访问，整体运行速度反而显著提升，同时显存占用降至线性复杂度。

### 案例模拟

面试官追问：FlashAttention 为什么能降低显存占用？ 回答示例：因为它避免了实例化巨大的注意力分数矩阵。在反向传播时，它不保存前向的中间注意力矩阵，而是利用保存的归一化统计量（如 softmax 的分母）在 SRAM 中快速重新计算注意力分数。这种以计算换显存的策略，使得显存复杂度从 O(N^2) 降到了 O(N)，彻底解决了长序列 OOM 问题。

### 21. 6.2 简单介绍一下 FlashAttention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- FlashAttention 是一种硬件感知的精确注意力算法。
- 通过 Tiling 分块技术在 SRAM 中完成注意力矩阵计算。
- 实现了显存复杂度的线性化和计算速度的显著提升。

### 详细解答

结论：FlashAttention 是一种硬件感知（IO-aware）的精确注意力计算算法，旨在解决 Transformer 在长序列下的显存和速度瓶颈。 原理与对比：与近似注意力（如 Sparse Attention）不同，FlashAttention 计算的是完全精确的注意力结果。它的核心思想是优化 GPU 的内存层级访问。GPU 的 SRAM 速度极快但容量小，HBM 容量大但速度慢。FlashAttention 将输入的 Q、K、V 矩阵切分成小块（Tiling），加载到 SRAM 中计算局部注意力，并利用在线 Softmax 算法合并局部结果。 工程权衡：这种设计将内存访问次数降到最低，实现了计算与访存的完美平衡。虽然算法实现上需要深度定制 CUDA Kernel，开发难度大，但为大模型长上下文训练提供了根本性的基础设施支持。

### 案例模拟

面试官追问：FlashAttention 和普通的稀疏注意力有什么区别？ 回答示例：稀疏注意力通过丢弃部分注意力连接来降低复杂度，属于近似算法，可能会损失模型精度。而 FlashAttention 是精确注意力算法，数学上与标准 Attention 结果完全一致。它通过优化底层硬件的显存读写逻辑来提升效率，既保证了模型效果，又突破了长序列的计算瓶颈。

### 22. 6.3 简单介绍一下 FlashAttention 核心？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- Tiling 分块技术将大矩阵切分为适合 SRAM 的小块。
- Online Softmax 支持在无全局最大值时计算局部结果。
- 重计算技术在反向传播时大幅节省中间显存占用。

### 详细解答

结论：FlashAttention 的核心在于 Tiling（分块）、Online Softmax 和 Recomputation（重计算）三大技术的结合。 原理与对比：首先，Tiling 技术将 Q、K、V 矩阵分块加载到高速 SRAM 中，避免了对慢速 HBM 的频繁访问。其次，标准 Softmax 需要全局最大值和分母，无法分块计算，FlashAttention 引入了 Online Softmax，通过维护局部的最大值和缩放因子，使得 Softmax 可以在分块计算时动态更新，最终得到精确的全局结果。最后，在反向传播时，通过 Recomputation 机制，利用前向保存的统计量重新计算注意力分数，避免了存储庞大的中间矩阵。 工程权衡：这三大核心技术共同作用，将显存访问量最小化。虽然重计算增加了约 15-20% 的 FLOPs，但由于打破了 IO 瓶颈，整体端到端速度依然获得了数倍的提升。

### 案例模拟

面试官追问：Online Softmax 是如何解决分块计算难题的？ 回答示例：标准 Softmax 需要遍历所有元素找到最大值以防止溢出，然后再计算指数和。Online Softmax 在处理每个分块时，记录当前块的最大值和指数和。当处理下一个分块时，如果发现更大的值，会利用数学恒等式对之前累积的指数和进行缩放修正。这样就实现了单趟遍历完成 Softmax 计算。

### 23. 6.4 介绍一下 FlashAttention 优点？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 显著提升计算速度，通常比标准注意力快两到四倍。
- 显存占用从序列长度的平方复杂度降低为线性复杂度。
- 保证计算结果的精确性，不引入任何近似误差或精度损失。

### 详细解答

结论：FlashAttention 的主要优点体现在速度快、省显存、精度无损以及支持超长上下文四个方面。 原理与对比：1. 速度快：通过减少 HBM 的读写次数，打破了内存带宽瓶颈，使得 GPU 计算单元能满载运行，训练和推理速度大幅提升。2. 省显存：得益于分块和重计算技术，无需存储 O(N^2) 的注意力分数矩阵，显存占用降至 O(N)。3. 精度无损：与各种稀疏注意力或低秩近似方法不同，FlashAttention 在数学上等价于标准 Attention，不会影响模型收敛和最终效果。4. 支持长序列：因为显存不再是瓶颈，模型可以轻松扩展到 32K、128K 甚至更长的上下文窗口。 工程权衡：其缺点是高度依赖特定的硬件架构（如 Nvidia GPU 的 SRAM 特性），跨平台移植（如到 TPU 或 AMD GPU）需要重新编写底层算子，工程维护成本较高。

### 案例模拟

面试官追问：在实际业务中，引入 FlashAttention 会带来哪些直接收益？ 回答示例：在我们的长文本问答项目中，引入 FlashAttention 后，首先是训练阶段的 OOM 问题得到了解决，使得我们可以将上下文长度从 4K 扩展到 32K。其次，推理时的首字延迟（TTFT）显著降低，吞吐量提升了约 3 倍，大幅降低了线上机器的推理成本，且模型效果完全没有打折。

### 24. 6.5 介绍一下 FlashAttention 代表模型？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- LLaMA 系列模型全面采用该技术提升长文本能力。
- Falcon 和 Mistral 等开源大模型均内置该算子支持。
- GPT-4 等闭源巨头模型底层也广泛应用了类似的高效算子。

### 详细解答

结论：FlashAttention 已经成为当前主流大语言模型（LLMs）的标配底层算子，代表模型包括 LLaMA 系列、Mistral、Qwen 等。 原理与对比：在 FlashAttention 出现之前，模型处理长序列通常采用 Longformer 等稀疏注意力结构，但效果往往不如全注意力。自从 FlashAttention 开源并集成到 PyTorch 2.0（作为 scaled_dot_product_attention 的后端）后，几乎所有现代大模型都转向了精确的全注意力机制。例如，LLaMA-2 和 LLaMA-3 在预训练和微调阶段都重度依赖 FlashAttention-2 来实现高效的计算；Mistral 结合 FlashAttention 和滑动窗口注意力（SWA）实现了极高的推理效率。 工程权衡：这些模型在设计时不再需要为了显存而妥协注意力结构，可以直接使用标准 Transformer 架构。开发者在部署这些模型时，只需确保环境支持相应的 CUDA 版本，即可免费获得巨大的性能提升。

### 案例模拟

面试官追问：如果我要在旧版 PyTorch 上跑 LLaMA 模型，没有 FlashAttention 会怎样？ 回答示例：如果没有 FlashAttention，当输入序列长度超过 2K 或 4K 时，极易触发显存溢出（OOM）。即使显存勉强够用，计算速度也会非常缓慢，因为标准注意力会产生巨大的中间矩阵读写开销。通常我们需要手动安装 flash-attn 库，或者升级到 PyTorch 2.0 以上版本来开启加速。

### 25. 7 并行 transformer block

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 并行架构将 Attention 和 FFN 层进行并行计算。
- 能够有效减少计算图深度，提升前向和反向传播的并行度。
- 代表模型如 PaLM，在极大规模下显著提升训练吞吐量。

### 详细解答

结论：并行 Transformer Block 是一种架构变体，它将传统的串行 Attention 和 FFN（前馈神经网络）改为并行执行，以提升计算效率。 原理与对比：在标准 Transformer 中，数据流是串行的：x = x + Attention(LayerNorm(x))，然后 x = x + FFN(LayerNorm(x))。而在并行结构中，输入 x 同时送入 Attention 和 FFN，即 x = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))。这种设计使得 Attention 和 FFN 的矩阵乘法可以合并或并行调度，减少了内核启动开销和数据依赖，缩短了关键路径长度。 工程权衡：并行设计显著提升了训练速度，特别是在超大规模集群上。但由于 Attention 和 FFN 共享相同的输入表示，可能会略微削弱模型的表达能力。不过在百亿或千亿参数规模下（如 PaLM），这种表达能力的损失微乎其微，而带来的吞吐量提升却是极其可观的。

### 案例模拟

面试官追问：并行 Transformer Block 对推理延迟有什么影响？ 回答示例：在推理阶段，并行 Block 可以降低延迟。因为 Attention 和 FFN 之间没有了数据依赖，硬件可以同时调度这两个模块的计算任务。特别是在张量并行（Tensor Parallelism）场景下，原本需要两次 All-Reduce 通信（Attention 一次，FFN 一次），并行结构可以将其合并为一次通信，大幅降低了分布式推理的通信开销。

### 26. 9.1 简单介绍一下 Paged Attention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 该技术借鉴了操作系统的虚拟内存分页管理机制。
- 将变长的 KV Cache 划分为固定大小的块进行非连续存储。
- 极大减少了显存碎片，提升了大模型推理的批处理能力。

### 详细解答

结论：Paged Attention 是 vLLM 框架提出的一种高效显存管理算法，专门用于解决大模型推理时 KV Cache 显存碎片化和利用率低的问题。 原理与对比：在传统的推理框架中，KV Cache 通常被分配为连续的显存空间。由于生成文本的长度不可预测，往往需要预先分配最大可能长度的显存，导致严重的内部碎片（预留未用）和外部碎片。Paged Attention 将 KV Cache 切分为固定大小的块（例如每个块包含 16 个 token 的 KV 值），并通过一张“页表”将逻辑上连续的 token 映射到物理上非连续的显存块中。 工程权衡：这种按需分配的机制将显存浪费降到了 4% 以下，使得同一台机器可以容纳更大的 Batch Size，吞吐量提升 2-4 倍。代价是引入了页表查找的微小开销，并且需要定制化的 CUDA Kernel 来支持非连续内存的注意力计算。

### 案例模拟

面试官追问：Paged Attention 如何支持复杂的解码策略，比如束搜索（Beam Search）？ 回答示例：在 Beam Search 或并行采样中，多个候选序列会共享相同的前缀。Paged Attention 通过页表机制，可以轻松实现内存共享（Copy-on-Write）。不同的序列可以指向相同的物理块，只有当某个序列生成新的 token 时，才分配新的物理块。这极大地节省了显存，使得复杂解码策略的成本大幅降低。

### 27. BEVFormer中的Spatial Cross-Attention的步骤？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/IndustryAlgorithm/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

Step 1 Lift each BEV query to be a pillar Step 2 Project the 3D points in pillar to 2D points in views Step 3 Sample features from regions in hit views Step 4 Fuse by weight

### 案例模拟

面试表达可以这样组织：先用一句话回答“BEVFormer中的Spatial Cross-Attention的步骤？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 28. Transformer/CNN/RNN的时间复杂度对比

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/DeepLearning/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

- https://zhuanlan.zhihu.com/p/264749298 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“Transformer/CNN/RNN的时间复杂度对比”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 29. Transformer中的Attention计算复杂度以及如何改进？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

在标准的Transformer中，attention计算的时间复杂度为O(N^2)，其中N是输入序列的长度。为了降低计算复杂度，可以采用以下几种方法： - 使用自注意力机制，减少计算复杂度。自注意力机制不需要计算输入序列之间的交叉关系，而是计算每个输入向量与自身之间的关系，从而减少计算量。 - 使用局部注意力机制，只计算输入序列中与当前位置相关的子序列的交互，从而降低计算复杂度。 - 采用基于近似的方法，例如使用随机化和采样等方法来近似计算，从而降低计算复杂度。 - 使用压缩注意力机制，通过将输入向量映射到低维空间来减少计算量，例如使用哈希注意力机制和低秩注意力机制等。

### 案例模拟

面试表达可以这样组织：先用一句话回答“Transformer中的Attention计算复杂度以及如何改进？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 30. Transformer为何使用多头注意力机制

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

多头保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。论文原作者发现这样效果确实好，更详细的解析可以查阅Multi-head Attention 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“Transformer为何使用多头注意力机制”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 31. Transformer的注意力机制常用softmax函数，可以使用sigmoid代替吗？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/VisionPerception/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

softmax有强制稀疏化的效果，sigmoid受到类别不均匀的影响。如果用sigmoid，就不是在做正则化和注意力调整。sigmoid的灵活度也更高，可以让“注意力”变成“什么都注意机制”或者“什么都不注意机制”。 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“Transformer的注意力机制常用softmax函数，可以使用sigmoid代替吗？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 32. 一、vLLM 用于大模型并行推理加速 存在什么问题？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——PagedAttention篇 / 未知](https://articles.zsxq.com/id_p22mjq881n3n.html)

### 基础知识补充

- 静态显存分配导致显存碎片化，浪费大量空间
- 预先分配最大长度的KV Cache，短请求利用率低
- 缺乏细粒度的内存管理，限制了高并发吞吐量

### 详细解答

结论：在vLLM提出PagedAttention之前，传统的大模型并行推理加速框架在显存管理上面临着严重的碎片化和利用率低下的问题。 原理与权衡：传统框架在处理请求时，通常会根据模型支持的最大序列长度为每个请求预先分配连续的显存空间用于存放KV Cache。然而，实际生成的文本长度往往不可预测且远小于最大长度，这导致了严重的内部显存碎片。同时，由于请求的生命周期不同，显存中还会产生外部碎片。据统计，传统方式下KV Cache的实际显存利用率通常不到30%。这种粗放的内存管理极大地限制了系统能够同时处理的Batch Size，进而导致GPU算力无法被充分利用，吞吐量遭遇瓶颈。

### 案例模拟

面试官追问：内部碎片和外部碎片在传统KV Cache管理中具体是怎么产生的？ 回答示例：内部碎片是因为系统按最大可能长度预分配显存，但实际请求只生成了少量Token就结束了，剩下的空间被浪费。外部碎片则是由于不同请求的到达和结束时间不同，释放显存后留下了许多不连续的小块空闲内存，导致新的长请求找不到连续大空间而分配失败。

### 33. 一、为什么需要 FasterTransformer？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——Faster Transformer篇 / 未知](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### 基础知识补充

- 原生PyTorch推理存在大量算子开销
- 缺乏针对Transformer架构的深度算子融合
- 动态图机制导致执行效率低下且显存占用高

### 详细解答

需要FasterTransformer（FT）的核心原因是为了突破原生深度学习框架在Transformer模型推理上的性能瓶颈。结论上，FT通过极致的算子融合和底层CUDA优化，大幅降低了推理延迟并提升了吞吐量。原理上，原生PyTorch等框架在执行Transformer时，会将LayerNorm、矩阵乘法等拆分成多个独立的小算子。这会导致频繁的GPU显存读写和大量的Kernel启动开销。FT针对这一问题，将多个小算子融合为一个大的CUDA Kernel，极大地减少了访存次数。工程权衡上，FT使用C++和CUDA重写了整个推理逻辑，虽然带来了极致的性能，但也牺牲了灵活性和易用性，导致新模型适配成本极高，且需要手动管理显存。

### 案例模拟

面试官追问：算子融合（Kernel Fusion）为什么能提升性能？ 回答：在GPU计算中，很多操作（如LayerNorm、激活函数）是访存密集型的，计算量小但需频繁读写显存。每次启动Kernel都需要从全局显存读写数据。算子融合将多个连续操作合并到一个Kernel中，数据读取到寄存器后直接完成所有计算再写回，省去了中间结果的显存读写开销，显著提升推理速度。

### 34. 一、为什么需要 跨注意力机制（Cross-Attention）？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 实现不同模态或不同序列之间的信息对齐与深度融合。
- 解决 Seq2Seq 任务中输入与输出序列长度不一致的问题。
- 突破固定长度上下文向量的瓶颈，实现动态信息检索。

### 详细解答

结论：需要 Cross-Attention 是为了在复杂的生成任务中，让模型能够动态、精准地从外部条件（如源文本、图像、音频）中提取所需信息。 原理与对比：在早期的 Seq2Seq 模型（如基础的 RNN）中，整个源序列被压缩成一个固定长度的上下文向量，这在处理长句子时会导致严重的信息丢失。Cross-Attention 打破了这一限制，它允许解码器在生成每一步时，直接查看编码器的所有输出状态。通过计算 Query（解码器状态）与 Key（编码器状态）的注意力分数，模型可以自适应地聚焦于当前生成步骤最相关的源序列部分。 工程权衡：虽然 Cross-Attention 极大地提升了翻译、摘要等任务的准确性，但它要求在解码的每一步都对整个源序列进行计算，增加了推理延迟。在实际部署中，通常会缓存编码器的 K 和 V 矩阵以避免重复计算。

### 案例模拟

面试官追问：在多模态模型（如 Stable Diffusion）中，Cross-Attention 扮演了什么角色？ 回答示例：在 Stable Diffusion 中，Cross-Attention 是实现文本控制图像生成的关键。文本提示（Prompt）经过 CLIP 编码后作为 Key 和 Value，而图像的潜变量（Latent）作为 Query。在去噪过程中，图像特征通过 Cross-Attention 不断查询文本特征，从而确保生成的图像内容与文本描述高度一致。

### 35. 七、Cross Attention 的优势和挑战？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 优势在于能够有效融合不同模态或不同来源的特征信息。
- 挑战主要体现在计算复杂度高以及对齐不同分布数据的难度。
- 常用于编码器-解码器架构或多模态大模型中的特征交互。

### 详细解答

结论：Cross Attention 的核心优势是跨序列信息融合能力强，但面临计算开销大和对齐困难的挑战。 原理与权衡：在 Cross Attention 中，Query 来自一个序列（如解码器目标），而 Key 和 Value 来自另一个序列（如编码器源）。这种机制使得模型能够根据目标需求动态提取源序列的上下文，在机器翻译、图文多模态任务中表现卓越。然而，其计算复杂度与两个序列长度的乘积成正比，导致处理长文本或高分辨率图像时显存占用极高。此外，不同模态的数据分布差异大，模型需要大量数据才能学习到有效的对齐关系。工程上常通过降采样、局部注意力或引入 Perceiver 架构来缓解计算压力。

### 案例模拟

面试官追问：在多模态大模型中，如何降低 Cross Attention 的计算复杂度？ 回答示例：在实际项目中，我们通常会在输入 Cross Attention 之前，对视觉特征进行池化或使用 Q-Former 等结构进行特征压缩，减少 Key 和 Value 的序列长度。另外，也可以采用线性注意力机制或分块注意力来替代标准的点积注意力，从而在保证跨模态融合效果的同时，显著降低显存占用和推理延迟。

### 36. 三、FasterTransformer 核心是什么？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——Faster Transformer篇 / 未知](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### 基础知识补充

- 极致的算子融合（Kernel Fusion）技术
- 高效的内存管理与KV Cache复用机制
- 针对不同GPU架构的底层CUDA/PTX优化

### 详细解答

FasterTransformer的核心在于通过底层硬件级别的优化来榨干GPU的计算和显存带宽潜力。结论上，其最核心的技术支柱是算子融合、高效的注意力机制实现以及分布式并行策略。原理上，FT将Transformer Block中原本零散的算子进行深度融合，大幅减少了Kernel Launch开销和Global Memory的读写次数。在注意力机制方面，FT针对Prefill和Decode阶段分别实现了高度优化的Fused Attention Kernel。此外，它对KV Cache进行了精细的内存管理，预先分配显存以避免动态申请的开销。工程权衡上，这种高度定制化的C++/CUDA实现使得FT在特定硬件上能达到理论性能上限，但代价是代码与硬件高度耦合，缺乏跨平台通用性。

### 案例模拟

面试官追问：在Decode阶段，FasterTransformer是如何优化Attention计算的？ 回答：Decode阶段每次只生成一个Token，Attention计算是典型的访存密集型操作。FT实现了专门的Masked Multi-Head Attention Kernel，将Query与KV Cache的计算融合。通过将KV Cache分块加载到GPU共享内存中，并利用寄存器进行高效归约计算，最大程度减少了对全局显存的访问。

### 37. 三、什么是 PagedAttention？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——PagedAttention篇 / 未知](https://articles.zsxq.com/id_p22mjq881n3n.html)

### 基础知识补充

- 借鉴操作系统虚拟内存分页思想的注意力机制
- 将连续的逻辑KV Cache映射到非连续的物理块
- 计算时动态读取分散的显存块，消除内存碎片

### 详细解答

结论：PagedAttention是vLLM框架的核心创新，它是一种允许在非连续的物理显存空间中存储和计算KV Cache的注意力机制。 原理与权衡：在传统的Attention计算中，要求Key和Value张量在显存中必须是连续存储的，这导致了严重的内存分配难题。PagedAttention打破了这一限制，它将每个序列的KV Cache切分成固定大小的Token块。在计算注意力得分时，PagedAttention的CUDA Kernel会根据系统维护的块表，逐个定位并读取分散在显存各处的物理块，完成点积运算。这种设计虽然在Kernel内部引入了查表的轻微开销，但换来了显存利用率的巨大提升，最终带来的吞吐量收益远超查表损耗。

### 案例模拟

面试官追问：PagedAttention是如何支持Beam Search（束搜索）优化的？ 回答示例：在Beam Search中，多个候选序列通常共享相同的前缀。PagedAttention通过引入类似操作系统的“写时复制”和引用计数机制，允许多个逻辑序列指向相同的物理Block。只有当某个序列生成了不同的新Token时，系统才会为它分配新的物理Block，极大地减少了显存占用。

### 38. 为什么transformer块使用LayerNorm而不是BatchNorm

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

Batch Normalization 是对这批样本的同一维度特征做归一化， Layer Normalization 是对这单个样本的所有维度特征做归一化。LN不依赖于batch的大小和输入sequence的长度，因此可以用于batchsize为1和RNN中sequence的normalize操作。 - 为什么BN在NLP中效果差 - BN计算特征的均值和方差是需要在batch_size维度，而这个维度表示一个特征，比如身高、体重、肤色等，如果将BN用于NLP中，其需要对每一个单词做处理，让每一个单词是对应到了MLP中的每一个特征明显是违背直觉得； - BN是对单词做缩放，在NLP中，单词由词向量来表达，本质上是对词向量进行缩放。词向量是什么？是我们学习出来的参数来表示词语语义的参数，不是真实存在的。 - 为什么LayerNorm单独对一个样本的所有单词做缩放可以起到效果 - layner-norm 针对每一个样本做特征的缩放。换句话讲，保留了N维度，在C/H/W维度上做缩放。 - layner-norm 也是在对同一个特征下的元素做归一化，只不过这里不再是对应N（或者说batch size），而是对应的文本长度。

### 案例模拟

面试表达可以这样组织：先用一句话回答“为什么transformer块使用LayerNorm而不是BatchNorm”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 39. 二、FasterTransformer 介绍一下？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——Faster Transformer篇 / 未知](https://articles.zsxq.com/id_dd2gowztxtfg.html)

### 基础知识补充

- NVIDIA开源的Transformer高性能推理库
- 基于C++和CUDA实现，提供极致性能
- 支持张量并行（TP）和流水线并行（PP）

### 详细解答

FasterTransformer（FT）是NVIDIA推出的专门针对Transformer架构进行极致优化的开源推理加速库。结论上，它是工业界早期最著名的高性能大模型推理解决方案之一。原理上，FT完全脱离了PyTorch等高级框架的动态图机制，底层采用C++和CUDA编写。它不仅实现了深度的算子融合，还针对不同的GPU架构进行了汇编级别的优化，并利用cuBLAS进行高效的矩阵运算。此外，FT内置了对分布式推理的强大支持，包括张量并行和流水线并行，使其能够跨多卡、多机部署超大模型。工程权衡上，FT的性能极佳，但代码复杂度极高，二次开发困难。目前NVIDIA已将其核心技术演进并整合到了新一代的TensorRT-LLM框架中。

### 案例模拟

面试官追问：FasterTransformer如何处理大模型的分布式推理？ 回答：FT主要通过张量并行（TP）和流水线并行（PP）实现。在TP中，FT将线性映射矩阵切分到不同GPU上，每张卡计算一部分后通过NCCL汇总，适合单机多卡。对于超大模型，FT结合PP将模型的不同层分配到不同机器上，通过点对点通信传递隐藏状态，实现多机多卡协同推理。

### 40. 二、vLLM 如何 优化 大模型并行推理加速？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLM（大语言模型）部署加速方法——PagedAttention篇 / 未知](https://articles.zsxq.com/id_p22mjq881n3n.html)

### 基础知识补充

- 引入PagedAttention，实现KV Cache非连续存储
- 动态按需分配显存块，彻底消除内部显存碎片
- 支持内存共享机制，高效处理束搜索和多输出

### 详细解答

结论：vLLM通过引入操作系统中虚拟内存分页的经典思想，设计了PagedAttention机制，从根本上解决了大模型推理中的显存碎片问题，实现了并行推理加速。 原理与权衡：vLLM将KV Cache划分为固定大小的块。在逻辑上，一个请求的KV Cache是连续的，但在物理显存中，这些块可以分散存储在任何非连续的空间中。vLLM维护一张块表来映射逻辑块到物理块。在生成过程中，系统按需动态分配新的物理块，而不是一次性预分配最大长度。这种机制将显存利用率从30%提升到了90%以上。此外，对于Beam Search任务，vLLM通过引用计数机制实现了不同序列间物理块的内存共享，进一步节省了显存。

### 案例模拟

面试官追问：PagedAttention中的Block大小设置对性能有什么影响？ 回答示例：Block大小是一个关键超参数。如果Block设置过大，会导致每个Block内部仍存在一定的内部碎片；如果设置过小，虽然碎片极小，但会导致块表变得非常庞大，增加CPU调度和查表的开销，同时可能影响GPU访存的连续性。通常vLLM默认设置为16或32。

### 41. 二、介绍一些 跨注意力机制（Cross-Attention）？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 标准 Cross-Attention：广泛用于编码器解码器结构。
- 多模态变体：如 Flamingo 模型中的门控跨注意力机制。
- 高效变体：用于处理超长外部序列的线性近似算法。

### 详细解答

结论：跨注意力机制在不同领域衍生出了多种变体，以适应特定的业务需求和数据模态。 原理与对比：1. 标准 Cross-Attention：最经典的形态，见于原始 Transformer，用于机器翻译等 NLP 任务。2. 多模态 Cross-Attention：在视觉-语言模型中，如 Flamingo 引入了门控跨注意力（Gated Cross-Attention），通过可学习的门控因子控制视觉信息注入文本的程度，保证预训练语言模型的稳定性。3. 空间/时间 Cross-Attention：在视频生成模型（如 Sora）中，Cross-Attention 被扩展到时空维度，以对齐文本与动态视频帧。 工程权衡：不同变体的核心权衡在于“融合效果”与“计算代价”。例如，门控机制增加了少量参数，但大幅降低了多模态微调的难度；而针对长序列的高效 Cross-Attention 则是在牺牲部分精度的前提下换取显存和速度的可行性。

### 案例模拟

面试官追问：Flamingo 中的 Gated Cross-Attention 为什么要加“门控（Gate）”？ 回答示例：Flamingo 是在冻结的预训练 LLM 基础上增加视觉能力的。如果在 LLM 层间直接插入标准的 Cross-Attention，初始随机化的参数会破坏 LLM 原本的特征分布。通过引入门控机制（初始化为 0），模型在训练初期表现得就像纯文本模型，随着训练进行，门控值逐渐增大，平滑地将视觉信息融入文本生成中，大大提升了训练的稳定性。

### 42. 五、Cross Attention 代码实现

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 需要定义三个独立的线性层分别生成 Q、K、V 特征矩阵。
- Q 的输入维度与 K/V 可不同，但内部特征维度必须对齐。
- 核心计算逻辑为缩悉点积注意力公式的矩阵乘法实现。

### 详细解答

结论：Cross Attention 的代码实现与 Self Attention 非常相似，关键区别在于前向传播（Forward）函数需要接收两个不同的输入张量。 原理与对比：在 PyTorch 中手写 Cross Attention 时，首先初始化 W_q, W_k, W_v 三个线性映射层。前向传播时，接收 x1（如文本）和 x2（如图像）。Q = W_q(x1)，而 K = W_k(x2)，V = W_v(x2)。接着计算注意力分数 scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)。如果需要，应用掩码（Mask），然后执行 attn = softmax(scores) @ V。最后通过一个输出线性层。 工程权衡：在实际工程中，为了极致的性能，我们极少手写上述逻辑，而是直接调用 torch.nn.functional.scaled_dot_product_attention。这不仅代码简洁，还能自动触发底层的 FlashAttention 或 Memory-Efficient Attention 算子，大幅降低显存占用并提升计算速度。

### 案例模拟

面试官追问：在 Cross Attention 中，如果 Q 的序列长度是 N，K 和 V 的序列长度是 M，输出的形状是什么？ 回答示例：输出的序列长度将与 Q 保持一致，即 N。因为注意力分数的形状是 (N, M)，表示 Q 中的每个元素对 K 中 M 个元素的关注度。这个分数矩阵与形状为 (M, D) 的 V 相乘后，得到的结果形状是 (N, D)。这体现了 Cross Attention 将源序列（M）的信息聚合到了目标序列（N）的维度上。

### 43. 介绍transformer算法

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

Transformer本身是一个典型的encoder-decoder模型，Encoder端和Decoder端均有6个Block，Encoder端的Block包括两个模块，多头self-attention模块以及一个前馈神经网络模块；Decoder端的Block包括三个模块，多头self-attention模块，多头Encoder-Decoder attention交互模块，以及一个前馈神经网络模块；需要注意：Encoder端和Decoder端中的每个模块都有残差层和Layer Normalization层。

### 案例模拟

面试表达可以这样组织：先用一句话回答“介绍transformer算法”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 44. 先做题，手写一个特殊的损失函数计算（带mask矩阵的要求的），手写rope

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[真实面经_nowcoder / 阿里大模型二面 / 阿里巴巴](https://www.nowcoder.com/feed/main/detail/37c33cc8ee9041ad91a35dc42f4ca239)

### 基础知识补充

- Mask矩阵常用于忽略Padding部分或实现因果注意力
- RoPE（旋转位置编码）通过复数乘法实现绝对位置的相对表达
- 需熟练掌握PyTorch的张量操作（如gather, einsum, view）

### 详细解答

结论：手写带Mask的损失函数和RoPE是考察候选人对Transformer底层细节和PyTorch工程实现能力的经典题目。 原理：1）带Mask的Loss：在NLP任务中，由于Batch内序列长度不同，需要用Padding对齐。计算Loss（如CrossEntropy）时，必须生成一个与目标等长的布尔Mask矩阵，将Padding位置的Loss置零，最后求和并除以有效Token数量。2）RoPE：其核心思想是将词向量的相邻两维两两分组，视为复数，然后乘以一个与位置相关的旋转矩阵。实现时，通常先生成位置索引，计算对应的sin和cos值，然后将输入张量切分、翻转符号并与sin/cos相乘，最后拼接。 工程权衡：在手写RoPE时，为了极致性能，通常会预计算sin和cos矩阵（Cos/Sin Cache），并在推理时直接切片使用，避免重复计算。

### 案例模拟

面试官追问：你手写的RoPE在处理超长上下文时，如何进行长度外推（Extrapolation）？ 回答：基础的RoPE在超过训练长度时性能会急剧下降。为了实现长度外推，我会在代码中引入位置插值（Position Interpolation）或NTK-Aware缩放。具体来说，在计算旋转角度时，不直接使用绝对位置索引 $m$，而是将其除以一个缩放因子 $s$（如 $s = L_{test} / L_{train}$），从而将未见过的长位置映射回模型训练时见过的区间内。在代码层面，只需在生成位置索引张量时除以这个缩放因子即可。

### 45. 六、Cross Attention 应用场景

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 机器翻译与文本摘要：编码器与解码器之间的文本对齐。
- 文本生成图像：如扩散模型中文本控制图像去噪过程。
- 多模态大模型：视觉特征与大型语言模型的跨模态融合。

### 详细解答

结论：Cross Attention 广泛应用于任何需要融合两个不同信息源的深度学习任务中，尤其是 Seq2Seq 和多模态领域。 原理与对比：在纯 NLP 领域，机器翻译、文本摘要、语音识别等任务依赖 Cross Attention 将编码器的源信息传递给解码器。在多模态领域，它的作用更加不可替代。例如在 Stable Diffusion 中，U-Net 的每一层都通过 Cross Attention 引入文本 Prompt 的特征，指导图像的生成方向；在视觉问答（VQA）模型中，文本 Query 通过 Cross Attention 从图像特征序列中提取与问题相关的视觉区域。 工程权衡：虽然 Cross Attention 融合效果好，但计算开销大。在一些对实时性要求极高的场景（如自动驾驶的多传感器融合），工程师可能会采用更轻量级的融合方式（如简单的特征拼接或 MLP）来替代 Cross Attention，以牺牲少量精度换取极低的推理延迟。

### 案例模拟

面试官追问：在语音识别（ASR）任务中，Cross Attention 是如何应用的？ 回答示例：在基于 Transformer 的 ASR 模型（如 Whisper）中，音频信号首先经过 Encoder 提取出声学特征序列（作为 K 和 V）。Decoder 在生成文本时，其隐藏状态作为 Q。通过 Cross Attention，Decoder 能够在生成每一个文字时，精准地对齐到音频序列中对应的发音片段，从而实现高准确率的语音转文本。

### 46. 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 跨注意力描述的是 Q 与 K/V 来源不同的注意力结构。
- 多头注意力描述的是将计算拆分为多个独立子空间的机制。
- 实际应用中，跨注意力几乎总是以多头注意力的形式存在。

### 详细解答

结论：Cross Attention 和多头注意力（MHA）并不是对立的概念，而是从不同维度描述注意力机制的术语，两者通常结合使用。 原理与对比：Cross Attention 关注的是“信息流向”，即 Query 来自一个模态/序列，Key 和 Value 来自另一个模态/序列，解决的是跨序列对齐问题。而多头注意力（MHA）关注的是“特征表达”，它将 Q、K、V 投影到多个低维空间并行计算注意力，最后拼接起来，解决的是单一注意力头容易忽略多维度特征的问题。在 Transformer Decoder 中，Cross Attention 层实际上就是一个“多头跨注意力层”。 工程权衡：将 Cross Attention 设计为多头结构，可以显著增强模型捕捉复杂对齐关系的能力（例如在翻译中，一个头关注语法，另一个头关注词义）。代价是增加了矩阵乘法的并行调度开销，但在现代 GPU 上，这种开销可以通过矩阵拼接被很好地掩盖。

### 案例模拟

面试官追问：在多头跨注意力中，各个头的计算是完全独立的吗？ 回答示例：是的，在计算注意力分数的阶段，各个头是完全并行的，它们拥有独立的 Q、K、V 投影权重矩阵，分别计算各自的相似度分布和加权结果。直到最后一步，所有头的输出才会被拼接（Concatenate）在一起，并通过一个输出线性层（Output Projection）进行融合，从而综合不同子空间提取到的跨序列信息。

### 47. 多头注意力机制MHA是Transformer模型中的核心组件, KV Cache和GQA优化的核心思想？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

KV Cache（Key-Value Cache）： - KV Cache是一种在自回归生成模型中使用的优化技术，它通过缓存历史输入的Key（K）和Value（V）来减少重复计算，从而提高推理效率。 - 在生成式模型的推理过程中，由于每个新生成的token都会与之前的token一起作为下一次输入，因此可以利用- KV Cache来避免对相同token的重复计算。 - KV Cache的显存占用会随着输入序列长度和输出序列长度的增加而线性增长，因此需要对其进行优化以适应更长的序列 Grouped Query Attention (GQA)： - GQA是一种介于MHA和MQA之间的折中方案，它将查询头（Query Heads）分组，并在每组中共享一个键头（Key Head）和一个值头（Value Head）。 - GQA既保留了多头注意力的一定表达能力，又通过减少内存访问压力来加速推理速度。 - GQA可以通过对已经训练好的模型进行微调来实现，使用mean pooling来生成共享的KV，这种方法在保持推理速度的同时也能保持较高的模型质量

### 案例模拟

面试表达可以这样组织：先用一句话回答“多头注意力机制MHA是Transformer模型中的核心组件, KV Cache和GQA优化的核心思想？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 48. 推理优化技术 Flash Attention 的作用是什么？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

Flash Attention 是一种高效的注意力机制实现，如共享张量核心和高效的内存使用，以减少内存占用并提高计算速度。这种方法特别适用于具有长序列和大型模型参数的场景，例如自然语言处理和推荐系统。 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“推理优化技术 Flash Attention 的作用是什么？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 49. 模型问题：为什么现在的llm大模型主要都是用RoPE位置编码而非其他？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

核心思想是将词嵌入向量的某些维度视为复数，然后通过旋转矩阵来编码位置信息。 https://www.zhihu.com/question/1821771428/answer/1961032470133737195 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“模型问题：为什么现在的llm大模型主要都是用RoPE位置编码而非其他？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 50. 模型问题：如何理解DETR中的object query的概念，要为 cross attention 提供更好的位置先验该如何设计模型？

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/VisionPerception/Reference.md)

### 基础知识补充

- 先讲模块组成
- 再讲计算路径
- 补复杂度与取舍

### 详细解答

- 位置部分则来自于 learnable queries； - 引入anchor box作为query提供位置先验，可参考DAB-DETR； 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“模型问题：如何理解DETR中的object query的概念，要为 cross attention 提供更好的位置先验该如何设计模型？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 51. 跨注意力机制（Cross-Attention）篇

- 主标签：Transformer结构
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Attention 升级面 / 未知](https://articles.zsxq.com/id_u67us9zex93d.html)

### 基础知识补充

- 该机制用于融合两个不同来源或模态的序列信息。
- Query 来自一个序列，Key 和 Value 来自另一个序列。
- 它是编码器与解码器架构中跨序列信息传递的桥梁。

### 详细解答

结论：跨注意力机制（Cross-Attention）是 Transformer 架构中用于处理双序列输入的关键组件，主要负责将外部上下文信息注入到当前处理的序列中。 原理与对比：在 Self-Attention 中，Q、K、V 均来自同一个输入序列，用于捕捉序列内部的依赖关系。而在 Cross-Attention 中，Q 通常来自目标序列（如 Decoder 的隐藏状态），而 K 和 V 来自源序列（如 Encoder 的输出）。通过这种方式，目标序列在生成每个元素时，都能动态地“关注”源序列中最相关的部分。 工程权衡：Cross-Attention 的计算复杂度取决于两个序列长度的乘积。在多模态大模型中，由于图像或音频序列通常较长，Cross-Attention 会成为计算瓶颈。工程上常通过降采样源序列（如 Perceiver Resampler）或引入局部注意力来优化性能。

### 案例模拟

面试官追问：在机器翻译任务中，Cross-Attention 是如何工作的？ 回答示例：在翻译时，Encoder 先把源语言（如英文）编码成一系列特征向量，作为 K 和 V。Decoder 在生成目标语言（如中文）时，其当前状态作为 Q。Cross-Attention 计算 Q 和 K 的相似度，找出当前要翻译的词对应英文句子中的哪些词，然后对 V 进行加权求和，将这些关键信息融入 Decoder，从而生成准确的中文词汇。
