# LLMs_interview_notes：一、大模型（LLMs）基础面

> 来源分组：LLMs_interview_notes
> 本页题目数：70
> 每题均包含基础知识补充、详细解答和案例模拟。

## 大模型（LLMs）基础面

### 1. 1 目前 主流的开源模型体系 有哪些？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- LLaMA系列：Meta开源的自回归语言模型，生态最为繁荣
- Qwen/ChatGLM系列：国内领先的中英双语优秀开源大模型
- MoE架构模型：如Mixtral，通过稀疏激活提升推理效率

### 详细解答

结论：当前主流开源大模型体系主要分为以LLaMA为代表的国际开源生态，以及以Qwen、ChatGLM为代表的国产双语生态。原理与对比：Meta的LLaMA系列（如LLaMA 2/3）确立了开源模型的基础架构标准，采用Decoder-only架构、SwiGLU激活函数和RoPE位置编码，其微调生态极其完善。国内方面，阿里的Qwen系列和智谱的ChatGLM系列在中文语境下表现优异，且在多模态和长文本处理上持续发力。此外，法国Mistral AI推动了MoE（混合专家）架构在开源界的普及。工程权衡：业务选型时，若主打英文或需丰富的社区插件，首选LLaMA系；若深耕中文业务，Qwen和GLM是首选；若对推理延迟和显存有严格限制，则应考虑MoE架构或较小参数量的模型。

### 案例模拟

面试官追问：如果公司要从零冷启动一个垂直领域的知识问答助手，你会选择哪个开源模型作为基座？回答示例：我会优先选择Qwen-7B或14B作为基座。首先，垂直领域问答主要面向国内用户，Qwen的中文原生词表和预训练语料能提供更好的中文理解力；其次，7B到14B的参数量在单张或双张A100上即可完成高效的SFT和部署，推理成本可控；最后，Qwen社区提供了完善的vLLM部署和微调脚本，能大幅缩短工程落地周期。

### 2. 2 prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Causal Decoder采用单向严格因果掩码，适合自回归生成
- Prefix Decoder对前缀部分双向可见，对生成部分单向可见
- Encoder-Decoder采用独立双向编码与单向解码，适合序列转换

### 详细解答

结论：这三种架构的核心区别在于注意力掩码（Attention Mask）的设计机制，分别对应不同的文本处理任务偏好。原理与对比：Causal Decoder（如GPT系列）使用严格的下三角掩码，每个Token只能看到之前的Token，零样本生成能力强，是目前大模型的主流。Prefix Decoder（如GLM）允许输入的前缀部分进行全双向注意力交互，而生成部分依然是单向自回归，这种设计在处理长上下文理解和生成结合的任务时更具优势。Encoder-Decoder（如T5、BART）则将双向理解和单向生成物理分离，通过交叉注意力连接，在机器翻译等Seq2Seq任务上表现最佳。工程权衡：Causal Decoder在训练效率和扩展性上占优，已成为千亿参数模型的事实标准；而Encoder-Decoder在同等参数下理解能力更强，但训练和部署成本较高。

### 案例模拟

面试官追问：为什么现在主流的百亿/千亿级大模型几乎都采用了Causal Decoder架构，而不是Encoder-Decoder？回答示例：主要有三个工程和理论原因。第一是训练效率，Causal Decoder无需区分输入和输出，所有Token都能参与预测，数据利用率高；第二是扩展性，实验表明Decoder-only架构在参数量扩大时，性能提升更稳定且容易实现并行化；第三是KV Cache的优化，单向架构在推理时只需缓存历史Token，而Encoder-Decoder需要同时维护交叉注意力的状态，显存管理更复杂。

### 3. 3 大模型LLM的 训练目标 是什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 预训练阶段的核心目标是Next Token Prediction
- SFT阶段目标是最小化生成回复与人类标注数据的交叉熵
- RLHF阶段目标是最大化奖励模型得分以对齐人类价值观

### 详细解答

结论：大模型（LLM）的训练目标随训练阶段的不同而变化，整体从“学习语言规律”向“对齐人类意图”演进。原理与解释：在预训练阶段，绝大多数主流模型采用自回归语言建模，其核心目标是Next Token Prediction，即通过最大化给定前文条件下下一个词出现的概率，让模型压缩并内化海量世界知识。在指令微调（SFT）阶段，目标转为行为克隆，通过计算模型输出与高质量人类指令回复之间的交叉熵损失，使模型学会遵循指令格式。在基于人类反馈的强化学习（RLHF）阶段，训练目标变为最大化奖励模型给出的标量评分，同时利用KL散度惩罚项防止模型偏离初始策略过远。工程权衡：Next Token Prediction极其简单且易于大规模并行，但可能导致模型产生幻觉；而RLHF能有效抑制有害输出，但训练极不稳定，工程实现难度极大。

### 案例模拟

面试官追问：Next Token Prediction 这种简单的训练目标，为什么能让模型产生复杂的逻辑推理能力？回答示例：这体现了“量变引起质变”的涌现现象。虽然目标只是预测下一个词，但为了在海量且复杂的语料（如代码、数学推导）中准确预测，模型被迫在隐藏层中构建对世界规律的内部表示，而不仅仅是统计词频。例如，要预测一段代码的下一个字符，模型必须隐式地理解变量作用域和语法树。当模型参数量和数据量突破一定阈值时，这种深层的模式识别就外化为了逻辑推理能力。

### 4. 4 涌现能力是啥原因？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 模型规模（参数量与训练数据）突破临界阈值引发的性能突变
- 复杂任务被隐式分解为多个基础子任务的联合概率乘积
- 评测指标的非平滑性（如精确匹配）在视觉上放大了突变效果

### 详细解答

结论：大模型的涌现能力是指当模型规模达到某一临界点时，在某些复杂任务上性能突然大幅提升的现象。其原因目前学术界尚无统一定论，但主要归结为任务复杂性、规模效应及评测指标特性。原理分析：首先，复杂任务（如多步推理）通常需要多个基础能力的组合。假设每个基础能力的准确率随模型规模线性增长，当所有步骤串联时，整体成功率是各步概率的乘积。只有当单步准确率极高时，联合概率才会突然从接近0飙升，形成涌现。其次，模型深度的增加使其能学习到更抽象的启发式特征。工程视角：斯坦福有研究指出，涌现能力部分可能是一种“海市蜃楼”，是因为采用了非平滑的评测指标（如Exact Match）。如果改用平滑指标，性能提升曲线会变得更加平滑。尽管如此，规模扩大带来的能力跃升在业务体感上是真实存在的。

### 案例模拟

面试官追问：既然涌现能力可能与评测指标有关，我们在实际业务中评估大模型时应该注意什么？回答示例：在业务评测中，不能仅依赖单一的精确匹配或选择题准确率。我会建立多维度的评估体系：首先，引入平滑指标，如计算生成文本与参考答案的语义相似度；其次，针对推理任务，评估中间步骤的合理性，而不仅仅是最终答案；最后，结合业务场景进行人工盲测，因为用户的真实体感往往比单纯的客观指标更能反映模型在特定任务上的“涌现”水平。

### 5. 为何现在的大模型大部分是Decoder only结构

- 主标签：LLM基础
- 来源条数：2
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Encoder-only如BERT擅长双向语言理解与特征表征。
- Decoder-only如GPT具备强大的自回归文本生成能力。
- Encoder双向注意力存在低秩问题可能削弱表达能力。

### 详细解答

结论：大模型普遍采用Decoder-only架构，是理论表达能力、训练效率与工程实现综合权衡的最优解。 原理与对比：NLP模型架构主要分三种。Encoder-only（如BERT）利用双向注意力擅长理解任务；Encoder-Decoder（如T5）适合翻译等Seq2Seq任务；Decoder-only（如GPT）通过单向自回归擅长生成任务。理论上，Encoder的双向注意力机制容易导致注意力矩阵的低秩问题，从而削弱模型的表达能力，而对于生成任务，引入双向注意力并无实质增益。 工程权衡：在同等参数量和推理成本下，Decoder-only架构无需像Encoder-Decoder那样处理复杂的交叉注意力，且在KV Cache优化、并行训练（如Megatron-LM）等工程实现上更加成熟高效。

### 案例模拟

面试追问：“Decoder-only架构在处理长文本输入时有什么工程挑战？” 回答示例：“主要挑战在于自注意力机制的计算复杂度随序列长度呈平方级增长，以及推理时KV Cache占用大量显存。在我们的长文本问答项目中，为了解决这个问题，我们引入了FlashAttention来优化显存读写并加速计算，同时采用了PagedAttention技术对KV Cache进行显存分页管理，使得单机能够支持的并发请求数提升了3倍以上。”

### 6. 6 简单 介绍一下 大模型【LLMs】？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 基于Transformer架构且参数量通常在百亿级别以上的神经网络
- 经历预训练、指令微调和人类价值观对齐三个核心训练阶段
- 具备强大的上下文学习、少样本泛化及复杂逻辑推理能力

### 详细解答

结论：大语言模型（LLMs）是基于深度学习（通常为Transformer架构）构建的超大规模自然语言处理模型，代表了当前人工智能在认知领域的最高水平。原理与演进：LLM的核心在于“大”，即海量的参数（通常10B以上）和海量的训练数据。通过自监督的预训练，模型将人类知识压缩进神经网络权重中，获得了对语言规律和世界常识的深刻理解。随后，通过SFT（指令微调）和RLHF（强化学习对齐），模型被塑造成能够听懂人类指令并给出安全、有用回复的AI助手。工程权衡：大模型虽然具备极强的零样本泛化能力和涌现能力，但其训练和推理成本极其高昂，且存在幻觉和知识更新滞后的问题。因此，在实际工程中，通常需要结合RAG（检索增强生成）或Agent架构，以弥补其在实时信息获取和垂直领域深度上的不足。

### 案例模拟

面试官追问：你提到大模型存在幻觉问题，在实际落地中，你们是如何缓解这一问题的？回答示例：在我们的知识库问答项目中，主要通过RAG（检索增强生成）技术来缓解幻觉。首先，我们将用户问题向量化，从本地向量数据库中检索出最相关的Top-K文档片段；然后，在构建Prompt时，明确要求大模型“仅根据以下参考资料回答问题，如果资料中没有相关信息，请回答不知道”。此外，我们还会引入后置校验模块，利用一个小参数量的NLI模型，检查大模型的输出是否与检索到的文档存在逻辑冲突，从而大幅降低了业务中的幻觉率。

### 7. 7 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- B代表Billion（十亿），指代模型包含的参数总数量。
- 参数量决定了模型的容量、表达能力和所需的计算资源。
- 175B对应GPT-3，是衡量大模型规模与算力需求的核心指标。

### 详细解答

结论：175B、60B等后缀代表大语言模型的参数量，其中“B”是Billion（十亿）的缩写，175B即1750亿个参数。 原理与工程权衡：参数是神经网络中通过训练学习到的权重和偏置。参数量越大，模型能够记忆的知识和捕捉的复杂模式就越多，通常表现出更强的泛化能力和涌现能力（如上下文学习、逻辑推理）。然而，庞大的参数量也带来了极高的工程代价。在显存占用上，以FP16精度计算，1B参数约占用2GB显存，175B模型仅加载权重就需要约350GB显存，必须依赖多卡分布式推理（如张量并行）。此外，训练成本、推理延迟和能耗也会随参数量呈线性或超线性增长。因此，当前工业界在追求大参数的同时，也在积极探索量化、剪枝等模型压缩技术。

### 案例模拟

面试官追问：“如果要在单张24G显存的RTX 3090上部署模型，最大能跑多大参数量的？” 回答示例：“在不使用量化的情况下，FP16精度下1B参数约占2G显存，24G显存最多只能加载10B左右的模型（还需预留KV Cache空间，实际通常跑7B模型）。如果采用INT8量化，可以部署约13B-14B的模型；若采用INT4量化，则勉强可以部署30B级别的模型，但会牺牲部分精度。”

### 8. 8 大模型【LLMs】具有什么优点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 具备强大的涌现能力，如上下文学习和复杂逻辑推理。
- 统一的生成范式，打破了传统NLP多任务定制的壁垒。
- 蕴含海量世界知识，能够实现跨领域、多语种的泛化。

### 详细解答

结论：大语言模型的核心优点在于其强大的泛化能力、涌现能力以及任务范式的统一性。 原理与对比：首先，随着参数量和训练数据的增加，大模型展现出小模型不具备的“涌现能力”，例如In-Context Learning（上下文学习）、思维链（Chain of Thought）推理和指令遵循能力，使其能在零样本或少样本下解决复杂问题。其次，大模型统一了NLP任务范式，将文本分类、翻译、摘要等传统上需要独立设计架构和微调的任务，全部转化为“Next-token Prediction”的生成任务，极大降低了下游任务的开发成本。最后，通过在海量语料上预训练，大模型内化了丰富的世界知识和常识，具备优秀的跨领域迁移能力和多轮对话交互能力，成为通用人工智能（AGI）的重要基石。

### 案例模拟

面试官追问：“大模型的上下文学习（ICL）和传统微调有什么区别？” 回答示例：“传统微调需要更新模型参数，依赖大量标注数据，且容易发生灾难性遗忘。而上下文学习不需要更新任何参数，只需在Prompt中提供几个示例，模型就能通过前向传播捕捉输入输出的模式并完成任务。这使得大模型在面对长尾需求或冷启动场景时，能够实现极低成本的快速适配，是其核心优势之一。”

### 9. 9 大模型【LLMs】具有什么缺点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 存在严重的幻觉问题，容易生成看似合理但错误的内容。
- 训练与推理成本极高，对算力、显存和带宽要求苛刻。
- 知识更新困难，难以实时获取最新信息且存在遗忘风险。

### 详细解答

结论：大语言模型的主要缺点集中在可靠性（幻觉）、计算成本、知识时效性以及安全性等方面。 原理与工程权衡：首先是“幻觉”问题，模型本质上是在做概率预测，缺乏对事实的真正理解，容易一本正经地胡说八道，这在医疗、法律等严谨领域是致命缺陷。其次是高昂的算力成本，千亿参数模型的预训练需要数千张GPU耗时数月，推理时也面临显存墙和内存带宽瓶颈，导致服务延迟高、吞吐量受限。第三，大模型的知识被固化在权重中，更新成本极高，无法像数据库那样实时增删改查，导致其对时效性强的信息无能为力。最后，大模型还面临数据隐私泄露、生成有害内容（如偏见、毒性）以及容易受到提示词注入攻击等安全合规风险。

### 案例模拟

面试官追问：“针对大模型的幻觉和知识滞后问题，工业界通常怎么解决？” 回答示例：“目前最主流的工程方案是RAG（检索增强生成）。在生成前，先通过向量数据库检索外部知识库中的相关文档，将其作为上下文拼接到Prompt中，让模型基于检索到的事实进行回答。这不仅解决了知识滞后问题，还能通过溯源降低幻觉。此外，也会结合Agent技术，让模型调用搜索引擎API来获取实时信息。”

### 10. 10 encoder-only, decoder-only, encoder-decoder的区别?

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Encoder-only采用双向注意力，擅长文本理解与分类任务。
- Decoder-only采用单向注意力，擅长自回归文本生成任务。
- Encoder-Decoder结合两者，适合翻译和摘要等序列到序列任务。

### 详细解答

结论：这三种架构是Transformer的不同变体，核心区别在于注意力机制的掩码方式和适用任务。 原理与对比：Encoder-only（如BERT）使用双向自注意力机制，每个Token都能看到上下文全局信息，因此在文本分类、命名实体识别等自然语言理解（NLU）任务上表现极佳，但无法直接用于自回归生成。Decoder-only（如GPT系列）使用因果掩码（Causal Mask），每个Token只能看到自己及之前的Token，严格遵循从左到右的生成逻辑，在自然语言生成（NLG）和零样本泛化上展现出统治力，是当前大模型的主流架构。Encoder-Decoder（如T5、BART）包含完整的编码和解码模块，Encoder双向理解输入，Decoder单向生成输出，两者通过交叉注意力连接，最适合机器翻译、文本摘要等Seq2Seq任务，但结构相对复杂，训练效率不如Decoder-only。

### 案例模拟

面试官追问：“为什么现在的大语言模型（如GPT-4、LLaMA）几乎都采用Decoder-only架构？” 回答示例：“首先，Decoder-only的自回归目标（Next-token Prediction）难度更大，迫使模型学习更深层的语言规律，有利于涌现能力的产生。其次，在工程实现上，Decoder-only结构统一，没有交叉注意力模块，KV Cache的管理和张量并行（TP）的切分更加高效。最后，Scaling Law在Decoder-only架构上得到了最充分的验证，扩展性极强。”

### 11. 11 BART、llama、gpt、t5、palm等主流模型异同点?

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- GPT、LLaMA、PaLM均采用Decoder-only架构，主导生成任务。
- BART和T5采用Encoder-Decoder架构，擅长Seq2Seq任务。
- LLaMA引入了RoPE旋转位置编码和RMSNorm等现代改进。

### 详细解答

结论：这些主流模型在架构选择、预训练目标和工程优化上各有侧重，反映了LLM从百花齐放到Decoder-only一统天下的演进。 对比与原理：BART和T5属于早期的Encoder-Decoder架构。BART通过破坏文本再重建（如掩码、打乱）进行预训练，T5则将所有NLP任务统一为文本到文本的格式，它们在翻译和摘要等微调任务上表现优异。GPT系列、PaLM和LLaMA则是Decoder-only架构的代表。GPT确立了自回归生成的范式；PaLM（Google）在多语言和推理任务上发力，使用了SwiGLU激活函数和并行Transformer层；LLaMA（Meta）则是开源界的标杆，它吸收了前沿优化，如使用RoPE（旋转位置编码）替换绝对位置编码，使用RMSNorm提升训练稳定性，并采用SwiGLU。目前，LLaMA的架构已成为开源大模型的事实标准。

### 案例模拟

面试官追问：“LLaMA相比于早期的GPT-3，在网络结构上做了哪些关键改进？” 回答示例：“LLaMA主要做了三点改进：一是前置归一化（Pre-normalization）并替换为RMSNorm，去除了均值计算，提升了训练稳定性与速度；二是采用SwiGLU激活函数替换ReLU，增强了非线性表达能力；三是使用RoPE（旋转位置编码）代替绝对位置编码，更好地捕捉了Token之间的相对距离，且具备一定的长度外推潜力。”

### 12. 12 prefix LM 和 causal LM 区别是什么?

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Causal LM采用严格的单向因果掩码，只能看到历史信息。
- Prefix LM对前缀部分使用双向掩码，生成部分使用单向掩码。
- Prefix LM结合了双向理解与单向生成的优势，常用于GLM。

### 详细解答

结论：Prefix LM（前缀语言模型）和 Causal LM（因果语言模型）的核心区别在于注意力掩码（Attention Mask）的设计机制不同。 原理与对比：Causal LM（如GPT系列）采用严格的下三角掩码矩阵，在整个序列中，任何一个Token在计算注意力时都只能看到它自己和它之前的Token。这种设计完全契合自回归生成，但在处理输入的Prompt时，无法利用下文信息，理解能力稍弱。Prefix LM（如ChatGLM、UniLM）则将序列分为“前缀（Prefix）”和“生成”两部分。对于前缀部分（通常是用户的输入Prompt），注意力掩码是全开的（双向可见），使得模型能像BERT一样充分理解上下文；而对于生成部分，则采用因果掩码以保证自回归生成。工程权衡上，Causal LM结构更简单，KV Cache管理更统一；Prefix LM在少样本理解任务上表现更好，但掩码矩阵的实现相对复杂。

### 案例模拟

面试官追问：“在实际训练中，Prefix LM 的掩码矩阵具体长什么样？” 回答示例：“假设序列总长度为N，其中前缀长度为P。掩码矩阵是一个N×N的矩阵。在左上角P×P的区域，全为1（双向可见）；在右上角P×(N-P)的区域，全为0（前缀看不到未来生成的词）；在下方的(N-P)×N区域，前P列全为1（生成部分能看到所有前缀），右侧的(N-P)×(N-P)区域是一个下三角矩阵（生成部分严格遵循单向自回归）。”

## Layer normalization 篇

### 13. Layer Norm 的计算公式写一下？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- Layer Norm在特征维度上计算均值和方差进行归一化。
- 公式包含减去均值、除以标准差，并引入可学习的缩放平移。
- 能够有效缓解内部协变量偏移，稳定深层Transformer训练。

### 详细解答

结论：Layer Normalization（层归一化）是对单个样本的所有特征维度进行归一化，其计算公式为：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$。 原理与解释：在公式中，$x$ 是输入向量，$\mu$ 是该向量在特征维度（Hidden Size）上的均值，$\sigma^2$ 是方差。$\epsilon$ 是一个极小的常数（如1e-5），用于防止分母为零。归一化后的结果会乘以可学习的缩放参数 $\gamma$（Gamma）并加上平移参数 $\beta$（Beta），以恢复网络原本的表达能力。 对比与工程权衡：与Batch Norm不同，Layer Norm的均值和方差是在单个样本的特征维度上计算的，因此它不依赖于Batch Size的大小，非常适合处理变长序列的NLP任务（如RNN、Transformer）。在Transformer中，LN通常放置在多头注意力和前馈网络前后，能有效控制梯度爆炸/消失，加速模型收敛。

### 案例模拟

面试官追问：“在Transformer中，Post-LN和Pre-LN有什么区别？大模型通常用哪个？” 回答示例：“Post-LN是将Layer Norm放在残差连接之后，早期Transformer用这种，但深层网络容易出现梯度消失，训练不稳定，通常需要Warm-up。Pre-LN是将Layer Norm放在注意力或FFN模块之前、残差连接内部。大模型（如GPT-3）普遍采用Pre-LN，因为它的主干网络有一条直通的残差路径，梯度回传更顺畅，即使在极深的网络下也能保持训练稳定，不需要复杂的学习率预热。”

### 14. RMS Norm 的计算公式写一下？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- RMS Norm是Layer Norm的简化版，去除了均值计算。
- 公式为输入除以均方根，再乘以可学习的缩放参数Gamma。
- 降低了计算开销，提升了训练速度，被LLaMA等大模型广泛采用。

### 详细解答

结论：RMS Norm（Root Mean Square Normalization）的计算公式为：$y = \frac{x}{\text{RMS}(x)} \cdot \gamma$，其中 $\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$。 原理与对比：在传统的Layer Norm中，需要先计算均值，将输入减去均值（中心化），再除以标准差。而RMS Norm的作者发现，Layer Norm的成功主要归功于缩放不变性（方差部分），而不是平移不变性（均值部分）。因此，RMS Norm直接去除了计算均值和减去均值的步骤，仅使用均方根（RMS）对输入进行缩放归一化。$\gamma$ 依然是可学习的缩放参数，且通常去除了偏置参数 $\beta$。 工程权衡：由于省去了均值的计算和广播操作，RMS Norm的计算图更简单，显存访问量减少。在实际工程中，RMS Norm相比Layer Norm能带来约10%到50%的计算提速，同时保持了几乎相同的模型性能和训练稳定性，因此成为LLaMA、Gemma等现代开源大模型的标配。

### 案例模拟

面试官追问：“在实现RMS Norm的前向传播时，如何避免计算平方和时出现数值溢出（FP16下）？” 回答示例：“在FP16精度下，直接对特征向量求平方和极易超出最大表示范围（65504）导致溢出（NaN）。工程上的标准做法是：在计算RMS之前，先将输入张量 $x$ 转换为FP32精度，在FP32下计算平方、求均值、加 $\epsilon$ 并开根号，得到缩放因子后，再将其转换回FP16与原输入相乘。这样既保证了数值稳定性，又不会显著增加计算耗时。”

### 15. RMS Norm 相比于 Layer Norm 有什么特点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 移除均值计算，仅保留均方根（RMS）进行归一化处理。
- 假设输入特征均值为零，大幅减少计算开销与内存访问。
- 在LLaMA等大模型中广泛使用，提升训练速度且性能不降。

### 详细解答

RMS Norm（Root Mean Square Normalization）相比Layer Norm最大的特点是计算效率更高且性能相当。结论上，它通过移除均值计算，将计算成本降低了约10%-50%。原理方面，Layer Norm需要计算输入特征的均值和方差来进行平移和缩放，而RMS Norm假设特征均值接近于零，直接计算均方根（RMS）进行缩放。对比来看，虽然去除了均值平移（中心化）操作，但大量实验表明，模型性能并未受到明显影响，反而因为减少了同步开销和内存带宽占用，显著提升了训练吞吐量。在工程权衡上，现代大语言模型（如LLaMA、Gemma）几乎全部采用RMS Norm替代Layer Norm，以在千亿参数规模下追求极致的训练效率。

### 案例模拟

面试官追问：“如果输入数据的均值偏移严重，RMS Norm还会有效吗？” 回答：“如果均值偏移极大，RMS Norm的假设失效，可能会导致激活值分布异常，影响训练稳定性。但在Transformer架构中，由于权重初始化和残差连接的特性，内部激活的均值通常保持在零附近。如果确实遇到偏移问题，可以在RMS Norm前引入轻量级的偏置项，或者在数据预处理阶段做好严格的零均值化。”

### 16. Deep Norm 思路？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 结合Post-LN的性能与Pre-LN的稳定性，缓解梯度消失。
- 通过缩放残差连接的权重，控制深层网络激活值的方差。
- 核心公式引入常数alpha对残差分支进行缩放以稳定训练。

### 详细解答

Deep Norm的核心思路是通过数学推导严格控制Transformer深层网络中激活值的方差爆炸问题，从而实现千层以上网络的稳定训练。结论上，它结合了Post-LN在性能上的优势和Pre-LN在训练稳定性上的优势。原理上，随着网络层数加深，残差连接会导致激活值的方差不断累积，引发梯度消失或爆炸。Deep Norm在执行残差连接前，引入一个与网络深度相关的缩放因子$\alpha$（通常大于1）对残差分支进行缩放，同时在初始化阶段对权重进行特定比例的缩放。对比传统Pre-LN，Deep Norm允许模型采用Post-LN架构而不崩溃，从而获得更好的最终泛化性能。在工程实践中，这种方法使得训练1000层以上的Transformer成为可能，极大拓展了模型深度的上限。

### 案例模拟

面试官追问：“Deep Norm中的缩放因子alpha是如何确定的？” 回答：“alpha的值是根据网络架构（如Encoder或Decoder）和层数N通过数学推导得出的。对于标准Transformer Decoder，alpha通常设为$(2N)^{0.25}$。在实际工程中，我们不需要手动调参，只需根据模型总层数套用公式即可。这在训练超深模型（如GLM-130B早期实验）时，能有效避免前期训练loss飞坡的问题。”

### 17. 写一下 Deep Norm 代码实现？

- 主标签：经典算法与编程
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 核心操作为在残差相加时对旧状态乘以缩放因子alpha。
- 需配合特定的权重初始化策略，缩放投影层和前馈层权重。
- 伪代码逻辑：LayerNorm(alpha * x + f(x))，注意缩放位置。

### 详细解答

Deep Norm的代码实现非常简洁，核心在于前向传播时的残差缩放以及初始化时的权重缩放。结论上，只需在标准Post-LN的基础上修改几行代码即可完成。原理实现上，前向传播公式为 x = LayerNorm(alpha * x + f(x))，其中 f(x) 是注意力层或前馈层，alpha 是根据层数计算的常数。对比标准残差 x = LayerNorm(x + f(x))，Deep Norm放大了恒等映射的权重。在工程实现中，除了前向传播的修改，还必须在模型初始化阶段对 f(x) 内部的权重（如Attention的输出投影层、FFN的第二层）乘以一个缩放因子 beta。这种实现方式计算开销极小，不需要引入新的可学习参数，且能完美兼容现有的深度学习框架（如PyTorch）的算子融合优化。

### 案例模拟

面试官追问：“能口述一下PyTorch中Deep Norm前向传播的核心代码吗？” 回答：“可以。假设输入为x，子模块为sublayer，alpha为预计算好的常数。代码大致为：residual = x * self.alpha; hidden_states = sublayer(x); output = self.layer_norm(residual + hidden_states)。同时在初始化函数中，我会遍历子模块的权重，将其乘以预设的beta值。这种实现既保持了代码简洁，又确保了深层梯度稳定。”

### 18. Deep Norm 有什么优点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 极大提升模型训练稳定性，支持训练千层以上的超深网络。
- 兼具Post-LN的高模型性能与Pre-LN的易收敛特性。
- 无需引入额外可学习参数，计算开销与标准LN完全一致。

### 详细解答

Deep Norm的优点主要集中在突破模型深度限制、兼顾性能与稳定性以及极低的工程改造成本上。结论上，它是目前解决超深Transformer训练不稳定问题的最优雅方案之一。原理上，它通过理论推导出的常数缩放因子，严格界定了深层激活值和梯度的方差边界，避免了梯度消失或爆炸。对比Pre-LN，Pre-LN虽然稳定但深层网络往往退化为恒等映射，导致性能受限；而Post-LN性能好但极难训练。Deep Norm完美融合了两者优点，使得模型既能享受Post-LN的性能红利，又能像Pre-LN一样稳定收敛。在工程权衡上，Deep Norm不需要像其他方法（如ReZero）那样引入新的可学习参数，也不增加前向和反向的计算复杂度，对算子融合极其友好，是极具性价比的架构改进。

### 案例模拟

面试官追问：“既然Deep Norm这么好，为什么现在的开源大模型（如LLaMA）更多使用Pre-RMSNorm而不是Deep Norm？” 回答：“这是因为当前大模型的发展趋势是‘做宽’而不是‘做深’。LLaMA等模型层数通常在32到80层之间，在这个深度下，Pre-RMSNorm已经足够稳定，且RMSNorm计算效率更高。Deep Norm的真正优势在于1000层以上的极端深度。如果未来业务需要极深的模型来增强逻辑推理能力，Deep Norm将是首选。”

### 19. 1 LN 在 LLMs 中的不同位置 有什么区别么？如果有，能介绍一下区别么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- Post-LN置于残差相加后，性能上限高但深层极易梯度爆炸。
- Pre-LN置于子层计算前，训练稳定但深层网络易发生退化。
- Sandwich-LN在残差分支内外均归一化，试图兼顾但易致训练崩溃。

### 详细解答

Layer Norm在LLM中的位置主要分为Post-LN、Pre-LN和Sandwich-LN，其核心区别在于对梯度流和模型深度的影响。结论上，Pre-LN是目前大模型的主流选择，因为它在工程上最稳定。原理解释上，Post-LN（x = LN(x + f(x))）在残差相加后归一化，导致主干网络的方差随层数累积，深层梯度极易爆炸，需要极小的学习率和Warmup；Pre-LN（x = x + f(LN(x))）在进入子层前归一化，主干网络保持纯粹的残差连接，梯度能无损直达浅层，训练极其稳定。对比来看，Post-LN的最终泛化性能通常略优于Pre-LN，因为Pre-LN的深层子层对最终输出的贡献会逐渐衰减。工程权衡上，为了在千亿参数规模下保证训练不崩溃，业界几乎一致牺牲了Post-LN微小的性能优势，选择了Pre-LN（或Pre-RMSNorm）。

### 案例模拟

面试官追问：“Pre-LN深层网络退化的问题在实际业务中有什么影响？如何缓解？” 回答：“深层退化意味着增加模型层数带来的收益递减，导致参数利用率下降。在实际业务中，这会表现为模型规模扩大但效果提升不明显。缓解方法包括使用Deep Norm替代Pre-LN，或者采用更复杂的初始化策略（如T-Fixup）。不过目前工业界更倾向于通过增加模型宽度（隐藏层维度）或引入MoE架构来提升容量，规避深层退化问题。”

### 20. LLMs 各模型分别用了 哪种 Layer normalization？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 早期模型如BERT和GPT-1采用Post-LN，追求极致泛化性能。
- GPT-3、PaLM等主流模型转向Pre-LN，确保大参数量训练稳定。
- LLaMA、Gemma、Qwen等现代开源模型全面采用Pre-RMSNorm。

### 详细解答

随着大语言模型的发展，Layer Normalization的类型和位置经历了明显的演进。结论上，演进路线为：Post-LN -> Pre-LN -> Pre-RMSNorm。早期模型（如BERT、Transformer原论文）采用Post-LN，因为当时模型层数较浅（12-24层），Post-LN能提供更好的性能上限。随着模型规模扩大到百亿、千亿参数（如GPT-2、GPT-3、Megatron-LM），训练稳定性成为首要矛盾，业界全面转向Pre-LN。近年来，为了进一步提升训练吞吐量，LLaMA、ChatGLM、Qwen、Mistral等绝大多数现代开源大模型都采用了Pre-RMSNorm。对比来看，Pre-RMSNorm去除了均值计算，在保持Pre-LN稳定性的同时，将归一化层的计算速度提升了约20%。工程权衡上，这种演进体现了算法设计向大规模分布式训练效率妥协的趋势。

### 案例模拟

面试官追问：“除了RMSNorm，还有哪些变体被用在现代大模型中？” 回答：“除了RMSNorm，还有DeepSpeed提出的DeepNorm（用于GLM-130B早期版本），以及Sandwich-LN（曾用于CogView，但因不稳定被弃用）。另外，有些模型为了解决RMSNorm可能带来的均值偏移问题，会使用带偏置的RMSNorm，或者在特定层（如QKV投影后）引入额外的LayerNorm（如Qwen2中的Q-K Norm）来稳定注意力机制的计算。”

## LLMs 激活函数篇

### 21. 1 介绍一下 FFN 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- 标准FFN包含两次线性变换与中间的非线性激活函数。
- 公式为 FFN(x) = Activation(xW1 + b1)W2 + b2。
- 现代LLM常采用SwiGLU变体，引入门控机制提升表达能力。

### 详细解答

FFN（Feed-Forward Network）是Transformer中负责非线性特征映射的核心模块。结论上，它通过升维再降维的操作，极大地丰富了模型的特征表达空间。原理上，标准FFN的计算公式为 $FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$（以ReLU为例）。输入 $x$ 首先通过权重 $W_1$ 投影到一个更高维度的空间（通常是输入维度的4倍），经过非线性激活函数后，再通过 $W_2$ 投影回原始维度。对比注意力机制（负责全局信息交互），FFN主要负责每个Token位置的局部特征非线性变换。在工程实践与现代LLM演进中，标准FFN逐渐被GLU（Gated Linear Unit）变体取代，如LLaMA使用的SwiGLU。SwiGLU的公式变为 $(Swish(xW_1) \otimes xW_3)W_2$，去除了偏置项，并通过门控机制显著提升了模型的收敛速度和最终性能。

### 案例模拟

面试官追问：“为什么FFN中间层的维度通常要放大4倍？可以不放大吗？” 回答：“放大维度是为了将特征映射到高维空间，使得非线性激活函数能更好地分离和提取复杂特征，这类似于SVM中核函数的作用。如果不放大，模型的拟合能力会大幅下降。在工程权衡上，4倍是一个经验值，能在参数量、计算量和模型性能之间取得较好的平衡。在现代使用SwiGLU的模型中，为了保持参数量与标准FFN一致，通常会将放大倍数调整为8/3。”

### 22. 2 介绍一下 GeLU 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- GeLU结合了Dropout、区域门控与ReLU的特性。
- 核心公式为 x 乘以标准正态分布的累积分布函数 Phi(x)。
- 工程上常使用Tanh或Sigmoid函数进行快速近似计算。

### 详细解答

GeLU（Gaussian Error Linear Unit）是现代大模型中广泛使用的非线性激活函数。结论上，它通过引入随机正则化思想，比ReLU表现出更平滑的梯度和更好的泛化能力。原理上，GeLU的计算公式为 $GeLU(x) = x \cdot \Phi(x)$，其中 $\Phi(x)$ 是标准正态分布的累积分布函数。它的直观理解是：输入 $x$ 越小，被保留的概率越低（趋于0）；输入越大，被保留的概率越高（趋于恒等映射）。对比ReLU的硬截断（小于0直接为0），GeLU在0附近是平滑过渡的，这使得模型在反向传播时梯度更加稳定，避免了神经元死亡问题。在工程实现上，由于计算高斯累积分布函数极其耗时，通常采用近似公式：$0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$。这种近似在保证精度的同时，大幅提升了GPU上的计算吞吐量。

### 案例模拟

面试官追问：“GeLU和Swish激活函数有什么联系和区别？” 回答：“Swish的公式是 $x \cdot \sigma(\beta x)$，当 $\beta=1.702$ 时，Swish极其接近GeLU。两者的核心思想非常相似，都是基于输入值自身来控制门控概率，且在负半轴都有一个非单调的‘小坑’（允许微小的负值通过）。区别在于GeLU有严格的概率论推导背景（基于高斯误差），而Swish是基于NAS（神经架构搜索）实验得出的。在实际业务中，两者的性能差异微乎其微，选择哪种更多取决于框架底层的算子优化程度。”

### 23. 3 介绍一下 Swish 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- Swish激活函数公式为f(x)=x*sigmoid(βx)。
- 具备平滑、非单调特性，能缓解梯度消失问题。
- 在深层网络中表现优于ReLU，提升模型收敛速度。

### 详细解答

Swish 是一种由谷歌提出的自门控激活函数，其计算公式为 $f(x) = x \cdot \sigma(\beta x)$，其中 $\sigma$ 是 Sigmoid 函数，$\beta$ 是可学习参数或常数（通常设为1，此时也称SiLU）。结论上，它在现代大模型中逐渐替代了传统的 ReLU。原理上，与 ReLU 相比，Swish 最大的特点是“平滑”且“非单调”。在 $x < 0$ 的区域，它不会像 ReLU 那样直接截断为0，而是保留了微小的负梯度，这有效缓解了神经元死亡问题（Dying ReLU）。同时，其处处可导的平滑特性使得损失景观更加平滑，有利于优化器寻找全局最优解。在工程权衡上，Swish 的计算开销比 ReLU 大，因为它引入了指数运算，但在现代大模型（如 LLaMA）中，这种计算成本的增加通常能被模型性能的显著提升所弥补。

### 案例模拟

面试官追问：“Swish中的参数β有什么作用？如果β趋于无穷大或0会怎样？” 回答：“β控制了Swish函数的形状。当β趋于0时，Swish退化为线性函数f(x)=x/2；当β趋于无穷大时，Sigmoid部分趋近于阶跃函数，Swish就退化成了ReLU。在实际工程中，β通常作为可学习参数，让模型在训练过程中自适应地调整非线性程度，从而在不同网络层中找到最适合的激活形态。”

### 24. 4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- GLU核心公式为(xW+b)⊗σ(xV+c)，引入门控机制。
- 结合FFN块，将传统两层MLP替换为带门控的线性变换。
- 能够有效缓解梯度消失，提升大语言模型的表达能力。

### 详细解答

门控线性单元（GLU，Gated Linear Unit）是一种通过门控机制控制信息流动的网络结构。结论上，GLU 变体在各大语言模型中被广泛采用，是目前架构优化的标配。在 Transformer 的 FFN（前馈神经网络）块中，传统的结构是两层线性映射中间加一个激活函数：$FFN(x) = f(xW_1 + b_1)W_2 + b_2$。而使用 GLU 的 FFN 块将其改进为：$GLU(x) = (xW_1 + b_1) \otimes \sigma(xW_2 + b_2)$，然后再乘以输出权重 $W_3$。其中 $\otimes$ 表示逐元素相乘。原理上，它通过门控机制动态决定哪些特征应该被激活或抑制。相比传统 FFN，GLU 增加了参数量和计算量（多了一次线性投影），但显著提升了模型的非线性表达能力和训练稳定性。

### 案例模拟

面试官追问：“在Transformer中引入GLU会带来哪些工程上的挑战？” 回答：“主要挑战在于显存和计算量的增加。标准FFN只需两次矩阵乘法，而GLU变体需要三次（两个输入投影，一个输出投影）。为了保持参数量与原模型一致，通常需要将隐藏层维度（如4d）按比例缩小（如缩减为8/3d）。在推理部署时，由于多了一个投影分支，对显存带宽的压力更大，常需要算子融合来优化性能。”

### 25. 5 介绍一下 使用 GeLU 的 GLU 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- GeGLU将标准GLU中的Sigmoid替换为GeLU激活函数。
- 计算公式为GeGLU(x)=(xW_1)⊗GeLU(xW_2)。
- 结合了GeLU的正则化效应与GLU的动态门控信息筛选能力。

### 详细解答

使用 GeLU 的 GLU 块（通常称为 GeGLU）是门控线性单元的一种重要变体。其核心结论是：将标准 GLU 中的 Sigmoid 激活函数替换为 GeLU（高斯误差线性单元），从而结合两者的优势。具体计算公式为：$GeGLU(x, W, V) = (xW) \otimes GeLU(xV)$，在 Transformer 的 FFN 中，完整输出为 $GeGLU(x)W_o$（通常省略偏置项）。原理上，GeLU 结合了 Dropout 的随机正则化思想和 ReLU 的非线性，其平滑特性比 Sigmoid 更适合深层网络的梯度传播；而 GLU 的逐元素相乘机制则提供了强大的特征筛选能力。对比传统 FFN 或标准 GLU，GeGLU 在多项 NLP 任务中展现出更优的收敛速度和最终困惑度。在工程实践中，T5 模型就广泛验证了 GeGLU 的有效性，性价比极高。

### 案例模拟

面试官追问：“GeLU的计算比较复杂，在实际工程中如何优化GeGLU的计算效率？” 回答：“GeLU的精确计算包含误差函数erf，计算代价较高。在实际工程中，通常采用基于Tanh或多项式的近似计算方法。此外，在CUDA算子层面，我们会将xW和xV的矩阵乘法合并为一个大的GEMM操作，然后编写一个融合算子（Fused Kernel）在显存中一次性完成切分、GeLU激活和逐元素相乘，大幅减少显存读写开销。”

### 26. 6 介绍一下 使用 Swish 的 GLU 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- SwiGLU将GLU的激活函数替换为Swish(或SiLU)。
- 公式为SwiGLU(x)=(xW_1)⊗Swish(xW_2)。
- 是LLaMA等主流开源大模型标配，显著提升模型收敛效果。

### 详细解答

使用 Swish 的 GLU 块（即 SwiGLU）是目前大语言模型（如 LLaMA 系列、ChatGLM 等）中最主流的 FFN 架构。结论上，它通过将 GLU 中的激活函数替换为 Swish（通常使用 $\beta=1$ 的 SiLU），实现了模型性能的显著跃升。其计算公式为：$SwiGLU(x) = (xW_1) \otimes Swish(xW_2)$，最终 FFN 输出为 $SwiGLU(x)W_3$。原理在于，Swish 函数的非单调性和平滑性使得梯度流动更加稳定，而门控机制允许模型动态地放大重要特征、抑制噪声。对比 GeGLU 或传统 ReLU FFN，SwiGLU 在同等参数量下能取得更低的困惑度（PPL）和更好的下游任务表现。在工程权衡上，为了保持与原 Transformer 相同的参数量和计算量，SwiGLU 通常会将隐藏层维度从 $4h$ 缩小到 $\frac{8}{3}h$，从而在不增加硬件负担的前提下最大化架构收益。

### 案例模拟

面试官追问：“为什么LLaMA系列模型要将SwiGLU的隐藏层维度设置为8/3倍的隐藏层大小，而不是传统的4倍？” 回答：“这是出于参数量和计算量对齐的工程考量。传统FFN有两个权重矩阵，参数量为8h^2。SwiGLU有三个权重矩阵。如果保持4h的中间维度，参数量会变成12h^2。为了公平对比并控制显存占用，将中间维度设为(8/3)h，这样三个矩阵的参数量总和刚好是3 * h * (8/3)h = 8h^2，与传统FFN严格对齐。”

### 27. 7 各LLMs 都使用哪种激活函数？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- 早期模型如GPT-2和BERT主要使用GeLU激活函数。
- LLaMA、Qwen等现代开源大模型普遍采用SwiGLU结构。
- T5模型和部分Google系模型倾向于使用GeGLU变体结构。

### 详细解答

在大语言模型（LLMs）的发展历程中，激活函数的选择经历了从简单到复杂、从单一到门控的演进。结论上，目前 SwiGLU 是开源大模型的绝对主流，而早期模型多用 GeLU。具体来看：1) GeLU：BERT、GPT-2、GPT-3 等早期经典模型广泛使用 GeLU，它结合了正则化与非线性，比 ReLU 表现更好。2) GeGLU：Google 的 T5 模型引入了 GeGLU，证明了门控机制在 FFN 中的优越性。3) SwiGLU (SiLU)：自 PaLM 和 LLaMA 系列发布以来，SwiGLU 凭借其在缩放定律下的优异表现，成为了 LLaMA、Qwen、Mistral 等几乎所有主流开源模型的标配。工程权衡上，虽然 SwiGLU 和 GeGLU 的计算复杂度高于普通 GeLU（需要三次矩阵乘法），但它们带来的非线性表达能力提升远超计算成本的增加，是目前性价比最高的架构选择。

### 案例模拟

面试官追问：“既然SwiGLU效果这么好，为什么不把所有网络层（比如Attention内部）的激活函数都换成SwiGLU？” 回答：“首先，Attention机制的核心是Softmax，用于计算注意力权重，其概率分布特性是SwiGLU无法替代的。其次，SwiGLU主要用于FFN层来做特征的非线性映射和升降维。如果在其他地方滥用，不仅会破坏原有的数学逻辑，还会导致参数量和计算量急剧膨胀。工程上，FFN占据了Transformer约2/3的参数，在这里使用SwiGLU收益最大。”

### 28. 8 Adam优化器和SGD的区别？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- SGD仅依赖当前梯度更新，容易陷入局部最优或鞍点。
- Adam结合了动量法和RMSProp，计算一阶与二阶矩估计。
- Adam能自适应调整各参数学习率，收敛速度通常远快于SGD。

### 详细解答

Adam 和 SGD 是深度学习中最常用的两种优化器，核心区别在于更新策略和学习率的自适应性。结论上，SGD 适合精调且泛化上限高，而 Adam 收敛极快、是大模型训练的标配。原理方面，SGD（随机梯度下降）每次仅根据当前批次的梯度进行参数更新，即使加上 Momentum（动量），所有参数也共享同一个全局学习率；而 Adam 同时维护了梯度的一阶矩（动量，指示方向）和二阶矩（未中心化的方差，指示步长），能够为每个参数动态计算自适应的学习率。对比来看，Adam 在面对稀疏梯度或复杂损失地形（如 Transformer 架构）时，能迅速穿越平缓区域并避免震荡，收敛速度远超 SGD。但在工程权衡上，Adam 需要额外保存一阶和二阶动量状态，显存占用是 SGD 的三倍，这也是大模型训练必须引入 ZeRO 优化等技术的原因。

### 案例模拟

面试官追问：“大模型训练中Adam优化器的显存占用很大，有什么工程手段可以优化？” 回答：“大模型训练中Adam的状态通常占显存的大头。工程上主要有三种优化手段：1. 使用DeepSpeed的ZeRO系列技术，将优化器状态切分到不同GPU上，打破显存墙；2. 使用低精度优化器，如8-bit Adam，通过量化技术将32位的状态压缩为8位，大幅降低显存；3. 使用Adafactor等变体优化器，通过矩阵分解的方式近似二阶矩，从而减少状态存储需求。”

## Attention 升级面

### 29. 1 传统 Attention 存在哪些问题？

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

### 30. 2 Attention 有哪些 优化方向？

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

### 31. 3 Attention 变体有哪些？

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

### 32. 4.1 Multi-head Attention 存在什么问题？

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

### 33. 4.2 介绍一下 Multi-Query Attention？

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

### 34. 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？

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

### 35. 4.4 Multi-Query Attention 这样做的好处是什么？

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

### 36. 4.5 有 哪些模型 是 使用 Multi-Query Attention？

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

### 37. 5.1 什么是 Grouped-query Attention？

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

### 38. 5.2 有哪些大模型使用 Grouped-query Attention？

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

### 39. 6.1 为什么需要 FlashAttention？

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

### 40. 6.2 简单介绍一下 FlashAttention？

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

### 41. 6.3 简单介绍一下 FlashAttention 核心？

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

### 42. 6.4 介绍一下 FlashAttention 优点？

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

### 43. 6.5 介绍一下 FlashAttention 代表模型？

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

### 44. 7 并行 transformer block

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

### 45. Attention计算复杂度以及如何改进

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

### 46. 9.1 简单介绍一下 Paged Attention？

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

### 47. 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?

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

### 48. 跨注意力机制（Cross-Attention）篇

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

### 49. 一、为什么需要 跨注意力机制（Cross-Attention）？

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

### 50. 二、介绍一些 跨注意力机制（Cross-Attention）？

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

### 51. 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？

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

### 52. 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇

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

### 53. 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？

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

### 54. 五、Cross Attention 代码实现

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

### 55. 六、Cross Attention 应用场景

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

### 56. 七、Cross Attention 的优势和挑战？

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

## transformers 操作篇

### 57. 1. 如何 利用 transformers 加载 Bert 模型？

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

### 58. 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？

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

## LLMs 损失函数篇

### 59. 一、介绍一下 KL 散度？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 衡量两个概率分布之间差异的非对称性度量指标。
- 公式为P分布下对数概率差的期望值。
- KL散度始终大于等于零，当且仅当两分布相同时为零。

### 详细解答

KL散度（Kullback-Leibler Divergence），又称相对熵，是信息论中用于衡量两个概率分布（如真实分布P和近似分布Q）之间差异的核心指标。结论上，它量化了使用分布Q来编码服从分布P的数据时，所产生的额外信息熵（即需要多耗费的平均编码长度）。原理上，其计算公式为 D_KL(P||Q) = Σ P(x) log(P(x)/Q(x))。需要特别注意的是，KL散度不具备对称性，即 D_KL(P||Q) ≠ D_KL(Q||P)，因此它不是严格意义上的距离度量。在工程实践中，大模型对齐（如RLHF中的PPO算法）常利用KL散度作为惩罚项，限制强化学习更新后的策略模型分布不要偏离初始的参考模型分布过远，从而防止模型产生过度优化或灾难性遗忘。

### 案例模拟

面试官追问：“在RLHF中，为什么计算KL散度惩罚项时通常用近似估计而不是精确计算？” 回答：“在PPO算法的工程实现中，精确计算KL散度需要遍历整个词表空间求和，对于动辄几万词表的大模型来说计算开销极大。因此，我们通常采用采样近似法，即直接计算当前策略与参考模型在生成Token上的对数概率差值作为KL散度的无偏估计。这不仅大幅降低了计算复杂度，还能满足梯度更新的需求。”

### 60. 二、交叉熵损失函数写一下，物理意义是什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 公式为真实分布与预测分布对数乘积的负期望。
- 物理意义是衡量使用预测分布编码真实分布的平均信息量。
- 在机器学习中等价于最大化训练数据的似然函数。

### 详细解答

交叉熵损失函数的离散形式公式为 H(P, Q) = -Σ P(x) log Q(x)，其中P是真实标签分布，Q是模型预测的概率分布。结论上，它是分类任务中最核心的损失函数，用于促使模型的预测分布尽可能逼近真实分布。从物理意义来看，根据信息论，熵代表消除不确定性所需的信息量；交叉熵则表示在真实分布为P的前提下，使用非最优的预测分布Q来进行数据编码时，所需要的平均比特数。当Q完全等于P时，交叉熵达到最小值，即等于P的信息熵。在工程权衡中，交叉熵配合Softmax激活函数能够提供非常平滑且梯度明确的优化曲面，避免了均方误差在概率预测中容易出现的梯度消失问题，是大模型预训练（如Next Token Prediction）的基石。

### 案例模拟

面试官追问：“在长文本生成任务中，标准交叉熵损失有什么局限性？” 回答：“在长文本生成场景下，标准交叉熵存在‘曝光偏差’（Exposure Bias）问题。因为训练时采用Teacher Forcing，每一步都基于真实前缀计算交叉熵；而推理时只能基于自身生成的Token，一旦前期生成错误，误差会迅速累积。业务中，我们常通过引入强化学习（如PPO优化序列级奖励）或采用计划采样（Scheduled Sampling）来缓解这一局限性。”

### 61. 三、KL 散度与交叉熵的区别？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 交叉熵等于真实分布的信息熵加上KL散度。
- 优化交叉熵在数学上完全等价于优化KL散度。
- 交叉熵常用于分类损失，KL散度常用于分布对齐。

### 详细解答

KL散度与交叉熵在数学上紧密相关但侧重点不同。结论是：交叉熵 H(P,Q) 等于真实分布的信息熵 H(P) 加上两者之间的KL散度 D_KL(P||Q)。原理上，在大多数监督学习任务中，真实标签分布 P 是固定的（例如One-hot编码），因此其信息熵 H(P) 是一个常数。这意味着最小化交叉熵在数学优化上完全等价于最小化KL散度。区别在于应用场景：交叉熵直接衡量预测分布与真实标签的整体差异，计算更直接，是分类任务和语言模型预训练的标准Loss；而KL散度剥离了目标分布自身的内在混乱度，纯粹衡量两个分布的“距离”，因此在需要对齐两个模型分布的场景（如知识蒸馏中对齐Teacher和Student的软标签，或RLHF中约束策略模型）更为常用。

### 案例模拟

面试官追问：“在知识蒸馏中，为什么使用KL散度而不是直接用交叉熵？” 回答：“在知识蒸馏中，Teacher模型输出的软标签（Soft Targets）包含了类别间的暗知识（如‘猫’和‘狗’的概率都较高）。此时真实分布P（Teacher输出）不再是One-hot，其信息熵H(P)不是常数。虽然优化交叉熵和KL散度的梯度相同，但使用KL散度能让Loss值在数值上更直观地反映Student与Teacher分布的纯粹差异，当Loss降为0时代表分布完全一致，便于工程上的监控与调试。”

### 62. 四、多任务学习各loss差异过大怎样处理？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 采用动态权重调整策略平衡各任务梯度量级。
- 使用不确定性加权，让高噪声任务权重降低。
- 梯度投影法（如PCGrad）解决任务间梯度冲突。

### 详细解答

在多任务学习中，各任务Loss差异过大会导致模型被大Loss任务主导，出现“跷跷板现象”。结论是：必须通过动态权重分配或梯度干预机制来平衡各任务的优化过程。原理上，Loss差异大通常源于任务难度、数据量级或量纲不同。工程实践中常用的处理方法有三种：一是基于同方差不确定性（Uncertainty Weighting），将Loss权重设为可学习参数，自动降低高方差任务的权重；二是动态任务优先级（Dynamic Weight Averaging），根据各任务Loss的下降速率动态调整权重，下降慢的赋予更高权重；三是梯度级别的干预，如PCGrad算法，当检测到两个任务的梯度方向夹角大于90度（存在冲突）时，将一个梯度投影到另一个梯度的法平面上，从而在物理层面消除梯度相互抵消的问题。

### 案例模拟

面试官追问：“如果业务场景中有一个主任务和一个辅助任务，辅助任务Loss很大影响了主任务怎么办？” 回答：“在实际推荐系统或大模型微调中，如果明确区分主次任务，我会采用非对称的梯度截断或停止梯度（Stop Gradient）策略。具体来说，可以限制辅助任务的梯度回传范围，仅更新其特定的网络分支，而不影响共享底座。或者采用梯度裁剪，强制将辅助任务的梯度范数缩放至主任务梯度范数的十分之一，确保主任务在优化方向上的绝对主导地位。”

### 63. 五、分类问题为什么用交叉熵损失函数不用均方误差（MSE）？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 交叉熵配合Softmax能提供更陡峭的梯度。
- MSE在预测概率接近0或1时容易出现梯度消失。
- 交叉熵从最大似然估计推导，更符合分类概率假设。

### 详细解答

分类问题中弃用MSE而选择交叉熵，核心原因在于优化曲面的性质与梯度表现。结论是：交叉熵配合Softmax激活函数能够保证高效的梯度回传，而MSE会导致严重的梯度消失问题。原理上，分类模型通常使用Sigmoid或Softmax输出概率。若使用MSE，其损失函数对网络输出的导数包含激活函数的导数项。当预测值严重错误但概率接近0或1时，激活函数导数趋于0，导致参数几乎无法更新。相反，交叉熵的对数运算恰好能与Softmax的指数运算相抵消，其梯度直接正比于预测概率与真实标签的差值，误差越大梯度越大，优化极其高效。此外，从统计学角度，交叉熵等价于多项分布下的最大似然估计，其概率假设比MSE隐含的高斯分布假设更契合离散的分类任务。

### 案例模拟

面试官追问：“在什么特殊情况下，分类问题也可以考虑使用MSE或者其变体？” 回答：“虽然标准分类不用MSE，但在某些特定工程场景下MSE有奇效。例如在标签存在严重噪声（Label Noise）的场景中，交叉熵由于对错误样本施加过大惩罚，容易导致模型过拟合噪声数据；而MSE的梯度有界，对异常值相对鲁棒（Mean Absolute Error更鲁棒）。此外，在知识蒸馏中，有时会直接用MSE去拟合Teacher模型输出的Logits（即Softmax之前的值），这能保留更丰富的负类分布信息。”

### 64. 六、什么是信息增益？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 信息熵衡量系统的不确定性，值越大越混乱。
- 信息增益是划分前后信息熵的差值，代表纯度提升。
- 决策树ID3算法核心，偏向选择取值较多的特征。

### 详细解答

信息增益（Information Gain）是衡量一个特征对数据集分类能力提升程度的指标。结论上，它等于数据集划分前的信息熵减去按某特征划分后的条件熵。原理方面，信息熵反映了数据的混乱程度，当引入某个特征进行分类后，如果数据变得更加纯粹（即不确定性降低），那么条件熵就会变小，两者的差值即为信息增益。增益越大，说明该特征的分类区分能力越强。在工程权衡中，信息增益存在一个天然缺陷：它倾向于选择取值较多的特征（如身份证号），因为分支越多往往子集越纯。为了解决这个问题，C4.5算法引入了信息增益比，通过特征的固有熵进行惩罚，从而在特征选择时更加稳健。

### 案例模拟

面试官追问：“信息增益和基尼指数有什么区别？” 回答：“信息增益基于信息论中的熵，计算涉及对数运算，计算成本相对较高，且偏向取值多的特征；而基尼指数（Gini Index）是CART算法使用的指标，衡量随机抽取两个样本类别不一致的概率。基尼指数仅涉及多项式计算，计算速度更快，在构建大规模决策树或随机森林时，工程效率更高，且对连续值和离散值的处理更加统一。”

### 65. 七、多分类的分类损失函数(Softmax)？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- Softmax将模型输出的Logits映射为概率分布。
- 交叉熵损失衡量预测概率分布与真实标签的差异。
- 结合Softmax与交叉熵可避免梯度消失并简化求导。

### 详细解答

多分类任务中最常用的损失函数是Softmax交叉熵损失（Softmax Cross-Entropy Loss）。结论上，它是将Softmax激活函数与交叉熵损失结合在一起的复合函数。原理上，模型首先输出未归一化的Logits，Softmax函数通过指数运算并归一化，将其转化为总和为1的概率分布。随后，交叉熵计算该预测分布与真实One-Hot标签分布之间的距离。在工程实现中，直接计算Softmax再求对数容易出现数值不稳定（如溢出或下溢），因此深度学习框架（如PyTorch的CrossEntropyLoss）通常在底层将Log-Softmax和NLLLoss合并计算。这种结合不仅提升了数值稳定性，还使得反向传播时的梯度计算极其简洁，即预测概率减去真实标签，有效避免了梯度消失。

### 案例模拟

面试官追问：“为什么多分类不用MSE而用交叉熵？” 回答：“在分类任务中，MSE搭配Softmax会导致非凸优化问题，且在预测值严重错误时，Softmax的导数趋于0，引发梯度消失，导致模型极难收敛。而交叉熵损失的对数操作恰好抵消了Softmax的指数操作，使得梯度直接正比于预测误差（p-y），误差越大梯度越大，从而加速模型收敛，更适合分类任务的概率分布拟合。”

### 66. 八、softmax和交叉熵损失怎么计算，二值交叉熵呢？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- Softmax计算为指数除以所有类别指数之和。
- 交叉熵计算为真实标签与预测概率对数乘积的负和。
- 二值交叉熵(BCE)针对二分类，分别计算正负类损失。

### 详细解答

Softmax和交叉熵的计算是分类任务的核心。结论上，Softmax通过指数归一化将Logits转化为概率；交叉熵通过真实标签与预测概率对数乘积的负和计算损失。对于多分类，真实标签通常是One-Hot编码，因此交叉熵退化为负的正确类别预测概率的对数。二值交叉熵（BCE）则是交叉熵在二分类下的特例，计算公式为正类与负类对数损失的加权和，预测概率通常由Sigmoid函数输出。工程对比上，Softmax交叉熵各类别概率互斥，概率和为1，适用于单标签多分类；而BCE独立计算每个类别的概率，适用于多标签分类任务。在实现时，框架均采用Log-Sum-Exp技巧来保证数值计算的稳定性。

### 案例模拟

面试官追问：“如果一个样本可能同时属于多个类别，应该用哪种损失？” 回答：“应该使用二值交叉熵（BCE）配合Sigmoid激活函数。因为Softmax会强制所有类别的概率和为1，导致类别之间产生互斥竞争，不适合多标签场景。而Sigmoid将每个类别的预测独立映射到0到1之间，BCE分别对每个类别计算二分类损失，能够完美适配一个样本同时具备多个标签的业务需求，比如图像的多属性识别。”

### 67. 九、如果softmax的e次方超过float的值了怎么办？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 指数运算极易导致数值溢出，产生NaN或Inf错误。
- 减去最大值技巧可有效防止Softmax上溢出问题。
- Log-Sum-Exp技巧在计算对数概率时保证数值稳定。

### 详细解答

当Softmax的指数运算超过浮点数表示范围时，会发生数值溢出（Overflow），导致结果变为Inf或NaN。结论上，工程中最标准的解决方案是引入平移不变性，即在计算指数前，将所有Logits减去其中的最大值。原理上，Softmax具有平移不变性：分子分母同乘一个常数的指数，等价于每个输入减去该常数。这样处理后，最大值变为0，其指数为1，其余值均为负数，指数在0到1之间，彻底杜绝了上溢出。虽然这可能导致极小值发生下溢出（变为0），但分母中至少有一个1，不会出现除以0的错误。在计算交叉熵时，进一步结合Log-Sum-Exp技巧，可以同时避免上溢和下溢，深度学习框架底层的算子均内置了此类数值稳定优化。

### 案例模拟

业务案例模拟：“在训练大语言模型时，由于FP16精度范围较小（最大约65504），计算Attention的Softmax时极易溢出。我们在自定义CUDA算子时，必须在分块计算（Tiling）过程中动态维护当前块的最大值，并利用在线Softmax（Online Softmax）算法，在遍历序列的同时不断更新最大值和分母，从而在不增加显存访问的前提下，完美解决FP16下的数值溢出问题。”

## 相似度函数篇

### 68. 一、除了cosin还有哪些算相似度的方法

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 相似度函数篇 / 未知](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

### 基础知识补充

- 欧氏距离衡量空间绝对距离，对向量模长非常敏感。
- 曼哈顿距离计算绝对轴距，常用于网格状空间寻路。
- 杰卡德相似度用于集合交并比，适合词袋或布尔特征。

### 详细解答

除了余弦相似度（Cosine Similarity），常用的相似度或距离度量方法还有多种。结论上，主要包括欧氏距离、曼哈顿距离、杰卡德相似度和皮尔逊相关系数等。原理与对比方面：欧氏距离衡量两点间的直线绝对距离，对向量的绝对大小（模长）非常敏感，适合需要考虑量级的场景；而余弦相似度只关注方向，常用于文本向量。皮尔逊相关系数本质上是中心化后的余弦相似度，能消除用户评分基准不同的偏差，常用于推荐系统。杰卡德相似度计算集合的交集与并集之比，非常适合处理布尔型特征或词袋模型。工程权衡上，余弦和内积在向量检索（如Faiss）中计算极快，而欧氏距离在未归一化时可能导致维度灾难，需谨慎使用。

### 案例模拟

面试官追问：“在做向量检索（如RAG中的召回）时，内积（Dot Product）和余弦相似度有什么区别？” 回答：“余弦相似度是归一化后的内积。如果所有向量在存入向量数据库前都已经进行了L2归一化，那么计算内积就完全等价于计算余弦相似度。在工程实现中，直接计算内积省去了实时计算模长和除法的开销，速度更快。如果向量未归一化，内积不仅考虑方向，还会受到向量模长的影响，模长越大的向量越容易被检索出。”

### 69. 二、了解对比学习嘛？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 相似度函数篇 / 未知](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

### 基础知识补充

- 对比学习通过拉近正样本、推远负样本学习表征。
- InfoNCE损失是对比学习中最核心的损失函数。
- SimCLR和MoCo是视觉领域经典的对比学习框架。

### 详细解答

对比学习（Contrastive Learning）是一种强大的自监督学习范式。结论上，它的核心思想是通过在特征空间中拉近相似样本（正样本），同时推远不相似样本（负样本），从而学习到具有良好区分度的特征表示。原理上，模型通常对同一数据进行不同的数据增强（如裁剪、加噪）构造正样本对，而批次内的其他样本作为负样本。通过优化InfoNCE损失函数，最大化正样本间的互信息。工程对比上，对比学习摆脱了对大规模人工标注的依赖，在CV和NLP领域均取得了突破。例如SimCLR依赖超大Batch Size来提供足够的负样本，而MoCo则通过动量编码器和队列机制，在常规Batch Size下实现了海量负样本的存储与更新，极大地降低了硬件门槛。

### 案例模拟

面试官追问：“在NLP领域，对比学习通常是怎么构造正负样本的？” 回答：“在NLP中，最经典的代表是SimCSE。无监督SimCSE极其巧妙地利用了Dropout作为数据增强，将同一个句子输入模型两次，由于Dropout的随机性，两次输出的向量作为正样本对，批次内其他句子作为负样本。有监督SimCSE则利用NLI数据集，将蕴含关系的句子作为正样本，矛盾关系的句子作为困难负样本（Hard Negatives），显著提升了句子向量的表征质量。”

### 70. 三、对比学习负样本是否重要？负样本构造成本过高应该怎么解决？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 相似度函数篇 / 未知](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

### 基础知识补充

- 负样本防止模型坍塌，避免所有输入映射到同一向量。
- 困难负样本能提供更大梯度，显著提升模型判别能力。
- 动量队列和In-Batch负采样是降低成本的有效手段。

### 详细解答

对比学习中负样本极其重要。结论上，负样本是防止模型发生“表征坍塌”（即所有输入都输出相同向量）的关键，且高质量的负样本能决定表征的上限。原理上，如果只有正样本，模型走捷径输出恒定值即可让损失降为0；负样本提供了排斥力，使得特征空间均匀分布。当负样本构造成本过高时，工程上有几种主流解决方案：1. In-Batch Negative：直接利用同一个Batch内的其他样本作为负样本，计算效率极高，但依赖大Batch Size；2. 动量队列（如MoCo）：维护一个异步更新的特征队列，解耦了负样本数量与Batch Size的绑定；3. 难负样本挖掘：利用BM25等廉价算法初步筛选出相似但不相关的样本作为Hard Negative，提高单个负样本的训练收益，从而减少对负样本总量的需求。

### 案例模拟

业务案例模拟：“在训练RAG系统的检索模型（如BGE）时，我们发现随机负样本效果很差。为了解决负样本构造成本问题，我们采用了两阶段策略：首先使用In-Batch负采样保证基础的特征空间均匀性；然后引入跨批次（Cross-Batch）负样本共享技术，并在离线阶段使用上一版模型挖掘出得分较高但非正确答案的文本作为困难负样本（Hard Negatives）。这样在不增加显存的情况下，大幅提升了模型的召回准确率。”
