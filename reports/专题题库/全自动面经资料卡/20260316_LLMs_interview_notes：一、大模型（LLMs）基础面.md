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

- LLaMA体系：Meta开源，奠定当前开源大模型生态基础。
- Qwen体系：阿里开源，多语言与多模态能力表现优异。
- Mistral/Mixtral体系：主打高效与MoE架构，小参数高性能。

### 详细解答

结论：当前主流的开源大模型体系主要由Meta的LLaMA系列、阿里的Qwen系列以及欧洲的Mistral系列主导，此外还有GLM、Baichuan等优秀体系。 原理：1. LLaMA系列（如LLaMA-3）：采用标准的Causal Decoder架构，凭借庞大的高质量训练数据和极佳的社区生态，成为大多数微调和二次开发的基座；2. Qwen系列（如Qwen2.5）：在中文及多语言支持上极具优势，且原生支持长上下文，其代码和数学能力在同量级中处于领先地位；3. Mistral系列：以Mistral-7B和Mixtral 8x7B MoE为代表，通过滑动窗口注意力和稀疏专家混合架构，在推理效率和性能之间取得了极佳的平衡。 工程权衡：选择开源基座时需综合考虑业务需求。如果面向海外市场或需要丰富的社区插件，LLaMA是首选；如果深耕中文业务或需要强大的多模态扩展，Qwen体系更合适；若对推理成本和显存极度敏感，Mistral的MoE架构则更具性价比。

### 案例模拟

面试官追问：如果要在单张24G显存的GPU上部署一个用于代码辅助的模型，你会推荐哪个体系？ 回答：我会推荐Qwen2.5-Coder-7B或DeepSeek-Coder-7B。首先，7B参数量的模型在FP16精度下占用约14GB显存，单张24G显卡完全可以流畅运行并留有足够的KV Cache空间。其次，Qwen和DeepSeek在代码预训练数据上做了深度优化，其代码补全能力非常拔尖。如果需要进一步降低显存占用，还可以采用AWQ或GPTQ进行4-bit量化。

### 2. 2 prefix Decoder 和 causal Decoder 和 Encoder-Decoder 区别是什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Causal Decoder：严格单向注意力，仅能看到历史Token。
- Prefix Decoder：前缀部分双向注意力，生成部分单向注意力。
- Encoder-Decoder：编码器双向，解码器单向，通过交叉注意力连接。

### 详细解答

结论：这三种架构的核心区别在于注意力掩码（Attention Mask）的设计不同，从而决定了模型处理输入和生成输出的方式。 原理：1. Causal Decoder（如GPT系列、LLaMA）：采用下三角掩码，每个Token只能关注自己及之前的Token。这种设计天然契合自回归生成任务，训练效率高，是目前大模型的主流；2. Prefix Decoder（如GLM）：输入的前缀部分（Prompt）采用全可见的双向注意力，能更充分地理解上下文，而生成部分采用单向注意力。它在Few-shot和理解任务上表现更好；3. Encoder-Decoder（如T5、BART）：拥有独立的编码器（全向注意力）和解码器（单向注意力），两者通过Cross-Attention交互。适合输入输出长度差异大的任务（如翻译、摘要）。 工程权衡：Encoder-Decoder在复杂理解任务上性能好，但参数效率低，推理时需计算两套网络；Causal Decoder虽然在双向理解上稍弱，但结构简单，易于扩展参数规模，且在KV Cache优化上极其成熟，因此统治了当前的LLM时代。

### 案例模拟

面试官追问：为什么现在的大模型几乎都放弃了Encoder-Decoder架构，转向纯Causal Decoder？ 回答：主要有三个原因：一是计算效率，Causal Decoder在训练时可以高度并行化，且不需要处理复杂的Cross-Attention；二是Scaling Law的验证，研究表明当参数量和数据量足够大时，Causal Decoder的上下文理解能力能弥补其缺乏双向注意力的劣势；三是工程生态，纯Decoder架构在推理加速（如FlashAttention）和显存管理上更容易实现极致优化。

### 3. 3 大模型LLM的 训练目标 是什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 核心目标是下一个词预测（Next Token Prediction）。
- 数学本质是最大化训练语料的联合概率分布。
- 采用交叉熵损失函数（Cross-Entropy Loss）进行优化。

### 详细解答

结论：大语言模型（LLM）在预训练阶段的核心训练目标是“下一个词预测（Next Token Prediction）”，即基于给定的历史上下文，预测词表中每个Token作为下一个词的概率。 原理：从统计语言模型的角度来看，一段文本的出现概率可以分解为各个词在给定前缀条件下的条件概率的连乘积。LLM通过自回归的方式，试图最大化庞大训练语料库的似然估计。在具体实现中，模型输出一个维度为词表大小的Logits向量，经过Softmax转化为概率分布，然后与真实的下一个Token（One-hot标签）计算交叉熵损失（Cross-Entropy Loss）。通过反向传播不断更新网络权重，使得模型预测真实Token的概率趋近于1。 工程权衡：虽然“下一个词预测”极其简单，但它迫使模型在海量数据中压缩知识、学习语法、逻辑甚至世界常识。这种自监督学习方式无需人工标注，能够无限制地利用互联网文本。缺点是模型容易产生幻觉，因为其本质只是在拟合概率分布，而非进行严谨的逻辑推理。

### 案例模拟

面试官追问：除了Next Token Prediction，还有其他的训练目标吗？ 回答：有的。虽然Next Token Prediction是Causal Decoder的主流，但在掩码语言模型（如BERT）中，训练目标是掩码语言建模（MLM），即预测句子中被随机遮挡的词。此外，在指令微调（SFT）阶段，训练目标依然是Next Token Prediction，但通常只对模型生成的回答部分计算Loss，而忽略Prompt部分的Loss。在对齐阶段（如RLHF），训练目标则转变为最大化人类偏好奖励模型的得分。

### 4. 4 涌现能力是啥原因？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 参数规模与训练数据量突破临界阈值（通常百亿参数以上）。
- 复杂任务被隐式分解为多个基础能力的组合。
- 评价指标的非线性（如精确匹配度量）放大了性能跃升感。

### 详细解答

结论：大模型的“涌现能力（Emergent Abilities）”是指当模型规模（参数量、计算量、数据量）达到一定阈值时，模型在某些复杂任务上的表现突然出现从随机猜测到接近人类水平的非线性跃升现象。 原理：涌现能力的成因目前学术界尚无定论，主要有几种假说：1. 组合爆炸假说：大模型在预训练中掌握了大量基础语法和事实。当规模足够大时，模型具备了将这些基础能力进行多步组合推理的能力，从而解决复杂任务（如思维链推理）；2. 记忆与泛化临界点：随着参数增加，模型从单纯的“记忆”训练数据，跨越到了提取深层抽象规律的阶段；3. 度量指标假说：有学者指出，涌现可能是一种“度量幻觉”。如果采用连续的评价指标（如交叉熵）而非离散指标（如准确率），性能的提升其实是平滑的。 工程权衡：为了激发涌现能力，工程上必须投入巨大的算力进行规模扩展（Scaling）。但这也带来了极高的训练成本。目前的研究趋势是通过更高质量的数据和更优的架构，试图在更小的参数规模下提前激发涌现能力。

### 案例模拟

面试官追问：思维链（Chain of Thought, CoT）为什么被认为是一种涌现能力？ 回答：因为在小规模模型（如1B或3B）中，即使在Prompt中给出了思维链的示例，模型也无法有效模仿，甚至强行输出中间步骤会导致最终答案的准确率下降。但当模型规模达到一定量级（如60B以上）时，CoT不仅能被模型自然掌握，还能极大地提升其在数学和逻辑推理任务上的准确率。这种“小模型无效，大模型突然有效”的现象，正是涌现能力的典型特征。

### 5. 为何现在的大模型大部分是Decoder only结构

- 主标签：LLM基础
- 来源条数：2
- 答案生成方式：原仓库答案
- 来源：[DeepLearing-Interview-Awesome-2024 / 原仓库题解 / 未知](https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/LLMs/Reference.md)
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 先定义模型范式
- 说明训练目标
- 补结构与能力边界

### 详细解答

大模型从模型架构上主要分为三种：Only-encoder, Only-Decoder, Encoder-Decoder三种模型架构 - Only-encoder：例如BERT，通过在大规模无标签文本上进行预训练，然后在下游任务上进行微调，具有强大的语言理解能力和表征能力。 - Only-Decoder: 例如GPT，通过在大规模无标签文本上进行预训练，然后在特定任务上进行微调，具有很强的生成能力和语言理解能力。 - Encoder-Decoder：例如T5（Text-to-Text Transfer Transformer）可以用于多种自然语言处理任务，如文本分类、机器翻译、问答等。 而LLM之所以主要都用Decoder-only架构，除了训练效率和工程实现上的优势外，在理论上是因为Encoder的双向注意力会存在低秩问题，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数。所以，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。

### 案例模拟

面试表达可以这样组织：先用一句话回答“为何现在的大模型大部分是Decoder only结构”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 6. 6 简单 介绍一下 大模型【LLMs】？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 基于Transformer架构的超大规模深度学习模型。
- 经历预训练、指令微调和人类偏好对齐三个核心阶段。
- 具备强大的上下文理解、少样本学习和泛化生成能力。

### 详细解答

结论：大语言模型（Large Language Models, LLMs）是具有数十亿至数万亿参数的神经网络，通过海量文本数据训练，能够理解、生成和处理自然语言，是当前通用人工智能（AGI）的核心技术路径。 原理：LLMs通常基于Transformer的Decoder架构。其生命周期包含三个关键阶段：1. 预训练（Pre-training）：在海量无标注互联网文本上进行自监督学习（预测下一个词），吸收世界知识并建立语言基础；2. 指令微调（SFT）：使用高质量的人工标注问答对进行有监督训练，使模型学会遵循人类指令并以对话形式输出；3. 对齐（Alignment）：通过RLHF或DPO等技术，使模型的输出符合人类的价值观，做到诚实、有用且无害。 工程权衡：大模型的优势在于“通用性”，一个模型可以解决翻译、摘要、代码、问答等多种任务，打破了过去NLP领域“一任务一模型”的碎片化局面。但其代价是极高的算力成本、显存占用以及难以彻底消除的幻觉问题，这要求在落地时结合RAG、Agent等工程手段进行弥补。

### 案例模拟

面试官追问：如果公司要落地一个垂直领域的大模型应用，你会选择从头预训练还是微调？ 回答：在绝大多数商业场景下，我会选择基于开源基座模型（如Qwen或LLaMA）进行微调，而不是从头预训练。从头预训练需要准备数万亿Token的清洗数据，并耗费数百万美元的算力，风险极高且周期长。垂直领域的落地，更具性价比的方案是：利用领域专有数据对基座模型进行全量微调或参数高效微调（如LoRA），注入行业知识；同时结合RAG技术外挂企业知识库，保证回答的专业性和时效性。

### 7. 7 大模型【LLMs】后面跟的 175B、60B、540B等 指什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- B代表Billion即十亿，表示模型参数量级。
- 参数量决定了模型的容量和拟合复杂数据的能力。
- 175B通常指GPT-3的参数规模，是经典大模型标杆。

### 详细解答

结论：这些数字代表大语言模型（LLMs）的参数量，B是Billion（十亿）的缩写，175B即1750亿参数。 原理解释：模型参数主要包括神经网络中各层的权重（Weights）和偏置（Biases）。参数量越大，模型能够记忆和学习到的知识就越多，涌现能力往往也越强。例如GPT-3是175B，PaLM是540B。 工程权衡：虽然参数量增加能提升模型性能，但也会带来巨大的算力开销和显存占用。在实际工程中，推理175B模型通常需要多张80G显存的A100进行张量并行部署。因此，当前开源社区更倾向于训练7B到70B规模的模型，以平衡性能与部署成本。

### 案例模拟

面试官追问：如果要在单张24G显存的消费级显卡上运行大模型，你会怎么选择？ 回答：我会选择7B或13B参数规模的模型。对于7B模型，FP16精度下大约需要14GB显存，单卡完全可以装下。如果选择13B模型，FP16需要26GB会超出显存，这时我会采用INT8或INT4量化技术，将显存占用降低到10GB左右，实现单卡流畅推理。

### 8. 8 大模型【LLMs】具有什么优点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 具备强大的泛化能力，能处理多种未见过的任务。
- 拥有涌现能力，在参数规模突破阈值后展现复杂推理。
- 支持少样本或零样本学习，减少对标注数据的依赖。

### 详细解答

结论：大模型的核心优点在于强大的泛化能力、涌现能力以及对多任务的统一建模能力。 原理解释：首先，通过海量数据预训练，大模型吸收了丰富的世界知识，能够通过零样本或少样本提示完成翻译、摘要、问答等多种任务，打破了传统NLP“一任务一模型”的范式。其次，当参数规模达到一定量级（如百亿以上），模型会展现出“涌现能力”，例如思维链（CoT）推理、复杂逻辑推演等。 工程权衡：大模型的通用性极大地降低了下游任务的适配成本，开发者只需通过Prompt工程或轻量级微调（如LoRA）即可快速上线新业务，显著缩短了AI应用的研发周期。

### 案例模拟

面试官追问：在实际业务中，大模型的通用性如何体现？ 回答：在我们的智能客服项目中，以前需要分别训练意图识别、情感分析和实体抽取三个小模型，维护成本极高。引入大模型后，只需设计不同的Prompt，同一个模型就能同时完成这三项任务。不仅减少了数据标注成本，还统一了技术栈，使得系统迭代更加敏捷高效。

### 9. 9 大模型【LLMs】具有什么缺点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- 存在幻觉问题，容易生成看似合理但错误的内容。
- 训练和推理的算力成本极高，对硬件资源要求苛刻。
- 知识更新困难，无法实时获取最新信息或动态数据。

### 详细解答

结论：大模型的主要缺点包括幻觉问题、高昂的算力成本、知识更新滞后以及缺乏可解释性。 原理解释：幻觉（Hallucination）是大模型最致命的弱点，由于其本质是基于概率的下一个词预测，容易生成一本正经的胡说八道。其次，千亿参数模型的预训练需要数千张GPU耗时数月，推理也需要昂贵的集群支持。此外，模型的知识被冻结在训练完成的那一刻，无法直接回答训练数据之后发生的新事件。 工程权衡：为了克服这些缺点，工程上常采用检索增强生成（RAG）来缓解幻觉并补充最新知识；使用量化、剪枝和知识蒸馏等技术来降低推理成本；通过对齐微调（RLHF）来提升输出的可靠性。

### 案例模拟

面试官追问：如何解决大模型在垂直领域的知识滞后和幻觉问题？ 回答：在医疗问答业务中，我们采用了RAG（检索增强生成）架构。当用户提问时，系统先从最新的医学知识库中检索相关文献片段，然后将这些片段作为上下文拼接到Prompt中，让大模型基于检索到的事实进行总结。这不仅解决了知识更新问题，还大幅降低了模型捏造医疗建议的风险。

### 10. 10 encoder-only, decoder-only, encoder-decoder的区别?

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Encoder-only采用双向注意力，适合文本理解任务。
- Decoder-only采用单向注意力，适合文本生成任务。
- Encoder-Decoder结合两者，适合序列到序列任务。

### 详细解答

结论：这三种架构的区别在于注意力机制的掩码方式及适用任务。Encoder-only双向可见，Decoder-only单向可见，Encoder-Decoder包含双向编码和单向解码。 原理解释：Encoder-only（如BERT）使用无掩码的自注意力，每个词能看到上下文，擅长分类、实体抽取等理解任务。Decoder-only（如GPT系列）使用因果掩码，每个词只能看到前面的词，严格遵循自回归生成，擅长对话和续写。Encoder-Decoder（如T5、BART）先用Encoder双向理解输入，再用Decoder自回归生成输出，适合机器翻译和文本摘要。 工程权衡：目前大语言模型几乎全部走向Decoder-only架构。因为在海量数据下，自回归目标具有更好的扩展性，且能通过Prompt统一所有NLP任务，工程实现也更简洁。

### 案例模拟

面试官追问：为什么现在的大模型（如GPT-4、LLaMA）都采用Decoder-only架构？ 回答：主要有三个原因：一是Decoder-only的自回归训练目标更难，迫使模型学习更深层的逻辑，从而更容易产生涌现能力；二是它在Zero-shot泛化上表现更好，能通过Prompt无缝切换任务；三是工程上KV Cache的优化在Decoder-only上更成熟，推理效率更高。

### 11. 11 BART、llama、gpt、t5、palm等主流模型异同点?

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- GPT、LLaMA、PaLM均为Decoder-only架构的生成模型。
- BART和T5属于Encoder-Decoder架构，适合Seq2Seq。
- LLaMA开源生态繁荣，GPT和PaLM多为闭源商业模型。

### 详细解答

结论：这些模型的异同主要体现在架构选择、训练目标和开源策略上。GPT、LLaMA、PaLM是Decoder-only，而BART、T5是Encoder-Decoder。 原理解释：BART通过破坏文本再重建来进行预训练，T5将所有NLP任务统一为文本到文本的形式，两者在翻译和摘要上表现优异。GPT系列开创了Decoder-only自回归预训练的范式；PaLM是Google推出的超大参数模型，引入了多路径推理；LLaMA则是Meta开源的模型，使用了RoPE旋转位置编码、RMSNorm和SwiGLU等现代改进。 工程权衡：在实际选型中，如果做特定领域的轻量级翻译或摘要，T5/BART微调成本低；如果是构建通用对话系统或复杂推理应用，基于LLaMA系列的开源模型进行二次开发是目前性价比最高的方案。

### 案例模拟

面试官追问：如果让你从头训练一个垂直领域的代码大模型，你会参考哪个模型的架构？ 回答：我会参考LLaMA的架构。它采用了Decoder-only结构，非常适合代码生成这种自回归任务。同时，LLaMA引入的RoPE位置编码对外推性支持较好，有利于处理长代码上下文；RMSNorm和SwiGLU也已被证明能显著提升训练稳定性和收敛速度，是目前业界公认的标杆架构。

### 12. 12 prefix LM 和 causal LM 区别是什么?

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）基础面 / 未知](https://articles.zsxq.com/id_mw52p1pfbzql.html)

### 基础知识补充

- Causal LM采用严格的下三角掩码，只能看到历史信息。
- Prefix LM对前缀部分双向可见，后续部分单向可见。
- Causal LM适合纯生成，Prefix LM兼顾理解与生成。

### 详细解答

结论：两者的核心区别在于注意力掩码（Attention Mask）的设计不同。Causal LM是纯单向的，而Prefix LM在给定的前缀（Prefix）部分是双向的。 原理解释：Causal LM（因果语言模型，如GPT）在计算自注意力时，每个Token只能和它之前的Token计算注意力，掩码矩阵是一个严格的下三角矩阵。Prefix LM（前缀语言模型，如GLM）允许输入的前缀序列内部进行双向注意力计算，即前缀中的Token可以互相看到，而生成部分的Token依然只能看到前面的内容。 工程权衡：Causal LM的优势在于训练效率高，KV Cache管理简单，是目前大模型的主流。Prefix LM在处理长篇阅读理解或复杂Prompt时，由于前缀部分信息交互更充分，理解能力往往更强，但在推理时KV Cache的实现相对复杂。

### 案例模拟

面试官追问：在实际微调中，如何将Causal LM改造成类似Prefix LM的效果？ 回答：在指令微调（SFT）阶段，我们可以通过修改Attention Mask来实现。对于输入的Prompt部分，我们不计算Loss，并且在某些实现中可以放开Prompt内部的Mask限制，使其双向可见。不过为了保持与预训练阶段的一致性，通常我们只屏蔽Prompt的Loss，依然保留Causal Mask，这在工程上最简单且效果稳定。

## Layer normalization 篇

### 13. Layer Norm 的计算公式写一下？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 对单个样本的所有特征维度求均值和方差。
- 公式包含减去均值并除以标准差的标准化过程。
- 引入可学习的缩放参数和平移参数恢复表达能力。

### 详细解答

结论：Layer Norm（层归一化）的计算公式为：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$。 原理解释：其中 $x$ 是输入向量，$\mu$ 是该向量所有特征维度的均值，$\sigma^2$ 是方差。$\epsilon$ 是一个极小的常数，用于防止除零错误。标准化后的结果会乘以一个可学习的缩放参数 $\gamma$，并加上一个可学习的平移参数 $\beta$。与Batch Norm不同，Layer Norm是在特征维度上进行归一化，不依赖于Batch Size，因此非常适合处理变长序列的Transformer模型。 工程权衡：Layer Norm能有效缓解深层网络中的梯度消失和爆炸问题，加速模型收敛。但计算均值和方差需要遍历所有特征，存在一定的计算开销，这也是后续RMS Norm对其进行简化的动机。

### 案例模拟

面试官追问：为什么Transformer中选择Layer Norm而不是Batch Norm？ 回答：NLP任务中句子长度通常不一致，如果使用Batch Norm，在Padding部分计算均值和方差会引入噪声；而且Batch Norm对Batch Size敏感，大模型训练时受显存限制Batch Size往往较小，导致统计量不准确。Layer Norm在单个样本的特征维度上计算，完全不受序列长度和Batch Size的影响，更适合文本数据。

### 14. RMS Norm 的计算公式写一下？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- RMS Norm全称为均方根归一化，是LN的简化版。
- 放弃了均值计算，直接使用均方根进行缩放。
- 公式为输入除以均方根后乘以可学习参数。

### 详细解答

结论：RMS Norm（Root Mean Square Normalization）的计算公式为：$y = \frac{x}{\text{RMS}(x)} \cdot \gamma$，其中 $\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$。 原理解释：RMS Norm去除了Layer Norm中的均值计算（即假设均值为0）和平移参数 $\beta$。它直接计算输入向量 $x$ 所有元素的平方和的平均值，再开根号得到均方根（RMS）。然后将输入向量除以这个均方根进行归一化，最后乘以可学习的缩放参数 $\gamma$。 工程权衡：由于省去了计算均值和减去均值的步骤，RMS Norm的计算量比Layer Norm减少了约10%到30%。在超大规模语言模型（如LLaMA）中，这种微小的计算优化在数百个Transformer层中累加起来，能显著提升训练和推理的吞吐量，同时保持与Layer Norm相当的模型性能。

### 案例模拟

面试官追问：在手写RMS Norm的CUDA算子时，有什么优化点？ 回答：在CUDA实现中，计算平方和是一个典型的Reduce操作。为了优化显存带宽，我们通常会使用Warp-level的归约原语（如__shfl_down_sync）来加速求和过程。此外，可以将除以RMS和乘以$\gamma$的操作融合（Kernel Fusion）在一个算子中完成，避免中间结果写回Global Memory，从而大幅降低访存延迟。

### 15. RMS Norm 相比于 Layer Norm 有什么特点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 计算效率更高，省去了均值计算和平移参数。
- 假设输入特征的均值趋于零，仅做方差缩放。
- 在大模型中表现出与Layer Norm相当的收敛效果。

### 详细解答

结论：RMS Norm相比Layer Norm最大的特点是计算更高效、实现更简单，同时在性能上几乎没有损失。 原理解释：Layer Norm的成功主要归功于其缩放不变性，而不是平移不变性。RMS Norm正是基于这一发现，大胆去除了计算均值（平移操作）和偏置参数 $\beta$。它仅通过均方根对输入进行缩放，保留了对权重和输入的缩放不变性。 工程权衡：在工程实践中，大模型的训练瓶颈往往在于显存带宽和计算速度。RMS Norm减少了同步点和计算量，非常适合算子融合（Kernel Fusion）。目前主流的开源大模型（如LLaMA系列、Qwen系列）几乎全部采用RMS Norm替代Layer Norm，这已经成为现代LLM架构的标准配置。

### 案例模拟

面试官追问：如果把一个预训练好的使用Layer Norm的模型直接改成RMS Norm，会发生什么？ 回答：直接替换会导致模型输出分布发生剧烈变化，原本的权重无法适应新的归一化尺度，模型性能会瞬间崩溃，表现为输出乱码或Loss激增。如果必须替换，需要重新进行预训练，或者在替换后冻结大部分参数，仅对RMS Norm的$\gamma$参数和部分层进行一定规模的持续预训练来对齐分布。

### 16. Deep Norm 思路？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 旨在解决极深Transformer模型的训练不稳定问题。
- 结合了Post-LN的性能优势和Pre-LN的稳定优势。
- 通过缩放残差连接和初始化权重来控制梯度范数。

### 详细解答

结论：Deep Norm是由微软提出的一种归一化策略，核心思路是在执行Layer Norm之前扩大残差连接的权重，并对网络权重进行特定的缩放初始化，以实现千层Transformer的稳定训练。 原理解释：在Transformer中，Post-LN性能好但深层容易梯度爆炸；Pre-LN训练稳定但深层容易出现表征崩塌。Deep Norm的公式为 $x = \text{LayerNorm}(\alpha \cdot x + f(x))$，其中 $\alpha$ 是大于1的常数。同时，它对FFN和Attention的权重初始化进行缩放，确保每一层的更新对整体的扰动被严格控制在安全范围内。 工程权衡：Deep Norm使得训练1000层以上的Transformer成为可能。但在目前的大模型工程中，由于增加宽度比增加深度更容易并行化，业界更倾向于使用Pre-LN/RMSNorm配合更宽的网络，Deep Norm的应用相对较少。

### 案例模拟

面试官追问：为什么现在主流的LLaMA等大模型没有使用Deep Norm，而是用了Pre-RMSNorm？ 回答：虽然Deep Norm能解决极深网络的训练问题，但当前大模型（如7B到70B）的层数通常在32到80层之间，并未达到必须使用Deep Norm的深度。在这个深度下，Pre-RMSNorm配合合理的初始化已经足够稳定，且计算效率更高。此外，过深的网络会增加流水线并行的切分难度和通信开销。

### 17. 写一下 Deep Norm 代码实现？

- 主标签：经典算法与编程
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 核心在于残差连接时的缩放因子alpha。
- 需要自定义权重初始化逻辑，缩放特定层的权重。
- 基于PyTorch实现时，只需修改残差相加的逻辑。

### 详细解答

结论：Deep Norm的代码实现主要分为两部分：前向传播中引入缩放因子 $\alpha$，以及在模型初始化时对权重进行缩放。 原理解释：前向传播代码简写为：residual = x * self.alpha; x = self.layer_norm(residual + self.sublayer(x))。在初始化阶段，需要对Attention的投影矩阵和FFN的权重进行缩放，缩放比例通常为 $\beta = (8N)^{-0.25}$，其中N为层数。 工程权衡：实现Deep Norm并不复杂，难点在于超参数 $\alpha$ 和 $\beta$ 的推导需要严格的数学证明。在工程落地时，这种非标准的残差结构可能会破坏某些现成的算子融合（如FlashAttention中的融合逻辑），需要重新编写定制化的CUDA Kernel以保证训练效率。

### 案例模拟

面试官追问：在代码中，self.sublayer(x) 具体指代什么？ 回答：在Transformer架构中，self.sublayer(x) 指代的是多头自注意力机制（Multi-Head Attention）模块或者前馈神经网络（FFN）模块。在Deep Norm中，无论是Attention层还是FFN层，其残差连接的方式都要统一替换为乘以 $\alpha$ 后的相加再进行Layer Norm。

### 18. Deep Norm 有什么优点？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- 极大地提升了深层Transformer的训练稳定性。
- 缓解了Post-LN的梯度爆炸和Pre-LN的表征崩塌。
- 理论上支持训练高达1000层以上的超深网络。

### 详细解答

结论：Deep Norm的核心优点是打破了Transformer的深度限制，使得训练超深层网络（如1000层）变得稳定且高效。 原理解释：传统的Post-LN在网络加深时，输出的方差会随层数累积，导致深层梯度过大，极易引发训练崩溃；而Pre-LN虽然稳定，但深层的残差分支贡献越来越小，导致网络退化。Deep Norm通过巧妙的数学设计（放大残差主干 $\alpha x$，缩小网络权重），将每一层的扰动控制在常数级别。它兼具了Post-LN的高性能和Pre-LN的高稳定性。 工程权衡：对于需要极强逻辑推理能力的任务，增加模型深度通常比增加宽度更有效。Deep Norm为探索“深而窄”的大模型架构提供了理论和工程基础。不过，超深网络在推理时会导致极高的延迟，这是实际部署时必须权衡的代价。

### 案例模拟

面试官追问：如果业务场景对推理延迟要求极高，你会选择用Deep Norm训练一个深层模型吗？ 回答：不会。如果对延迟要求极高，我会倾向于选择“宽而浅”的模型架构，并使用Pre-RMSNorm。因为深层模型在推理时，Token必须依次穿过每一层，无法并行，延迟与层数成正比。而宽模型可以通过张量并行将计算分摊到多张卡上，显著降低单次前向传播的耗时。

### 19. 1 LN 在 LLMs 中的不同位置 有什么区别么？如果有，能介绍一下区别么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- Post-LN放在残差相加之后，性能好但深层难训练。
- Pre-LN放在子层计算之前，训练稳定但深层易退化。
- Sandwich-LN在子层前后都加，试图结合两者优点。

### 详细解答

结论：Layer Norm在LLM中的位置主要有Post-LN和Pre-LN两种，它们的区别直接影响模型的训练稳定性和最终性能。 原理解释：Post-LN的公式是 $x = \text{LN}(x + \text{Sublayer}(x))$。优点是每一层输出都被强制归一化，性能上限高；缺点是靠近输出层的梯度非常大，必须依赖Warm-up，且层数加深时极易崩溃。Pre-LN的公式是 $x = x + \text{Sublayer}(\text{LN}(x))$。优点是残差连接直达深层，梯度流动顺畅，不需要Warm-up也能稳定训练；缺点是深层网络的贡献会被削弱，存在表征退化。 工程权衡：在当前的大模型时代，训练的稳定性压倒一切。因为一次千亿参数模型的崩溃会导致巨大的算力浪费。因此，几乎所有现代LLM都毫不犹豫地选择了Pre-LN（或Pre-RMSNorm）架构。

### 案例模拟

面试官追问：既然Pre-LN会导致深层网络退化，为什么现在的大模型还要用它？ 回答：因为在大规模分布式训练中，稳定性是第一优先级的。Post-LN的梯度爆炸问题在百亿参数规模下几乎无法通过调参解决。虽然Pre-LN有深层退化的风险，但工程上可以通过增加模型的宽度、使用更好的激活函数（如SwiGLU）以及海量的高质量数据来弥补这一缺陷，最终达到极高的综合性能。

### 20. LLMs 各模型分别用了 哪种 Layer normalization？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / Layer normalization 篇 / 未知](https://articles.zsxq.com/id_pzcgd4ovk098.html)

### 基础知识补充

- GPT-2/GPT-3采用传统的Pre-Layer Norm。
- LLaMA系列全面采用Pre-RMS Norm以提升效率。
- GLM系列早期使用Deep Norm，后转向RMS Norm。

### 详细解答

结论：随着大模型技术的发展，归一化策略经历了从Post-LN到Pre-LN，再到Pre-RMSNorm的演进。 原理解释：早期的模型如BERT和初代Transformer使用的是Post-Layer Norm。为了解决训练不稳定的问题，GPT-2、GPT-3以及T5等模型转向了Pre-Layer Norm。近年来，为了进一步压榨计算效率，Meta推出的LLaMA系列率先大规模使用了Pre-RMS Norm。由于LLaMA的巨大成功，后续的开源模型如Qwen、Baichuan、Mistral等几乎全部跟进，将Pre-RMS Norm作为标配。 工程权衡：从Layer Norm转向RMS Norm，本质上是算法向工程妥协并取得双赢的经典案例。RMS Norm去除了均值计算，不仅减少了计算量，更重要的是减少了GPU内部的同步开销，使得算子融合更加高效，白嫖了训练速度。

### 案例模拟

面试官追问：如果让你设计一个全新的百亿参数模型，你会选择哪种Norm？ 回答：我会毫不犹豫地选择Pre-RMS Norm。首先，Pre架构保证了大规模训练的稳定性，避免梯度爆炸；其次，RMS Norm在LLaMA等模型中已被充分验证，生态支持极好，像FlashAttention、vLLM等主流训练和推理框架都对RMS Norm有深度的底层算子优化，能直接降低工程开发成本。

## LLMs 激活函数篇

### 21. 1 介绍一下 FFN 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- FFN是两层全连接网络，中间夹着非线性激活函数。
- 公式通常为：FFN(x) = Activation(xW1 + b1)W2 + b2。
- 在大模型中，常被无偏置的SwiGLU变体所取代。

### 详细解答

结论：传统Transformer中的前馈神经网络（FFN）由两个线性变换和一个非线性激活函数组成，计算公式为 $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$（以ReLU为例）。 原理解释：在Transformer中，Attention层负责捕捉Token之间的全局上下文关系，而FFN层则负责对每个Token的特征进行非线性映射和记忆存储。通常，第一层线性变换 $W_1$ 会将特征维度放大（如放大4倍），激活函数引入非线性，第二层 $W_2$ 再将维度映射回原来的大小。 工程权衡：在现代大模型（如LLaMA）中，传统的FFN已被门控线性单元（GLU）的变体（如SwiGLU）取代。SwiGLU去除了偏置项 $b$，并引入了额外的门控矩阵。虽然参数量增加了，但由于其更强的表达能力，在同等计算量下能取得更低的困惑度。

### 案例模拟

面试官追问：有研究说FFN层是大模型的“知识库”，你怎么理解？ 回答：是的，很多研究表明，FFN的第一层（升维）可以看作是Key，第二层（降维）可以看作是Value。输入特征通过激活函数激活特定的Key，然后提取对应的Value。大模型在预训练阶段学习到的丰富事实性知识，很大一部分就隐式地存储在FFN的巨大权重矩阵中，而Attention更多是负责信息的路由和组合。

### 22. 2 介绍一下 GeLU 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- GeLU全称高斯误差线性单元，结合了Dropout和ReLU。
- 公式为：x * P(X <= x)，其中X服从标准正态分布。
- 常用近似公式：0.5x(1 + tanh(sqrt(2/pi)(x + 0.0447x^3)))。

### 详细解答

结论：GeLU（Gaussian Error Linear Unit）是一种平滑的非线性激活函数，其数学定义为 $x \cdot \Phi(x)$，其中 $\Phi(x)$ 是标准正态分布的累积分布函数。 原理解释：ReLU是直接根据输入是否大于0进行硬截断，而GeLU则引入了随机正则化的思想。它将输入 $x$ 乘以一个概率值，这个概率值是输入在正态分布中的累积概率。当 $x$ 越小，保留的概率越趋近于0；当 $x$ 越大，保留的概率越趋近于1。这使得GeLU在0附近具有平滑的非线性过渡，避免了ReLU的“死神经元”问题。 工程权衡：由于计算正态分布的累积函数非常耗时，工程上通常使用基于 $\tanh$ 的近似公式来加速计算。GeLU在BERT、GPT-2等经典模型中被广泛使用。不过在最新的大模型中，GeLU正逐渐被计算效率更高、表现更好的Swish或SwiGLU所替代。

### 案例模拟

面试官追问：相比于ReLU，GeLU为什么在NLP任务中表现更好？ 回答：NLP任务中的词向量分布通常比较复杂，ReLU在0处的不可导和硬截断会丢失部分微小的负值特征，并可能导致神经元死亡。GeLU在负半轴提供了一个平滑的、非零的梯度，允许模型保留并学习到细微的负面特征。这种平滑特性使得模型在优化过程中梯度下降更加稳定，从而在复杂的语言建模任务中获得更好的泛化能力。

### 23. 3 介绍一下 Swish 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- Swish 公式为 f(x) = x * sigmoid(βx)。
- 具有平滑、非单调特性，缓解梯度消失。
- 常作为 ReLU 的替代方案提升模型性能。

### 详细解答

Swish 是一种由谷歌提出的自门控激活函数，其计算公式为 $f(x) = x \cdot \sigma(\beta x)$，其中 $\sigma$ 是 Sigmoid 函数，$\beta$ 是可学习参数或固定常数（常设为1，此时也称 SiLU）。结论上，Swish 结合了线性函数的无上界特性和 Sigmoid 的平滑非线性。原理在于，当 $x$ 很大时，Swish 趋近于线性函数 $x$，避免了梯度消失；当 $x$ 为负且较小时，趋近于 0，具有稀疏性。与 ReLU 相比，Swish 在负半轴具有非单调性（存在一个极小值），这种平滑且非单调的特性使得梯度流更加稳定，有助于深层网络的优化。在工程实践中，虽然计算量比 ReLU 稍大，但通常能带来模型精度的显著提升，广泛应用于现代大语言模型中。

### 案例模拟

面试官追问：Swish 和 ReLU 相比，计算开销如何权衡？ 回答：Swish 包含 Sigmoid 计算，硬件开销确实比 ReLU 大。但在大模型训练中，激活函数的计算时间占比相对较小，主要瓶颈在矩阵乘法。工程上常使用近似计算或查表法优化 Sigmoid。综合来看，Swish 带来的收敛速度和最终精度的提升，完全能够弥补其微小的计算开销增加。

### 24. 4 介绍一下 使用 GLU 线性门控单元的 FFN 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- GLU 通过门控机制动态控制信息流动。
- 公式为 GLU(x) = (xW+b) ⊗ σ(xV+c)。
- 在 Transformer 的 FFN 中引入可增强表达能力。

### 详细解答

GLU（门控线性单元）是一种通过逐元素乘法实现信息过滤的结构。在 Transformer 的 FFN（前馈神经网络）块中，使用 GLU 的计算公式通常表示为 $\text{FFN}_{GLU}(x) = ( (xW_1) \otimes \sigma(xW_2) ) W_3$，其中 $\otimes$ 表示逐元素相乘，$\sigma$ 是 Sigmoid 激活函数。结论上，GLU 机制通过一个线性变换作为主干，另一个线性变换经过激活函数作为门控信号，动态控制特征的传递。原理上，这种乘法交互引入了更强的非线性表达能力，相比传统 FFN 的单路激活，GLU 能够更精细地筛选重要特征。工程权衡方面，GLU 引入了额外的权重矩阵 $W_2$，增加了参数量和计算量，但通常可以通过减小隐藏层维度来保持整体参数量不变，从而在同等参数规模下获得更好的模型性能。

### 案例模拟

面试官追问：引入 GLU 后参数量增加了，如何保持模型参数规模不变？ 回答：在标准 Transformer 中，FFN 的隐藏层维度通常是输入维度的 4 倍。引入 GLU 后，由于多了一个投影矩阵，为了保持总参数量一致，我们通常会将隐藏层维度缩小，例如从 4 倍降至 8/3 倍。这样既利用了 GLU 强大的门控表达能力，又严格控制了显存占用和计算开销。

### 25. 5 介绍一下 使用 GeLU 的 GLU 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- GeGLU 将 GLU 的 Sigmoid 替换为 GeLU。
- 计算公式为 GeGLU(x) = (xW_1) ⊗ GeLU(xW_2)。
- GeLU 结合了 Dropout 和 ReLU 的特性。

### 详细解答

GeGLU 是 GLU 架构的一种变体，它将门控分支的激活函数从 Sigmoid 替换为 GeLU。其在 FFN 块中的计算公式为 $\text{FFN}_{GeGLU}(x) = ( (xW_1) \otimes \text{GeLU}(xW_2) ) W_3$。结论上，GeGLU 结合了门控机制的特征筛选能力和 GeLU 平滑非线性的优势。原理在于，GeLU（高斯误差线性单元）通过正态分布的累积分布函数对输入进行加权，相比 Sigmoid 具有更宽的非饱和区，能够有效缓解深层网络中的梯度消失问题。对比标准 GLU，GeGLU 在负区间的非单调性提供了更丰富的梯度信息。在工程应用中，GeGLU 被广泛证明在自然语言处理任务中优于传统的 FFN 结构，尽管计算复杂度略有增加，但通过算子融合技术可以有效降低显存读写开销，提升训练吞吐量。

### 案例模拟

面试官追问：GeLU 的计算比较复杂，工程上如何加速？ 回答：GeLU 的精确计算涉及误差函数（erf），计算代价较高。在实际工程中，我们通常采用近似公式，如基于 Tanh 的近似，或者直接使用更简单的 $x \cdot \sigma(1.702x)$。此外，在底层实现时，会将线性投影、GeLU 激活和逐元素乘法融合成一个 CUDA Kernel，大幅减少显存带宽占用。

### 26. 6 介绍一下 使用 Swish 的 GLU 块 计算公式？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- SwiGLU 将 GLU 激活函数替换为 Swish。
- 公式为 SwiGLU(x) = (xW_1) ⊗ Swish(xW_2)。
- LLaMA 等主流开源大模型广泛采用 SwiGLU。

### 详细解答

SwiGLU 是目前大语言模型中最流行的 FFN 变体之一，其计算公式为 $\text{FFN}_{SwiGLU}(x) = ( (xW_1) \otimes \text{Swish}(xW_2) ) W_3$，其中 Swish 通常指 $\beta=1$ 时的 SiLU 函数。结论上，SwiGLU 凭借其卓越的性能，成为了 LLaMA、PaLM 等顶级大模型的标配。原理上，SwiGLU 结合了门控机制的动态特征选择和 Swish 函数平滑、非单调的梯度特性。与 GeGLU 相比，SwiGLU 在经验上往往能取得略微更好的收敛效果和困惑度。工程权衡方面，SwiGLU 同样需要三个权重矩阵，因此通常将隐藏层维度设置为输入维度的 8/3 倍以对齐参数量。由于其广泛应用，主流推理框架都对 SwiGLU 进行了深度的算子级优化，确保了极高的执行效率。

### 案例模拟

面试官追问：为什么 LLaMA 选择 SwiGLU 而不是 GeGLU？ 回答：在 PaLM 和 LLaMA 的消融实验中，SwiGLU 在同等计算量和参数量下，展现出了比 GeGLU 和传统 ReLU FFN 更低的验证集困惑度。虽然 Swish 和 GeLU 形状相似，但 Swish 的计算在某些硬件架构上比 GeLU 更容易优化。综合性能收益和工程实现便利性，SwiGLU 成为了最优解。

### 27. 7 各LLMs 都使用哪种激活函数？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- 早期模型如 GPT-2、BERT 多采用 GeLU。
- LLaMA、Qwen 等现代模型普遍采用 SwiGLU。
- PaLM 模型也采用了 SwiGLU 结构提升表达能力。

### 详细解答

在大语言模型的发展历程中，激活函数的选择经历了明显的演进。结论上，早期模型偏爱 GeLU，而现代主流大模型几乎全面转向了 SwiGLU。具体来说，BERT、GPT-2 等经典模型主要使用 GeLU，因为它在当时被证明比 ReLU 更平滑，能带来更好的泛化性能。随着 GLU 变体的兴起，研究发现门控机制能显著提升 Transformer 的表达能力。因此，谷歌的 PaLM、Meta 的 LLaMA 系列、阿里的 Qwen 系列等现代模型，均采用了基于 Swish（SiLU）的 SwiGLU 结构。工程权衡上，虽然 SwiGLU 增加了参数矩阵的数量，但通过调整隐藏层维度保持了总参数量不变，且在同等规模下取得了更优的评测指标。这种统一的趋势也促使底层硬件对 SwiGLU 进行了极致优化。

### 案例模拟

面试官追问：如果让你从头训练一个百亿参数模型，你会选哪个激活函数？ 回答：我会毫不犹豫地选择 SwiGLU。首先，学术界和工业界已经充分验证了它在收敛速度和最终效果上的优势。其次，现有的生态系统（如 Megatron-LM、vLLM）对 SwiGLU 的支持极其完善，算子融合优化非常成熟，不会成为性能瓶颈。选择 SwiGLU 是兼顾模型上限和工程稳定性的最佳实践。

### 28. 8 Adam优化器和SGD的区别？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 激活函数篇 / 未知](https://articles.zsxq.com/id_6xm3wzzice2s.html)

### 基础知识补充

- SGD 仅依赖当前梯度更新，易陷入局部最优。
- Adam 结合了动量（一阶矩）和自适应学习率。
- Adam 收敛极快，是大模型训练的首选优化器。

### 详细解答

Adam 和 SGD 是深度学习中最常用的两种优化算法，它们在更新机制和适用场景上存在显著差异。结论上，SGD 简单直接但收敛慢，Adam 收敛极快且对超参数不敏感，是大模型训练的首选。原理上，SGD 每次仅根据当前批次的梯度更新参数，容易在峡谷状地形中震荡；而 Adam 不仅计算梯度的指数移动平均（一阶矩）来指引更新方向，还计算梯度平方的指数移动平均（二阶矩）来为每个参数自适应调整学习率。对比来看，Adam 能够自动缓解梯度稀疏和尺度不一的问题，极大地加速了 Transformer 等复杂模型的训练。工程权衡上，Adam 需要额外保存一阶和二阶动量状态，显存占用是 SGD 的三倍，这也是大模型训练中需要引入 ZeRO 优化等技术的核心原因。

### 案例模拟

面试官追问：大模型训练中 Adam 的显存占用太大，有什么解决办法？ 回答：在千亿参数模型训练中，Adam 的优化器状态确实是显存大头。工程上通常有几种解法：一是使用 DeepSpeed ZeRO 将优化器状态切分到不同 GPU 上；二是采用混合精度训练；三是使用显存友好的变体，如 8-bit Adam 或 Adafactor，通过量化或矩阵分解技术大幅压缩动量状态的显存占用。

## Attention 升级面

### 29. 1 传统 Attention 存在哪些问题？

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

### 30. 2 Attention 有哪些 优化方向？

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

### 31. 3 Attention 变体有哪些？

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

### 32. 4.1 Multi-head Attention 存在什么问题？

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

### 33. 4.2 介绍一下 Multi-Query Attention？

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

### 34. 4.3 对比一下 Multi-head Attention 和 Multi-Query Attention？

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

### 35. 4.4 Multi-Query Attention 这样做的好处是什么？

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

### 36. 4.5 有 哪些模型 是 使用 Multi-Query Attention？

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

### 37. 5.1 什么是 Grouped-query Attention？

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

### 38. 5.2 有哪些大模型使用 Grouped-query Attention？

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

### 39. 6.1 为什么需要 FlashAttention？

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

### 40. 6.2 简单介绍一下 FlashAttention？

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

### 41. 6.3 简单介绍一下 FlashAttention 核心？

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

### 42. 6.4 介绍一下 FlashAttention 优点？

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

### 43. 6.5 介绍一下 FlashAttention 代表模型？

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

### 44. 7 并行 transformer block

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

### 45. Attention计算复杂度以及如何改进

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

### 46. 9.1 简单介绍一下 Paged Attention？

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

### 47. 1、MHA，GQA，MQA 三种注意力机制是否了解?区别是什么?

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

### 48. 跨注意力机制（Cross-Attention）篇

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

### 49. 一、为什么需要 跨注意力机制（Cross-Attention）？

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

### 50. 二、介绍一些 跨注意力机制（Cross-Attention）？

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

### 51. 3.1 Cross Attention 和 Self Attention 都是基于注意力机制的，有什么相同点？

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

### 52. 四、Cross Attention 和 多头注意力（Multi-Head Attention）篇

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

### 53. 4.2 Cross Attention 和 多头注意力（Multi-Head Attention） 都是基于注意力机制的，有什么异同点？

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

### 54. 五、Cross Attention 代码实现

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

### 55. 六、Cross Attention 应用场景

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

### 56. 七、Cross Attention 的优势和挑战？

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

## transformers 操作篇

### 57. 1. 如何 利用 transformers 加载 Bert 模型？

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

### 58. 2. 如何 利用 transformers 输出 Bert 指定 hidden\_state？

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

## LLMs 损失函数篇

### 59. 一、介绍一下 KL 散度？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- KL散度又称相对熵，用于衡量两个概率分布之间的差异程度。
- 其具有非负性，当且仅当两个分布完全相同时 KL 散度为零。
- KL散度是不对称的，即 D_KL(P||Q) 通常不等于 D_KL(Q||P)。

### 详细解答

结论：KL 散度（Kullback-Leibler Divergence）是信息论中用于度量一个概率分布 Q 相对于另一个真实分布 P 的信息损失的指标。 原理与对比：公式为 $D_{KL}(P||Q) = \sum P(x) \log(P(x)/Q(x))$。在机器学习中，P 通常代表数据的真实分布，Q 代表模型的预测分布。KL 散度衡量了用 Q 来近似 P 时引入的额外信息量。由于其不对称性，前向 KL 散度（P||Q）倾向于让 Q 覆盖 P 的所有模式（可能导致均值回归），而后向 KL 散度（Q||P）倾向于让 Q 锁定 P 的某一个模式（常用于变分推断）。在工程实践中，由于真实分布 P 的熵是常数，最小化 KL 散度在数学上等价于最小化交叉熵损失。

### 案例模拟

面试官追问：既然 KL 散度不对称，在强化学习（如 PPO 算法）中通常使用哪种方向的 KL 散度？ 回答示例：在 PPO 算法中，我们通常计算旧策略和新策略之间的 KL 散度，即 $D_{KL}(\pi_{old} || \pi_{new})$。这是因为我们希望新策略在旧策略采样的状态分布下，不要发生过大的偏离。通过将其作为惩罚项加入目标函数，可以限制策略更新的步长，保证训练的稳定性。如果反过来计算，可能会导致新策略在未探索区域产生不可控的概率突变。

### 60. 二、交叉熵损失函数写一下，物理意义是什么？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 离散分布的交叉熵公式为 H(P,Q) = -∑ P(x) log Q(x)。
- 物理意义是使用预测分布 Q 编码真实分布 P 所需的平均信息量。
- 在分类任务中，最小化交叉熵等价于最大化正确类别的预测概率。

### 详细解答

结论：交叉熵损失函数定义为 $H(P,Q) = -\sum P(x) \log Q(x)$，其物理意义是衡量用预测分布 Q 来表示真实分布 P 时的平均编码长度。 原理与工程权衡：在信息论中，熵 $H(P)$ 是真实分布的最优编码长度，而交叉熵 $H(P,Q) = H(P) + D_{KL}(P||Q)$。因为真实标签的分布 P 是固定的（其熵为常数），优化交叉熵就等价于最小化模型分布 Q 与真实分布 P 之间的 KL 散度。在深度学习分类任务中，P 通常是 One-hot 编码，此时交叉熵退化为 $-\log Q(y_{true})$，即负对数似然。这种形式能够对错误预测施加极大的对数惩罚，促使模型快速收敛，是目前分类模型和语言模型（如 LLM 的 Next-token prediction）最核心的损失函数。

### 案例模拟

面试官追问：在语言模型预训练中，交叉熵损失的值一般在什么范围？如何直观理解这个值？ 回答示例：在 LLM 预训练初期，交叉熵损失通常较大（如 10 左右），随着训练会降至 1.5 到 2.5 之间。直观上，交叉熵与困惑度（Perplexity, PPL）直接相关，$PPL = \exp(CrossEntropy)$。如果 Loss 为 2.0，PPL 约为 7.4，意味着模型在预测下一个词时，平均在 7.4 个候选词中犹豫。在业务中，我们常通过监控 Loss 和 PPL 的下降趋势来判断模型是否正常收敛及是否出现过拟合。

### 61. 三、KL 散度与交叉熵的区别？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 交叉熵等于真实分布的熵加上两者之间的 KL 散度。
- 当真实分布固定时，优化交叉熵与优化 KL 散度完全等价。
- KL散度衡量分布差异，交叉熵衡量编码总长度，侧重点不同。

### 详细解答

结论：KL 散度和交叉熵在数学上紧密相关，公式关系为 $H(P,Q) = H(P) + D_{KL}(P||Q)$，主要区别在于是否包含真实分布的熵。 原理与对比：交叉熵 $H(P,Q)$ 衡量的是用模型分布 Q 编码真实分布 P 的总信息量，而 KL 散度 $D_{KL}(P||Q)$ 衡量的是这种编码带来的“额外”信息量（即差异）。在大多数监督学习任务中，真实标签分布 P（如 One-hot 向量）是固定的，其信息熵 $H(P)$ 为常数 0。因此，对交叉熵求导和对 KL 散度求导的结果完全一致，优化两者是等价的。但在知识蒸馏或软标签训练中，真实分布 P 是教师模型的输出，其熵不再是 0，此时直接使用 KL 散度能更直观地反映学生模型与教师模型分布的拟合程度。

### 案例模拟

面试官追问：在知识蒸馏中，为什么通常使用 KL 散度而不是交叉熵作为损失函数？ 回答示例：在知识蒸馏中，教师模型输出的是软标签（Soft targets），其本身包含丰富的类间关系信息，熵不为零。如果使用交叉熵，损失值会包含教师模型输出的固有熵，这部分是不可优化的常数，可能导致 Loss 数值偏大，不利于直观评估蒸馏效果。使用 KL 散度可以剔除这部分常数，直接反映学生模型与教师模型分布的相对差异，使得 Loss 降到 0 时代表完全拟合，便于工程上的监控和调试。

### 62. 四、多任务学习各loss差异过大怎样处理？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 可以采用手动静态加权，根据任务重要性或量级调整权重。
- 使用动态权重调整策略，如基于不确定性加权或梯度归一化。
- 采用梯度投影（PCGrad）避免不同任务梯度方向冲突。

### 详细解答

结论：多任务学习中 Loss 差异过大会导致模型偏向大 Loss 任务，通常通过动态权重分配或梯度干预来解决。 原理与工程权衡：不同任务的 Loss 量级可能相差数倍，直接相加会使小 Loss 任务被忽略。基础方案是手动调参赋予不同权重，但成本高且不鲁棒。进阶方案包括：1. 基于同方差不确定性（Uncertainty Weighting），让模型自动学习各任务的噪声方差，Loss 波动大的任务权重降低；2. 动态任务优先级（DWA），根据各任务 Loss 的下降速率动态调整权重；3. 梯度级别的干预，如 GradNorm（统一各任务梯度范数）或 PCGrad（当任务梯度夹角为钝角时进行投影，消除冲突）。工程上，通常先尝试简单的静态加权或不确定性加权，若效果不佳再引入复杂的梯度操作。

### 案例模拟

面试官追问：在推荐系统的多任务模型（如同时预测点击和转化）中，你是如何处理 Loss 差异的？ 回答示例：在我们的 CTR/CVR 多任务模型中，点击任务的样本量大且 Loss 相对平稳，而转化任务样本稀疏且 Loss 波动大。我们最初尝试了固定权重，但效果不稳定。后来引入了基于不确定性的动态加权策略（Uncertainty Weighting），将各任务的权重设为可学习参数。模型在训练过程中自动降低了高方差 CVR 任务的权重，使得两个任务的梯度量级趋于一致，最终不仅免去了繁琐的调参工作，双任务的 AUC 也都有了显著提升。

### 63. 五、分类问题为什么用交叉熵损失函数不用均方误差（MSE）？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- MSE 配合 Sigmoid/Softmax 容易导致梯度消失问题。
- 交叉熵的对数特性能够抵消激活函数的导数，保持梯度稳定。
- 交叉熵更符合分类任务中概率分布度量的最大似然估计逻辑。

### 详细解答

结论：分类问题使用交叉熵而非 MSE，主要是因为交叉熵能有效避免梯度消失，且在数学上等价于最大似然估计。 原理与对比：在分类任务中，输出层通常使用 Sigmoid 或 Softmax 激活函数。如果使用 MSE 损失，其梯度计算中会包含激活函数的导数项（如 $\sigma(x)(1-\sigma(x))$）。当预测值严重错误（如预测为 0 但真实为 1）时，激活函数处于饱和区，导数趋近于 0，导致模型参数几乎无法更新（梯度消失）。而交叉熵损失包含对数操作，求导时恰好能与 Sigmoid/Softmax 的指数形式抵消，最终梯度正比于预测值与真实值的误差（$p - y$）。这意味着误差越大，梯度越大，模型收敛越快。此外，交叉熵直接衡量概率分布的差异，物理意义更契合分类任务。

### 案例模拟

面试官追问：在什么特殊情况下，分类问题可能会考虑使用 MSE 损失？ 回答示例：虽然交叉熵是主流，但在某些特定场景下 MSE 也有应用。例如在标签噪声极大的分类任务中，交叉熵对错误标签的惩罚是对数级的，可能会导致模型过度拟合噪声数据；而 MSE 的惩罚是平方级的，相对更平缓，具有一定的抗噪鲁棒性。另外，在知识蒸馏中，如果直接对教师和学生的 Logits（未经过 Softmax）进行拟合，有时也会使用 MSE 损失来拉近两者的绝对数值差异。

### 64. 六、什么是信息增益？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 信息增益是划分数据集前后，系统信息熵的减少量。
- 常用于决策树算法（如 ID3）中作为特征选择的评价指标。
- 信息增益越大，说明使用该特征划分数据带来的纯度提升越明显。

### 详细解答

结论：信息增益（Information Gain）衡量的是在已知某个特征的条件下，数据集不确定性（信息熵）减少的程度。 原理与工程权衡：在信息论中，熵代表随机变量的不确定性。对于分类任务，数据集的经验熵 $H(D)$ 反映了类别分布的混乱程度。当我们根据某个特征 $A$ 将数据集划分为多个子集时，可以计算条件熵 $H(D|A)$。信息增益即为 $IG(D, A) = H(D) - H(D|A)$。在构建决策树（如 ID3 算法）时，我们总是优先选择信息增益最大的特征进行节点分裂，因为这能最大程度地提高子节点的类别纯度。然而，信息增益存在一个固有缺陷：它天然偏好取值较多的特征（如 ID 字段）。为了解决这个问题，C4.5 算法引入了信息增益比，通过特征本身的熵进行惩罚。

### 案例模拟

面试官追问：为什么信息增益会偏好取值多的特征？在实际业务中如何避免？ 回答示例：如果一个特征（比如用户 ID）每个取值只对应一个样本，那么划分后的子集纯度极高，条件熵为 0，信息增益达到最大。但这显然没有泛化能力。在实际风控业务中构建树模型时，我们通常不直接使用 ID3 算法，而是使用基于信息增益比的 C4.5，或者基于基尼指数（Gini Impurity）的 CART 算法。此外，在特征工程阶段，我们会对高基数类别特征（如城市、设备号）进行 Target Encoding 或频数截断，从根本上避免模型对多值特征的过拟合。

### 65. 七、多分类的分类损失函数(Softmax)？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 多分类通常使用 Softmax 激活函数结合交叉熵损失。
- Softmax 将模型输出的 logits 转换为和为 1 的概率分布。
- 损失函数仅对真实类别对应的预测概率计算负对数似然。

### 详细解答

结论：多分类任务的标准损失函数是 Softmax 交叉熵损失（Categorical Cross-Entropy），它将模型输出归一化为概率并计算与真实标签的对数损失。 原理与工程权衡：在多分类中，模型最后一层输出未归一化的得分（Logits）。首先通过 Softmax 函数 $p_i = \exp(z_i) / \sum \exp(z_j)$ 将其映射为概率分布，突出最大值并抑制其他值。接着计算交叉熵损失 $L = -\sum y_i \log(p_i)$。由于真实标签 $y$ 通常是 One-hot 编码，公式简化为 $L = -\log(p_{true})$，即只关注正确类别的预测概率。在深度学习框架（如 PyTorch 的 CrossEntropyLoss）中，通常将 Softmax 和 Log-Likelihood 合并计算，利用 Log-Sum-Exp 技巧避免指数运算带来的数值溢出问题，从而保证训练的数值稳定性。

### 案例模拟

面试官追问：如果多分类任务中存在类别极度不平衡的情况，直接使用 Softmax 交叉熵会有什么问题？如何改进？ 回答示例：在类别极度不平衡时（如医疗诊断），多数类样本会主导 Loss 的计算，导致模型对少数类的预测能力极差。在实际项目中，我们通常会采用加权交叉熵（Weighted Cross-Entropy），为少数类赋予更大的权重。更进一步，我们会使用 Focal Loss，在交叉熵的基础上增加一个调制系数 $(1-p_t)^\gamma$，不仅解决类别不平衡，还能让模型在训练时自动降低易分样本的权重，强迫模型关注那些难以分类的少数类样本。

### 66. 八、softmax和交叉熵损失怎么计算，二值交叉熵呢？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- Softmax 计算为 exp(z_i) / sum(exp(z_j))，交叉熵为 -log(p_true)。
- 二值交叉熵（BCE）用于二分类，公式为 -[y log(p) + (1-y) log(1-p)]。
- BCE 通常配合 Sigmoid 激活函数使用，独立计算每个类别的概率。

### 详细解答

结论：Softmax 交叉熵用于互斥的多分类任务，而二值交叉熵（BCE）用于二分类或多标签分类任务。 原理与对比：对于多分类，先对 Logits 进行 Softmax 操作得到概率分布，然后计算真实类别对应概率的负对数，即 $Loss = -\log(\frac{\exp(z_{true})}{\sum \exp(z_j)})$。这种机制下，各类别的概率相互竞争，和为 1。而对于二分类或多标签分类（一个样本可同时属于多个类），使用 Sigmoid 函数 $p = 1 / (1 + \exp(-z))$ 将每个 Logit 独立映射到 0-1 之间。BCE 损失公式为 $Loss = -[y \log(p) + (1-y) \log(1-p)]$，它分别惩罚正样本预测不到 1 和负样本预测不到 0 的情况。工程上，PyTorch 提供了 CrossEntropyLoss 和 BCEWithLogitsLoss，两者都在底层融合了激活函数以提升数值稳定性。

### 案例模拟

面试官追问：在多标签文本分类任务中（比如一篇文章既属于“科技”又属于“财经”），应该用哪种损失函数？ 回答示例：在多标签分类任务中，类别之间不再是互斥关系，因此不能使用 Softmax 交叉熵。我们会将模型输出层的激活函数改为 Sigmoid，并使用二值交叉熵（BCEWithLogitsLoss）。这样模型会对每个类别独立进行二分类判断，计算各自的 BCE 损失然后求平均。在业务实践中，如果某些标签出现频率极低，我们还会在 BCE 的基础上引入正负样本权重（pos_weight）来缓解多标签场景下的正负样本不平衡问题。

### 67. 九、如果softmax的e次方超过float的值了怎么办？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / LLMs 损失函数篇 / 未知](https://articles.zsxq.com/id_q0ajjlbc8493.html)

### 基础知识补充

- 这会导致数值溢出（Overflow），产生 NaN 结果。
- 常用 Log-Sum-Exp 技巧，减去输入向量中的最大值。
- 减去最大值不改变 Softmax 的最终概率分布结果。

### 详细解答

结论：当 Softmax 输入的 Logits 过大时，指数运算会导致浮点数溢出（Overflow）。工程上通过减去 Logits 中的最大值来保证数值稳定性。 原理与工程实践：Softmax 的公式为 $p_i = \exp(z_i) / \sum \exp(z_j)$。如果 $z_i$ 很大（例如超过 88，对于 float32 来说 $\exp(88)$ 就会溢出），计算结果会变成 inf，进而导致除法出现 NaN。为了解决这个问题，利用指数函数的性质，可以在分子分母同乘一个常数 $C = \exp(-\max(z))$。公式变为 $p_i = \exp(z_i - \max(z)) / \sum \exp(z_j - \max(z))$。这样处理后，输入的最大值变为 0，其指数为 1，其余值均为负数，指数在 0 到 1 之间，彻底避免了上溢出。同时，即使出现下溢出（趋于 0），分母至少为 1，也不会导致除以零的错误。现代深度学习框架底层均已内置此优化。

### 案例模拟

面试官追问：除了上溢出，Softmax 还会遇到下溢出问题吗？如何解决？ 回答示例：是的，如果 Logits 全是极小的负数，减去最大值后虽然避免了上溢出，但计算交叉熵时需要对 Softmax 结果取对数。如果某个 $p_i$ 下溢出变为 0，$\log(0)$ 会导致负无穷（-inf）。为了解决这个问题，框架在计算交叉熵时通常不显式计算 Softmax，而是直接使用 Log-Sum-Exp 技巧计算 $\log(p_i) = z_i - \max(z) - \log(\sum \exp(z_j - \max(z)))$，从而在数学层面上完全避开了 $\log(0)$ 的风险，保证了反向传播的稳定性。

## 相似度函数篇

### 68. 一、除了cosin还有哪些算相似度的方法

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 相似度函数篇 / 未知](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

### 基础知识补充

- 欧氏距离（L2距离），衡量空间中两点的绝对直线距离。
- 曼哈顿距离（L1距离），衡量在标准坐标系上的绝对轴距总和。
- 杰卡德相似度（Jaccard），用于衡量两个集合的交并比。

### 详细解答

结论：除了余弦相似度（Cosine），常用的相似度/距离度量还包括欧氏距离、曼哈顿距离、杰卡德相似度和皮尔逊相关系数等。 原理与对比：余弦相似度关注向量方向的差异，对绝对数值大小不敏感，常用于文本向量匹配。欧氏距离（Euclidean Distance）衡量两点间的绝对距离，受向量模长影响大，适用于需要考虑量级差异的场景（如图像像素对比）。如果对向量进行 L2 归一化，欧氏距离的平方与余弦相似度是等价的。曼哈顿距离计算各维度差值的绝对值之和，对异常值比欧氏距离更鲁棒。杰卡德相似度（Jaccard Similarity）主要用于离散集合，计算交集与并集的比值，常用于推荐系统中的用户行为重合度分析。皮尔逊相关系数则相当于中心化后的余弦相似度。

### 案例模拟

面试官追问：在向量数据库（如 Milvus/Faiss）进行大规模检索时，为什么通常首选内积（Inner Product）或余弦相似度，而不是欧氏距离？ 回答示例：在实际的 RAG 或推荐系统中，文本/图像特征通常会被 L2 归一化。归一化后，内积、余弦相似度和欧氏距离的排序结果是完全一致的。但从计算效率来看，内积只需要进行乘加运算，而欧氏距离需要计算平方差再求和，计算指令更多。此外，现代 CPU/GPU 对矩阵乘法（即批量内积）有极高的硬件级优化（如 Tensor Core），因此使用内积进行相似度检索在工程上速度最快，吞吐量最高。

### 69. 二、了解对比学习嘛？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 相似度函数篇 / 未知](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

### 基础知识补充

- 对比学习是一种自监督学习范式，旨在拉近正样本，推远负样本。
- 核心依赖于数据增强构建正样本对，以及 InfoNCE 等对比损失函数。
- 经典模型包括 SimCLR、MoCo 和 CLIP，广泛用于表征学习。

### 详细解答

结论：对比学习（Contrastive Learning）通过在特征空间中“拉近相似样本（正样本）、推远不相似样本（负样本）”来学习数据的通用表征，无需人工标注。 原理与工程权衡：其核心思想是构建代理任务。以 SimCLR 为例，对同一图像进行两次不同的随机数据增强（如裁剪、翻转、颜色抖动）得到正样本对，同一批次内的其他图像均视为负样本。模型通过优化 InfoNCE 损失，最大化正样本间的互信息。对比学习的难点在于对负样本数量和质量的极高要求：Batch Size 越大，负样本越多，表征学习越好，但这会带来巨大的显存压力。为此，MoCo 引入了动量编码器和内存队列，解耦了负样本数量与 Batch Size 的限制，成为工程上极具性价比的解决方案。

### 案例模拟

面试官追问：在 NLP 领域（如句向量训练），对比学习是如何构建正负样本的？ 回答示例：在 NLP 领域，经典的对比学习模型是 SimCSE。在无监督 SimCSE 中，正样本的构建非常巧妙：将同一个句子输入 BERT 两次，利用 Transformer 内部标准的 Dropout 机制作为“数据增强”，得到两个不同的句向量作为正样本对，批次内其他句子作为负样本。在有监督场景下，我们会利用 NLI 数据集，将蕴含关系的句子对作为正样本，矛盾关系的句子作为困难负样本（Hard Negatives），这能显著提升句向量在语义匹配任务中的区分度。

### 70. 三、对比学习负样本是否重要？负样本构造成本过高应该怎么解决？

- 主标签：LLM基础
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 相似度函数篇 / 未知](https://articles.zsxq.com/id_wp25j5xr8ocw.html)

### 基础知识补充

- 负样本极其重要，防止模型陷入所有输出相同的“模式崩溃”。
- 负样本数量越多、难度越大（Hard Negatives），表征边界越清晰。
- 可通过内存队列（MoCo）或批内负样本（In-batch）降低构造成本。

### 详细解答

结论：负样本在对比学习中至关重要，它提供了排斥力，防止模型将所有输入映射到同一点（模式崩溃）。解决负样本构造成本高的方法主要包括内存队列机制和无负样本架构。 原理与工程权衡：InfoNCE 损失的分母是所有负样本相似度的指数和，负样本越多，模型越能学到细粒度的特征区分。但大 Batch Size 会耗尽显存。工程解决方案有：1. 动量队列（MoCo）：维护一个包含大量历史特征的队列作为负样本池，用动量更新的编码器保证特征一致性，从而在小 Batch Size 下实现海量负样本。2. In-batch Negatives：直接复用当前批次内的其他样本作为负样本，常用于双塔检索模型。3. 放弃负样本：如 BYOL 和 SimSiam 算法，通过引入非对称的网络结构（预测器）和停止梯度（Stop-Gradient）操作，在完全没有负样本的情况下也能避免模式崩溃，极大降低了显存开销。

### 案例模拟

面试官追问：在推荐系统的召回双塔模型中，如果只使用随机采样的负样本会有什么问题？如何优化？ 回答示例：在推荐召回中，如果只使用全局随机采样的负样本（Easy Negatives），模型很容易就能区分正负样本，导致 Loss 迅速下降但实际排序能力很差。为了提升模型对相似物品的区分度，我们在业务中会引入“困难负样本（Hard Negatives）”。例如，将用户曝光但未点击的物品，或者与正样本属于同类目但用户未交互的物品作为负样本。通常我们会保持 Easy 和 Hard 负样本的比例（如 4:1），既保证训练稳定，又提升模型的细粒度排序能力。
