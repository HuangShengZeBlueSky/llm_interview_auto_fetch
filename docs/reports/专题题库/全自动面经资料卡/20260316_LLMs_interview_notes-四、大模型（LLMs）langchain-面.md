# LLMs_interview_notes：四、大模型（LLMs）langchain 面

> 来源分组：LLMs_interview_notes
> 本页题目数：10
> 每题均包含基础知识补充、详细解答和案例模拟。

## 大模型（LLMs）langchain 面

### 1. 一、什么是 LangChain?

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- 核心概念：LangChain是开发大语言模型应用的开源框架。
- 关键模块：包含Model、Prompt、Memory、Chain、Agent等组件。
- 工程价值：通过标准化接口降低LLM应用开发门槛与切换成本。

### 详细解答

结论：LangChain 是一个用于开发由大型语言模型（LLMs）驱动的应用程序的开源框架，它通过提供标准化的组件和接口，极大地简化了构建复杂AI应用的流程。 原理与模块：其核心在于将LLM的交互过程进行抽象和封装。主要模块包括：Models（统一不同大模型API的调用接口）、Prompts（管理和优化提示词模板）、Memory（为无状态的LLM提供上下文记忆能力）、Retrieval（连接外部数据源实现RAG）、Chains（将多个组件串联成固定工作流）以及Agents（赋予LLM使用工具和自主决策的能力）。 对比与工程权衡：相比于直接调用原生API，LangChain的优势在于高度的模块化和生态丰富度，开发者可以轻松实现模型切换或接入外部工具。但在工程实践中，其高度封装也可能带来调试困难、运行开销增加的问题。对于简单任务，直接调用API可能更高效；而对于需要复杂逻辑编排、多工具协同的系统，LangChain则是极佳的脚手架。

### 案例模拟

面试官追问：“在实际业务中，LangChain的Chain和Agent有什么本质区别？” 回答示例：“Chain是硬编码的确定性工作流，执行步骤和顺序在代码编写时就已经固定，适合流程明确的任务，比如先总结文章再翻译。而Agent是动态的，它利用LLM作为推理引擎，根据当前输入和环境反馈，自主决定调用哪些工具及执行顺序。在我们的客服系统中，我们用Chain处理标准FAQ，用Agent处理需要查询订单、调用退款接口的复杂客诉。”

### 2. 二、LangChain 包含哪些 核心概念？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- 模型与提示词：负责大语言模型交互与输入指令的模板化管理。
- 链与记忆：实现多步任务逻辑串联与多轮对话上下文状态保持。
- 代理与检索：赋予模型工具调用能力与外部私有知识库的接入。

### 详细解答

LangChain 的核心概念围绕大模型应用开发的生命周期，主要包含模型（Models）、提示词（Prompts）、记忆（Memory）、检索（Retrieval）、链（Chains）和代理（Agents）六大核心模块。 原理与作用：1. 模型与提示词：提供统一接口适配各类底层 LLM，并通过模板化技术动态生成指令，提升复用性。2. 记忆组件：由于大模型本身无状态，该模块用于在多轮交互中存储历史对话，维持上下文连贯性。3. 检索系统：包含文档加载、文本切分、向量化及向量数据库，是构建 RAG 应用的基础，有效解决模型知识滞后和幻觉问题。4. 链与代理：链将多个组件硬编码串联成固定工作流；代理则利用 LLM 的推理能力作为大脑，动态决定执行步骤并调用外部工具。 工程权衡：在实际开发中，链（Chains）适合逻辑确定、稳定性要求高的场景；代理（Agents）灵活性极强，但受限于底层模型的推理能力，容易出现死循环或调用失败，通常需要配合严格的异常捕获与重试机制。

### 案例模拟

面试官追问：在实际业务中，你是如何选择使用 Chain 还是 Agent 的？ 回答：这主要取决于任务的确定性。在我们的智能客服项目中，对于常规的退换货流程，步骤是固定的（查订单->验规则->退款），我会使用 Chain 将这些步骤硬编码串联，确保执行的绝对稳定性和低延迟。而对于开放式的复杂数据查询，由于无法预知需要调用哪些分析工具，我会使用 Agent，让大模型自主规划调用 SQL 和 Python 工具。虽然 Agent 延迟较高，但能极大拓展业务边界。

### 3. 2.1 LangChain 中 Components and Chains 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Components 是构建大语言模型应用的基础模块化组件。
- Chains 是将多个组件按特定顺序组合执行的端到端工作流。
- 常见组件包括提示词模板、语言模型接口和输出解析器等。

### 详细解答

在 LangChain 中，Components（组件）和 Chains（链）是构建 LLM 应用的核心抽象。结论上，Components 是提供单一功能的积木，而 Chains 是将积木拼接以完成特定任务的流水线。原理上，Components 涵盖与大模型交互的独立环节，如 Models（封装 LLM API）、Prompts（格式化提示词）和 Memory（上下文记忆），它们高度解耦且可复用。Chains 则是对组件的编排，例如基础的 LLMChain 将 PromptTemplate、LLM 和 OutputParser 串联，实现“输入变量->生成提示词->模型推理->解析输出”的自动化流程。工程权衡方面，直接调用 Components 适合简单任务或极致自定义，但代码冗余；使用 Chains 能大幅降低复杂任务的开发门槛。当前，传统 Chains 正逐渐被更灵活的 LCEL 替代，以更好地支持动态路由和流式输出。

### 案例模拟

面试官追问：“在实际项目中，如果现有的 Chains 无法满足复杂的业务逻辑，你会怎么做？” 回答示例：“如果内置的 Chains（如 RetrievalQAChain）不够灵活，我通常有两种方案。第一种是继承 Chain 基类自定义，重写 _call 方法来硬编码业务逻辑。第二种，也是目前推荐的做法，是使用 LCEL（LangChain表达式语言）。通过管道符将各个 Components 组合起来，不仅代码更简洁，还能原生支持异步调用、流式输出和 Fallback 机制，极大提升了工程鲁棒性。”

### 4. 2.2 LangChain 中 Prompt Templates and Values 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Prompt Template 是用于动态生成提示词的参数化模板类。
- Prompt Value 是模板实例化后的对象，可转为字符串或消息。
- 包含 FewShot 等模板类型，满足复杂工程提示词构建需求。

### 详细解答

在 LangChain 中，Prompt Templates（提示词模板）和 Prompt Values（提示词值）是构建和管理大模型输入的两个核心概念。结论是：Templates 是定义提示词结构的“模具”，而 Values 是注入具体数据后生成的“成品”，负责适配不同类型的大模型接口。 原理与对比：Prompt Template 允许开发者通过占位符定义提示词的骨架。它不仅支持简单的字符串替换，还支持针对聊天模型的结构化消息模板（ChatPromptTemplate），如区分 System、Human 等角色。当向 Template 传入具体参数后，就会生成一个 Prompt Value。Prompt Value 的核心作用是解耦：因为不同大模型 API 接收的输入格式不同（纯文本或消息列表），Prompt Value 提供统一接口（如 to_string 和 to_messages），在底层自动完成格式转换。 工程权衡：硬编码提示词会导致代码难以维护。使用 Templates 可实现提示词的版本控制、复用和动态组装，极大地提升了应用的可扩展性和跨模型兼容性。

### 案例模拟

面试官追问：在构建 RAG（检索增强生成）应用时，你会如何使用 Prompt Templates？ 回答示例：在 RAG 项目中，我会使用 ChatPromptTemplate 来构建输入。首先定义 System Message 模板规定 AI 人设；然后定义 Human Message 模板，包含 {context}（检索文档）和 {question}（用户问题）占位符。运行时，检索器返回的文档动态填充到 {context}，生成最终的 Prompt Value 传给 LLM，既保证结构清晰，又能缓解提示词注入风险。

### 5. 2.3 LangChain 中 Example Selectors 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：近似题匹配：DeepLearing-Interview-Awesome-2024_手撕代码专题:CodeAnything/Reference.md
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- 先讲规划与调用
- 再讲状态维护
- 补异常处理策略

### 详细解答

compute_error_for_line_given_points(1, 2, [[3,6],[6,9],[12,18]]) 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。

### 案例模拟

面试表达可以这样组织：先用一句话回答“2.3 LangChain 中 Example Selectors 是什么？”的核心结论，再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。

### 6. 2.4 LangChain 中 Output Parsers 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- 先定义核心概念
- 再解释关键机制
- 最后补工程取舍

### 详细解答

这道题建议先给一句结论：2.4 LangChain 中 Output Parsers 是什么？ 的回答要围绕定义、核心原理和工程权衡展开。如果你有项目经验，可以补充自己实际做过的方案、踩过的坑和最终指标变化；如果没有项目经验，就用模型结构、训练目标、推理成本、效果收益这四个角度来组织回答。

### 案例模拟

面试表达可以这样收尾：先说清楚这个方法解决了什么问题，再补一句它的代价是什么，最后用“如果让我在项目里落地，我会先做小规模验证”来体现工程意识。

### 7. 2.5 LangChain 中 Indexes and Retrievers 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Indexes 负责文档加载、文本分割与向量化存储。
- Retrievers 是根据用户查询返回相关文档的标准化接口。
- 两者结合构成了检索增强生成（RAG）的核心数据管道。

### 详细解答

在 LangChain 中，Indexes（索引）和 Retrievers（检索器）是构建 RAG（检索增强生成）系统的两大核心组件，分别负责数据的“存”与“取”。 Indexes 是一个数据处理管道，包含文档加载（Document Loaders）、文本切分（Text Splitters）、向量化（Embeddings）以及存储到向量数据库（VectorStores）的过程，目的是将非结构化数据转化为大模型可高效检索的结构化格式。Retrievers 则是独立于存储的检索接口，它接收用户的自然语言查询，并返回最相关的文档列表。 在工程实践中，单纯的向量相似度往往不够。因此 LangChain 提供了高级检索器，如 MultiQueryRetriever（多查询扩展提高召回率）、ContextualCompressionRetriever（上下文压缩减少冗余 token 消耗）以及 EnsembleRetriever（混合检索）。实际开发中，需根据延迟要求、成本预算和召回率目标，权衡选择合适的索引策略与高级检索器组合。

### 案例模拟

面试官追问：如果检索出来的文档包含太多无关信息，导致大模型上下文超限，你会怎么优化 Retriever？ 回答示例：我会引入 LangChain 的 ContextualCompressionRetriever（上下文压缩检索器）。首先，使用基础的向量检索器召回初步相关的文档；然后，将这些文档传递给一个基于 LLM 的文档压缩器。压缩器会根据用户的原始查询，提取文档中真正相关的片段，剔除冗余信息。这样不仅能有效避免上下文窗口超限，还能降低大模型的 Token 消耗，从而提升最终生成的准确性。

### 8. 2.6 LangChain 中 Chat Message History 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- 它是LangChain中管理多轮对话上下文的核心基础组件。
- 支持内存存储及Redis、MongoDB等持久化数据库后端。
- 通过统一接口封装，便于与各类Memory记忆模块无缝集成。

### 详细解答

Chat Message History 是 LangChain 框架中负责捕获、存储和检索人类用户与 AI 模型之间对话记录的基础抽象组件。大模型本身是无状态的，该组件赋予了模型“记忆”能力。在多轮对话中，每次交互的 HumanMessage 和 AIMessage 都会被追加到历史记录中。当发起下一次请求时，系统会将这些历史消息与当前提示词拼接，一并发送给大模型，从而让模型理解上下文。LangChain 提供了多种实现。最基础的是基于内存的 ChatMessageHistory，读写速度极快，但进程重启后数据会丢失，仅适用于简单测试。在生产环境中，通常会使用基于 Redis、PostgreSQL 或 MongoDB 的持久化实现。工程实践中需要权衡上下文窗口限制与存储成本，通常会结合 ConversationSummaryMemory 或窗口截断策略，避免历史记录过长导致 Token 超出限制或推理成本激增。

### 案例模拟

面试官追问：在高并发客服系统中，你会如何设计 Chat Message History 的存储方案？ 回答：在生产级高并发场景下，我会选择 RedisChatMessageHistory 作为后端，利用 Redis 的高性能读写满足实时对话需求，并为每个 Session ID 设置合理的过期时间（如24小时）以自动清理冷数据。同时，为防止单次请求 Token 溢出，我会引入窗口记忆模块仅保留最近5轮对话，或用异步任务定期将长对话摘要化，以控制 API 成本。

### 9. 2.7 LangChain 中 Agents and Toolkits 是什么？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 大模型（LLMs）langchain 面 / 未知](https://articles.zsxq.com/id_ve2dgaiqrjzv.html)

### 基础知识补充

- Agent是大脑，负责根据输入决定调用哪些工具
- Toolkit是工具包，包含特定任务所需的工具集合
- 核心机制是ReAct模式，结合推理与行动实现任务

### 详细解答

结论：在LangChain中，Agent（代理）是利用大语言模型作为推理引擎来决定行动策略的核心组件，而Toolkit（工具包）则是为Agent提供执行特定任务所需的一系列工具集合。 原理与工程权衡：Agent通过ReAct（Reasoning and Acting）等框架，将复杂任务拆解，并根据当前状态动态选择工具。Toolkit将相关工具（如数据库查询、API调用、文件操作）打包，方便Agent直接加载使用。例如SQL Database Toolkit包含了获取表结构、执行查询等工具。在工程实践中，将工具按领域封装成Toolkit，能有效降低Agent的认知负荷，避免工具过多导致LLM上下文溢出或选择错误。同时，合理设计工具的描述（Description）对Agent的准确调用至关重要。

### 案例模拟

面试官追问：如果Agent频繁调用错误的工具，你会如何优化？ 回答：首先，我会检查工具的描述（Description）是否足够清晰明确，避免歧义。其次，可以通过Few-shot Prompting在系统提示词中提供正确调用的示例。如果工具数量过多，我会引入工具检索机制（Tool Retrieval），根据用户Query动态召回最相关的Top-K个工具供Agent选择，从而降低模型的决策难度。

## 多轮对话中让AI保持长期记忆的8种优化方式篇

### 10. 二、Agent 如何获取上下文对话信息？

- 主标签：Agent与工具调用
- 来源条数：1
- 答案生成方式：模型自动补全
- 来源：[LLMs_interview_notes / 多轮对话中让AI保持长期记忆的8种优化方式篇 / 未知](https://articles.zsxq.com/id_3qgicwcwzjpi.html)

### 基础知识补充

- 通过Memory组件存储和检索历史对话记录
- 常用机制包括Buffer Memory和Summary Memory
- 历史信息通常作为Prompt的一部分注入到当前请求中

### 详细解答

结论：Agent主要通过内存（Memory）组件来获取和管理上下文对话信息，将历史交互记录动态注入到大模型的Prompt中。 原理与工程权衡：在LangChain等框架中，Memory组件负责拦截输入输出并保存。常见的实现有ConversationBufferMemory（保存完整历史）和ConversationSummaryMemory（利用LLM对历史进行摘要）。工程实践中，直接拼接完整历史会导致Token消耗过大且容易触发上下文长度限制。因此，通常采用滑动窗口（仅保留最近N轮对话）、向量数据库检索（将历史对话向量化，根据当前Query检索相关历史）或摘要机制来权衡上下文丰富度与Token成本。这些机制确保Agent在多轮对话中保持连贯性。

### 案例模拟

面试官追问：在超长多轮对话场景下，如何设计Memory机制？ 回答：在超长对话场景中，我会采用分层记忆架构。短期记忆使用滑动窗口保留最近3-5轮的原始对话，保证即时上下文的准确性；长期记忆则将历史对话进行实体抽取和摘要，并存入向量数据库。当用户提问时，Agent不仅读取短期记忆，还会通过向量检索召回相关的长期记忆片段，拼接进Prompt中，从而在控制Token消耗的同时实现长期记忆。
